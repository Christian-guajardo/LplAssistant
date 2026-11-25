#!/usr/bin/env python3
"""Writes what the GGUF converter of LplAssistant is checked against, from its primary sources.

Run from the repository's root:

    uv run --no-project --with clang-format==18.1.3 python -I tools/gguf_reference.py <command>

Commands:
    fixtures  write tests/convert/fixtures/gguf_reference.hpp: GPT-2's byte-level alphabet, the rotary tables of
              SmolLM2 (base 100000) and of LLaMA (base 10000) over heads of 64 channels in 60-digit decimal
              arithmetic, and the first base past 10000 whose table puts a word within 10^-6 of a rounding tie
    gguf      build SmolLM2-135M-Instruct's GGUF, bf16 and q8_0, from its official safetensors with llama.cpp b3775's
              own convert_hf_to_gguf.py, and check that each has the SHA-256 pinned below
    check     compute the fixture again and fail when it differs from the committed one, writing nothing

The inputs are downloaded once into --cache (~/.cache/lplassistant/gguf-reference), each source into a directory of its
own, and refused when their SHA-256 is not the pinned one. llama.cpp's archive is extracted with tarfile's "data"
filter. The converter runs offline under uv, on the Python packages of tools/gguf_requirements.txt, each pinned by
version and hash; it reads model.safetensors through safetensors' safe_open, never a pickle. `gguf` prints, for each
file, its bytes, its SHA-256, and the wall time and peak resident memory llama.cpp's converter took.

GPT-2's alphabet is built as openai/gpt-2's encoder.py builds it (bytes_to_unicode, from byte_pair_reference.py, which
reads tokenizer.json through it).
"""

import argparse
import decimal
import hashlib
import importlib.util
import os
import pathlib
import resource
import subprocess
import sys
import tarfile
import time

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
FIXTURE_HEADER = REPOSITORY / "tests" / "convert" / "fixtures" / "gguf_reference.hpp"
REQUIREMENTS = REPOSITORY / "tools" / "gguf_requirements.txt"

sys.dont_write_bytecode = True
_SPEC = importlib.util.spec_from_file_location("byte_pair_reference", REPOSITORY / "tools" / "byte_pair_reference.py")
byte_pair_reference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(byte_pair_reference)

SMOLLM2 = byte_pair_reference.SMOLLM2
SMOLLM2_DIRECTORY = "smollm2-135m-instruct-" + byte_pair_reference.SMOLLM2_COMMIT[:7]
LLAMA_CPP_COMMIT = "503147a9f9d195d6a14e7c998df23b6eb61f2bae"
LLAMA_CPP_DIRECTORY = "llama.cpp-" + LLAMA_CPP_COMMIT[:7]
LLAMA_CPP_ARCHIVE = f"llama.cpp-{LLAMA_CPP_COMMIT}.tar.gz"
MODEL_FILES = ("config.json", "generation_config.json", "model.safetensors", "special_tokens_map.json",
               "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
byte_pair_reference.INPUTS.update({
    "model.safetensors": (SMOLLM2 + "model.safetensors",
                          "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"),
    "special_tokens_map.json": (SMOLLM2 + "special_tokens_map.json",
                                "2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52"),
    "vocab.json": (SMOLLM2 + "vocab.json", "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79"),
    "merges.txt": (SMOLLM2 + "merges.txt", "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510"),
    LLAMA_CPP_ARCHIVE: (f"https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{LLAMA_CPP_COMMIT}",
                        "b01b4485762785dc421b9df4f9f67506068e07cb4d6756db40840eda43c5c7e5"),
})

# Two runs of the converter give these bytes, so the chain from the official weights to an image is pinned end to end.
GGUF_DIGESTS = {
    "bf16": "472ef778b38554e4fd3926e57df5cbe337ca440925e40e910ed368709557debe",
    "q8_0": "53c47dfb6d63ebd3266dd9984757537fa2bd38a44189766f11173a1fc675aba4",
}

ROTARY_DIMENSIONS = 64
SMOLLM2_BASE = 100000
LLAMA_BASE = 10000
NEAR_TIE = decimal.Decimal("1e-6")
decimal.getcontext().prec = 60
PI = decimal.Decimal("3.14159265358979323846264338327950288419716939937510582097494459")


def turns_per_position(base, dimensions, pair):
    """base^(-2 pair / dimensions) / (2 pi) * 2^32, exactly to 60 digits: how far pair @p pair turns per position."""
    theta = (decimal.Decimal(base).ln() * decimal.Decimal(-2 * pair) / decimal.Decimal(dimensions)).exp()
    return theta / (2 * PI) * decimal.Decimal(2) ** 32


def rotary_table(base, dimensions):
    """The words of a rotary table: each pair's turns per position, rounded half up, as the converter writes them."""
    return [int(turns_per_position(base, dimensions, pair).to_integral_value(rounding=decimal.ROUND_HALF_UP))
            for pair in range(dimensions // 2)]


def distance_to_tie(value):
    """How far @p value lies from the nearest x.5."""
    return abs(value - value.to_integral_value(rounding=decimal.ROUND_FLOOR) - decimal.Decimal("0.5"))


def first_near_tie(after, dimensions):
    """The first integer base past @p after whose table puts a word within NEAR_TIE of a tie, its pair and distance.

    A double screens each base, so that only its candidates are worked in 60 digits."""
    base = after
    while True:
        base += 1
        for pair in range(dimensions // 2):
            screened = base ** (-2.0 * pair / dimensions) / (2.0 * 3.141592653589793) * 2.0 ** 32
            if abs(screened - int(screened) - 0.5) > 1e-4:
                continue
            distance = distance_to_tie(turns_per_position(base, dimensions, pair))
            if distance < NEAR_TIE:
                return base, pair, distance


def fixture_header():
    """The text of tests/convert/fixtures/gguf_reference.hpp."""
    to_character, _ = byte_pair_reference.byte_level_alphabet()
    code_points = [ord(to_character[byte]) for byte in range(256)]
    smollm2 = rotary_table(SMOLLM2_BASE, ROTARY_DIMENSIONS)
    llama = rotary_table(LLAMA_BASE, ROTARY_DIMENSIONS)
    smollm2_closest = min(distance_to_tie(turns_per_position(SMOLLM2_BASE, ROTARY_DIMENSIONS, pair))
                          for pair in range(ROTARY_DIMENSIONS // 2))
    tie_base, tie_pair, tie_distance = first_near_tie(LLAMA_BASE, ROTARY_DIMENSIONS)
    numbers = byte_pair_reference.numbers
    return f"""/**
 * @file gguf_reference.hpp
 * @brief What the GGUF converter's pure parts are checked against, computed apart from them.
 *
 * Written by tools/gguf_reference.py fixtures: they change there, never here. GPT-2's byte-level alphabet is built as
 * openai/gpt-2's encoder.py builds it (bytes_to_unicode). Each rotary word is base^(-2i/{ROTARY_DIMENSIONS}) / (2 pi) * 2^32,
 * rounded half up, in 60-digit decimal arithmetic; SmolLM2's closest word to a tie lies {smollm2_closest:.6f} of a unit
 * from it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_TESTS_CONVERT_FIXTURES_GGUF_REFERENCE_HPP
#    define LPL_TESTS_CONVERT_FIXTURES_GGUF_REFERENCE_HPP

#    include <cstdint>

namespace gguf_reference {{

/** The character GPT-2's byte-level alphabet writes each byte as, by the byte. */
inline constexpr std::uint32_t kByteLevelCodePoints[256] = {{
{numbers([f"{value}u" for value in code_points], 12)}
}};

/** Channels of a head the tables below turn: SmolLM2's. */
inline constexpr std::uint32_t kRotaryDimensions = {ROTARY_DIMENSIONS}u;

/** Pairs of such a head. */
inline constexpr std::uint32_t kRotaryPairs = {ROTARY_DIMENSIONS // 2}u;

/** SmolLM2's rope_theta. */
inline constexpr std::uint32_t kSmolLm2RotaryBase = {SMOLLM2_BASE}u;

/** How far each pair turns per position at SmolLM2's base, in units of 2^-32 turn. */
inline constexpr std::uint32_t kSmolLm2Turns[kRotaryPairs] = {{
{numbers([f"{value}u" for value in smollm2], 8)}
}};

/** LLaMA's rope_theta. */
inline constexpr std::uint32_t kLlamaRotaryBase = {LLAMA_BASE}u;

/** How far each pair turns per position at LLaMA's base, in units of 2^-32 turn. */
inline constexpr std::uint32_t kLlamaTurns[kRotaryPairs] = {{
{numbers([f"{value}u" for value in llama], 8)}
}};

/** The first integer base past LLaMA's whose table puts a word within 10^-6 of a rounding tie. */
inline constexpr std::uint32_t kNearTieBase = {tie_base}u;

/** The pair whose word lies there, {tie_distance:.3e} of a unit from the tie. */
inline constexpr std::uint32_t kNearTiePair = {tie_pair}u;

}} // namespace gguf_reference

#endif /* LPL_TESTS_CONVERT_FIXTURES_GGUF_REFERENCE_HPP */
"""


def extracted_llama_cpp(cache):
    """llama.cpp b3775's source tree, extracted once from its pinned archive."""
    directory = cache / LLAMA_CPP_DIRECTORY
    tree = directory / f"llama.cpp-{LLAMA_CPP_COMMIT}"
    byte_pair_reference.fetch(directory, LLAMA_CPP_ARCHIVE)
    if not tree.exists():
        with tarfile.open(directory / LLAMA_CPP_ARCHIVE) as archive:
            archive.extractall(directory, filter="data")
    return tree


def build_gguf(cache):
    """Builds SmolLM2's GGUF in each type, checks each digest, and returns whether every one is the pinned one."""
    model = cache / SMOLLM2_DIRECTORY
    for name in MODEL_FILES:
        byte_pair_reference.fetch(model, name)
    converter = extracted_llama_cpp(cache) / "convert_hf_to_gguf.py"
    output = cache / (SMOLLM2_DIRECTORY + "-gguf")
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    every_pinned = True
    for kind, digest in GGUF_DIGESTS.items():
        target = output / f"smollm2-135m-instruct-{kind}.gguf"
        command = ["uv", "run", "--no-project", "--python", "3.12.13", "--index-strategy", "unsafe-best-match",
                   "--extra-index-url", "https://download.pytorch.org/whl/cpu", "--with-requirements",
                   str(REQUIREMENTS), "python", "-I", str(converter), str(model), "--outtype", kind, "--outfile",
                   str(target)]
        started = time.monotonic()
        run = subprocess.run(command, env=environment, cwd=output, capture_output=True, text=True, check=False)
        seconds = time.monotonic() - started
        if run.returncode != 0:
            sys.exit(f"convert_hf_to_gguf.py --outtype {kind} failed:\n{run.stderr[-2000:]}")
        data = target.read_bytes()
        found = hashlib.sha256(data).hexdigest()
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        print(f"gguf {kind} path={target} bytes={len(data)} sha256={found} pinned={found == digest} "
              f"seconds={seconds:.1f} peak_resident_kib={peak}")
        every_pinned = every_pinned and found == digest
    return every_pinned


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fixtures", "gguf", "check"])
    parser.add_argument("--cache", type=pathlib.Path,
                        default=pathlib.Path(os.path.expanduser("~/.cache/lplassistant/gguf-reference")))
    parser.add_argument("--style", type=pathlib.Path, default=REPOSITORY.parent / "LplKernel" / ".clang-format")
    arguments = parser.parse_args()

    if arguments.command == "gguf":
        return 0 if build_gguf(arguments.cache) else 1
    text = byte_pair_reference.formatted(fixture_header(), FIXTURE_HEADER.name, arguments.style)
    if arguments.command == "fixtures":
        FIXTURE_HEADER.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE_HEADER.write_text(text)
        return 0
    if not FIXTURE_HEADER.exists() or FIXTURE_HEADER.read_text() != text:
        print(f"{FIXTURE_HEADER.relative_to(REPOSITORY)}: differs from what its sources give; run `fixtures`")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

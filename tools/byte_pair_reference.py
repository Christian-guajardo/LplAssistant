#!/usr/bin/env python3
"""Writes what the byte-pair encoder of lpl::infer is checked against, from its primary sources.

Run from the repository's root, with Hugging Face tokenizers as the reference:

    uv run --no-project --with tokenizers==0.22.1 --with clang-format==18.1.3 \
        python -I tools/byte_pair_reference.py <command>

Commands:
    classes        write infer/src/UnicodeCharacterClasses.hpp from the Unicode Character Database 16.0.0
    fixture        write tests/infer/fixtures/smollm2_byte_pairs.hpp: SmolLM2's merges that a set of prompts reaches,
                   and the words and ids Hugging Face tokenizers gives each prompt
    check-classes  compare the class of every code point with what Hugging Face's pre-tokenizer makes of it
    check          write both files into a temporary directory and fail when they differ from the committed ones

The inputs are downloaded once into --cache (~/.cache/lplassistant/byte-pairs) and refused when their SHA-256 is not
the pinned one. Each file is formatted as every C++ file of LplAssistant is, by clang-format 18.1.3 with LplKernel's
.clang-format (--style, the sibling checkout's by default). Hugging Face tokenizers 0.22.1 reads Unicode 16.0: it classes U+1C89 as a letter and U+16D70 as a
number, both new in 16.0, and U+10940 and U+323B0, new in 17.0, as neither.
"""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import urllib.request

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
CLASSES_HEADER = REPOSITORY / "infer" / "src" / "UnicodeCharacterClasses.hpp"
FIXTURE_HEADER = REPOSITORY / "tests" / "infer" / "fixtures" / "smollm2_byte_pairs.hpp"

UCD = "https://www.unicode.org/Public/16.0.0/ucd/"
SMOLLM2_COMMIT = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
SMOLLM2 = f"https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/{SMOLLM2_COMMIT}/tokenizer.json"
INPUTS = {
    "DerivedGeneralCategory.txt": (
        UCD + "extracted/DerivedGeneralCategory.txt",
        "7676ab755a41ef82108460238569e60ad65c191ddafe61b36c6765ec1353f293",
    ),
    "PropList.txt": (UCD + "PropList.txt", "53d614508e2a0b2305a8aa21cd60d993de9326cdf65993660dfcce4503548583"),
    "tokenizer.json": (SMOLLM2, "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"),
}

OTHER, LETTER, NUMBER, SPACE = 0, 1, 2, 3
CODE_POINTS = 0x110000
INDEXED_CODE_POINTS = 0x40000
BLOCK_SHIFT = 7
BLOCK_CODE_POINTS = 1 << BLOCK_SHIFT
FNV1A_OFFSET_BASIS = 0x811C9DC5
FNV1A_PRIME = 0x01000193

# Each prompt exercises something the encoder must agree on; none holds the text of an added token, which Hugging
# Face would match before pre-tokenising (#106).
PROMPTS = [
    "Hello, world! This is a test of the tokenizer.",
    "In 2024, 3.14159 and 1,000,000 people paid $42.50 for 7 items.",
    "a  b   c    d\tend   \n    indented\n\n\nthree newlines   ",
    "I'm sure they'll say it's what we've done, isn't it? You'd, we're, DON'T, O'Reilly's.",
    "x 'quoted' 's 'S 're 've 'm 'll 'd 'x ''s",
    "Café crème, naïve façade, Übermäßig groß, señor, déjà vu.",
    "日本語のテキストと中文文本，한국어 텍스트。",
    "Emoji \U0001F600\U0001F44D\U0001F3FD and flags \U0001F1EB\U0001F1F7 with a family "
    "\U0001F468‍\U0001F469‍\U0001F467 and ❤️.",
    "#" * 80 + "\n" + "-" * 64 + " done",
    "\n" + " " * 80 + "\n",
    "nbsp here, ideographic　space, line separator, em space",
    "٣٤٥ ²³ Ⅻ ⑤ 一二三 ¾",
    "control\x04bytes\x06and !\x04? mixed\x13\x14\x16\x1D end",
    'def main():\n    return {"key": [1, 2, 3]}  # comment\r\n',
    "ASCII art: ╔════╗ and " + "─" * 24,
    "Ελληνικά, русский, עברית, العربية, हिन्दी, ไทย",
    "  leading spaces and trailing  ",
]


def formatted(text, name, style):
    """@p text as clang-format writes it with the style file @p style, for a file named @p name."""
    run = subprocess.run(["clang-format", f"--style=file:{style}", f"--assume-filename={name}"], input=text,
                         capture_output=True, text=True, check=False)
    if run.returncode != 0:
        sys.exit(f"clang-format failed on {name}: {run.stderr.strip()}")
    return run.stdout


def fetch(cache, name):
    """Returns the bytes of input @p name, downloaded into @p cache once and refused when its digest is not pinned."""
    url, digest = INPUTS[name]
    path = cache / name
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            path.write_bytes(response.read())
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        sys.exit(f"{path}: SHA-256 {hashlib.sha256(data).hexdigest()}, pinned {digest}")
    return data


def ranges(text, wanted=None):
    """Yields (first, last, value) for each line `first..last ; value` of a UCD file, or only those of @p wanted."""
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        span, value = [field.strip() for field in line.split(";")[:2]]
        if wanted is not None and value != wanted:
            continue
        first, _, last = span.partition("..")
        yield int(first, 16), int(last or first, 16), value


def character_classes(cache):
    """The class of every code point: Space for White_Space, Letter for a category L*, Number for N*, else Other."""
    classes = [OTHER] * CODE_POINTS
    for first, last, category in ranges(fetch(cache, "DerivedGeneralCategory.txt").decode()):
        value = {"L": LETTER, "N": NUMBER}.get(category[0], OTHER)
        for code_point in range(first, last + 1):
            classes[code_point] = value
    for first, last, _ in ranges(fetch(cache, "PropList.txt").decode(), "White_Space"):
        for code_point in range(first, last + 1):
            classes[code_point] = SPACE
    if any(classes[INDEXED_CODE_POINTS:]):
        sys.exit("a code point at or past U+40000 is not Other: the index must cover it")
    return classes


def fold(words):
    """The FNV-1a fold of @p words, one 32-bit word at a time, as the tests fold."""
    value = FNV1A_OFFSET_BASIS
    for word in words:
        value = ((value ^ word) * FNV1A_PRIME) & 0xFFFFFFFF
    return value


def two_level_table(classes):
    """The index of each 128-code-point block below U+40000 and the distinct blocks, 2 bits per code point."""
    blocks, index = [], []
    seen = {}
    for start in range(0, INDEXED_CODE_POINTS, BLOCK_CODE_POINTS):
        packed = bytes(
            sum(classes[start + 4 * byte + lane] << (2 * lane) for lane in range(4))
            for byte in range(BLOCK_CODE_POINTS // 4)
        )
        if packed not in seen:
            seen[packed] = len(blocks)
            blocks.append(packed)
        index.append(seen[packed])
    if len(blocks) > 256:
        sys.exit(f"{len(blocks)} distinct blocks: the index no longer holds in a byte")
    return index, blocks


def numbers(values, per_line, width=120, indent="    "):
    """Lines of comma-separated @p values, @p per_line to a line at most, each line under @p width columns."""
    lines, line = [], []
    for value in values:
        line.append(str(value))
        if len(line) == per_line:
            lines.append(indent + ", ".join(line) + ",")
            line = []
    if line:
        lines.append(indent + ", ".join(line) + ",")
    return "\n".join(lines)


def classes_header(cache):
    """The text of infer/src/UnicodeCharacterClasses.hpp."""
    classes = character_classes(cache)
    index, blocks = two_level_table(classes)
    counts = [classes.count(value) for value in (OTHER, LETTER, NUMBER, SPACE)]
    rows = "\n".join("    {" + ", ".join(str(byte) for byte in block) + "}," for block in blocks)
    return f"""/**
 * @file UnicodeCharacterClasses.hpp
 * @brief The class of every Unicode code point a pre-tokenizer reads: letter, number, space or other.
 *
 * Written by tools/byte_pair_reference.py classes, from DerivedGeneralCategory.txt and PropList.txt of the Unicode
 * Character Database 16.0.0 (https://www.unicode.org/Public/16.0.0/ucd/): a class changes there, never here. A space
 * has the property White_Space, a letter a general category L*, a number a category N*; every other code point,
 * the surrogates and the unassigned ones included, is other. These are the classes \\s, \\p{{L}} and \\p{{N}} of the
 * patterns Hugging Face tokenizers 0.22.1 runs, which `byte_pair_reference.py check-classes` compares on every code
 * point.
 *
 * A code point's class is 2 bits of the block of 128 code points it falls in; blocks that hold the same classes are
 * stored once. Every code point at or past U+{INDEXED_CODE_POINTS:X} is other, so the index stops there:
 * {len(index)} block numbers and {len(blocks)} blocks, {len(index) + len(blocks) * BLOCK_CODE_POINTS // 4} bytes.
 * Code points by class: {counts[0]} other, {counts[1]} letters, {counts[2]} numbers, {counts[3]} spaces.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_INFER_SRC_UNICODECHARACTERCLASSES_HPP
#    define LPL_INFER_SRC_UNICODECHARACTERCLASSES_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

namespace lpl::infer::unicode {{

/** Code points of one block are those that share their bits above this shift. */
inline constexpr core::u32 kClassBlockShift = {BLOCK_SHIFT}u;

/** The first code point past the index: it and every code point after it are other. */
inline constexpr core::u32 kFirstUnindexedCodePoint = 0x{INDEXED_CODE_POINTS:X}u;

/** Bytes of one block: 128 code points at 2 bits each. */
inline constexpr core::u32 kClassBlockBytes = {BLOCK_CODE_POINTS // 4}u;

/** The block that holds the classes of each run of 128 code points below kFirstUnindexedCodePoint. */
inline constexpr core::u8 kClassBlockOfCodePoints[{len(index)}] = {{
{numbers(index, 28)}
}};

/** The distinct blocks; code point c of a block is bits 2 (c % 4) and up of byte c / 4. */
inline constexpr core::u8 kClassBlocks[{len(blocks)}][kClassBlockBytes] = {{
{rows}
}};

}} // namespace lpl::infer::unicode

#    endif /* LPL_HAS_FOUNDATION */

#endif /* LPL_INFER_SRC_UNICODECHARACTERCLASSES_HPP */
"""


def byte_level_alphabet():
    """GPT-2's map from a byte to the character tokenizer.json writes it as, and back."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    characters = printable[:]
    shifted = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            characters.append(256 + shifted)
            shifted += 1
    to_character = {byte: chr(character) for byte, character in zip(printable, characters)}
    return to_character, {character: byte for byte, character in to_character.items()}


class SmolLM2:
    """SmolLM2's vocabulary and ranked merges, the byte mapping undone."""

    def __init__(self, data):
        model = json.loads(data)["model"]
        if model["type"] != "BPE" or model.get("ignore_merges") or model.get("byte_fallback"):
            sys.exit("tokenizer.json is not the byte-level BPE this fixture was written for")
        _, to_byte = byte_level_alphabet()
        self.bytes_of = {}
        for text, token in model["vocab"].items():
            self.bytes_of[token] = bytes(to_byte[character] for character in text)
        self.token_of = {raw: token for token, raw in self.bytes_of.items()}
        self.merges = []
        for line in model["merges"]:
            left, right = line.split(" ") if isinstance(line, str) else line
            left, right = self.token_of[self.raw(left)], self.token_of[self.raw(right)]
            self.merges.append((left, right, self.token_of[self.bytes_of[left] + self.bytes_of[right]]))
        self.rank_of = {(left, right): rank for rank, (left, right, _) in enumerate(self.merges)}
        if len(self.rank_of) != len(self.merges):
            sys.exit("a pair is merged twice")

    @staticmethod
    def raw(text):
        _, to_byte = byte_level_alphabet()
        return bytes(to_byte[character] for character in text)

    def encode_word(self, word, ranks, reached):
        """GPT-2's merging of one word: lowest rank first, every occurrence left to right; the ranks of the table
        that were ever adjacent go into @p reached."""
        symbols = [self.token_of[bytes([byte])] for byte in word if bytes([byte]) in self.token_of]
        while len(symbols) > 1:
            adjacent = [ranks.get(pair) for pair in zip(symbols, symbols[1:])]
            reached.update(rank for rank in adjacent if rank is not None)
            present = [rank for rank in adjacent if rank is not None]
            if not present:
                break
            lowest, merged, at = min(present), [], 0
            while at < len(symbols):
                if at + 1 < len(symbols) and ranks.get((symbols[at], symbols[at + 1])) == lowest:
                    merged.append(self.merges[lowest][2])
                    at += 2
                else:
                    merged.append(symbols[at])
                    at += 1
            symbols = merged
        return symbols


def reference_encodings(smollm2, data):
    """Hugging Face's words and ids of each prompt, checked against this script's own merging."""
    from tokenizers import Tokenizer

    reference = Tokenizer.from_str(data.decode())
    encodings, reached = [], set()
    for prompt in PROMPTS:
        words = [smollm2.raw(piece) for piece, _ in reference.pre_tokenizer.pre_tokenize_str(prompt)]
        ids = reference.encode(prompt, add_special_tokens=False).ids
        mine = [token for word in words for token in smollm2.encode_word(word, smollm2.rank_of, reached)]
        if mine != ids:
            sys.exit(f"this script's merging disagrees with Hugging Face on {prompt!r}")
        if b"".join(words) != prompt.encode():
            sys.exit(f"the words of {prompt!r} do not spell it")
        encodings.append((prompt.encode(), words, ids))
    return encodings, sorted(reached)


def subset(smollm2, encodings, reached):
    """The tokens and merges the prompts reach, renumbered: every single byte, every merge whose pair was ever
    adjacent, in rank order, and the tokens those merges make. Encoding with them is checked to give the same ids."""
    tokens = sorted({token for raw, token in smollm2.token_of.items() if len(raw) == 1} |
                    {token for rank in reached for token in smollm2.merges[rank]})
    renumbered = {token: index for index, token in enumerate(tokens)}
    merges = [tuple(renumbered[token] for token in smollm2.merges[rank]) for rank in reached]
    ranks = {(smollm2.merges[rank][0], smollm2.merges[rank][1]): rank for rank in reached}
    for raw, words, ids in encodings:
        if [token for word in words for token in smollm2.encode_word(word, ranks, set())] != ids:
            sys.exit(f"the subset does not encode {raw!r} as the whole table does")
    return tokens, merges


def string_literal(raw):
    """@p raw as C string literals, an octal escape for every byte that is not printable ASCII, split by line."""
    pieces, piece = [], ""
    for byte in raw:
        character = chr(byte)
        if 0x20 <= byte < 0x7F and character not in '"\\?':
            piece += character
        else:
            piece += f"\\{byte:03o}"
        if len(piece) >= 100:
            pieces.append(piece)
            piece = ""
    if piece or not pieces:
        pieces.append(piece)
    return "\n".join(f'    "{piece}"' for piece in pieces)


def fixture_header(cache):
    """The text of tests/infer/fixtures/smollm2_byte_pairs.hpp."""
    data = fetch(cache, "tokenizer.json")
    smollm2 = SmolLM2(data)
    encodings, reached = reference_encodings(smollm2, data)
    tokens, merges = subset(smollm2, encodings, reached)
    text, entries, offset = b"", [], 0
    for token in tokens:
        raw = smollm2.bytes_of[token]
        entries.append((offset, len(raw)))
        text += raw
        offset += len(raw)
    prompts, ids, words, prompt_text = [], [], [], b""
    for raw, prompt_words, prompt_ids in encodings:
        skipped = sum(1 for byte in raw if bytes([byte]) not in smollm2.token_of)
        prompts.append((len(prompt_text), len(raw), len(ids), len(prompt_ids), len(words), len(prompt_words), skipped))
        prompt_text += raw
        ids += prompt_ids
        words += [len(word) for word in prompt_words]
    longest = max(len(smollm2.bytes_of[token]) for token in ids)
    classes = character_classes(cache)
    counts = [classes.count(value) for value in (OTHER, LETTER, NUMBER, SPACE)]
    entry_rows = numbers([f"{{{start}u, {length}u}}" for start, length in entries], 8)
    merge_rows = numbers([f"{{{left}u, {right}u, {result}u}}" for left, right, result in merges], 5)
    prompt_rows = numbers(["{" + ", ".join(f"{field}u" for field in prompt) + "}" for prompt in prompts], 2)
    return f"""/**
 * @file smollm2_byte_pairs.hpp
 * @brief What SmolLM2-135M-Instruct's tokenizer makes of a set of prompts, and the part of its tables they reach.
 *
 * Written by tools/byte_pair_reference.py fixture, from the tokenizer.json of SmolLM2-135M-Instruct at
 * {SMOLLM2_COMMIT}, and from Hugging Face tokenizers 0.22.1, which gave every word and id: they change there, never
 * here.
 *
 * The tables are a subset of the model's: its {len(tokens)} tokens are every single byte that has one and every token
 * a merge below makes, numbered in the order of their ids (kOriginalIds gives each one's id in the model); its
 * {len(merges)} merges are every merge of the model whose pair is adjacent at some point while the prompts are
 * encoded, in the order of their ranks. A merge that was never adjacent can change no encoding, so the subset encodes
 * each prompt as the model's {len(smollm2.merges)} merges do, which the script checks. The longest token a prompt
 * encodes to is {longest} bytes.
 *
 * The classes of every code point, folded one word each with FNV-1a, come from the Unicode Character Database 16.0.0
 * as tools/byte_pair_reference.py reads it, apart from the encoder's table.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_TESTS_INFER_FIXTURES_SMOLLM2_BYTE_PAIRS_HPP
#    define LPL_TESTS_INFER_FIXTURES_SMOLLM2_BYTE_PAIRS_HPP

#    include <lpl/infer/Tokenizer.hpp>

namespace smollm2_byte_pairs {{

/**
 * @struct Prompt
 * @brief Where one prompt's bytes, ids and words lie in the tables below, and the bytes it holds that no token spells.
 */
struct Prompt {{
    lpl::core::u32 textOffset; ///< First byte in kPromptText.
    lpl::core::u32 textBytes;  ///< Bytes of the prompt.
    lpl::core::u32 firstId;    ///< First id in kReferenceIds.
    lpl::core::u32 idCount;    ///< Ids the reference encodes it to.
    lpl::core::u32 firstWord;  ///< First word in kReferenceWordBytes.
    lpl::core::u32 wordCount;  ///< Words the reference pre-tokenizer splits it into.
    lpl::core::u32 unspellable; ///< Bytes of it no single-byte token spells.
}};

/** Tokens of the subset. */
inline constexpr lpl::core::u32 kTokenCount = {len(tokens)}u;

/** Merges of the subset. */
inline constexpr lpl::core::u32 kMergeCount = {len(merges)}u;

/** Prompts. */
inline constexpr lpl::core::u32 kPromptCount = {len(prompts)}u;

/** The longest token, in bytes, any prompt encodes to. */
inline constexpr lpl::core::u32 kLongestEncodedTokenBytes = {longest}u;

/** The FNV-1a fold of the class of every code point from U+0000 to U+10FFFF: 0 other, 1 letter, 2 number, 3 space. */
inline constexpr lpl::core::u32 kCharacterClassSignature = 0x{fold(classes):08X}u;

/** Code points that are letters. */
inline constexpr lpl::core::u32 kLetterCodePoints = {counts[1]}u;

/** Code points that are numbers. */
inline constexpr lpl::core::u32 kNumberCodePoints = {counts[2]}u;

/** Code points that are spaces. */
inline constexpr lpl::core::u32 kSpaceCodePoints = {counts[3]}u;

/** The tokens' bytes, one after the other. */
inline constexpr char kTokenText[] =
{string_literal(text)};

/** Bytes of kTokenText. */
inline constexpr lpl::core::u32 kTokenTextBytes = {len(text)}u;

/** Where each token's bytes lie in kTokenText. */
inline constexpr lpl::infer::VocabEntry kTokenEntries[kTokenCount] = {{
{entry_rows}
}};

/** Each token's id in the model. */
inline constexpr lpl::core::u32 kOriginalIds[kTokenCount] = {{
{numbers([f"{token}u" for token in tokens], 12)}
}};

/** The merges: left, right, result, numbered as the tokens above, in the order of their ranks. */
inline constexpr lpl::infer::BytePairMerge kMerges[kMergeCount] = {{
{merge_rows}
}};

/** The prompts' bytes, one after the other. */
inline constexpr char kPromptText[] =
{string_literal(prompt_text)};

/** The prompts. */
inline constexpr Prompt kPrompts[kPromptCount] = {{
{prompt_rows}
}};

/** The ids Hugging Face gives each prompt, as the model numbers them. */
inline constexpr lpl::core::u32 kReferenceIds[] = {{
{numbers([f"{token}u" for token in ids], 12)}
}};

/** The bytes of each word Hugging Face's pre-tokenizer splits each prompt into. */
inline constexpr lpl::core::u32 kReferenceWordBytes[] = {{
{numbers([f"{length}u" for length in words], 16)}
}};

}} // namespace smollm2_byte_pairs

#endif /* LPL_TESTS_INFER_FIXTURES_SMOLLM2_BYTE_PAIRS_HPP */
"""


def check_classes(cache):
    """Compares the class of every code point but the surrogates with Hugging Face's pre-tokenizer: two of a number
    are two words, a letter after x is one, an other after ! is one, and a space is none of these."""
    from tokenizers import Tokenizer

    pre_tokenizer = Tokenizer.from_str(fetch(cache, "tokenizer.json").decode()).pre_tokenizer
    classes = character_classes(cache)

    def words(text):
        return len(pre_tokenizer.pre_tokenize_str(text))

    differences, compared = [], 0
    for code_point in range(CODE_POINTS):
        if 0xD800 <= code_point <= 0xDFFF:
            continue
        character = chr(code_point)
        if words(character * 2) == 2:
            seen = NUMBER
        elif words("x" + character) == 1:
            seen = LETTER
        elif words("!" + character) == 1:
            seen = OTHER
        else:
            seen = SPACE
        compared += 1
        if seen != classes[code_point]:
            differences.append(code_point)
    print(f"code points compared: {compared}, differences: {len(differences)}",
          " ".join(f"U+{code_point:04X}" for code_point in differences[:20]))
    return not differences


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["classes", "fixture", "check-classes", "check"])
    parser.add_argument("--cache", type=pathlib.Path,
                        default=pathlib.Path(os.path.expanduser("~/.cache/lplassistant/byte-pairs")))
    parser.add_argument("--style", type=pathlib.Path, default=REPOSITORY.parent / "LplKernel" / ".clang-format")
    arguments = parser.parse_args()
    writers = {CLASSES_HEADER: classes_header, FIXTURE_HEADER: fixture_header}

    def text_of(path):
        return formatted(writers[path](arguments.cache), path.name, arguments.style)

    if arguments.command in ("classes", "fixture"):
        path = CLASSES_HEADER if arguments.command == "classes" else FIXTURE_HEADER
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text_of(path))
    elif arguments.command == "check-classes":
        return 0 if check_classes(arguments.cache) else 1
    else:
        stale = [path.relative_to(REPOSITORY) for path in writers
                 if not path.exists() or path.read_text() != text_of(path)]
        for path in stale:
            print(f"{path}: differs from what its sources give; run the command that writes it")
        return 1 if stale else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

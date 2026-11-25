/**
 * @file gguf_reference.hpp
 * @brief What the GGUF converter's pure parts are checked against, computed apart from them.
 *
 * Written by tools/gguf_reference.py fixtures: they change there, never here. GPT-2's byte-level alphabet is built as
 * openai/gpt-2's encoder.py builds it (bytes_to_unicode). Each rotary word is base^(-2i/64) / (2 pi) * 2^32,
 * rounded half up, in 60-digit decimal arithmetic; SmolLM2's closest word to a tie lies 0.036727 of a unit
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

namespace gguf_reference {

/** The character GPT-2's byte-level alphabet writes each byte as, by the byte. */
inline constexpr std::uint32_t kByteLevelCodePoints[256] = {
    256u, 257u, 258u, 259u, 260u, 261u, 262u, 263u, 264u, 265u, 266u, 267u, 268u, 269u, 270u, 271u, 272u, 273u, 274u,
    275u, 276u, 277u, 278u, 279u, 280u, 281u, 282u, 283u, 284u, 285u, 286u, 287u, 288u, 33u,  34u,  35u,  36u,  37u,
    38u,  39u,  40u,  41u,  42u,  43u,  44u,  45u,  46u,  47u,  48u,  49u,  50u,  51u,  52u,  53u,  54u,  55u,  56u,
    57u,  58u,  59u,  60u,  61u,  62u,  63u,  64u,  65u,  66u,  67u,  68u,  69u,  70u,  71u,  72u,  73u,  74u,  75u,
    76u,  77u,  78u,  79u,  80u,  81u,  82u,  83u,  84u,  85u,  86u,  87u,  88u,  89u,  90u,  91u,  92u,  93u,  94u,
    95u,  96u,  97u,  98u,  99u,  100u, 101u, 102u, 103u, 104u, 105u, 106u, 107u, 108u, 109u, 110u, 111u, 112u, 113u,
    114u, 115u, 116u, 117u, 118u, 119u, 120u, 121u, 122u, 123u, 124u, 125u, 126u, 289u, 290u, 291u, 292u, 293u, 294u,
    295u, 296u, 297u, 298u, 299u, 300u, 301u, 302u, 303u, 304u, 305u, 306u, 307u, 308u, 309u, 310u, 311u, 312u, 313u,
    314u, 315u, 316u, 317u, 318u, 319u, 320u, 321u, 322u, 161u, 162u, 163u, 164u, 165u, 166u, 167u, 168u, 169u, 170u,
    171u, 172u, 323u, 174u, 175u, 176u, 177u, 178u, 179u, 180u, 181u, 182u, 183u, 184u, 185u, 186u, 187u, 188u, 189u,
    190u, 191u, 192u, 193u, 194u, 195u, 196u, 197u, 198u, 199u, 200u, 201u, 202u, 203u, 204u, 205u, 206u, 207u, 208u,
    209u, 210u, 211u, 212u, 213u, 214u, 215u, 216u, 217u, 218u, 219u, 220u, 221u, 222u, 223u, 224u, 225u, 226u, 227u,
    228u, 229u, 230u, 231u, 232u, 233u, 234u, 235u, 236u, 237u, 238u, 239u, 240u, 241u, 242u, 243u, 244u, 245u, 246u,
    247u, 248u, 249u, 250u, 251u, 252u, 253u, 254u, 255u,
};

/** Channels of a head the tables below turn: SmolLM2's. */
inline constexpr std::uint32_t kRotaryDimensions = 64u;

/** Pairs of such a head. */
inline constexpr std::uint32_t kRotaryPairs = 32u;

/** SmolLM2's rope_theta. */
inline constexpr std::uint32_t kSmolLm2RotaryBase = 100000u;

/** How far each pair turns per position at SmolLM2's base, in units of 2^-32 turn. */
inline constexpr std::uint32_t kSmolLm2Turns[kRotaryPairs] = {
    683565276u, 477012756u, 332874091u, 232289721u, 162098872u, 113117551u, 78936887u, 55084574u,
    38439700u,  26824399u,  18718886u,  13062611u,  9115489u,   6361067u,   4438947u,  3097633u,
    2161623u,   1508447u,   1052640u,   734565u,    512602u,    357709u,    249620u,   174193u,
    121557u,    84826u,     59194u,     41308u,     28826u,     20115u,     14037u,    9796u,
};

/** LLaMA's rope_theta. */
inline constexpr std::uint32_t kLlamaRotaryBase = 10000u;

/** How far each pair turns per position at LLaMA's base, in units of 2^-32 turn. */
inline constexpr std::uint32_t kLlamaTurns[kRotaryPairs] = {
    683565276u, 512601642u, 384397003u, 288257087u, 216162320u, 162098872u, 121557005u, 91154895u,
    68356528u,  51260164u,  38439700u,  28825709u,  21616232u,  16209887u,  12155701u,  9115489u,
    6835653u,   5126016u,   3843970u,   2882571u,   2161623u,   1620989u,   1215570u,   911549u,
    683565u,    512602u,    384397u,    288257u,    216162u,    162099u,    121557u,    91155u,
};

/** The first integer base past LLaMA's whose table puts a word within 10^-6 of a rounding tie. */
inline constexpr std::uint32_t kNearTieBase = 25068u;

/** The pair whose word lies there, 7.685e-7 of a unit from the tie. */
inline constexpr std::uint32_t kNearTiePair = 31u;

} // namespace gguf_reference

#endif /* LPL_TESTS_CONVERT_FIXTURES_GGUF_REFERENCE_HPP */

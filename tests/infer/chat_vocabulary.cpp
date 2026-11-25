#include "chat_vocabulary.hpp"

#include <lpl/infer/Parity.hpp>

namespace lpl::infer::tests {

namespace {

/** The merges of a vocabulary segmented by longest match: none. */
const BytePairMerges kNoMerges{};

/**
 * @brief Bytes of the terminated @p text.
 */
[[nodiscard]] core::u32 textBytes(const char *text) noexcept
{
    core::u32 bytes = 0u;

    while (text[bytes] != '\0')
        ++bytes;
    return bytes;
}

} // namespace

bool buildParityVocabularyWith(TensorArena &arena, const char *const *texts, core::u32 count, Vocab &out)
{
    Vocab parity;
    if (!buildParityVocab(arena, parity))
        return false;

    core::u32 appendedBytes = 0u;
    for (core::u32 i = 0u; i < count; ++i)
        appendedBytes += textBytes(texts[i]);
    char *const blob = arena.claim<char>(parity.blobBytes() + appendedBytes);
    VocabEntry *const entries = arena.claim<VocabEntry>(parity.size() + count);
    if (blob == nullptr || entries == nullptr)
        return false;

    for (core::u32 i = 0u; i < parity.blobBytes(); ++i)
        blob[i] = parity.blob()[i];
    for (core::u32 i = 0u; i < parity.size(); ++i)
        entries[i] = parity.entries()[i];
    core::u32 cursor = parity.blobBytes();
    for (core::u32 i = 0u; i < count; ++i)
    {
        const core::u32 bytes = textBytes(texts[i]);
        entries[parity.size() + i] = VocabEntry{cursor, bytes};
        for (core::u32 b = 0u; b < bytes; ++b)
            blob[cursor++] = texts[i][b];
    }
    return out.build(arena, blob, cursor, entries, parity.size() + count);
}

bool synthesiseParityModelOver(TensorArena &arena, const Vocab &vocab, ChatFormat format, core::u32 contextLength,
                               Model &out)
{
    ModelConfig shape = parityModelConfig();

    shape.vocabSize = vocab.size();
    shape.contextLength = contextLength;
    return out.synthesise(arena, shape, vocab, kNoMerges, PreTokenizer::WholeText, format, parityModelSeed());
}

void putChatFormatWord(core::u8 *image, core::u32 value) noexcept
{
    for (core::u32 i = 0u; i < 4u; ++i)
        image[kHeaderChatFormatByte + i] = static_cast<core::u8>((value >> (8u * i)) & 0xFFu);
}

} // namespace lpl::infer::tests

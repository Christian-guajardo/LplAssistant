/**
 * @file ModelFile.hpp
 * @brief A GGUF file read through ggml's gguf API, mapped, and turned into a SourceModel.
 *
 * ggml b3775 parses the file's header, keys and tensor table, and aborts on some malformed ones instead of refusing
 * them (ggml.c:22378-22379): a file the converter reads comes from the pinned chain of tools/gguf_reference.py. Past
 * that parser, every value is read by the type the file gives it, and every tensor's bytes are bounded by the file.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CONVERT_MODELFILE_HPP
#    define LPL_LPL_CONVERT_MODELFILE_HPP

#    include <lpl/convert/Source.hpp>

#    include <cstddef>
#    include <optional>
#    include <string>

namespace lpl::convert {

/**
 * @class ModelFile
 * @brief A GGUF file mapped read-only, and the model it holds; the tensors' bytes lie in the mapping.
 */
class ModelFile {
public:
    ModelFile() = default;
    ModelFile(const ModelFile &) = delete;
    ModelFile &operator=(const ModelFile &) = delete;
    ~ModelFile();

    /**
     * @brief Maps the file at @p path and reads its keys and tensors.
     * @param path The file.
     * @return Nothing when it is read; otherwise `file_missing` when no file is there, `file_unreadable` when it cannot
     *         be mapped or ggml cannot parse it as a GGUF, `metadata_repeated` when a key is written twice, which
     *         llama.cpp reads as its first, `tensor_outside_file` when a tensor's offset lies past the file's end.
     */
    [[nodiscard]] std::optional<Refusal> open(const std::string &path);

    /**
     * @brief The model read.
     * @return Its keys and tensors, as far as open() read them.
     */
    [[nodiscard]] const SourceModel &source() const noexcept { return _source; }

    /**
     * @brief The file's bytes, as mapped.
     * @return The first byte; nullptr before a file is mapped.
     */
    [[nodiscard]] const std::uint8_t *bytes() const noexcept { return _bytes; }

    /**
     * @brief The file's length.
     * @return Bytes; 0 before a file is mapped.
     */
    [[nodiscard]] std::size_t size() const noexcept { return _size; }

private:
    SourceModel _source{};
    const std::uint8_t *_bytes{nullptr};
    std::size_t _size{0u};
};

} // namespace lpl::convert

#endif /* LPL_LPL_CONVERT_MODELFILE_HPP */

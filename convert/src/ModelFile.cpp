#include <lpl/convert/ModelFile.hpp>

#include <ggml.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lpl::convert {

namespace {

/**
 * @brief A refusal named @p name about the file at @p path, for @p why.
 */
[[nodiscard]] Refusal fileRefusal(const char *name, const std::string &path, const std::string &why)
{
    return Refusal{name, path + ": " + why};
}

/**
 * @brief The tensor type @p type is, for the converter.
 */
[[nodiscard]] TensorType tensorTypeOf(ggml_type type) noexcept
{
    switch (type)
    {
    case GGML_TYPE_F32: return TensorType::Float32;
    case GGML_TYPE_F16: return TensorType::Float16;
    case GGML_TYPE_BF16: return TensorType::BrainFloat16;
    case GGML_TYPE_Q8_0: return TensorType::QuantisedEightBit;
    default: return TensorType::Other;
    }
}

/**
 * @brief Key @p id of @p context, by the type the file gives it; a getter is only called on its own type.
 */
[[nodiscard]] MetadataValue keyOf(const gguf_context *context, int id)
{
    MetadataValue value;
    const gguf_type type = gguf_get_kv_type(context, id);
    value.typeName = gguf_type_name(type);

    if (type == GGUF_TYPE_UINT32)
    {
        value.kind = MetadataKind::UnsignedWord;
        value.unsignedWord = gguf_get_val_u32(context, id);
    }
    else if (type == GGUF_TYPE_FLOAT32)
    {
        value.kind = MetadataKind::Real;
        value.real = gguf_get_val_f32(context, id);
    }
    else if (type == GGUF_TYPE_BOOL)
    {
        value.kind = MetadataKind::Boolean;
        value.boolean = gguf_get_val_bool(context, id);
    }
    else if (type == GGUF_TYPE_STRING)
    {
        value.kind = MetadataKind::Text;
        value.text = gguf_get_val_str(context, id);
    }
    else if (type == GGUF_TYPE_ARRAY)
    {
        const gguf_type element = gguf_get_arr_type(context, id);
        const int count = gguf_get_arr_n(context, id);
        value.typeName = std::string("arr[") + gguf_type_name(element) + "]";
        if (element == GGUF_TYPE_STRING)
        {
            value.kind = MetadataKind::TextList;
            for (int i = 0; i < count; ++i)
                value.texts.emplace_back(gguf_get_arr_str(context, id, i));
        }
        else if (element == GGUF_TYPE_INT32)
        {
            value.kind = MetadataKind::SignedWordList;
            const std::int32_t *const words = static_cast<const std::int32_t *>(gguf_get_arr_data(context, id));
            value.signedWords.assign(words, words + count);
        }
    }
    return value;
}

/**
 * @brief Tensor @p id of @p context, its bytes in the @p size bytes mapped at @p bytes, or the refusal of an offset
 *        past their end.
 */
[[nodiscard]] std::optional<Refusal> tensorOf(gguf_context *context, ggml_context *meta, int id,
                                              const std::uint8_t *bytes, std::size_t size, const std::string &path,
                                              SourceTensor &out)
{
    const char *const name = gguf_get_tensor_name(context, id);
    const ggml_tensor *const tensor = ggml_get_tensor(meta, name);
    const std::size_t offset = gguf_get_data_offset(context) + gguf_get_tensor_offset(context, id);
    if (tensor == nullptr || offset > size)
        return fileRefusal("tensor_outside_file", path,
                           std::string(name) + " starts at " + std::to_string(offset) + ", past the file's " +
                               std::to_string(size) + " bytes");

    out.type = tensorTypeOf(gguf_get_tensor_type(context, id));
    out.typeName = ggml_type_name(gguf_get_tensor_type(context, id));
    for (int d = 0; d < ggml_n_dims(tensor); ++d)
        out.extents.push_back(static_cast<std::uint64_t>(tensor->ne[d]));
    out.bytes = bytes + offset;
    out.byteCount = size - offset;
    return std::nullopt;
}

} // namespace

ModelFile::~ModelFile()
{
    if (_bytes != nullptr)
        munmap(const_cast<std::uint8_t *>(_bytes), _size);
}

std::optional<Refusal> ModelFile::open(const std::string &path)
{
    struct stat status {};
    if (stat(path.c_str(), &status) != 0)
        return fileRefusal(errno == ENOENT ? "file_missing" : "file_unreadable", path, std::strerror(errno));
    if (!S_ISREG(status.st_mode) || status.st_size == 0)
        return fileRefusal("file_unreadable", path, "not a regular file of at least one byte");

    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0)
        return fileRefusal("file_unreadable", path, std::strerror(errno));
    const std::size_t size = static_cast<std::size_t>(status.st_size);
    void *const mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, descriptor, 0);
    close(descriptor);
    if (mapped == MAP_FAILED)
        return fileRefusal("file_unreadable", path, std::strerror(errno));
    _bytes = static_cast<const std::uint8_t *>(mapped);
    _size = size;

    ggml_context *meta = nullptr;
    gguf_context *const context = gguf_init_from_file(path.c_str(), gguf_init_params{true, &meta});
    if (context == nullptr)
        return fileRefusal("file_unreadable", path, "ggml cannot parse it as a GGUF file");

    std::optional<Refusal> refusal;
    for (int id = 0; !refusal && id < gguf_get_n_kv(context); ++id)
    {
        const std::string key = gguf_get_key(context, id);
        if (!_source.metadata.emplace(key, keyOf(context, id)).second)
            refusal = fileRefusal("metadata_repeated", path,
                                  key + " is written twice: llama.cpp would read the first, and the converter refuses "
                                        "to choose");
    }
    for (int id = 0; !refusal && id < gguf_get_n_tensors(context); ++id)
        refusal = tensorOf(context, meta, id, _bytes, _size, path, _source.tensors[gguf_get_tensor_name(context, id)]);
    gguf_free(context);
    ggml_free(meta);
    return refusal;
}

} // namespace lpl::convert

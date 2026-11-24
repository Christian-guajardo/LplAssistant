#include "model_image_writer.hpp"

namespace lpl::infer::tests {

core::u8 *writeModelImage(TensorArena &arena, const Model &model, core::u32 alignmentLog2, core::u32 shift,
                          core::u32 &size)
{
    size = model.blobBytes(alignmentLog2);
    core::u32 *const words = size != 0u ? arena.claim<core::u32>(size / sizeof(core::u32) + 2u) : nullptr;
    if (words == nullptr)
        return nullptr;

    core::u8 *const image = reinterpret_cast<core::u8 *>(words) + shift;
    for (core::u32 i = 0u; i < size; ++i)
        image[i] = kUnwrittenByte;
    return model.writeBlob(image, size, alignmentLog2) == size ? image : nullptr;
}

} // namespace lpl::infer::tests

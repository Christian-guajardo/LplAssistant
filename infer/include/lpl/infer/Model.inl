namespace lpl::infer {

constexpr core::u32 ModelConfig::headsPerKeyValueHead() const noexcept
{
    return keyValueHeads == 0u ? 0u : heads / keyValueHeads;
}

constexpr bool ModelConfig::valid() const noexcept
{
    return dim != 0u && layers != 0u && heads != 0u && ffnHidden != 0u && contextLength != 0u && vocabSize != 0u &&
           keyValueHeads != 0u && (dim % heads) == 0u && (heads % keyValueHeads) == 0u;
}

} // namespace lpl::infer

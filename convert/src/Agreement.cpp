#include <lpl/convert/Agreement.hpp>

#include <lpl/convert/Rotary.hpp>
#include <lpl/infer/Attention.hpp>
#include <lpl/infer/FeedForward.hpp>
#include <lpl/infer/KvCache.hpp>
#include <lpl/infer/Transformer.hpp>

#include <llama.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>

namespace lpl::convert {

namespace {

constexpr std::size_t kArenaSlack = 1u << 20;         ///< Room an arena keeps past what a run claims.
constexpr const char *kLayerOutput = "l_out-";        ///< The name llama.cpp gives each layer's output.
constexpr const char *kFeedForwardInput = "ffn_inp-"; ///< And the residual stream after each layer's attention.
constexpr double kFixedUnits = 65536.0;               ///< Units of Q16.16 in 1.

/**
 * @brief Passes llama.cpp's errors to the standard error, and discards its other lines, so that a test's or a tool's
 *        own records are all the standard output holds and a refusal to load keeps its cause.
 */
void keepErrors(ggml_log_level level, const char *text, void *)
{
    if (level == GGML_LOG_LEVEL_ERROR)
        std::fputs(text, stderr);
}

/**
 * @brief Initialises llama.cpp's backend once for the process.
 */
void initialiseBackendOnce()
{
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(keepErrors, nullptr);
        llama_backend_init();
    });
}

/**
 * @brief Whether @p name begins with @p prefix.
 */
[[nodiscard]] bool startsWith(const char *name, const char *prefix)
{
    return std::strncmp(name, prefix, std::strlen(prefix)) == 0;
}

/**
 * @brief Keeps the largest |value| of the residual stream llama.cpp computes, after each layer's attention and after
 *        its feed-forward, where lpl::infer's is observed too, in the double @p peak points at.
 */
bool observeLayerOutput(ggml_tensor *tensor, bool ask, void *peak)
{
    if (ask)
        return startsWith(tensor->name, kLayerOutput) || startsWith(tensor->name, kFeedForwardInput);
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor))
        return true;
    const float *const values = static_cast<const float *>(tensor->data);
    double &largest = *static_cast<double *>(peak);
    for (std::int64_t i = 0; i < ggml_nelements(tensor); ++i)
        largest = std::max(largest, static_cast<double>(std::fabs(values[i])));
    return true;
}

/**
 * @brief The first index of the largest of @p count values at @p values.
 */
[[nodiscard]] std::uint32_t mostLikely(const float *values, std::uint32_t count)
{
    return static_cast<std::uint32_t>(std::max_element(values, values + count) - values);
}

/**
 * @brief The largest of @p count values less the second largest.
 */
[[nodiscard]] double marginOf(const float *values, std::uint32_t count)
{
    float first = -INFINITY;
    float second = -INFINITY;
    for (std::uint32_t i = 0u; i < count; ++i)
    {
        if (values[i] > first)
        {
            second = first;
            first = values[i];
        }
        else if (values[i] > second)
            second = values[i];
    }
    return static_cast<double>(first) - static_cast<double>(second);
}

/**
 * @brief Keeps @p residual's largest |value| in @p peak and counts in @p saturations its values at Q16.16's bounds.
 */
void observeResidual(infer::ConstVectorView residual, double &peak, std::uint64_t &saturations)
{
    for (core::u32 i = 0u; i < residual.count; ++i)
    {
        const core::i32 raw = residual.at(i).raw();
        if (raw == math::Fixed32::max().raw() || raw == math::Fixed32::min().raw())
            ++saturations;
        peak = std::max(peak, std::fabs(static_cast<double>(raw)) / kFixedUnits);
    }
}

/**
 * @struct ImageRun
 * @brief What running an image layer by layer claims: a cache, and the buffers Transformer::forward claims for itself.
 */
struct ImageRun {
    infer::KvCache cache{};             ///< The layer-by-layer run's.
    infer::VectorView residual{};       ///< The residual stream.
    infer::VectorView normed{};         ///< A normalised copy.
    infer::VectorView query{};          ///< A position's queries.
    infer::VectorView blend{};          ///< The heads' blended values.
    infer::VectorView scores{};         ///< A head's scores over the positions.
    infer::VectorView gate{};           ///< The feed-forward's gate.
    infer::VectorView up{};             ///< And its up projection.
    infer::VectorView logits{};         ///< A position's logits.
    infer::QuantBlock *blocks{nullptr}; ///< An activation, quantised.
};

/**
 * @brief Claims a vector of @p count values from @p arena.
 */
[[nodiscard]] infer::VectorView claimVector(infer::TensorArena &arena, core::u32 count)
{
    return infer::VectorView{arena.claim<math::Fixed32>(count), count};
}

/**
 * @brief Claims what running @p model over @p positions takes, layer by layer.
 */
[[nodiscard]] bool claimRun(infer::TensorArena &arena, const infer::Model &model, core::u32 positions, ImageRun &run)
{
    infer::ModelConfig shape = model.config();
    shape.contextLength = positions;
    const core::u32 widest = std::max(shape.dim, shape.ffnHidden);

    run.residual = claimVector(arena, shape.dim);
    run.normed = claimVector(arena, shape.dim);
    run.query = claimVector(arena, shape.dim);
    run.blend = claimVector(arena, shape.dim);
    run.scores = claimVector(arena, positions);
    run.gate = claimVector(arena, shape.ffnHidden);
    run.up = claimVector(arena, shape.ffnHidden);
    run.logits = claimVector(arena, shape.vocabSize);
    run.blocks = arena.claim<infer::QuantBlock>(infer::quantBlockCount(widest));
    return run.cache.allocate(arena, shape, infer::CachePolicy::Refuse) && run.residual.values != nullptr &&
           run.normed.values != nullptr && run.query.values != nullptr && run.blend.values != nullptr &&
           run.scores.values != nullptr && run.gate.values != nullptr && run.up.values != nullptr &&
           run.logits.values != nullptr && run.blocks != nullptr;
}

/**
 * @brief Runs @p token at @p position through @p model as Transformer::forward does, observing the residual stream
 *        after each block.
 */
[[nodiscard]] bool stepImage(const infer::Model &model, std::uint32_t token, std::uint32_t position, ImageRun &run,
                             SequenceAgreement &out)
{
    const infer::ModelConfig &shape = model.config();
    const core::u32 slot = run.cache.reserve();
    if (slot == infer::kNoSlot)
        return false;

    model.embed(token, run.residual);
    for (core::u32 l = 0u; l < shape.layers; ++l)
    {
        infer::attentionBlock(shape, model.layer(l), model.rotaryFrequencies(), run.cache, l, slot, position,
                              run.residual, run.normed, run.query, run.scores, run.blend, run.blocks);
        observeResidual(run.residual, out.imageResidualPeak, out.imageSaturations);
        infer::feedForwardBlock(shape, model.layer(l), run.residual, run.normed, run.gate, run.up, run.blocks);
        observeResidual(run.residual, out.imageResidualPeak, out.imageSaturations);
    }
    infer::rmsNorm(run.residual, model.finalNorm(), shape.normEpsilon, run.normed);
    infer::quantiseRow(run.normed.values, run.normed.count, run.blocks);
    infer::quantisedMatVecPrepared(model.embedding(), run.blocks, run.logits);
    return true;
}

/**
 * @brief Bytes an arena needs to run @p model over @p positions twice, layer by layer and by Transformer::forward.
 */
[[nodiscard]] std::size_t runArenaBytes(const infer::Model &model, core::u32 positions)
{
    const infer::ModelConfig &shape = model.config();
    const std::size_t cache = 2u * shape.layers * static_cast<std::size_t>(positions) * shape.keyValueWidth() * 4u;
    const std::size_t buffers =
        4u * (8u * shape.dim + 4u * shape.ffnHidden + 2u * shape.vocabSize + shape.contextLength + positions);
    return 2u * cache + 2u * buffers + kArenaSlack;
}

/**
 * @brief The largest |difference| of a logit over every position of @p sequences.
 */
[[nodiscard]] double largestLogitDifferenceOf(const std::vector<SequenceAgreement> &sequences) noexcept
{
    double largest = 0.0;
    for (const SequenceAgreement &sequence : sequences)
    {
        for (const PositionAgreement &position : sequence.positions)
            largest = std::max(largest, position.largestLogitDifference);
    }
    return largest;
}

} // namespace

/**
 * @struct ReferenceModel::Loaded
 * @brief The model llama.cpp loaded.
 */
struct ReferenceModel::Loaded {
    llama_model *model{nullptr}; ///< llama.cpp's.
};

std::uint32_t SequenceAgreement::judged() const noexcept
{
    return static_cast<std::uint32_t>(
        std::count_if(positions.begin(), positions.end(),
                      [](const PositionAgreement &position) { return position.referenceMargin >= kAgreementMargin; }));
}

std::uint32_t SequenceAgreement::judgedDisagreements() const noexcept
{
    return static_cast<std::uint32_t>(
        std::count_if(positions.begin(), positions.end(), [](const PositionAgreement &position) {
            return position.referenceMargin >= kAgreementMargin && position.imageToken != position.referenceToken;
        }));
}

std::uint32_t SequenceAgreement::agreements() const noexcept
{
    return static_cast<std::uint32_t>(
        std::count_if(positions.begin(), positions.end(), [](const PositionAgreement &position) {
            return position.imageToken == position.referenceToken;
        }));
}

ReferenceModel::ReferenceModel() : _loaded(std::make_unique<Loaded>()) {}

ReferenceModel::~ReferenceModel()
{
    if (_loaded->model != nullptr)
        llama_free_model(_loaded->model);
}

std::optional<Refusal> ReferenceModel::open(const std::string &path, std::uint32_t threads)
{
    initialiseBackendOnce();
    _loaded->model = llama_load_model_from_file(path.c_str(), llama_model_default_params());
    _threads = threads;
    if (_loaded->model == nullptr)
        return Refusal{"reference_unloadable", path + ": llama.cpp b3775 does not load it"};
    return std::nullopt;
}

bool ReferenceModel::run(const std::vector<std::uint32_t> &tokens, std::vector<float> &logits, double &residualPeak)
{
    const std::uint32_t count = static_cast<std::uint32_t>(tokens.size());
    if (count == 0u)
        return false;
    llama_context_params parameters = llama_context_default_params();
    parameters.n_ctx = count;
    parameters.n_batch = count;
    parameters.n_ubatch = count;
    parameters.n_threads = static_cast<std::int32_t>(_threads);
    parameters.n_threads_batch = static_cast<std::int32_t>(_threads);
    parameters.type_k = GGML_TYPE_F32;
    parameters.type_v = GGML_TYPE_F16;
    parameters.cb_eval = observeLayerOutput;
    parameters.cb_eval_user_data = &residualPeak;
    residualPeak = 0.0;

    llama_context *const context = llama_new_context_with_model(_loaded->model, parameters);
    if (context == nullptr)
        return false;
    llama_batch batch = llama_batch_init(static_cast<std::int32_t>(count), 0, 1);
    for (std::uint32_t i = 0u; i < count; ++i)
    {
        batch.token[i] = static_cast<llama_token>(tokens[i]);
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = static_cast<std::int32_t>(count);
    const bool decoded = llama_decode(context, batch) == 0;
    const std::uint32_t vocabulary = vocabularySize();
    logits.clear();
    for (std::uint32_t i = 0u; decoded && i < count; ++i)
    {
        const float *const row = llama_get_logits_ith(context, static_cast<std::int32_t>(i));
        logits.insert(logits.end(), row, row + vocabulary);
    }
    llama_batch_free(batch);
    llama_free(context);
    return decoded;
}

std::vector<std::uint32_t> ReferenceModel::tokenize(const std::string &text, bool parseSpecial) const
{
    std::vector<llama_token> ids(text.size() + 8u);
    std::int32_t count = llama_tokenize(_loaded->model, text.data(), static_cast<std::int32_t>(text.size()), ids.data(),
                                        static_cast<std::int32_t>(ids.size()), false, parseSpecial);
    if (count < 0)
    {
        ids.resize(static_cast<std::size_t>(-count));
        count = llama_tokenize(_loaded->model, text.data(), static_cast<std::int32_t>(text.size()), ids.data(),
                               static_cast<std::int32_t>(ids.size()), false, parseSpecial);
    }
    return std::vector<std::uint32_t>(ids.begin(), ids.begin() + std::max(count, 0));
}

std::uint32_t ReferenceModel::vocabularySize() const noexcept
{
    return _loaded->model == nullptr ? 0u : static_cast<std::uint32_t>(llama_n_vocab(_loaded->model));
}

bool runImage(const infer::Model &model, const std::vector<std::uint32_t> &tokens, SequenceAgreement &out,
              std::vector<float> &logits)
{
    const infer::ModelConfig &shape = model.config();
    const core::u32 positions = static_cast<core::u32>(tokens.size());
    if (!model.ready() || positions == 0u || positions > shape.contextLength)
        return false;

    infer::TensorArena arena{runArenaBytes(model, positions)};
    ImageRun run;
    infer::Transformer forward;
    infer::KvCache forwardCache;
    infer::ModelConfig cacheShape = shape;
    cacheShape.contextLength = positions;
    infer::VectorView forwardLogits = claimVector(arena, shape.vocabSize);
    if (!claimRun(arena, model, positions, run) || !forward.allocate(arena, model) ||
        !forwardCache.allocate(arena, cacheShape, infer::CachePolicy::Refuse) || forwardLogits.values == nullptr)
        return false;

    out.imageResidualPeak = 0.0;
    out.imageSaturations = 0u;
    out.imageMatchesForward = true;
    logits.clear();
    for (core::u32 position = 0u; position < positions; ++position)
    {
        if (tokens[position] >= shape.vocabSize || !stepImage(model, tokens[position], position, run, out) ||
            !forward.forward(tokens[position], position, forwardCache, forwardLogits))
            return false;
        for (core::u32 i = 0u; i < shape.vocabSize; ++i)
        {
            out.imageMatchesForward = out.imageMatchesForward && run.logits.at(i).raw() == forwardLogits.at(i).raw();
            logits.push_back(static_cast<float>(static_cast<double>(run.logits.at(i).raw()) / kFixedUnits));
        }
    }
    return true;
}

bool compareSequence(ReferenceModel &reference, const infer::Model &model, const std::vector<std::uint32_t> &tokens,
                     SequenceAgreement &out)
{
    std::vector<float> referenceLogits;
    std::vector<float> imageLogits;
    const std::uint32_t vocabulary = reference.vocabularySize();
    if (vocabulary != model.config().vocabSize || !reference.run(tokens, referenceLogits, out.referenceResidualPeak) ||
        !runImage(model, tokens, out, imageLogits))
        return false;

    out.positions.clear();
    for (std::size_t position = 0u; position < tokens.size(); ++position)
    {
        const float *const theirs = referenceLogits.data() + position * vocabulary;
        const float *const ours = imageLogits.data() + position * vocabulary;
        PositionAgreement agreement;
        agreement.imageToken = mostLikely(ours, vocabulary);
        agreement.referenceToken = mostLikely(theirs, vocabulary);
        agreement.referenceMargin = marginOf(theirs, vocabulary);
        for (std::uint32_t i = 0u; i < vocabulary; ++i)
            agreement.largestLogitDifference =
                std::max(agreement.largestLogitDifference, std::fabs(static_cast<double>(ours[i]) - theirs[i]));
        out.positions.push_back(agreement);
    }
    return true;
}

bool agreesAtEveryJudgedPosition(const std::vector<SequenceAgreement> &sequences) noexcept
{
    std::uint32_t judged = 0u;
    for (const SequenceAgreement &sequence : sequences)
    {
        if (sequence.positions.empty() || !sequence.imageMatchesForward || sequence.judgedDisagreements() != 0u)
            return false;
        judged += sequence.judged();
    }
    return judged != 0u;
}

bool controlFails(const std::vector<SequenceAgreement> &control, const std::vector<SequenceAgreement> &image) noexcept
{
    std::uint32_t disagreements = 0u;
    for (const SequenceAgreement &sequence : control)
        disagreements += sequence.judgedDisagreements();
    return disagreements != 0u && largestLogitDifferenceOf(control) > largestLogitDifferenceOf(image);
}

bool assembleControl(const infer::Model &image, Control control, double base, ControlModel &out)
{
    const infer::ModelConfig &shape = image.config();
    out.layers.clear();
    for (core::u32 l = 0u; l < shape.layers; ++l)
        out.layers.push_back(image.layer(l));
    const infer::RotaryFrequencies rotary = image.rotaryFrequencies();
    out.rotary.assign(rotary.turnsPerPosition, rotary.turnsPerPosition + rotary.pairs);
    if (control == Control::KeyValueSwapped)
    {
        for (infer::LayerWeights &layer : out.layers)
            std::swap(layer.key, layer.value);
    }
    else if (rotaryTurnsPerPosition(base, shape.headDim(), out.rotary).has_value())
        return false;

    infer::ModelParts parts;
    parts.shape = shape;
    parts.vocabularyText = image.vocab().blob();
    parts.vocabularyTextBytes = image.vocab().blobBytes();
    parts.vocabularyEntries = image.vocab().entries();
    parts.merges = image.merges().merges();
    parts.mergeCount = image.merges().size();
    parts.preTokenizer = image.preTokenizer();
    parts.chatFormat = image.chatTokens().format;
    parts.embedding = image.embedding();
    parts.layers = out.layers.data();
    parts.finalNorm = image.finalNorm();
    parts.rotary = infer::RotaryFrequencies{out.rotary.data(), static_cast<core::u32>(out.rotary.size())};
    out.arena =
        std::make_unique<infer::TensorArena>(16u * (shape.vocabSize + 4u * image.merges().size()) + kArenaSlack);
    return out.model.assemble(*out.arena, parts).verdict == infer::ModelReadVerdict::Read;
}

} // namespace lpl::convert

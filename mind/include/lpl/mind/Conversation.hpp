/**
 * @file Conversation.hpp
 * @brief One turn: recall, prompt, generate, remember.
 *
 * The loop the sovereign actually talks to. Retrieval narrows the corpus, the
 * prompt is assembled from what survived, generation streams out token by token,
 * and the turn is filed away for later recall.
 * 
 * Cancellation is a first-class parameter rather than an afterthought, because
 * barge-in is the defining behaviour of a voice assistant: a new question must be
 * able to abandon the answer in flight, and the abandoned turn is still remembered
 * with whatever partial text it produced. Forgetting it would leave the
 * conversation with a hole where the interruption happened.
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_MIND_CONVERSATION_HPP
#    define LPL_MIND_CONVERSATION_HPP

#    include <lpl/backend/Embedder.hpp>
#    include <lpl/backend/HostInference.hpp>
#    include <lpl/backend/VectorStore.hpp>
#    include <lpl/infer/Types.hpp>

#    include <functional>
#    include <string>
#    include <vector>

namespace lpl::mind {

/// Recall, prompt, generate, remember — the four steps of one turn.
class Conversation {
  public:
    Conversation(backend::VectorStore &store, backend::Embedder &embedder,
                 backend::HostInference &model, int topMemoryCount, float minimumSimilarity,
                 int maximumNewTokens);

    /// Handles one turn. @p onToken streams the answer as it is produced.
    ///
    /// @p shouldCancel is consulted before each token so a new question can abandon
    /// the answer in flight (barge-in). The abandoned turn is still remembered with
    /// whatever partial text it produced: dropping it would leave a hole in the
    /// conversation exactly where the interruption happened.
    std::string ask(const std::string &userInput,
                    const std::function<void(const std::string &)> &onToken,
                    const std::function<bool()> &shouldCancel = nullptr);

  private:
    backend::VectorStore &_store;
    backend::Embedder &_embedder;
    backend::HostInference &_model;
    int _topMemoryCount;
    float _minimumSimilarity;
    int _maximumNewTokens;
    std::vector<infer::ChatMessage> _session;
};

} // namespace lpl::mind

#endif // LPL_MIND_CONVERSATION_HPP

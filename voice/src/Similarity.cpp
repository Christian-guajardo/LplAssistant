/**
 * @file Similarity.cpp
 * @brief Implementation of cosine similarity between vocal signatures.
 *
 * Accumulated in double even though the inputs are float: the vectors are short
 * but the coefficients span several orders of magnitude, and the norms are where
 * precision is actually lost.
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/voice/Similarity.hpp>

#include <cmath>

namespace lpl::voice {

float cosineSimilarity(const Signature &left, const Signature &right)
{
    if (left.size() != right.size() || left.empty())
        return -1.0f;

    double dot = 0.0;
    double normLeft = 0.0;
    double normRight = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index)
    {
        dot += static_cast<double>(left[index]) * right[index];
        normLeft += static_cast<double>(left[index]) * left[index];
        normRight += static_cast<double>(right[index]) * right[index];
    }

    if (normLeft <= 0.0 || normRight <= 0.0)
        return -1.0f;

    return static_cast<float>(dot / (std::sqrt(normLeft) * std::sqrt(normRight)));
}

} // namespace lpl::voice

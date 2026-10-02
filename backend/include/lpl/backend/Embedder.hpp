/**
 * @file Embedder.hpp
 * @brief Semantic embeddings, L2-normalised.
 *
 * Vectors are normalised on the way out, so a dot product IS the cosine
 * similarity. That saves a square root on every comparison in the retrieval path,
 * and it means the database index and the in-process comparison agree by
 * construction rather than by convention.
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_BACKEND_EMBEDDER_HPP
#    define LPL_BACKEND_EMBEDDER_HPP

#    include <memory>
#    include <string>
#    include <vector>

namespace lpl::backend {

// Embeddings sémantiques via llama.cpp (modèle bge-m3, multilingue).
// Vecteurs normalisés L2 -> le produit scalaire == similarité cosinus.
class Embedder {
public:
    Embedder(const std::string& modelPath, int threadCount);
    ~Embedder();

    int dim() const;
    std::vector<float> embed(const std::string& text);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace lpl::backend

#endif // LPL_BACKEND_EMBEDDER_HPP

#pragma once
#include <memory>
#include <string>
#include <vector>

namespace laplace {

// Embeddings sémantiques via llama.cpp (modèle bge-m3, multilingue).
// Vecteurs normalisés L2 -> le produit scalaire == similarité cosinus.
class Embedder {
public:
    Embedder(const std::string& model_path, int n_threads);
    ~Embedder();

    int dim() const;
    std::vector<float> embed(const std::string& text);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace laplace

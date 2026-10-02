/**
 * @file VectorStore.hpp
 * @brief Long-term memory: structured filter first, similarity second.
 *
 * The composite retrieval rule, made concrete: a relational predicate narrows the
 * corpus before any vector is consulted, and only then does cosine ordering apply
 * to what remains. Filtering in SQL and ranking in vector space is what keeps the
 * prompt small, and a small prompt is what keeps a local model fast.
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_BACKEND_VECTORSTORE_HPP
#    define LPL_BACKEND_VECTORSTORE_HPP

#    include <memory>
#    include <string>
#    include <vector>

namespace lpl::backend {

struct MemoryRecord {
    long long   id;
    std::string category;
    std::string content;
    std::string createdAt;
    float       similarity; // 1 - distance cosinus (0 si non applicable)
};

// Couche PostgreSQL + pgvector : "SQL pour filtrer, vecteurs pour rapprocher".
class VectorStore {
public:
    VectorStore(const std::string& connectionString, int embeddingDimensions);
    ~VectorStore();

    void addMemory(const std::string& category, const std::string& content,
                    const std::vector<float>& embedding);

    // Composite Retrieval : filtre SQL optionnel (catégorie) puis tri par
    // similarité cosinus sur le sous-ensemble filtré.
    std::vector<MemoryRecord> search(const std::vector<float>& embedding,
                                  int limit, float minimumSimilarity,
                                  const std::string& category = "");

    std::vector<MemoryRecord> recent(int limit);
    void forgetAll();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace lpl::backend

#endif // LPL_BACKEND_VECTORSTORE_HPP

#pragma once
#include <memory>
#include <string>
#include <vector>

namespace laplace {

struct MemoryRow {
    long long   id;
    std::string category;
    std::string content;
    std::string created_at;
    float       similarity; // 1 - distance cosinus (0 si non applicable)
};

// Couche PostgreSQL + pgvector : "SQL pour filtrer, vecteurs pour rapprocher".
class Db {
public:
    Db(const std::string& conn_str, int embed_dim);
    ~Db();

    void add_memory(const std::string& category, const std::string& content,
                    const std::vector<float>& embedding);

    // Composite Retrieval : filtre SQL optionnel (catégorie) puis tri par
    // similarité cosinus sur le sous-ensemble filtré.
    std::vector<MemoryRow> search(const std::vector<float>& embedding,
                                  int limit, float min_similarity,
                                  const std::string& category = "");

    std::vector<MemoryRow> recent(int limit);
    void forget_all();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace laplace

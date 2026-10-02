/**
 * @file VectorStore.cpp
 * @brief Implementation of the vector-backed memory store.
 *
 *  
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/backend/VectorStore.hpp>
#include <pqxx/pqxx>
#include <sstream>
#include <stdexcept>

namespace lpl::backend {

struct VectorStore::Impl {
    pqxx::connection conn;
    Impl(const std::string& s) : conn(s) {}
};

// pgvector attend un littéral texte "[x1,x2,...]".
static std::string vec_to_pg(const std::vector<float>& v) {
    std::ostringstream os;
    os << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ',';
        os << v[i];
    }
    os << ']';
    return os.str();
}

VectorStore::VectorStore(const std::string& connectionString, int embeddingDimensions)
    : impl(new Impl(connectionString)) {
    pqxx::work txn(impl->conn);
    txn.exec("CREATE EXTENSION IF NOT EXISTS vector");
    txn.exec(
        "CREATE TABLE IF NOT EXISTS memories ("
        "  id BIGSERIAL PRIMARY KEY,"
        "  category TEXT NOT NULL DEFAULT 'conversation',"
        "  content TEXT NOT NULL,"
        "  embedding vector(" + std::to_string(embeddingDimensions) + "),"
        "  created_at TIMESTAMPTZ NOT NULL DEFAULT now())");
    txn.exec("CREATE INDEX IF NOT EXISTS memories_category_idx ON memories (category)");
    txn.exec("CREATE INDEX IF NOT EXISTS memories_created_idx ON memories (created_at)");
    // HNSW : recherche approximative rapide dès que la table grossit.
    txn.exec("CREATE INDEX IF NOT EXISTS memories_embedding_idx "
             "ON memories USING hnsw (embedding vector_cosine_ops)");
    txn.commit();
}

VectorStore::~VectorStore() = default;

void VectorStore::addMemory(const std::string& category, const std::string& content,
                    const std::vector<float>& embedding) {
    pqxx::work txn(impl->conn);
    txn.exec("INSERT INTO memories (category, content, embedding) VALUES ($1, $2, $3::vector)",
             pqxx::params{category, content, vec_to_pg(embedding)});
    txn.commit();
}

std::vector<MemoryRecord> VectorStore::search(const std::vector<float>& embedding,
                                  int limit, float minimumSimilarity,
                                  const std::string& category) {
    pqxx::work txn(impl->conn);
    std::string q =
        "SELECT id, category, content, created_at::text, "
        "       1 - (embedding <=> $1::vector) AS sim "
        "FROM memories WHERE embedding IS NOT NULL ";
    if (!category.empty()) q += "AND category = $4 ";
    q += "AND 1 - (embedding <=> $1::vector) >= $2 "
         "ORDER BY embedding <=> $1::vector LIMIT $3";

    pqxx::result r =
        category.empty()
            ? txn.exec(q, pqxx::params{vec_to_pg(embedding), minimumSimilarity, limit})
            : txn.exec(q, pqxx::params{vec_to_pg(embedding), minimumSimilarity, limit, category});

    std::vector<MemoryRecord> out;
    for (const auto& row : r)
        out.push_back({row[0].as<long long>(), row[1].as<std::string>(),
                       row[2].as<std::string>(), row[3].as<std::string>(),
                       row[4].as<float>()});
    return out;
}

std::vector<MemoryRecord> VectorStore::recent(int limit) {
    pqxx::work txn(impl->conn);
    pqxx::result r = txn.exec("SELECT id, category, content, created_at::text FROM memories "
                              "ORDER BY created_at DESC LIMIT $1",
                              pqxx::params{limit});
    std::vector<MemoryRecord> out;
    for (const auto& row : r)
        out.push_back({row[0].as<long long>(), row[1].as<std::string>(),
                       row[2].as<std::string>(), row[3].as<std::string>(), 0.f});
    return out;
}

void VectorStore::forgetAll() {
    pqxx::work txn(impl->conn);
    txn.exec("TRUNCATE memories");
    txn.commit();
}

} // namespace lpl::backend

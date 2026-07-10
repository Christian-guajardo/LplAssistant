#include "state.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

using nlohmann::json;
namespace fs = std::filesystem;

namespace laplace::research {

const char* phase_name(Phase p) {
    switch (p) {
        case Phase::initialized: return "initialized";
        case Phase::planned:     return "planned";
        case Phase::researching: return "researching";
        case Phase::answered:    return "answered";
        case Phase::reported:    return "reported";
        case Phase::complete:    return "complete";
        case Phase::error:       return "error";
    }
    return "initialized";
}

Phase phase_from_name(const std::string& name) {
    for (Phase p : {Phase::initialized, Phase::planned, Phase::researching,
                    Phase::answered, Phase::reported, Phase::complete, Phase::error})
        if (name == phase_name(p)) return p;
    return Phase::initialized; // phase inconnue : repli doux (star-hengxing)
}

void RunState::to_phase(Phase target) {
    if (target == Phase::error) { phase = target; return; }
    if (phase == Phase::error) { phase = target; error.clear(); return; }
    if ((int)target <= (int)phase)
        throw std::runtime_error(std::string("transition de phase refusée : ") +
                                 phase_name(phase) + " -> " + phase_name(target));
    phase = target;
}

std::string RunState::to_json() const {
    json j;
    j["version"]      = version;
    j["topic"]        = topic;
    j["guidance"]     = guidance;
    j["run_dir"]      = run_dir;
    j["phase"]        = phase_name(phase);
    j["gaps"]         = std::vector<std::string>(gaps.begin(), gaps.end());
    j["knowledge"]    = knowledge;
    j["queries_done"] = queries_done;
    j["step"]         = step;
    j["tokens_used"]  = tokens_used;
    j["token_budget"] = token_budget;
    j["beast_mode"]   = beast_mode;
    j["final_answer"] = final_answer;
    j["error"]        = error;
    j["sources"]      = json::array();
    for (const auto& s : sources)
        j["sources"].push_back({{"id", s.id}, {"url", s.url}, {"title", s.title},
                                {"snippet", s.snippet}, {"provider", s.provider},
                                {"cache_id", s.cache_id}, {"read", s.read},
                                {"fetch_ok", s.fetch_ok}});
    j["attempts"] = json::array();
    for (const auto& a : attempts)
        j["attempts"].push_back({{"answer", a.answer}, {"reason", a.reason}});
    j["diags"] = json::object();
    for (const auto& [name, d] : diags)
        j["diags"][name] = {{"results", d.results}, {"status", d.status}};
    return j.dump(2);
}

RunState RunState::from_json(const std::string& text) {
    json j = json::parse(text);
    RunState st;
    st.version      = j.value("version", 1);
    st.topic        = j.value("topic", "");
    st.guidance     = j.value("guidance", "");
    st.run_dir      = j.value("run_dir", "");
    st.phase        = phase_from_name(j.value("phase", "initialized"));
    st.step         = j.value("step", 0);
    st.tokens_used  = j.value("tokens_used", 0L);
    st.token_budget = j.value("token_budget", 60000L);
    st.beast_mode   = j.value("beast_mode", false);
    st.final_answer = j.value("final_answer", "");
    st.error        = j.value("error", "");
    for (const auto& g : j.value("gaps", json::array()))
        st.gaps.push_back(g.get<std::string>());
    for (const auto& k : j.value("knowledge", json::array()))
        st.knowledge.push_back(k.get<std::string>());
    for (const auto& q : j.value("queries_done", json::array()))
        st.queries_done.push_back(q.get<std::string>());
    for (const auto& s : j.value("sources", json::array())) {
        Source src;
        src.id       = s.value("id", 0);
        src.url      = s.value("url", "");
        src.title    = s.value("title", "");
        src.snippet  = s.value("snippet", "");
        src.provider = s.value("provider", "");
        src.cache_id = s.value("cache_id", "");
        src.read     = s.value("read", false);
        src.fetch_ok = s.value("fetch_ok", true);
        st.sources.push_back(std::move(src));
    }
    for (const auto& a : j.value("attempts", json::array()))
        st.attempts.push_back({a.value("answer", ""), a.value("reason", "")});
    if (j.contains("diags"))
        for (auto it = j["diags"].begin(); it != j["diags"].end(); ++it)
            st.diags[it.key()] = {it.value().value("results", 0),
                                  it.value().value("status", "")};
    return st;
}

void save_state(const RunState& st) {
    fs::create_directories(st.run_dir);
    const std::string path = st.run_dir + "/state.json";
    const std::string tmp  = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        f << st.to_json();
    }
    fs::rename(tmp, path); // atomique sur le même fs : jamais de checkpoint corrompu
}

RunState load_state(const std::string& run_dir) {
    std::ifstream f(run_dir + "/state.json", std::ios::binary);
    if (!f) throw std::runtime_error("pas de state.json dans " + run_dir);
    std::string text((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    RunState st  = RunState::from_json(text);
    st.run_dir   = run_dir;
    return st;
}

bool state_exists(const std::string& run_dir) {
    return fs::exists(run_dir + "/state.json");
}

} // namespace laplace::research

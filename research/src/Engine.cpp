/**
 * @file Engine.cpp
 * @brief Implementation of the agent loop: search, read, reflect, answer.
 *
 *  
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/research/Engine.hpp>
#include <lpl/research/Grammar.hpp>
#include <lpl/research/WebFetch.hpp>
#include <lpl/research/OpenAccess.hpp>
#include <lpl/research/Reader.hpp>
#include <lpl/research/Report.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sstream>

using nlohmann::json;

namespace lpl::research {

namespace {

// En deçà de ce nombre de caractères de texte extrait, une « lecture » n'a
// ramené qu'un stub (redirection JS, page d'atterrissage) : on la rejette.
constexpr size_t kMinReadableChars = 200;

constexpr const char* kSystem =
    "You are a rigorous research agent. You gather facts from real sources, "
    "never invent information, and always answer in valid JSON when asked to. "
    "Be dense and factual: entities, numbers, dates. No filler.";

// Filet défensif : certains petits modèles emballent la réponse finale dans un
// objet JSON {"answer": "...", "sources": [...]} malgré la consigne de prose.
// On récupère alors le seul champ "answer" ; sinon on renvoie le texte tel quel.
std::string unwrap_answer_json(const std::string& text) {
    // Repérer un objet JSON dominant dans la sortie (éventuel fence ```json).
    auto start = text.find('{');
    auto end   = text.rfind('}');
    if (start == std::string::npos || end == std::string::npos || end <= start)
        return text;
    try {
        auto j = nlohmann::json::parse(text.substr(start, end - start + 1));
        if (j.is_object() && j.contains("answer") && j["answer"].is_string()) {
            std::string ans = j["answer"].get<std::string>();
            if (!ans.empty()) return ans;
        }
    } catch (...) {
        // Pas du JSON valide : c'est de la prose, on la garde intacte.
    }
    return text;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

std::string slugify(const std::string& s, size_t max_len = 40) {
    std::string out;
    for (char c : s) {
        if (out.size() >= max_len) break;
        if (std::isalnum((unsigned char)c)) out += (char)std::tolower((unsigned char)c);
        else if (!out.empty() && out.back() != '-') out += '-';
    }
    while (!out.empty() && out.back() == '-') out.pop_back();
    return out.empty() ? "run" : out;
}

std::string timestamp_now() {
    char buf[32];
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
    return buf;
}

// Bloc de contexte commun à tous les prompts de la boucle : c'est la mémoire
// de l'agent (jina : connaissances / requêtes faites / échecs), bornée.
std::string context_block(const RunState& st, size_t knowledge_tail = 18) {
    std::ostringstream c;
    c << "RESEARCH TOPIC: " << st.topic << "\n";
    if (!st.guidance.empty()) c << "USER GUIDANCE: " << st.guidance << "\n";
    if (!st.gaps.empty()) c << "CURRENT QUESTION: " << st.gaps.front() << "\n";
    if (!st.knowledge.empty()) {
        c << "\nKNOWLEDGE GATHERED SO FAR:\n";
        size_t start = st.knowledge.size() > knowledge_tail
                           ? st.knowledge.size() - knowledge_tail : 0;
        for (size_t i = start; i < st.knowledge.size(); ++i)
            c << "- " << st.knowledge[i] << "\n";
    }
    if (!st.queries_done.empty()) {
        c << "\nSEARCH QUERIES ALREADY DONE (do not repeat them):\n";
        for (const auto& q : st.queries_done) c << "- " << q << "\n";
    }
    if (!st.attempts.empty()) {
        c << "\nPREVIOUS ANSWER ATTEMPTS REJECTED BECAUSE:\n";
        for (const auto& a : st.attempts) c << "- " << a.reason << "\n";
    }
    return c.str();
}

std::vector<std::string> parse_string_list(const json& j, const char* key,
                                           size_t max_items) {
    std::vector<std::string> out;
    for (const auto& it : j.value(key, json::array())) {
        if (out.size() >= max_items) break;
        std::string s = it.get<std::string>();
        if (!s.empty()) out.push_back(s);
    }
    return out;
}

} // namespace

ResearchOptions ResearchOptions::fromEnvironment() {
    ResearchOptions o;
    auto env = [](const char* k) -> std::string {
        const char* v = std::getenv(k);
        return v ? v : "";
    };
    if (auto v = env("LAPLACE_RESEARCH_DIR"); !v.empty()) o.outputRoot = v;
    if (auto v = env("LAPLACE_RESEARCH_BUDGET"); !v.empty()) o.tokenBudget = std::atol(v.c_str());
    if (auto v = env("LAPLACE_RESEARCH_MAX_STEPS"); !v.empty()) o.maximumSteps = std::atoi(v.c_str());
    o.search.searxngUrl  = env("LAPLACE_SEARXNG_URL");
    o.search.githubToken = env("GITHUB_TOKEN");
    o.unpaywall_email     = env("LAPLACE_UNPAYWALL_EMAIL");
    if (auto v = env("LAPLACE_RESEARCH_PROVIDERS"); !v.empty()) {
        std::istringstream ss(v);
        std::string item;
        while (std::getline(ss, item, ','))
            if (!item.empty()) o.search.providers.push_back(item);
    }
    return o;
}

Engine::Engine(LanguageModelClient& llm, ResearchOptions opt) : llm_(llm), opt_(std::move(opt)) {}

void Engine::log(const std::string& msg) const {
    if (opt_.verbose) std::fprintf(stderr, "[recherche] %s\n", msg.c_str());
}

std::string Engine::run(const std::string& topic, const std::string& guidance) {
    RunState st;
    st.topic        = topic;
    st.guidance     = guidance;
    st.tokenBudget = opt_.tokenBudget;
    st.run_dir      = opt_.outputRoot + "/" + timestamp_now() + "-" + slugify(topic);
    st.gaps.push_back(topic);
    save_state(st);
    CompressCacheRetrieve ccr(st.run_dir + "/cache");
    return execute(st, ccr);
}

std::string Engine::resume(const std::string& run_dir) {
    RunState st = load_state(run_dir);
    if (st.phase == Phase::complete)
        return st.run_dir + "/report.md";
    if (st.phase == Phase::error) st.to_phase(Phase::researching);
    log("reprise du run « " + st.topic + " » au pas " + std::to_string(st.step));
    CompressCacheRetrieve ccr(st.run_dir + "/cache");
    return execute(st, ccr);
}

std::string Engine::execute(RunState& st, CompressCacheRetrieve& ccr) {
    try {
        // Plan : décomposition initiale en sous-questions (une seule fois).
        if (st.phase == Phase::initialized) {
            log("plan : décomposition de « " + st.topic + " »");
            auto reply = llm_.complete(
                kSystem,
                context_block(st) +
                    "\nBreak this research topic into 2 or 3 specific sub-questions "
                    "that together cover it. Output JSON: {\"subquestions\": [...]}",
                stringListsGrammar({"subquestions"}), 300, 0.3f);
            st.tokens_used += reply.approximateTokens;
            try {
                auto subs = parse_string_list(json::parse(reply.text), "subquestions", 3);
                // Sous-questions d'abord, question d'origine en dernier.
                st.gaps.clear();
                for (const auto& s : subs) st.gaps.push_back(s);
                st.gaps.push_back(st.topic);
                for (const auto& s : subs) log("  sous-question : " + s);
            } catch (...) {
                log("plan illisible, on continue avec la question brute");
            }
            st.to_phase(Phase::planned);
            save_state(st);
            st.to_phase(Phase::researching);
            save_state(st);
        }

        if (st.phase == Phase::researching) loop(st, ccr);

        if (st.phase == Phase::answered) {
            std::string path = writeReport(st, llm_, opt_);
            st.to_phase(Phase::reported);
            save_state(st);
            st.to_phase(Phase::complete);
            save_state(st);
            log("rapport écrit : " + path);
            return path;
        }
    } catch (const std::exception& e) {
        st.error = e.what();
        st.to_phase(Phase::error);
        save_state(st);
        throw;
    }
    return st.run_dir + "/report.md";
}

void Engine::loop(RunState& st, CompressCacheRetrieve& ccr) {
    int steps_on_gap = 0;
    while (st.step < opt_.maximumSteps) {
        save_state(st);

        // Budget : 10 % réservés au beast mode (jina).
        if (!st.beast_mode && st.tokens_used > st.tokenBudget * 9 / 10) {
            st.beast_mode = true;
            log("budget épuisé à 90 % : BEAST MODE, réponse forcée");
        }

        // Gating : n'exposer que les actions légitimes à cet instant.
        std::vector<std::string> allowed;
        bool has_unread = std::any_of(st.sources.begin(), st.sources.end(),
                                      [](const Source& s) { return !s.read && s.fetchOk; });
        if (st.beast_mode) {
            allowed = {"answer"};
        } else {
            if ((int)st.sources.size() < opt_.maximumSources) allowed.push_back("search");
            if (has_unread) allowed.push_back("read");
            if ((int)st.gaps.size() < opt_.maximumGaps && !st.knowledge.empty())
                allowed.push_back("reflect");
            if (!st.knowledge.empty()) allowed.push_back("answer");
            if (allowed.empty()) allowed.push_back("search");
        }

        // Prompt de décision : contexte + catalogue des actions autorisées.
        std::ostringstream user;
        user << context_block(st);
        if (has_unread) {
            user << "\nUNREAD SOURCES (id: title — provider — snippet):\n";
            for (const auto& s : st.sources)
                if (!s.read && s.fetchOk)
                    user << s.id << ": " << s.title << " — " << s.provider
                         << " — " << s.snippet << "\n";
        }
        user << "\nChoose ONE action among: ";
        for (size_t i = 0; i < allowed.size(); ++i)
            user << (i ? ", " : "") << allowed[i];
        user << ".\n"
             << "- search: arg = a NEW specific web search query.\n"
             << "- read: arg = the id number of ONE unread source worth reading.\n"
             << "- reflect: arg = what is still missing (one sentence).\n"
             << "- answer: arg = empty string. Choose it only when the knowledge "
                "covers the topic (or when forced).\n"
             << "Output JSON: {\"action\": \"...\", \"arg\": \"...\"}";

        std::string action, arg;
        try {
            auto reply = llm_.complete(kSystem, user.str(),
                                       decisionGrammar(allowed), 200, 0.2f);
            st.tokens_used += reply.approximateTokens;
            auto j = json::parse(reply.text);
            action = j.value("action", "");
            arg    = j.value("arg", "");
        } catch (const std::exception& e) {
            log(std::string("décision illisible (") + e.what() + "), repli");
            action = has_unread ? "read" : "search";
            arg    = has_unread ? "" : (st.gaps.empty() ? st.topic : st.gaps.front());
        }
        ++st.step;
        log("pas " + std::to_string(st.step) + " [" +
            std::to_string(st.tokens_used) + "/" + std::to_string(st.tokenBudget) +
            " tok] action=" + action + (arg.empty() ? "" : " arg=" + arg.substr(0, 80)));

        if (action == "search")       do_search(st, arg);
        else if (action == "read")    do_read(st, ccr, arg);
        else if (action == "reflect") do_reflect(st);
        else if (action == "answer") {
            if (do_answer(st)) {
                st.to_phase(Phase::answered);
                save_state(st);
                return;
            }
        }

        // Rotation des lacunes : aucune question ne monopolise la boucle.
        if (++steps_on_gap >= 3 && st.gaps.size() > 1) {
            st.gaps.push_back(st.gaps.front());
            st.gaps.pop_front();
            steps_on_gap = 0;
        }
    }
    // Pas de réponse acceptée avant maximumSteps : réponse forcée.
    log("max_steps atteint : réponse forcée");
    st.beast_mode = true;
    do_answer(st);
    st.to_phase(Phase::answered);
    save_state(st);
}

void Engine::do_search(RunState& st, const std::string& query) {
    std::string q = query.empty() ? (st.gaps.empty() ? st.topic : st.gaps.front())
                                  : query;
    // Anti-répétition : une requête déjà faite est remplacée par la question
    // courante ; si c'est encore un doublon, le pas est perdu (et journalisé).
    auto is_dup = [&](const std::string& s) {
        for (const auto& done : st.queries_done)
            if (lower(done) == lower(s)) return true;
        return false;
    };
    if (is_dup(q)) {
        std::string alt = st.gaps.empty() ? st.topic : st.gaps.front();
        if (is_dup(alt)) { log("recherche dupliquée ignorée : " + q); return; }
        q = alt;
    }
    st.queries_done.push_back(q);
    auto results = aggregate_search(q, opt_.search, st.diags);
    int  added   = 0;
    for (auto& r : results) {
        bool known = std::any_of(st.sources.begin(), st.sources.end(),
                                 [&](const Source& s) { return s.url == r.url; });
        if (known || (int)st.sources.size() >= opt_.maximumSources) continue;
        Source s;
        s.id       = (int)st.sources.size() + 1;
        s.url      = r.url;
        s.title    = r.title;
        s.snippet  = r.snippet;
        s.provider = r.provider;
        st.sources.push_back(std::move(s));
        ++added;
    }
    log("recherche « " + q + " » : " + std::to_string(added) + " nouvelle(s) source(s)");
}

void Engine::do_read(RunState& st, CompressCacheRetrieve& ccr, const std::string& arg) {
    // Choix de la source : id demandé s'il est valide, sinon première non lue.
    Source* src = nullptr;
    int wanted  = std::atoi(arg.c_str());
    for (auto& s : st.sources)
        if (!s.read && s.fetchOk && (s.id == wanted || !src)) {
            src = &s;
            if (s.id == wanted) break;
        }
    if (!src) { log("read : aucune source non lue"); return; }

    // DOI / éditeur muré -> tenter une copie Open Access en dépôt (Unpaywall).
    std::string fetch_url = resolve_open_access(src->url, opt_.unpaywall_email);
    if (fetch_url != src->url)
        log("OA [S" + std::to_string(src->id) + "] -> " + fetch_url);

    auto r = fetchUrl(fetch_url, 25);
    if (r.status < 200 || r.status >= 300 || r.body.empty()) {
        src->fetchOk = false;
        src->read     = true;
        log("lecture échouée [S" + std::to_string(src->id) + "] " + fetch_url +
            " (HTTP " + std::to_string(r.status) + ")");
        return;
    }

    // Routage par type de contenu : PDF -> pdftotext, HTML -> reader, sinon brut.
    PageText page;
    if (looksLikePortableDocument(r.body)) {
        page = portableDocumentToText(r.body);
        if (page.text.empty()) { // pdftotext absent ou PDF illisible : échec propre.
            src->fetchOk = false;
            src->read     = true;
            log("PDF illisible [S" + std::to_string(src->id) + "] " + fetch_url +
                " (pdftotext absent ou extraction vide)");
            return;
        }
        page.title = src->title;
    } else {
        page = looksLikeHtml(r.body) ? htmlToText(r.body)
                                       : PageText{src->title, r.body};
    }
    if (page.title.empty()) page.title = src->title;

    // Garde-fou anti-lecture-maigre : les pages JS de redirection/atterrissage
    // (« Redirecting », landing d'un dépôt type MPG.Pure/hdl.handle) ne rendent
    // que quelques caractères. Les compter comme sources fausserait la synthèse
    // et polluerait les citations : on les traite en échec propre.
    if (page.text.size() < kMinReadableChars) {
        src->fetchOk = false;
        src->read     = true;
        log("contenu trop maigre [S" + std::to_string(src->id) + "] " + fetch_url +
            " (" + std::to_string(page.text.size()) + " chars)");
        return;
    }

    auto stored   = ccr.store(src->url, page.title, page.text, (size_t)opt_.skimCharacters);
    src->cacheId = stored.id;
    src->read     = true;

    // Extraction : learnings denses + questions de suivi (map de map-reduce).
    std::ostringstream user;
    user << context_block(st, 8)
         << "\nSOURCE [S" << src->id << "] " << page.title << " (" << src->url << ")\n"
         << "CONTENT:\n" << stored.skim << "\n\n"
         << "Extract up to " << opt_.maximumLearningsPerRead
         << " unique, information-dense learnings relevant to the topic "
            "(entities, numbers, dates), and up to 2 follow-up questions this "
            "source raises. Output JSON: {\"learnings\": [...], \"followups\": [...]}";
    try {
        auto reply = llm_.complete(kSystem, user.str(),
                                   stringListsGrammar({"learnings", "followups"}),
                                   500, 0.3f);
        st.tokens_used += reply.approximateTokens;
        auto j = json::parse(reply.text);
        for (auto& l : parse_string_list(j, "learnings",
                                         (size_t)opt_.maximumLearningsPerRead))
            st.knowledge.push_back("[S" + std::to_string(src->id) + "] " + l);
        for (auto& f : parse_string_list(j, "followups", 2)) {
            bool dup = std::any_of(st.gaps.begin(), st.gaps.end(),
                                   [&](const std::string& g) { return lower(g) == lower(f); });
            if (!dup && (int)st.gaps.size() < opt_.maximumGaps) st.gaps.push_back(f);
        }
        log("lu [S" + std::to_string(src->id) + "] " + page.title + " (" +
            std::to_string(stored.totalCharacters) + " chars, cache " + stored.id + ")");
    } catch (const std::exception& e) {
        log(std::string("extraction illisible : ") + e.what());
    }
}

void Engine::do_reflect(RunState& st) {
    std::ostringstream user;
    user << context_block(st)
         << "\nName up to 3 NEW sub-questions that would fill the most important "
            "knowledge gaps on this topic. They must not repeat existing "
            "questions. Output JSON: {\"subquestions\": [...]}";
    try {
        auto reply = llm_.complete(kSystem, user.str(),
                                   stringListsGrammar({"subquestions"}), 300, 0.4f);
        st.tokens_used += reply.approximateTokens;
        auto subs = parse_string_list(json::parse(reply.text), "subquestions", 3);
        int added = 0;
        for (auto& s : subs) {
            bool dup = std::any_of(st.gaps.begin(), st.gaps.end(),
                                   [&](const std::string& g) { return lower(g) == lower(s); });
            if (!dup && (int)st.gaps.size() < opt_.maximumGaps) {
                st.gaps.push_back(s);
                ++added;
            }
        }
        log("réflexion : " + std::to_string(added) + " lacune(s) ajoutée(s)");
    } catch (const std::exception& e) {
        log(std::string("réflexion illisible : ") + e.what());
    }
}

bool Engine::do_answer(RunState& st) {
    std::ostringstream user;
    user << context_block(st, 60)
         << "\nWrite the final answer to the RESEARCH TOPIC using ONLY the "
            "knowledge above. Cite sources inline with their [S<id>] tags. "
            "Write in the language of the topic. Be complete but dense. "
            "Output prose in Markdown ONLY — do NOT wrap the answer in JSON, "
            "code fences, or a \"answer\" field.";
    auto draft = llm_.complete(kSystem, user.str(), "", 900, 0.4f);
    st.tokens_used += draft.approximateTokens;
    draft.text = unwrap_answer_json(draft.text);

    if (st.beast_mode || (int)st.attempts.size() >= opt_.maximumAnswerAttempts) {
        st.final_answer = draft.text;
        return true;
    }
    // Évaluateur (jina) : définitude, attribution, couverture.
    std::ostringstream ev;
    ev << "RESEARCH TOPIC: " << st.topic << "\n\nCANDIDATE ANSWER:\n"
       << draft.text
       << "\n\nJudge this answer: is it definitive (no evasion), grounded in the "
          "cited sources, and does it cover the topic? "
          "Output JSON: {\"pass\": true|false, \"reason\": \"...\"}";
    try {
        auto verdict = llm_.complete(kSystem, ev.str(), evaluationGrammar(), 200, 0.2f);
        st.tokens_used += verdict.approximateTokens;
        auto j = json::parse(verdict.text);
        if (j.value("pass", false)) {
            st.final_answer = draft.text;
            return true;
        }
        st.attempts.push_back({draft.text, j.value("reason", "raison inconnue")});
        log("réponse rejetée : " + st.attempts.back().reason);
        return false;
    } catch (...) {
        // Évaluateur illisible : ne pas bloquer la sortie.
        st.final_answer = draft.text;
        return true;
    }
}

} // namespace lpl::research

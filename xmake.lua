-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for LplAssistant — the voice and the thought.
--
-- Same dual contract as LplPlugin: the modules a constrained target must run
-- compile both for Linux and -ffreestanding into the kernel's libassistant.a,
-- because the demon runs in ring 0 on the server profile. The line between the two
-- halves is the one the project already drew for editor/ versus procgen/ — writer
-- versus reader, hosted versus freestanding.
--
--   infer/      freestanding   how the demon computes
--   mind/       freestanding   who the demon is
--   satellite/  freestanding   the wire format of a room node (three consumers)
--   voice/      freestanding   telling members of a household apart
--   research/   hosted         deep research: search, read, reflect, answer
--   backend/    hosted         inference runtime, database, audio, agent protocol
--
-- @author Christian-guajardo, MasterLaplace
-- @copyright MIT License
-- /////////////////////////////////////////////////////////////////////////////

add_rules("mode.debug", "mode.release")

set_languages("c++20")

-- Warnings at the project standard. The two OTHER project-wide flags
-- (`-fno-rtti -fno-exceptions`) are applied PER MODULE rather than globally, and the
-- reason is the same in both cases and worth stating once:
--
--   an audit of this repository found ZERO uses of dynamic_cast, typeid, or a throw
--   that our own code depends on — but the hosted dependencies use both. pqxx calls
--   typeid in a header, and pqxx/nlohmann_json/cpp-httplib all throw.
--
-- So the flags land where they are true: the FREESTANDING modules forbid both,
-- exactly like everything else linked into the kernel, and the hosted module allows
-- both with a written justification. That is not a derogation, it is the
-- reader/writer boundary as the compiler sees it — and bci/ set the precedent in
-- LplPlugin for precisely this situation.
-- `allextra` + erreur, le standard du projet, et atteignable parce que la mesure dit
-- ZÉRO warning après migration. Le seul qui restait — un appel pqxx déprécié — a été
-- porté sur `exec(query, pqxx::params{...})` plutôt que réduit au silence.
set_warnings("allextra", "error")

-- ─────────────────────────────────────────────────────────────────────────────
-- Le socle LplPlugin (Fixed32, CORDIC, les ombrelles lpl::pmr) est une AMÉLIORATION
-- détectée, jamais une exigence. Ce dépôt doit se construire, tourner et être testé
-- seul — comme LplPlugin se construit sans LplKernel. Même dispositif que le noyau,
-- qui boote sans le moteur quand le sous-module manque (LPL_PLUGIN_UNAVAILABLE).
--
--   présent  -> LPL_HAS_FOUNDATION : contrat de déterminisme accessible, et ces
--               modules peuvent être compilés -ffreestanding pour le ring 0 ;
--   absent   -> build AUTONOME, hôte uniquement. Fixed32 n'est pas émulé : une
--               fausse virgule fixe laisserait un build autonome revendiquer une
--               parité qu'il ne peut pas avoir.
-- ─────────────────────────────────────────────────────────────────────────────
option("foundation")
    set_default("auto")
    set_values("auto", "y", "n")
    set_showmenu(true)
    set_description("Use the LplPlugin foundation when available (auto|y|n)")
option_end()

local kFoundationRoot = "../LplKernel/LplPlugin"

local function hasFoundation()
    local mode = get_config("foundation") or "auto"
    if mode == "n" then
        return false
    end
    if os.isdir(path.join(kFoundationRoot, "core/include")) then
        return true
    end
    if mode == "y" then
        raise("--foundation=y was requested but " .. kFoundationRoot .. " is not present")
    end
    return false
end

-- L'ombrelle du dépôt : le seul endroit qui sait laquelle des deux situations on est.
add_includedirs("include")

if hasFoundation() then
    add_includedirs(path.join(kFoundationRoot, "core/include"))
    add_includedirs(path.join(kFoundationRoot, "math/include"))
    add_defines("LPL_HAS_FOUNDATION")
else
    print("[%s] standalone build: LplPlugin foundation absent, host only", "LplAssistant")
end

add_requires("nlohmann_json")
add_requires("cpp-httplib")
add_requires("llama.cpp", {configs = {shared = false}})

-- ─────────────────────────────────────────────────────────────────────────────
-- Le STT vit dans un binaire séparé : le ggml embarqué de whisper.cpp 1.6.2 est
-- incompatible au link avec celui de llama.cpp b3775 (segfault sinon). C'est aussi
-- pourquoi backend/ EXCLUT SpeechInput.cpp de sa bibliothèque : les réunir dans un
-- même artefact reproduirait exactement l'incompatibilité que la séparation évite.
--
-- Déclaré en OPTION et non en dépendance « optionnelle » : `optional = true`
-- empêche xmake d'échouer à récupérer le paquet, mais la cible tente quand même de
-- compiler contre `whisper.h` et le build casse plus loin, avec un message qui ne
-- dit nulle part que la dépendance était facultative.
-- ─────────────────────────────────────────────────────────────────────────────
option("stt")
    set_default(true)
    set_showmenu(true)
    set_description("Build the speech-to-text binary (requires whisper.cpp)")
option_end()

-- Nœud de développement : linke PulseAudio, qui existe sur un poste de travail et
-- nulle part ailleurs. Sans ce garde-fou, une machine sans serveur de son fait
-- échouer le build de TOUT le dépôt pour un satellite de secours.
option("satellite")
    set_default(false)
    set_showmenu(true)
    set_description("Build the local microphone node (requires PulseAudio)")
option_end()

if has_config("stt") then
    add_requires("whisper.cpp", {configs = {shared = false}})
end

includes("infer", "mind", "satellite", "voice", "research", "backend")

-- ─────────────────────────────────────────────────────────────────────────────
-- Applications
-- ─────────────────────────────────────────────────────────────────────────────
target("lpl-assistant")
    set_kind("binary")
    set_group("apps")
    add_deps("lpl-infer", "lpl-mind", "lpl-voice", "lpl-research", "lpl-assistant-backend")
    add_files("apps/laplace/main.cpp")
    add_packages("llama.cpp", "nlohmann_json", "cpp-httplib")
    add_syslinks("pqxx", "pq", "pthread")
target_end()

-- Pas de binaire autonome pour la recherche : `research/` est une BIBLIOTHÈQUE
-- D'OUTILS que l'intelligence appelle (voir research/Tools.hpp), pas une commande
-- qu'un humain lance. C'est la même inversion que partout ailleurs dans le projet —
-- l'IA est le directeur, le C++ déterministe est l'exécutant — appliquée ici à la
-- recherche documentaire.

if has_config("stt") then
    target("lpl-stt")
        set_kind("binary")
        set_group("apps")
        add_files("backend/src/SpeechInput.cpp", "apps/speech/main.cpp")
        add_includedirs("backend/include", "infer/include")
        add_packages("whisper.cpp")
        add_syslinks("pthread")
    target_end()
end

if has_config("satellite") then
    target("lpl-mic")
        set_kind("binary")
        set_group("apps")
        add_deps("lpl-satellite")
        add_files("apps/satellite/main.cpp")
        add_syslinks("pulse-simple", "pulse", "pthread")
    target_end()
end

target("lpl-demon")
    set_kind("binary")
    set_group("apps")
    add_deps("lpl-infer", "lpl-mind", "lpl-assistant-backend")
    add_files("apps/demon/main.cpp")
    add_packages("llama.cpp", "nlohmann_json", "cpp-httplib")
    add_syslinks("pqxx", "pq", "pthread")
target_end()

-- ─────────────────────────────────────────────────────────────────────────────
-- Gates. Le même contrat que partout : ce que l'hôte calcule et ce que le ring 0
-- calcule doivent folder à l'identique.
-- ─────────────────────────────────────────────────────────────────────────────
target("test-infer-parity")
    set_kind("binary")
    set_group("tests")
    set_default(false)
    add_deps("lpl-infer")
    add_files("tests/test_infer_parity.cpp")
target_end()

target("test-grammar-constraint")
    set_kind("binary")
    set_group("tests")
    set_default(false)
    add_deps("lpl-infer", "lpl-mind")
    add_files("tests/test_grammar_constraint.cpp")
target_end()

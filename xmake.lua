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
    set_default("detect")
    -- ⚠ The values used to be auto|y|n, and that was unusable: xmake COERCES an option
    -- whose values look boolean, so `--foundation=n` AND `--foundation=auto` were both
    -- stored as the boolean false — the documented default and the opt-out became the
    -- same setting. Measured, not guessed: y stored true, n stored false, auto stored
    -- false — and so was "auto", which xmake also reads as a boolean. Three words it has
    -- no boolean reading for keep the three settings apart.
    set_values("detect", "force", "off")
    set_showmenu(true)
    set_description("Use the LplPlugin foundation when available (detect|force|off)")
option_end()

-- LplPlugin is a sibling of this repository. LPLPLUGIN_ROOT names another checkout; the
-- kernel's submodule path is still tried while it exists (MasterLaplace/LplKernel#431).
local function foundationRoot()
    local root = os.getenv("LPLPLUGIN_ROOT")
    if root and root ~= "" then
        return root
    end
    for _, candidate in ipairs({"../LplPlugin", "../LplKernel/LplPlugin"}) do
        local candidatePath = path.join(os.scriptdir(), candidate)
        if os.isdir(path.join(candidatePath, "core/include")) then
            return candidatePath
        end
    end
    return path.join(os.scriptdir(), "../LplPlugin")
end

local kProjectRoot = os.scriptdir()
local kFoundationRoot = foundationRoot()
local kConfigHeader = path.join(os.scriptdir(), "include/lplassistant/config.h")

rule("laplace.version")
    on_load(function (target)
        local text = io.readfile(kConfigHeader)
        local version = {}
        for _, part in ipairs({"MAJOR", "MINOR", "PATCH"}) do
            table.insert(version, text:match("#define LPLASSISTANT_VERSION_" .. part .. " (%d+)"))
        end
        target:set("version", table.concat(version, "."))
    end)
rule_end()

add_rules("laplace.version")

local function hasFoundation()
    -- Booleans are still handled, because a configuration stored by an older checkout
    -- carries them: true reads as force, false reads as off. Without that a stale
    -- .xmake/ would flip a build to standalone with no message saying why.
    local mode = get_config("foundation")
    if mode == nil then
        mode = "detect"
    elseif mode == true then
        mode = "force"
    elseif mode == false then
        -- A stored boolean can only come from an older checkout, where it meant either
        -- "auto" or "n" and there is no way left to tell which. Read as DETECT, because
        -- the failure modes are not symmetric: detecting a sibling that is there costs
        -- nothing, while silently skipping one makes every gate target vanish.
        mode = "detect"
    end
    if mode == "off" then
        return false
    end
    -- Both are checked because both are USED: core/math carry the determinism
    -- contract, and agent/ carries the one decision seam the hosted demon and the
    -- ring-0 one share. Detecting only core/ and then including agent/ would fail at
    -- compile time on a checkout that has one and not the other, which is a worse
    -- error than an honest standalone build.
    if os.isdir(path.join(kFoundationRoot, "core/include")) and
       os.isdir(path.join(kFoundationRoot, "agent/include")) then
        return true
    end
    if mode == "force" then
        raise("--foundation=force was requested but " .. kFoundationRoot .. " is not present")
    end
    return false
end

-- L'ombrelle du dépôt : le seul endroit qui sait laquelle des deux situations on est.
add_includedirs("include")

local kHasFoundation = hasFoundation()

-- Stamps the one translation unit that prints a tool's identity with what the source cannot
-- know: the commits this build came from, and the build itself. Only that target recompiles
-- when a commit changes.
rule("laplace.identity")
    on_load(function (target)
        local function git(directory, arguments)
            local output = try { function () return os.iorunv("git", table.join({"-C", directory}, arguments)) end }
            return output and output:trim() or ""
        end
        local function commit(directory)
            local sha = git(directory, {"rev-parse", "--short=7", "HEAD"})
            if sha == "" then
                return "unknown"
            end
            local dirty = git(directory, {"status", "--porcelain", "--untracked-files=no"})
            return dirty ~= "" and (sha .. "-dirty") or sha
        end
        target:add("defines", 'LPLASSISTANT_COMMIT="' .. commit(kProjectRoot) .. '"')
        target:add("defines", 'LPLASSISTANT_BUILD="' .. target:plat() .. "." .. (get_config("mode") or "debug") .. '"')
        if kHasFoundation then
            target:add("defines", 'LPLPLUGIN_COMMIT="' .. commit(kFoundationRoot) .. '"')
        end
    end)
rule_end()

if kHasFoundation then
    add_includedirs(path.join(kFoundationRoot, "core/include"))
    add_includedirs(path.join(kFoundationRoot, "math/include"))
    add_includedirs(path.join(kFoundationRoot, "memory/include"))
    -- agent/, for its headers only and never its library: lpl/agent/Decision.hpp is
    -- self-contained (it includes core/Types.hpp and nothing else) and declares pure
    -- interfaces, so consuming it costs an include path and no link.
    add_includedirs(path.join(kFoundationRoot, "agent/include"))
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

-- ─────────────────────────────────────────────────────────────────────────────
-- Le socle, COMPILÉ. Les `add_includedirs` ci-dessus donnent les en-têtes ; trois
-- unités de traduction de LplPlugin ont en plus du code hors-ligne dont le chemin
-- freestanding a besoin : CORDIC (les rotations rotatives), l'arène (UNE seule
-- implémentation dans tout le projet, dont le compte d'octets est foldé par la
-- gate) et Log (le puits de journalisation).
--
-- C'est la décision 2 de LplKernel/docs/ARCHITECTURE_cible.md, tranchée ici de la
-- façon la moins engageante : une cible locale qui compile trois fichiers, et non
-- un paquet xmake. En ring 0 la question ne se pose pas — ces objets sont déjà dans
-- libengine.a, et libassistant.a ne les redéfinit pas.
if kHasFoundation then
    target("lpl-foundation")
        set_kind("static")
        set_group("modules")
        add_cxxflags("-fno-rtti", "-fno-exceptions", { force = true })
        add_files(path.join(kFoundationRoot, "math/src/Cordic.cpp"))
        add_files(path.join(kFoundationRoot, "memory/src/ArenaAllocator.cpp"))
        add_files(path.join(kFoundationRoot, "core/src/Log.cpp"))
    target_end()
end

includes("infer", "mind", "satellite", "voice", "research", "backend")

-- ─────────────────────────────────────────────────────────────────────────────
-- Applications
-- ─────────────────────────────────────────────────────────────────────────────
target("lpl-tool-identity")
    set_kind("static")
    set_group("apps")
    add_rules("laplace.identity")
    add_includedirs("apps", {public = true})
    add_files("apps/Identity.cpp")
target_end()

target("lpl-assistant")
    set_kind("binary")
    set_group("apps")
    add_deps("lpl-tool-identity", "lpl-infer", "lpl-mind-hosted", "lpl-voice", "lpl-research", "lpl-assistant-backend")
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
--
-- Déclarées seulement quand le socle est là, et c'est le point honnête de
-- Foundation.hpp : sans Fixed32 il n'y a pas de contrat de déterminisme à éprouver,
-- donc la cible est ABSENTE plutôt que stubée. Une gate qui passe en n'ayant rien
-- vérifié est pire que pas de gate.
if kHasFoundation then

target("test-infer-parity")
    set_kind("binary")
    set_group("tests")
    set_default(false)
    add_deps("lpl-infer", "lpl-foundation")
    add_files("tests/test_infer_parity.cpp")
target_end()

target("test-grammar-constraint")
    set_kind("binary")
    set_group("tests")
    set_default(false)
    add_deps("lpl-infer", "lpl-foundation")
    add_files("tests/test_grammar_constraint.cpp")
target_end()

target("test-satellite-parity")
    set_kind("binary")
    set_group("tests")
    set_default(false)
    add_deps("lpl-satellite", "lpl-foundation")
    add_files("tests/test_satellite_parity.cpp")
target_end()

target("test-agency-parity")
    set_kind("binary")
    set_group("tests")
    set_default(false)
    add_deps("lpl-mind", "lpl-foundation")
    add_files("tests/test_agency_parity.cpp")
target_end()

end -- if kHasFoundation

-- The format of a research report, held to the contract ANOTHER repository reads.
--
-- Outside the `kHasFoundation` block: what it checks is text, not arithmetic, so a
-- standalone checkout still has something real to run, the same policy as
-- `test-corpus-identity` in LplKnowledge.
--
-- Warning: it does NOT depend on `lpl-research`, whose package fetch can fail (it is
-- already why the maintainer's local `validate.sh` script, not part of this repository,
-- configures with `--stt=n`). A gate that does not build is red for a reason unrelated
-- to what it checks, so it compiles ReportMarkdown.cpp alone.
target("test-research-report")
    set_kind("binary")
    set_group("tests")
    set_default(false)
    add_includedirs("research/include")
    add_files("tests/test_research_report.cpp", "research/src/ReportMarkdown.cpp")
target_end()

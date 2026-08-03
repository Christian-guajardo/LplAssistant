-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::research module.
-- research/ build configuration — autonomous deep research: a typed state machine
-- whose every decision is physically constrained by a grammar.
--
-- HOST ONLY. This half speaks HTTP, writes a disk cache and parses HTML; none of it
-- belongs in ring 0. What a constrained target consumes is the REPORT it produces,
-- not the machinery that produced it — the same reader/writer line as everywhere
-- else in the project.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-research")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-assistant-backend")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/research/**.hpp)")
    add_packages("llama.cpp", "nlohmann_json", "cpp-httplib")
target_end()

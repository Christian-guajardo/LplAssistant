-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::mind module.
-- mind/ build configuration — identity, memory and the loop that turns thought into action.
-- Freestanding, and separate from infer/ on purpose: one module is how the demon
-- computes, this one is who it is. Everything here is data the sovereign can read,
-- edit and version — a persona, a set of notes, a transcript — rather than weights
-- nobody can inspect.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-mind")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-infer", "lpl-assistant-backend")

    -- The kernel's rules, because these sources are compiled into the kernel. Not
    -- inherited from the root on purpose: the hosted module next door cannot obey
    -- them (its dependencies use RTTI and throw), and a flag set globally then
    -- overridden in one place reads as an exception rather than as a boundary.
    add_cxxflags("-fno-rtti", "-fno-exceptions", { force = true })

    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/mind/**.hpp)")
target_end()

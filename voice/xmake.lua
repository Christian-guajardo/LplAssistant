-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::voice module.
-- voice/ build configuration — telling members of a household apart by ear.
--
-- Freestanding: this is long-term spectral statistics and a pitch estimate over a
-- PCM buffer, compared by cosine. No neural model, which is the point — the need
-- came before the solution, and the need is only to know which of a few people is
-- talking so the right conversation continues. It is explicitly NOT security
-- biometrics, and the code should keep saying so.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-voice")
    set_kind("static")
    set_group("modules")

    -- The kernel's rules, because these sources are compiled into the kernel. Not
    -- inherited from the root on purpose: the hosted module next door cannot obey
    -- them (its dependencies use RTTI and throw), and a flag set globally then
    -- overridden in one place reads as an exception rather than as a boundary.
    add_cxxflags("-fno-rtti", "-fno-exceptions", { force = true })

    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/voice/**.hpp)")
target_end()

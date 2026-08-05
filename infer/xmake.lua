-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::infer module.
-- infer/ build configuration — the forward pass, freestanding.
-- The demon thinks in ring 0 on the server profile, so this module obeys the
-- kernel's rules and not a framework's: no libc, no exceptions, no heap after
-- init, every buffer drawn from a preallocated arena. Integer quantisation is not
-- a size optimisation here — it is what makes inference expressible with the same
-- arithmetic discipline as the rest of the authoritative state.
--
-- The old objection to inference in ring 0 was that a freestanding kernel does not
-- preserve SSE/FPU registers across interrupts. That objection was answered by the
-- P3 gate: FXSAVE/FXRSTOR in the ISR shipped long ago.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-infer")
    set_kind("static")
    set_group("modules")
    -- No add_deps on lpl-core / lpl-math: those targets belong to LplPlugin's xmake
    -- project. Their headers arrive through the root add_includedirs, and the three
    -- translation units that have out-of-line code (CORDIC, the arena, the log sink)
    -- through the root's `lpl-foundation` target — decision 2 of
    -- LplKernel/docs/ARCHITECTURE_cible.md, settled as a local target rather than a
    -- package. Not depended on from here: this module is also compiled into
    -- libassistant.a, where those objects already live in libengine.a.

    -- The kernel's rules, because these sources are compiled into the kernel. Not
    -- inherited from the root on purpose: the hosted module next door cannot obey
    -- them (its dependencies use RTTI and throw), and a flag set globally then
    -- overridden in one place reads as an exception rather than as a boundary.
    add_cxxflags("-fno-rtti", "-fno-exceptions", { force = true })

    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/infer/**.hpp)")
target_end()

-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::mind module.
-- mind/ build configuration — identity, memory and the loop that turns thought into action.
-- Freestanding, and separate from infer/ on purpose: one module is how the demon
-- computes, this one is who it is. Everything here is data the sovereign can read,
-- edit and version — a persona, a set of notes, a transcript — rather than weights
-- nobody can inspect.
--
-- TWO targets, and the split is the header above made true. `lpl-mind` is the agency
-- core: it depends on the foundation and on `lpl-infer`, both of which are
-- freestanding and both of which are already in ring 0, so the whole target compiles
-- into the kernel and is folded by a gate. `lpl-mind-hosted` is the one turn that
-- talks to a real model over a real vector store, so it drags in llama.cpp and libpq
-- — things no freestanding build can have. They were one target until the gate needed
-- to be built, at which point "freestanding" and "links a database client" could no
-- longer both be true.
--
-- The infer/ dependency is the model-backed reasoner alone (`Reasoning.cpp`), and it
-- does not undo the separation stated above: computing and being somebody are still
-- different modules, and every other file here still compiles without a transformer.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-mind")
    set_kind("static")
    set_group("modules")
    -- infer/, for Reasoning.cpp alone: the reasoner that decides by generation rather
    -- than by rule. Freestanding, so it comes into ring 0 with the rest.
    add_deps("lpl-infer")

    -- The kernel's rules, because these sources are compiled into the kernel. Not
    -- inherited from the root on purpose: the hosted module next door cannot obey
    -- them (its dependencies use RTTI and throw), and a flag set globally then
    -- overridden in one place reads as an exception rather than as a boundary.
    add_cxxflags("-fno-rtti", "-fno-exceptions", { force = true })

    add_includedirs("include", { public = true })
    add_files("src/Budget.cpp", "src/Dialogue.cpp", "src/Intent.cpp", "src/Memory.cpp",
              "src/Parity.cpp", "src/Persona.cpp", "src/ReAct.cpp", "src/Reasoning.cpp",
              "src/Recall.cpp")
    add_headerfiles("include/(lpl/mind/**.hpp)")
target_end()

target("lpl-mind-hosted")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-mind", "lpl-infer", "lpl-assistant-backend")

    add_includedirs("include", { public = true })
    add_files("src/Conversation.cpp")
target_end()

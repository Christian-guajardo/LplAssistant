-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::satellite module.
-- satellite/ build configuration — the peripheral nervous system's wire format
-- and its local decisions, with no dependency on how the audio was acquired.
--
-- Naming, since the word invites the wrong picture: a satellite here is a small
-- node you put in a ROOM — a microphone, a speaker, a relay, a few tens of euros,
-- one per living space. Nothing orbital. It is called a satellite because it orbits
-- the central node, which holds the model and does the thinking.
-- (That said, the constraints are genuinely the same ones spacecraft designers
-- work under — bounded memory, no allocation, sleep between events, survive a
-- link that drops — so the joke about it being space-ready is only half a joke.)
--
-- Freestanding, and it has to be: this module has THREE consumers that share
-- nothing else — and, unlike everything else in this project, two of them will not
-- be x86. The hosted development satellite talks to PulseAudio, the kernel
-- satellite profile talks to a codec through the HAL, and the eventual
-- microcontroller firmware talks to an I2S peripheral. One protocol, one voice
-- activity detector, one wake-word gate — because three implementations of a wire
-- format is three chances for them to disagree about what END means.
--
-- Nothing here allocates, and nothing here knows what a socket is. Acquisition and
-- transport are injected; this module decides WHEN to speak and WHAT the bytes mean.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-satellite")
    set_kind("static")
    set_group("modules")
    -- No add_deps on lpl-core / lpl-math: those targets belong to LplPlugin's
    -- xmake project. Their headers arrive through the root add_includedirs.

    -- The kernel's rules, because these sources are compiled into the kernel. Not
    -- inherited from the root on purpose: the hosted module next door cannot obey
    -- them (its dependencies use RTTI and throw), and a flag set globally then
    -- overridden in one place reads as an exception rather than as a boundary.
    add_cxxflags("-fno-rtti", "-fno-exceptions", { force = true })

    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/satellite/**.hpp)")
target_end()

-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::convert module.
-- convert/ — the host tool that turns a GGUF model into a .lplmind image. Hosted
-- and never compiled into the kernel: it reads files through ggml's gguf API,
-- and its comparison runs llama.cpp. It writes images through lpl::infer, the one
-- writer of the layout.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-converter")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-infer")
    add_packages("llama.cpp", { public = true })
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/convert/**.hpp)")
target_end()

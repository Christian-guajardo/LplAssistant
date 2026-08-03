-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::backend module.
-- backend/ build configuration — hosted backends: the comfortable path, for development.
-- HOST ONLY. Everything that assumes an operating system lives here — an existing
-- inference runtime, a vector database, audio capture, an agent protocol server.
-- The freestanding modules define the contracts; this implements them the easy way
-- on Linux so the two can be compared. When the ring-0 path and this one disagree,
-- one of them is wrong and the fold says which.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-assistant-backend")
    set_kind("static")
    set_group("modules")
    -- HOST ONLY: never listed in a kernel make.config.
    add_deps("lpl-infer")
    add_packages("llama.cpp", "nlohmann_json", "cpp-httplib")

    -- `-fexceptions`, en connaissance de cause et avec un précédent dans le projet :
    -- bci/ fait exactement pareil parce qu'eigen/liblsl/brainflow en ont besoin. Ici
    -- ce sont pqxx, nlohmann_json et cpp-httplib, et 54 sites d'appel en dépendent.
    -- La ligne juste n'est pas « LplAssistant déroge » : c'est que le côté HÔTE peut
    -- lever, et que les modules freestanding (infer/, mind/, satellite/, voice/) ne
    -- peuvent pas — exactement la frontière lecteur/écrivain, vue par le compilateur.
    -- RTTI et exceptions, tous deux en connaissance de cause. pqxx appelle `typeid`
    -- dans un en-tête, et pqxx/nlohmann_json/cpp-httplib lèvent — 54 sites d'appel en
    -- dépendent. Notre propre code n'utilise ni l'un ni l'autre : la mesure a compté
    -- zéro `dynamic_cast` et zéro `typeid` écrits ici. Le précédent existe déjà dans
    -- le projet : bci/ force `-fexceptions` pour exactement la même raison.
    add_cxxflags("-fexceptions", "-frtti", { force = true })

    -- SpeechInput reste HORS de la bibliothèque : son moteur embarque une copie de
    -- ggml incompatible au link avec celle du moteur de langage. Les réunir ici
    -- reproduirait le segfault que la séparation en deux binaires évite.
    remove_files("src/SpeechInput.cpp")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/backend/**.hpp)")
target_end()

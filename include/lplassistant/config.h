/**
 * @file config.h
 * @brief Who LplAssistant is, how it was built, what it needs, and where it runs.
 *
 * A copy of the Laplace config.h template (MasterLaplace/.github, templates/config.h)
 * under the LPLASSISTANT_ prefix. The version below is the only place it is written.
 * The requirement on LplPlugin applies when the foundation is present, and the compiler
 * checks it in every translation unit that includes lpl/Foundation.hpp, ring 0 included.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-10-05
 * @copyright MIT License
 */
/* clang-format off */
#ifndef LPLASSISTANT_CONFIG_H_
    #define LPLASSISTANT_CONFIG_H_

/**
 * @name Identity
 *
 * The version is written here and nowhere else: the build, the release workflow
 * and CITATION.cff read it from these three lines.
 * @{
 */
#define LPLASSISTANT_NAME "LplAssistant"
#define LPLASSISTANT_VERSION_MAJOR 0
#define LPLASSISTANT_VERSION_MINOR 2
#define LPLASSISTANT_VERSION_PATCH 1
/** @} */

/** The shared part, down to the Requirements group: laplace-config v1, from MasterLaplace/.github templates/config.h. */
#define LPLASSISTANT_CONFIG_TEMPLATE 1

#ifdef __cplusplus
    #include <cstddef>
    #include <cstdint>
#else
    #include <stddef.h>
    #include <stdint.h>
#endif

#ifndef LAPLACE_CONFIG_UTILS
    #define LAPLACE_CONFIG_UTILS

/**
 * @name Portable macros, defined once per translation unit whichever copies it includes
 * @{
 */
#define LPL_NEED_COMMA struct _
#define LPL_UNUSED(x) (void)(x)

#if defined(__GNUC__) || defined(__clang__)
    #define LPL_ATTRIBUTE(key) __attribute__((key))
    #define LPL_UNUSED_ATTRIBUTE LPL_ATTRIBUTE(unused)
    #define LPL_LIKELY(x)   __builtin_expect(!!(x), 1)
    #define LPL_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define LPL_ATTRIBUTE(key)
    #define LPL_UNUSED_ATTRIBUTE
    #define LPL_LIKELY(x)   (x)
    #define LPL_UNLIKELY(x) (x)
#endif
/** @} */

/**
 * @name Converting a macro to a string
 * @{
 */
#define LPL_STRINGIFY(x) #x
#define LPL_TOSTRING(x) LPL_STRINGIFY(x)
/** @} */

/** Emits a TODO message during compilation, portably. */
#if defined(_MSC_VER)
    #define LPL_TODO(msg) __pragma(message("TODO: " msg))
#else
    #define LPL_TODO(msg) _Pragma(LPL_STRINGIFY(message ("TODO: " msg)))
#endif

/** Portable null pointer: the C++11 nullptr keyword where it exists. */
#if defined(__cplusplus) && __cplusplus >= 201103L
    #define lpl_nullptr nullptr
#elif !defined(NULL)
    #define lpl_nullptr ((void*)0)
#else
    #define lpl_nullptr NULL
#endif

/** Boolean type and values, for C translation units that did not include <stdbool.h>. */
#if !defined(__bool_true_false_are_defined) && !defined(__cplusplus)
    #define bool _Bool
    #define true 1
    #define false 0
    #define __bool_true_false_are_defined 1
#endif

#if defined __GNUC__ && defined __GNUC_MINOR__
# define __GNUC_PREREQ(maj, min) \
    ((__GNUC__ << 16) + __GNUC_MINOR__ >= ((maj) << 16) + (min))
#elif !defined(__GNUC_PREREQ)
# define __GNUC_PREREQ(maj, min) 0
#endif

/**
 * @name Portable structure packing
 *
 * @code
 * LPL_PACKED(struct MyStruct
 * {
 *     int a;
 *     char b;
 * });
 * @endcode
 * @{
 */
#if defined(_MSC_VER) || defined(_MSVC_LANG)
    #define LPL_PACKED( __Declaration__ ) __pragma(pack(push, 1)) __Declaration__ __pragma(pack(pop))
    #define LPL_PACKED_START __pragma(pack(push, 1))
    #define LPL_PACKED_END   __pragma(pack(pop))
#elif defined(__GNUC__) || defined(__GNUG__)
    #define LPL_PACKED( __Declaration__ ) __Declaration__ __attribute__((__packed__))
    #define LPL_PACKED_START _Pragma("pack(1)")
    #define LPL_PACKED_END   _Pragma("pack()")
#else
    #define LPL_PACKED( __Declaration__ ) __Declaration__
    #define LPL_PACKED_START
    #define LPL_PACKED_END
#endif
/** @} */

#endif /* !LAPLACE_CONFIG_UTILS */


/**
 * @brief Identifies the compiler as LPLASSISTANT_COMPILER_<name> and LPLASSISTANT_COMPILER_STRING.
 *
 * @details Clang and MinGW both define __GNUC__, so they are tested before GCC.
 */
#if defined(_MSC_VER) && !defined(__clang__)
    #define LPLASSISTANT_COMPILER_MSVC
    #define LPLASSISTANT_COMPILER_STRING "MSVC"
#elif defined(__clang__)
    #define LPLASSISTANT_COMPILER_CLANG
    #define LPLASSISTANT_COMPILER_STRING "Clang"
#elif defined(__MINGW32__) || defined(__MINGW64__)
    #define LPLASSISTANT_COMPILER_MINGW
    #define LPLASSISTANT_COMPILER_STRING "MinGW"
#elif defined(__CYGWIN__)
    #define LPLASSISTANT_COMPILER_CYGWIN
    #define LPLASSISTANT_COMPILER_STRING "Cygwin"
#elif defined(__GNUC__) || defined(__GNUG__)
    #define LPLASSISTANT_COMPILER_GCC
    #define LPLASSISTANT_COMPILER_STRING "GCC"
#else
    #error [Config@Distribution]: This compiler is not known to the Laplace config.h template.
#endif


/**
 * @brief Identifies the target system as LPLASSISTANT_SYSTEM_<name> and LPLASSISTANT_SYSTEM_STRING.
 *
 * @details The Laplace Kernel is tested first: code compiled for it is compiled for it,
 *          whatever the compiler would otherwise suggest. Android is tested before Linux
 *          because it defines __linux__. The kernel target also defines
 *          LPLASSISTANT_MODE_STRING, the real-time or standard suffix.
 */
#if defined(__LPL_KERNEL__) || defined(__is_kernel) || (defined(LPL_TARGET_KERNEL) && LPL_TARGET_KERNEL)

    #define LPLASSISTANT_SYSTEM_LAPLACE_KERNEL
    #define LPLASSISTANT_SYSTEM_STRING "Laplace Kernel"

    #if defined(LPL_KERNEL_REAL_TIME_MODE)
        #define LPLASSISTANT_MODE_STRING " (Real-Time)"
    #else
        #define LPLASSISTANT_MODE_STRING " (Standard)"
    #endif

#elif defined(_WIN32) || defined(__WIN32__) || defined(__MINGW32__) || defined(__CYGWIN__)

    #define LPLASSISTANT_SYSTEM_WINDOWS
    #define LPLASSISTANT_SYSTEM_STRING "Windows"

#elif defined(__ANDROID__)

    #define LPLASSISTANT_SYSTEM_ANDROID
    #define LPLASSISTANT_SYSTEM_STRING "Android"

#elif defined(__linux__) || defined(__linux) || defined(linux)

    #define LPLASSISTANT_SYSTEM_LINUX
    #define LPLASSISTANT_SYSTEM_STRING "Linux"

#elif defined(__APPLE__)

    #define LPLASSISTANT_SYSTEM_MACOS
    #define LPLASSISTANT_SYSTEM_STRING "macOS"

#elif defined(__FreeBSD__) || defined(__FreeBSD_kernel__)

    #define LPLASSISTANT_SYSTEM_FREEBSD
    #define LPLASSISTANT_SYSTEM_STRING "FreeBSD"

#elif defined(__unix) || defined(__unix__)

    #define LPLASSISTANT_SYSTEM_UNIX
    #define LPLASSISTANT_SYSTEM_STRING "Unix"

#else
    #error [Config@Distribution]: This operating system is not known to the Laplace config.h template.
#endif

#ifndef LPLASSISTANT_MODE_STRING
    #define LPLASSISTANT_MODE_STRING
#endif


/** Identifies the processor as LPLASSISTANT_ARCH_<name> and LPLASSISTANT_ARCH_STRING. */
#if defined(__x86_64__) || defined(_M_X64)
    #define LPLASSISTANT_ARCH_X64
    #define LPLASSISTANT_ARCH_STRING "x86_64"
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define LPLASSISTANT_ARCH_ARM64
    #define LPLASSISTANT_ARCH_STRING "arm64"
#elif defined(__i386__) || defined(_M_IX86)
    #define LPLASSISTANT_ARCH_X86
    #define LPLASSISTANT_ARCH_STRING "i686"
#elif defined(__riscv) && (__riscv_xlen == 64)
    #define LPLASSISTANT_ARCH_RISCV64
    #define LPLASSISTANT_ARCH_STRING "riscv64"
#else
    #define LPLASSISTANT_ARCH_UNKNOWN
    #define LPLASSISTANT_ARCH_STRING "unknown"
#endif


#ifdef __cplusplus
    #define LPLASSISTANT_EXTERN_C extern "C"

    #if __cplusplus >= 202302L
        #define LPLASSISTANT_CPP23(_) _
        #define LPLASSISTANT_CPP20(_) _
        #define LPLASSISTANT_CPP17(_) _
        #define LPLASSISTANT_CPP14(_) _
        #define LPLASSISTANT_CPP11(_) _
        #define LPLASSISTANT_CPP99(_) _
    #elif __cplusplus >= 202002L
        #define LPLASSISTANT_CPP23(_)
        #define LPLASSISTANT_CPP20(_) _
        #define LPLASSISTANT_CPP17(_) _
        #define LPLASSISTANT_CPP14(_) _
        #define LPLASSISTANT_CPP11(_) _
        #define LPLASSISTANT_CPP99(_) _
    #elif __cplusplus >= 201703L
        #define LPLASSISTANT_CPP23(_)
        #define LPLASSISTANT_CPP20(_)
        #define LPLASSISTANT_CPP17(_) _
        #define LPLASSISTANT_CPP14(_) _
        #define LPLASSISTANT_CPP11(_) _
        #define LPLASSISTANT_CPP99(_) _
    #elif __cplusplus >= 201402L
        #define LPLASSISTANT_CPP23(_)
        #define LPLASSISTANT_CPP20(_)
        #define LPLASSISTANT_CPP17(_)
        #define LPLASSISTANT_CPP14(_) _
        #define LPLASSISTANT_CPP11(_) _
        #define LPLASSISTANT_CPP99(_) _
    #elif __cplusplus >= 201103L
        #define LPLASSISTANT_CPP23(_)
        #define LPLASSISTANT_CPP20(_)
        #define LPLASSISTANT_CPP17(_)
        #define LPLASSISTANT_CPP14(_)
        #define LPLASSISTANT_CPP11(_) _
        #define LPLASSISTANT_CPP99(_) _
    #elif __cplusplus >= 199711L
        #define LPLASSISTANT_CPP23(_)
        #define LPLASSISTANT_CPP20(_)
        #define LPLASSISTANT_CPP17(_)
        #define LPLASSISTANT_CPP14(_)
        #define LPLASSISTANT_CPP11(_)
        #define LPLASSISTANT_CPP99(_) _
    #else
        #define LPLASSISTANT_CPP23(_)
        #define LPLASSISTANT_CPP20(_)
        #define LPLASSISTANT_CPP17(_)
        #define LPLASSISTANT_CPP14(_)
        #define LPLASSISTANT_CPP11(_)
        #define LPLASSISTANT_CPP99(_)
    #endif

    /**
     * @brief Keeps its argument only when the C++ standard in use is at least @p version.
     *
     * @code
     * void func() LPLASSISTANT_CPP14([[deprecated]]);
     * void func() LPLASSISTANT_CPP([[deprecated]], 14);
     * @endcode
     */
    #define LPLASSISTANT_CPP(_, version) LPLASSISTANT_CPP##version(_)

#else
    #define LPLASSISTANT_EXTERN_C extern

    #define LPLASSISTANT_CPP23(_)
    #define LPLASSISTANT_CPP20(_)
    #define LPLASSISTANT_CPP17(_)
    #define LPLASSISTANT_CPP14(_)
    #define LPLASSISTANT_CPP11(_)
    #define LPLASSISTANT_CPP99(_)
    #define LPLASSISTANT_CPP(_, version)
#endif

/**
 * @name Portable import / export macros for each module
 *
 * Windows compilers need specific (and different) keywords for export and import, and
 * Visual C++ also needs warning C4251 turned off. GCC 4 and later mark symbols visible
 * with one keyword used for both directions; older GCC cannot hide symbols at all, so
 * everything is exported.
 * @{
 */
#if defined(LPLASSISTANT_SYSTEM_WINDOWS)

    #define LPLASSISTANT_API_EXPORT LPLASSISTANT_EXTERN_C __declspec(dllexport)
    #define LPLASSISTANT_API_IMPORT LPLASSISTANT_EXTERN_C __declspec(dllimport)

    #ifdef _MSC_VER

        #pragma warning(disable : 4251)

    #endif

#elif defined(__GNUC__) && __GNUC__ >= 4

    #define LPLASSISTANT_API_EXPORT LPLASSISTANT_EXTERN_C __attribute__ ((__visibility__ ("default")))
    #define LPLASSISTANT_API_IMPORT LPLASSISTANT_EXTERN_C __attribute__ ((__visibility__ ("default")))

#else

    #define LPLASSISTANT_API_EXPORT LPLASSISTANT_EXTERN_C
    #define LPLASSISTANT_API_IMPORT LPLASSISTANT_EXTERN_C

#endif
/** @} */


/**
 * @name Portable entry point
 *
 * Windows GUI programs enter through WinMain, Android through android_main with no
 * main function at all, and macOS through a Unix main that also receives the Apple
 * strings. Every other platform uses the standard main.
 * @{
 */
#ifdef LPLASSISTANT_SYSTEM_WINDOWS

    #define LPLASSISTANT_GUI_MAIN(hInstance, hPrevInstance, lpCmdLine, nCmdShow) WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
    #define LPLASSISTANT_MAIN(ac, av, env) main(int ac, char *av[], char *env[])

#elif defined(LPLASSISTANT_SYSTEM_ANDROID)

    #define LPLASSISTANT_GUI_MAIN(app) android_main(struct android_app* app)
    #define LPLASSISTANT_MAIN

#elif defined(LPLASSISTANT_SYSTEM_MACOS)

    #define LPLASSISTANT_MAIN(ac, av, env, apple) main(int ac, char *av[], char *env[], char *apple[])

#else

    #define LPLASSISTANT_MAIN(ac, av, env) main(int ac, char *av[], char *env[])
#endif
/** @} */

/** LPLASSISTANT_DEBUG and LPLASSISTANT_DEBUG_STRING, from the usual debug flags (LPL_DEBUG included) and NDEBUG. */
#if (defined(_DEBUG) || defined(DEBUG) || defined(LPL_DEBUG)) && !defined(NDEBUG)

    #define LPLASSISTANT_DEBUG
    #define LPLASSISTANT_DEBUG_STRING "Debug"

#else
    #define LPLASSISTANT_DEBUG_STRING "Release"
#endif

/**
 * @name Portable deprecation markers
 *
 * @code
 * LPLASSISTANT_DEPRECATED void func();
 * struct LPLASSISTANT_DEPRECATED MyStruct { ... };
 * enum LPLASSISTANT_DEPRECATED MyEnum { ... };
 * enum MyEnum {
 *     MyEnum1 = 0,
 *     MyEnum2 LPLASSISTANT_DEPRECATED,
 *     MyEnum3
 * };
 * class LPLASSISTANT_DEPRECATED MyClass { ... };
 * @endcode
 * @{
 */
#ifdef LPLASSISTANT_DISABLE_DEPRECATION

    #define LPLASSISTANT_DEPRECATED
    #define LPLASSISTANT_DEPRECATED_MSG(message)
    #define LPLASSISTANT_DEPRECATED_VMSG(version, message)

#elif defined(__cplusplus) && (__cplusplus >= 201402)

    #define LPLASSISTANT_DEPRECATED [[deprecated]]
    #define LPLASSISTANT_DEPRECATED_MSG(message) [[deprecated(message)]]
    #define LPLASSISTANT_DEPRECATED_VMSG(version, message) [[deprecated("since " # version ". " message)]]

#elif defined(LPLASSISTANT_COMPILER_MSVC) && (_MSC_VER >= 1900)

    #define LPLASSISTANT_DEPRECATED __declspec(deprecated)
    #define LPLASSISTANT_DEPRECATED_MSG(message) __declspec(deprecated(message))
    #define LPLASSISTANT_DEPRECATED_VMSG(version, message) __declspec(deprecated("since " # version ". " message))

#elif defined(__GNUC__) && __GNUC_PREREQ(4, 9)

    #define LPLASSISTANT_DEPRECATED __attribute__((deprecated))
    #define LPLASSISTANT_DEPRECATED_MSG(message) __attribute__((deprecated(message)))
    #define LPLASSISTANT_DEPRECATED_VMSG(version, message) __attribute__((deprecated("since " # version ". " message)))

#else

    #define LPLASSISTANT_DEPRECATED
    #define LPLASSISTANT_DEPRECATED_MSG(message)
    #define LPLASSISTANT_DEPRECATED_VMSG(version, message)
#endif
/** @} */

/**
 * @name Version
 *
 * The version packs into one integer the way Vulkan's VK_MAKE_API_VERSION does: 7 bits
 * of major, 10 of minor and 12 of patch. Unlike Vulkan's, the macro has no cast, so it
 * also works inside #if, which is where a repository checks the version of another.
 *
 * @code
 * #if !OTHER_COMPATIBLE_WITH(0, 3, 0)
 *     #error "This needs the other repository at 0.3.0 or a later 0.x"
 * #endif
 * @endcode
 * @{
 */
#define LPLASSISTANT_MAKE_VERSION(major, minor, patch) (((major) << 22) | ((minor) << 12) | (patch))

#define LPLASSISTANT_VERSION \
        LPLASSISTANT_MAKE_VERSION(LPLASSISTANT_VERSION_MAJOR, LPLASSISTANT_VERSION_MINOR, \
                                      LPLASSISTANT_VERSION_PATCH)

/** At least this version. */
#define LPLASSISTANT_PREREQ_VERSION(major, minor, patch) \
        (LPLASSISTANT_VERSION >= LPLASSISTANT_MAKE_VERSION(major, minor, patch))

/** At least this version, and the same major: a new major is a break, never accepted in silence. */
#define LPLASSISTANT_COMPATIBLE_WITH(major, minor, patch) \
        (LPLASSISTANT_VERSION_MAJOR == (major) && LPLASSISTANT_PREREQ_VERSION(major, minor, patch))

#define LPLASSISTANT_VERSION_STRING \
        LPL_TOSTRING(LPLASSISTANT_VERSION_MAJOR) "." \
        LPL_TOSTRING(LPLASSISTANT_VERSION_MINOR) "." \
        LPL_TOSTRING(LPLASSISTANT_VERSION_PATCH)
/** @} */

/**
 * @name Build stamp
 *
 * What the source cannot know: the commit it was built from and the build it went into
 * (a profile and a mode, such as "server.debug"). A build passes them with -D to the one
 * translation unit that prints them, so a new commit does not recompile every file.
 * @{
 */
#ifndef LPLASSISTANT_COMMIT
    #define LPLASSISTANT_COMMIT "unknown"
#endif

#ifndef LPLASSISTANT_BUILD
    #define LPLASSISTANT_BUILD "unknown"
#endif
/** @} */

/** Compile-time configuration, one KEY=value per line. */
#define LPLASSISTANT_CONFIG_STRING \
        "LPLASSISTANT_VERSION=" LPLASSISTANT_VERSION_STRING "+" LPLASSISTANT_BUILD " " LPLASSISTANT_COMMIT "\n" \
        "LPLASSISTANT_SYSTEM=" LPLASSISTANT_SYSTEM_STRING LPLASSISTANT_MODE_STRING "\n" \
        "LPLASSISTANT_ARCH=" LPLASSISTANT_ARCH_STRING "\n" \
        "LPLASSISTANT_COMPILER=" LPLASSISTANT_COMPILER_STRING "\n" \
        "LPLASSISTANT_DEBUG=" LPLASSISTANT_DEBUG_STRING "\n"

/** @name Requirements: what this repository needs, checked by the compiler whatever the build system @{ */
#if defined(LPL_HAS_FOUNDATION)
    #if defined(__has_include)
        #if !__has_include(<lplplugin/config.h>)
            #error "LplAssistant needs LplPlugin 0.4.0 or later, and the LplPlugin found has no lplplugin/config.h: update it"
        #endif
    #endif
    #include <lplplugin/config.h>
    #if !LPLPLUGIN_COMPATIBLE_WITH(0, 4, 0)
        #pragma message("found LplPlugin " LPLPLUGIN_VERSION_STRING)
        #if LPLPLUGIN_VERSION_MAJOR != 0
            #error "LplAssistant was written for LplPlugin 0.x: read what broke in its CHANGELOG, then adapt"
        #else
            #error "LplAssistant needs LplPlugin 0.4.0 or later: update it, or build with --foundation=off"
        #endif
    #endif
#endif
/** @} */

#endif /* !LPLASSISTANT_CONFIG_H_ */
/* clang-format on */

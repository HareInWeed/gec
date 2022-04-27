#pragma once
#ifndef GEC_UTILS_BASIC_HPP
#define GEC_UTILS_BASIC_HPP

#ifdef GEC_DEBUG
#include <cstdio>
#endif // GEC_DEBUG

// architecture macro
#if defined(__x86_64__) || defined(_M_X64)
#define GEC_AMD64
#endif
#if defined(__i386__) || defined(_M_IX86)
#define GEC_X86
#endif

// c++ concepts flag, currently takes no effects
#if __cplusplus >= 202002L && !defined(GEC_CONCEPTS_DISABLE) &&                \
    !defined(GEC_CONCEPTS)
#define GEC_CONCEPTS
#endif

// compiler macro
#if defined(__NVCC__)
#define GEC_NVCC __NVCC__
#endif
#if defined(__clang__)
#define GEC_CLANG __clang__
#else
// clang might use compiler below as host
#if defined(_MSC_VER)
#define GEC_MSVC _MSC_VER
#elif defined(__GNUC__)
#define GEC_GCC _MSC_VER
#endif
#endif

// inline attribute modifier

#if defined(GEC_NVCC)
// see:
// <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#noinline-and-forceinline>
#define GEC_INLINE __forceinline__

#elif defined(GEC_CLANG)
// see:
// <https://clang.llvm.org/docs/AttributeReference.html#always-inline-force-inline>
#define GEC_INLINE __forceinline

#elif defined(GEC_GCC)
// see:
// - <https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html>
// - <https://gcc.gnu.org/onlinedocs/gcc/Inline.html>
#define GEC_INLINE inline __attribute__((always_inline))

#elif defined(GEC_MSVC)
// see:
// <https://docs.microsoft.com/en-us/cpp/cpp/inline-functions-cpp?view=msvc-170#inline-__inline-and-__forceinline>
#define GEC_INLINE __forceinline

#else
// this won't force compiler to inline a function, but it's good to give a hint.
#define GEC_INLINE inline

#endif

// restrict pointer marker modifier

#if defined(GEC_NVCC)
// see:
// <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#restrict>
#define GEC_RSTRCT __restrict__

#elif defined(GEC_CLANG)
// AFAIK, there is no restricted pointer support in clang yet.
#define GEC_RSTRCT

#elif defined(GEC_GCC)
// see:
// <https://gcc.gnu.org/onlinedocs/gcc-3.1.1/gcc/Restricted-Pointers.html>
#define GEC_RSTRCT __restrict__

#elif defined(GEC_MSVC)
// see:
// <https://docs.microsoft.com/en-us/cpp/cpp/extension-restrict?view=msvc-170>
#define GEC_RSTRCT __restrict

#else
// don't do anything, in case `restrict` keyword is not supported
#define GEC_RSTRCT

#endif

// CUDA function attribute modifier
#if !defined(GEC_NVCC)
#define __host__
#define __device__
#endif

#include <cinttypes>
#include <limits>

namespace gec {

namespace utils {

/** @brief comparsion results
 */
enum CmpEnum {
    Eq = 0,
    Lt = 1,
    Gt = 2,
};

} // namespace utils

} // namespace gec

#endif // !GEC_UTILS_BASIC_HPP
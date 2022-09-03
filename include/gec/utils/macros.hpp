#pragma once
#ifndef GEC_UTILS_MACROS_HPP
#define GEC_UTILS_MACROS_HPP

#ifdef __CUDACC__
#define GEC_DECL(name, prefix)                                                 \
    prefix name;                                                               \
    __constant__ prefix d_##name
#else
#define GEC_DECL(name, prefix) prefix name
#endif

#ifdef __CUDACC__
#define GEC_DEF(name, prefix, ...)                                             \
    prefix name(__VA_ARGS__);                                                  \
    __constant__ prefix d_##name(__VA_ARGS__)
#else
#define GEC_DEF(name, prefix, ...) prefix name(__VA_ARGS__)
#endif

#ifdef __CUDACC__
#define GEC_DECL_GLOBAL(name, T)                                               \
    extern const T name;                                                       \
    extern __constant__ const T d_##name
#else
#define GEC_DECL_GLOBAL(name, T) extern const T name
#endif

#ifdef __CUDACC__
#define GEC_DEF_GLOBAL(name, T, ...)                                           \
    const T name(__VA_ARGS__);                                                 \
    __constant__ const T d_##name(__VA_ARGS__)
#else
#define GEC_DEF_GLOBAL(name, T, ...) const T name(__VA_ARGS__)
#endif

// ---------- Array ----------

#if defined(__CUDACC__)
#define GEC_DECL_ARRAY(name, T, S)                                             \
    extern const T name[(S)];                                                  \
    extern __constant__ const T d_##name[(S)]
#define GEC_DECL_ALIGNED_ARRAY(name, T, S, align)                              \
    alignas((align)) extern const T name[(S)];                                 \
    alignas((align)) extern __constant__ const T d_##name[(S)]
#else
#define GEC_DECL_ARRAY(name, T, S) extern const T name[(S)]
#define GEC_DECL_ALIGNED_ARRAY(name, T, S, align)                              \
    alignas((align)) extern const T name[(S)]
#endif // __CUDACC__

#ifdef __CUDACC__
#define GEC_DEF_ARRAY(name, T, S, ...)                                         \
    const T name[(S)] = {__VA_ARGS__};                                         \
    __constant__ const T d_##name[(S)] = {__VA_ARGS__}
#define GEC_DEF_ALIGNED_ARRAY(name, T, S, align, ...)                          \
    alignas((align)) const T name[(S)] = {__VA_ARGS__};                        \
    alignas((align)) __constant__ const T d_##name[(S)] = {__VA_ARGS__}
#else
#define GEC_DEF_ARRAY(name, T, S, ...) const T name[(S)] = {__VA_ARGS__}
#define GEC_DEF_ALIGNED_ARRAY(name, T, S, align, ...)                          \
    alignas((align)) const T name[(S)] = {__VA_ARGS__}
#endif // __CUDACC__

#endif // !GEC_UTILS_MACROS_HPP

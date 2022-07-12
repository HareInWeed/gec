#pragma once
#ifndef GEC_TEST_COMMON_HPP
#define GEC_TEST_COMMON_HPP

// #define GEC_DEBUG

#include <gec/utils/basic.hpp>

#include <iostream>

#if defined(__CUDACC__)
#define decl_array(name, T, S)                                                 \
    extern const T name[(S)];                                                  \
    extern __constant__ const T d_##name[(S)]
#define decl_aligned_array(name, T, S, align)                                  \
    alignas((align)) extern const T name[(S)];                                 \
    alignas((align)) extern __constant__ const T d_##name[(S)]
#else
#define decl_array(name, T, S) extern const T name[(S)]
#define decl_aligned_array(name, T, S, align)                                  \
    alignas((align)) extern const T name[(S)]
#endif // __CUDACC__

#ifdef __CUDACC__
#define def_array(name, T, S, ...)                                             \
    const T name[(S)] = {__VA_ARGS__};                                         \
    __constant__ const T d_##name[(S)] = {__VA_ARGS__}
#define def_aligned_array(name, T, S, align, ...)                              \
    alignas((align)) const T name[(S)] = {__VA_ARGS__};                        \
    alignas((align)) __constant__ const T d_##name[(S)] = {__VA_ARGS__}
#else
#define def_array(name, T, S, ...) const T name[(S)] = {__VA_ARGS__}
#define def_aligned_array(name, T, S, align, ...)                              \
    alignas((align)) const T name[(S)] = {__VA_ARGS__}
#endif // __CUDACC__

using namespace std;

using LIMB_T = uint32_t;
using LIMB2_T = uint64_t;

/// @brief number of limbs for 160 bits bigint with 32-bit limb
constexpr size_t LN_160 = 5;
/// @brief -MOD^{-1} (mod 2^32)
constexpr LIMB_T MOD_P_160 = 0x96c9e927u;
/// @brief the modulus with 160 bits
decl_aligned_array(MOD_160, LIMB_T, LN_160, 8);
/// @brief R^2 (mod MOD)
decl_aligned_array(RR_160, LIMB_T, LN_160, 8);
/// @brief R (mod MOD)
decl_aligned_array(OneR_160, LIMB_T, LN_160, 8);

/// @brief number of limbs for 160 bits bigint with 64-bit limb
constexpr size_t LN2_160 = 3;
/// @brief -MOD^{-1} (mod 2^64)
constexpr LIMB2_T MOD2_P_160 = 0x1c23727c96c9e927u;
/// @brief the modulus with 160 bits
decl_aligned_array(MOD2_160, LIMB2_T, LN2_160, 8);
/// @brief R^2 (mod MOD)
decl_aligned_array(RR2_160, LIMB2_T, LN2_160, 8);
/// @brief R (mod MOD)
decl_aligned_array(OneR2_160, LIMB2_T, LN2_160, 8);

/// @brief number of limbs for 256 bits bigint with 32-bit limb
constexpr size_t LN_256 = 8;
/// @brief -MOD^{-1} (mod 2^32)
constexpr LIMB_T MOD_P_256 = 0xd2253531u;
/// @brief the modulus with 256 bits
decl_aligned_array(MOD_256, LIMB_T, LN_256, 32);
/// @brief R^2 (mod MOD)
decl_aligned_array(RR_256, LIMB_T, LN_256, 32);
/// @brief R (mod MOD)
decl_aligned_array(OneR_256, LIMB_T, LN_256, 32);

/// @brief number of limbs for 256 bits bigint with 64-bit limb
constexpr size_t LN2_256 = 4;
/// @brief -MOD^{-1} (mod 2^64)
constexpr LIMB2_T MOD2_P_256 = 0xd838091dd2253531u;
/// @brief the modulus with 256 bits
decl_aligned_array(MOD2_256, LIMB2_T, LN2_256, 32);
/// @brief R^2 (mod MOD)
decl_aligned_array(RR2_256, LIMB2_T, LN2_256, 32);
/// @brief R (mod MOD)
decl_aligned_array(OneR2_256, LIMB2_T, LN2_256, 32);

#endif // !GEC_TEST_COMMON_HPP

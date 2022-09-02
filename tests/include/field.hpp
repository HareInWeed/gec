#pragma once
#ifndef GEC_TEST_FIELD_HPP
#define GEC_TEST_FIELD_HPP

#include "common.hpp"

#include <gec/bigint.hpp>

#include <type_traits>

#ifdef __CUDACC__
#define ADD_GROUP(T, N, align, MOD)                                            \
    ::gec::bigint::RawAddGroup<T, N, &MOD, &d_##MOD, align>
#else
#define ADD_GROUP(T, N, align, MOD)                                            \
    ::gec::bigint::RawAddGroup<T, N, &MOD, nullptr, align>
#endif

#ifdef __CUDACC__
#define FIELD(T, N, align, MOD, MOD_P, RR, ONE_R)                              \
    ::gec::bigint::RawField<T, N, &MOD, MOD_P, &RR, &ONE_R, &d_##MOD, &d_##RR, \
                            &d_##ONE_R, align>
#else
#define FIELD(T, N, align, MOD, MOD_P, RR, ONE_R)                              \
    ::gec::bigint::RawField<T, N, &MOD, MOD_P, &RR, &ONE_R, nullptr, nullptr,  \
                            nullptr, align>
#endif

#ifdef __CUDACC__
#define decl_field(name, F)                                                    \
    extern const F name;                                                       \
    extern __constant__ const F d_##name
#else
#define decl_field(name, F) extern const F name
#endif

#ifdef __CUDACC__
#define def_field(name, F, ...)                                                \
    const F name(__VA_ARGS__);                                                 \
    __constant__ const F d_##name(__VA_ARGS__)
#else
#define def_field(name, F, ...) const F name(__VA_ARGS__)
#endif

#ifdef GEC_ENABLE_AVX2

#define AVX2FIELD(T, N, align, MOD, MOD_P, RR, ONE_R)                          \
    ::gec::bigint::RawAVX2Field<T, N, &MOD, MOD_P, &RR, &ONE_R, align>

#endif // GEC_ENABLE_AVX2

using Field160 = FIELD(LIMB_T, LN_160, alignof(LIMB_T), MOD_160, MOD_P_160,
                       RR_160, OneR_160);

using Field160_2 = FIELD(LIMB2_T, LN2_160, alignof(LIMB2_T), MOD2_160,
                         MOD2_P_160, RR2_160, OneR2_160);

#endif // !GEC_TEST_FIELD_HPP
#pragma once
#ifndef GEC_TEST_FIELD_HPP
#define GEC_TEST_FIELD_HPP

#include "common.hpp"

#include <gec/bigint.hpp>

#include <type_traits>

template <typename Core, typename T, size_t N, const T (*MOD)[N],
          const T (*d_MOD)[N]>
class GEC_EMPTY_BASES AddGroupMixin
    : public gec::bigint::AddGroupParams<Core, T, N, MOD, d_MOD>,
      public gec::bigint::Constants<Core, T, N>,
      public gec::bigint::VtCompare<Core, T, N>,
      public gec::bigint::BitOps<Core, T, N>,
      public gec::bigint::ModAddSub<Core, T, N>,
      public gec::bigint::ModRandom<Core, T, N>,
      public gec::bigint::WithBigintContext<Core>,
      public gec::bigint::WithArrayHasher<Core>,
      public gec::bigint::ArrayOstream<Core, T, N>,
      public gec::bigint::ArrayPrint<Core, T, N> {};

template <typename T, size_t N, size_t align, const T (*MOD)[N],
          const T (*d_MOD)[N] = MOD>
class alignas(align) GEC_EMPTY_BASES AddGroup
    : public gec::bigint::ArrayBE<T, N>,
      public AddGroupMixin<AddGroup<T, N, align, MOD, d_MOD>, T, N, MOD,
                           d_MOD> {
    using gec::bigint::ArrayBE<T, N>::ArrayBE;
};

#ifdef __CUDACC__
#define ADD_GROUP(T, N, align, MOD) AddGroup<T, N, align, &MOD, &d_##MOD>
#else
#define ADD_GROUP(T, N, align, MOD) AddGroup<T, N, align, &MOD>
#endif

template <typename Core, typename T, size_t N, const T (*MOD)[N], T MOD_P,
          const T (*RR)[N], const T (*ONE_R)[N], const T (*d_MOD)[N],
          const T (*d_RR)[N], const T (*d_ONE_R)[N]>
class GEC_EMPTY_BASES FieldMixin
    : public AddGroupMixin<Core, T, N, MOD, d_MOD>,
      public gec::bigint::MontgomeryParams<Core, T, N, MOD_P, RR, ONE_R, d_RR,
                                           d_ONE_R>,
      public gec::bigint::MontgomeryOps<Core, T, N>,
      public gec::bigint::Exponentiation<Core> {};

template <typename T, size_t N, size_t align, const T (*MOD)[N], T MOD_P,
          const T (*RR)[N], const T (*ONE_R)[N], const T (*d_MOD)[N] = MOD,
          const T (*d_RR)[N] = RR, const T (*d_ONE_R)[N] = ONE_R>
class alignas(align) GEC_EMPTY_BASES Field
    : public gec::bigint::ArrayBE<T, N>,
      public FieldMixin<
          Field<T, N, align, MOD, MOD_P, RR, ONE_R, d_MOD, d_RR, d_ONE_R>, T, N,
          MOD, MOD_P, RR, ONE_R, d_MOD, d_RR, d_ONE_R> {
    using gec::bigint::ArrayBE<T, N>::ArrayBE;
};

#ifdef __CUDACC__
#define FIELD(T, N, align, MOD, MOD_P, RR, ONE_R)                              \
    Field<T, N, align, &MOD, MOD_P, &RR, &ONE_R, &d_##MOD, &d_##RR, &d_##ONE_R>
#else
#define FIELD(T, N, align, MOD, MOD_P, RR, ONE_R)                              \
    Field<T, N, align, &MOD, MOD_P, &RR, &ONE_R>
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
    __constant__ const F d_##name
#else
#define def_field(name, F, ...) const F name(__VA_ARGS__)
#endif

#ifdef GEC_ENABLE_AVX2

template <typename Core, typename T, size_t N, const T (*MOD)[N], T MOD_P,
          const T (*RR)[N], const T (*ONE_R)[N], const T (*d_MOD)[N],
          const T (*d_RR)[N], const T (*d_ONE_R)[N]>
class GEC_EMPTY_BASES AVX2FieldMixin
    : public AddGroupMixin<Core, T, N, MOD, d_MOD>,
      public gec::bigint::MontgomeryParams<Core, T, N, MOD_P, RR, ONE_R, d_RR,
                                           d_ONE_R>,
      public gec::bigint::AVX2MontgomeryOps<Core, T, N>,
      public gec::bigint::Exponentiation<Core> {};

template <typename T, size_t N, size_t align, const T (*MOD)[N], T MOD_P,
          const T (*RR)[N], const T (*ONE_R)[N], const T (*d_MOD)[N] = nullptr,
          const T (*d_RR)[N] = nullptr, const T (*d_ONE_R)[N] = nullptr>
class alignas(align) GEC_EMPTY_BASES AVX2Field
    : public gec::bigint::ArrayBE<T, N>,
      public AVX2FieldMixin<
          AVX2Field<T, N, align, MOD, MOD_P, RR, ONE_R, d_MOD, d_RR, d_ONE_R>,
          T, N, MOD, MOD_P, RR, ONE_R, d_MOD, d_RR, d_ONE_R> {
    using gec::bigint::ArrayBE<T, N>::ArrayBE;
};

#define AVX2FIELD(T, N, align, MOD, MOD_P, RR, ONE_R)                          \
    AVX2Field<T, N, align, &MOD, MOD_P, &RR, &ONE_R>

#endif // GEC_ENABLE_AVX2

using Field160 = FIELD(LIMB_T, LN_160, alignof(LIMB_T), MOD_160, MOD_P_160,
                       RR_160, OneR_160);

using Field160_2 = FIELD(LIMB2_T, LN2_160, alignof(LIMB2_T), MOD2_160,
                         MOD2_P_160, RR2_160, OneR2_160);

#endif // !GEC_TEST_FIELD_HPP
#pragma once
#ifndef GEC_TEST_FIELD_HPP
#define GEC_TEST_FIELD_HPP

#include "common.hpp"

#include <gec/bigint.hpp>

#include <type_traits>

template <typename Core, typename T, size_t N, const T *MOD>
class AddGroupMixin : public gec::bigint::Constants<Core, T, N>,
                      public gec::bigint::VtCompare<Core, T, N>,
                      public gec::bigint::BitOps<Core, T, N>,
                      public gec::bigint::ModAddSub<Core, T, N, MOD>,
                      public gec::bigint::ModRandom<Core, T, N, MOD>,
                      public gec::bigint::WithBigintContext<Core>,
                      public gec::bigint::WithArrayHasher<Core>,
                      public gec::bigint::ArrayOstream<Core, T, N>,
                      public gec::bigint::ArrayPrint<Core, T, N> {};

template <typename T, size_t N, const T *MOD,
          size_t align = std::alignment_of_v<T>>
class alignas(align) AddGroup
    : public gec::bigint::ArrayBE<T, N>,
      public AddGroupMixin<AddGroup<T, N, MOD, align>, T, N, MOD> {
    using gec::bigint::ArrayBE<T, N>::ArrayBE;
};

template <typename T, size_t N, const T *MOD, T MOD_P, const T *RR,
          const T *ONE_R, size_t align = std::alignment_of_v<T>>
class alignas(align) Field
    : public gec::bigint::ArrayBE<T, N>,
      public AddGroupMixin<Field<T, N, MOD, MOD_P, RR, ONE_R, align>, T, N,
                           MOD>,
      public gec::bigint::Montgomery<Field<T, N, MOD, MOD_P, RR, ONE_R, align>,
                                     T, N, MOD, MOD_P, RR, ONE_R>,
      public gec::bigint::Exponentiation<
          Field<T, N, MOD, MOD_P, RR, ONE_R, align>> {
    using gec::bigint::ArrayBE<T, N>::ArrayBE;
};

#ifdef GEC_ENABLE_AVX2

template <typename T, size_t N, const T *MOD, T MOD_P, const T *RR,
          const T *ONE_R, size_t align = std::alignment_of_v<T>>
class alignas(align) AVX2Field
    : public gec::bigint::ArrayBE<T, N>,
      public AddGroupMixin<AVX2Field<T, N, MOD, MOD_P, RR, ONE_R, align>, T, N,
                           MOD>,
      public gec::bigint::AVX2Montgomery<
          AVX2Field<T, N, MOD, MOD_P, RR, ONE_R, align>, T, N, MOD, MOD_P, RR,
          ONE_R>,
      public gec::bigint::Exponentiation<
          AVX2Field<T, N, MOD, MOD_P, RR, ONE_R, align>> {
    using gec::bigint::ArrayBE<T, N>::ArrayBE;
};

#endif // GEC_ENABLE_AVX2

using Field160 = Field<LIMB_T, LN_160, MOD_160, MOD_P_160, RR_160, OneR_160>;

using Field160_2 =
    Field<LIMB2_T, LN2_160, MOD2_160, MOD2_P_160, RR2_160, OneR2_160>;

#endif // !GEC_TEST_FIELD_HPP
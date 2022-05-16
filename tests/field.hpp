#pragma once
#ifndef GEC_TEST_FIELD_HPP
#define GEC_TEST_FIELD_HPP

#include "common.hpp"

#include <gec/bigint.hpp>

template <typename T, size_t N, const T *MOD>
class AddGroup : public gec::bigint::ArrayBE<T, N>,
                 public gec::bigint::Constants<AddGroup<T, N, MOD>, T, N>,
                 public gec::bigint::VtCompare<AddGroup<T, N, MOD>, T, N>,
                 public gec::bigint::BitOps<AddGroup<T, N, MOD>, T, N>,
                 public gec::bigint::ModAddSub<AddGroup<T, N, MOD>, T, N, MOD>,
                 public gec::bigint::ModRandom<AddGroup<T, N, MOD>, T, N, MOD>,
                 public gec::bigint::WithBigintContext<AddGroup<T, N, MOD>>,
                 public gec::bigint::ArrayOstream<AddGroup<T, N, MOD>, T, N>,
                 public gec::bigint::ArrayPrint<AddGroup<T, N, MOD>, T, N> {
    using gec::bigint::ArrayBE<T, N>::ArrayBE;
};

template <typename T, size_t N, const T *MOD, T MOD_P, const T *RR,
          const T *ONE_R>
class Field
    : public gec::bigint::ArrayBE<T, N>,
      public gec::bigint::Constants<Field<T, N, MOD, MOD_P, RR, ONE_R>, T, N>,
      public gec::bigint::VtCompare<Field<T, N, MOD, MOD_P, RR, ONE_R>, T, N>,
      public gec::bigint::BitOps<Field<T, N, MOD, MOD_P, RR, ONE_R>, T, N>,
      public gec::bigint::ModAddSub<Field<T, N, MOD, MOD_P, RR, ONE_R>, T, N,
                                    MOD>,
      public gec::bigint::ModRandom<Field<T, N, MOD, MOD_P, RR, ONE_R>, T, N,
                                    MOD>,
      public gec::bigint::Montgomery<Field<T, N, MOD, MOD_P, RR, ONE_R>, T, N,
                                     MOD, MOD_P, RR, ONE_R>,
      public gec::bigint::Exponentiation<Field<T, N, MOD, MOD_P, RR, ONE_R>>,
      public gec::bigint::WithBigintContext<Field<T, N, MOD, MOD_P, RR, ONE_R>>,
      public gec::bigint::ArrayOstream<Field<T, N, MOD, MOD_P, RR, ONE_R>, T,
                                       N>,
      public gec::bigint::ArrayPrint<Field<T, N, MOD, MOD_P, RR, ONE_R>, T, N> {
    using gec::bigint::ArrayBE<T, N>::ArrayBE;
};

using Field160 = Field<LIMB_T, LN_160, MOD_160, MOD_P_160, RR_160, OneR_160>;

using Field160_2 =
    Field<LIMB2_T, LN2_160, MOD2_160, MOD2_P_160, RR2_160, OneR2_160>;

#endif // !GEC_TEST_FIELD_HPP
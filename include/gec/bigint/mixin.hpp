#pragma once
#ifndef GEC_BIGINT_MIXIN_HPP
#define GEC_BIGINT_MIXIN_HPP

#include "mixin/add_sub.hpp"
#include "mixin/bit_ops.hpp"
#include "mixin/constants.hpp"
#include "mixin/context.hpp"
#include "mixin/division.hpp"
#include "mixin/exp.hpp"
#include "mixin/hasher.hpp"
#include "mixin/mod_add_sub.hpp"
#include "mixin/montgomery.hpp"
#include "mixin/ostream.hpp"
#include "mixin/params.hpp"
#include "mixin/print.hpp"
#include "mixin/quadratic_residue.hpp"
#include "mixin/random.hpp"
#include "mixin/vt_compare.hpp"

namespace gec {

namespace bigint {

// ---------- Bigint ----------

template <typename Core, typename T, size_t N>
class GEC_EMPTY_BASES BigintMixin : public Constants<Core, T, N>,
                                    public VtCompare<Core, T, N>,
                                    public BitOps<Core, T, N>,
                                    public AddSub<Core, T, N>,
                                    public Division<Core, T, N>,
                                    public BigintRandom<Core, T, N>,
                                    public WithArrayHasher<Core>,
                                    public WithBigintContext<Core>,
                                    public ArrayOstream<Core, T, N>,
                                    public ArrayPrint<Core, T, N> {};

// ---------- AddGroup ----------

template <typename Core, typename T, size_t N>
class GEC_EMPTY_BASES AddGroupFunctions : public VtCompare<Core, T, N>,
                                          public BitOps<Core, T, N>,
                                          public ModAddSub<Core, T, N>,
                                          public Division<Core, T, N>,
                                          public ModRandom<Core, T, N>,
                                          public WithArrayHasher<Core>,
                                          public WithBigintContext<Core>,
                                          public ArrayOstream<Core, T, N>,
                                          public ArrayPrint<Core, T, N> {};

template <typename Core, typename T, size_t N, const T (*MOD)[N],
          const T (*d_MOD)[N] = nullptr>
class GEC_EMPTY_BASES AddGroupRawMixin
    : public AddGroupRawParams<Core, T, N, MOD, d_MOD>,
      public Constants<Core, T, N>,
      public AddGroupFunctions<Core, T, N> {};

template <typename Core, typename T, size_t N, typename Base, const Base *MOD,
          const Base *d_MOD = nullptr>
class GEC_EMPTY_BASES AddGroupMixin
    : public AddGroupParams<Core, Base, MOD, d_MOD>,
      public Constants<Core, T, N>,
      public AddGroupFunctions<Core, T, N> {};

// ---------- Field ----------

template <typename Core, typename T, size_t N>
class GEC_EMPTY_BASES BasicFieldFunctions
    : public AddGroupFunctions<Core, T, N>,
      public MonConstants<Core, T, N>,
      public Exponentiation<Core>,
      public MonQuadraticResidue<Core> {};

template <typename Core, typename T, size_t N>
class GEC_EMPTY_BASES FieldFunctions : public BasicFieldFunctions<Core, T, N>,
                                       public MontgomeryOps<Core, T, N> {};

template <typename Core, typename T, size_t N, const T (*MOD)[N], T MOD_P,
          const T (*RR)[N], const T (*ONE_R)[N], const T (*d_MOD)[N] = nullptr,
          const T (*d_RR)[N] = nullptr, const T (*d_ONE_R)[N] = nullptr>
class GEC_EMPTY_BASES FieldRawMixin
    : public AddGroupRawParams<Core, T, N, MOD, d_MOD>,
      public MontgomeryRawParams<Core, T, N, MOD_P, RR, ONE_R, d_RR, d_ONE_R>,
      public FieldFunctions<Core, T, N> {};

template <typename Core, typename T, size_t N, typename Base, const Base *MOD,
          T MOD_P, const Base *RR, const Base *ONE_R,
          const Base *d_MOD = nullptr, const Base *d_RR = nullptr,
          const Base *d_ONE_R = nullptr>
class GEC_EMPTY_BASES FieldMixin
    : public AddGroupParams<Core, Base, MOD, d_MOD>,
      public MontgomeryParams<Core, T, Base, MOD_P, RR, ONE_R, d_RR, d_ONE_R>,
      public FieldFunctions<Core, T, N> {};

#ifdef GEC_ENABLE_AVX2

template <typename Core, typename T, size_t N>
class GEC_EMPTY_BASES AVX2FieldFunctions
    : public BasicFieldFunctions<Core, T, N>,
      public AVX2MontgomeryOps<Core, T, N> {};

template <typename Core, typename T, size_t N, const T (*MOD)[N], T MOD_P,
          const T (*RR)[N], const T (*ONE_R)[N], const T (*d_MOD)[N] = nullptr,
          const T (*d_RR)[N] = nullptr, const T (*d_ONE_R)[N] = nullptr>
class GEC_EMPTY_BASES AVX2FieldRawMixin
    : public AddGroupRawParams<Core, T, N, MOD, d_MOD>,
      public MontgomeryRawParams<Core, T, N, MOD_P, RR, ONE_R, d_RR, d_ONE_R>,
      public AVX2FieldFunctions<Core, T, N> {};

template <typename Core, typename T, size_t N, typename Base, const Base *MOD,
          T MOD_P, const Base *RR, const Base *ONE_R,
          const Base *d_MOD = nullptr, const Base *d_RR = nullptr,
          const Base *d_ONE_R = nullptr>
class GEC_EMPTY_BASES AVX2FieldMixin
    : public AddGroupParams<Core, Base, MOD, d_MOD>,
      public MontgomeryParams<Core, T, Base, MOD_P, RR, ONE_R, d_RR, d_ONE_R>,
      public AVX2FieldFunctions<Core, T, N> {};

#endif

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_HPP

#pragma once
#ifndef GEC_BIGINT_PRESET_HPP
#define GEC_BIGINT_PRESET_HPP

#include "data.hpp"
#include "mixin.hpp"

namespace gec {

namespace bigint {

// ---------- Bigint ----------

template <typename T, size_t N, size_t align = alignof(T)>
class GEC_EMPTY_BASES BigintLE
    : public ArrayLE<T, N, align>,
      public BigintMixin<BigintLE<T, N, align>, T, N> {
    using ArrayLE<T, N, align>::ArrayLE;
};

template <typename T, size_t N, size_t align = alignof(T)>
class GEC_EMPTY_BASES BigintBE
    : public ArrayBE<T, N, align>,
      public BigintMixin<BigintBE<T, N, align>, T, N> {
    using ArrayBE<T, N, align>::ArrayBE;
};

template <typename T, size_t N, size_t align = alignof(T)>
using Bigint = BigintBE<T, N, align>;

// ---------- AddGroup ----------

template <typename T, size_t N, const T (*MOD)[N],
          const T (*d_MOD)[N] = nullptr, size_t align = alignof(T)>
class GEC_EMPTY_BASES RawAddGroup
    : public ArrayBE<T, N, align>,
      public AddGroupRawMixin<RawAddGroup<T, N, MOD, d_MOD, align>, T, N, MOD,
                              d_MOD> {
    using ArrayBE<T, N, align>::ArrayBE;
};

template <typename Base, const Base *MOD, const Base *d_MOD = nullptr>
class GEC_EMPTY_BASES BaseAddGroup
    : public Base,
      public AddGroupMixin<BaseAddGroup<Base, MOD, d_MOD>, typename Base::LimbT,
                           Base::LimbN, Base, MOD, d_MOD> {
    using Base::Base;
};

template <typename T, size_t N, const ArrayBE<T, N> *MOD,
          const ArrayBE<T, N> *d_MOD = nullptr>
using AddGroup = BaseAddGroup<ArrayBE<T, N>, MOD, d_MOD>;

template <typename T, size_t N, size_t align, const ArrayBE<T, N, align> *MOD,
          const ArrayBE<T, N, align> *d_MOD = nullptr>
using AlignedAddGroup = BaseAddGroup<ArrayBE<T, N, align>, MOD, d_MOD>;

// ---------- Field ----------

template <typename T, size_t N, const T (*MOD)[N], T MOD_P, const T (*RR)[N],
          const T (*ONE_R)[N], const T (*d_MOD)[N] = nullptr,
          const T (*d_RR)[N] = nullptr, const T (*d_ONE_R)[N] = nullptr,
          size_t align = alignof(T)>
class GEC_EMPTY_BASES RawField
    : public ArrayBE<T, N, align>,
      public FieldRawMixin<
          RawField<T, N, MOD, MOD_P, RR, ONE_R, d_MOD, d_RR, d_ONE_R, align>, T,
          N, MOD, MOD_P, RR, ONE_R, d_MOD, d_RR, d_ONE_R> {
    using ArrayBE<T, N, align>::ArrayBE;
};

template <typename Base, const Base *MOD, typename Base::LimbT MOD_P,
          const Base *RR, const Base *ONE_R, const Base *d_MOD = nullptr,
          const Base *d_RR = nullptr, const Base *d_ONE_R = nullptr>
class GEC_EMPTY_BASES BaseField
    : public Base,
      public FieldMixin<
          BaseField<Base, MOD, MOD_P, RR, ONE_R, d_MOD, d_RR, d_ONE_R>,
          typename Base::LimbT, Base::LimbN, Base, MOD, MOD_P, RR, ONE_R, d_MOD,
          d_RR, d_ONE_R> {
    using Base::Base;
};

template <typename T, size_t N, const ArrayBE<T, N> *MOD, T MOD_P,
          const ArrayBE<T, N> *RR, const ArrayBE<T, N> *ONE_R,
          const ArrayBE<T, N> *d_MOD = nullptr,
          const ArrayBE<T, N> *d_RR = nullptr,
          const ArrayBE<T, N> *d_ONE_R = nullptr>
using Field =
    BaseField<ArrayBE<T, N>, MOD, MOD_P, RR, ONE_R, d_MOD, d_RR, d_ONE_R>;

template <typename T, size_t N, size_t align, const ArrayBE<T, N, align> *MOD,
          T MOD_P, const ArrayBE<T, N, align> *RR,
          const ArrayBE<T, N, align> *ONE_R,
          const ArrayBE<T, N, align> *d_MOD = nullptr,
          const ArrayBE<T, N, align> *d_RR = nullptr,
          const ArrayBE<T, N, align> *d_ONE_R = nullptr>
using AlignedField = BaseField<ArrayBE<T, N, align>, MOD, MOD_P, RR, ONE_R,
                               d_MOD, d_RR, d_ONE_R>;

#ifdef GEC_ENABLE_AVX2

template <typename T, size_t N, const T (*MOD)[N], T MOD_P, const T (*RR)[N],
          const T (*ONE_R)[N], size_t align = alignof(T)>
class GEC_EMPTY_BASES RawAVX2Field
    : public ArrayBE<T, N, align>,
      public AVX2FieldRawMixin<RawAVX2Field<T, N, MOD, MOD_P, RR, ONE_R, align>,
                               T, N, MOD, MOD_P, RR, ONE_R> {
    using ArrayBE<T, N, align>::ArrayBE;
};

template <typename Base, const Base *MOD, typename Base::LimbT MOD_P,
          const Base *RR, const Base *ONE_R>
class GEC_EMPTY_BASES BaseAVX2Field
    : public Base,
      public AVX2FieldMixin<BaseAVX2Field<Base, MOD, MOD_P, RR, ONE_R>,
                            typename Base::LimbT, Base::LimbN, Base, MOD, MOD_P,
                            RR, ONE_R> {
    using Base::Base;
};

template <typename T, size_t N, const ArrayBE<T, N> *MOD, T MOD_P,
          const ArrayBE<T, N> *RR, const ArrayBE<T, N> *ONE_R>
using AVX2Field = BaseAVX2Field<ArrayBE<T, N>, MOD, MOD_P, RR, ONE_R>;

template <typename T, size_t N, size_t align, const ArrayBE<T, N, align> *MOD,
          T MOD_P, const ArrayBE<T, N, align> *RR,
          const ArrayBE<T, N, align> *ONE_R>
using AlignedAVX2Field =
    BaseAVX2Field<ArrayBE<T, N, align>, MOD, MOD_P, RR, ONE_R>;

#endif // GEC_ENABLE_AVX2

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_PRESET_HPP

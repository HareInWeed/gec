#pragma once
#ifndef GEC_CURVE_SM2_HPP
#define GEC_CURVE_SM2_HPP

#include <gec/bigint/preset.hpp>
#include <gec/curve/preset.hpp>

namespace gec {

namespace curve {

namespace sm2 {

// ----- Field -----
namespace _sm2_ {

using FT = uint64_t;
constexpr size_t FN = 4;

using FBase = bigint::ArrayBE<FT, FN>;

extern const FBase MOD;
constexpr FBase::LimbT MOD_P = 0x1;
extern const FBase RR;
extern const FBase ONE_R;

#ifdef __CUDACC__
__constant__ extern const FBase d_MOD;
__constant__ extern const FBase d_RR;
__constant__ extern const FBase d_ONE_R;
#endif // __CUDACC__

} // namespace _sm2_
using Field = bigint::BaseField<_sm2_::FBase, &_sm2_::MOD, _sm2_::MOD_P,
                                &_sm2_::RR, &_sm2_::ONE_R
#ifdef __CUDACC__
                                ,
                                &_sm2_::d_MOD, &_sm2_::d_RR, &_sm2_::d_ONE_R
#endif // __CUDACC__
                                >;

// ----- Scaler -----
namespace _sm2_ {
// same base between finite field and scaler
using SBase = FBase;
extern const SBase CARD;
constexpr FT CARD_P = 0x327f9e8872350975;
extern const SBase CARD_RR;
extern const SBase CARD_ONE_R;

#ifdef __CUDACC__
__constant__ extern const SBase d_CARD;
__constant__ extern const SBase d_CARD_RR;
__constant__ extern const SBase d_CARD_ONE_R;
#endif // __CUDACC__
} // namespace _sm2_
using Scaler =
    bigint::BaseField<_sm2_::SBase, &_sm2_::CARD, _sm2_::CARD_P,
                      &_sm2_::CARD_RR, &_sm2_::CARD_ONE_R
#ifdef __CUDACC__
                      ,
                      &_sm2_::d_CARD, &_sm2_::d_CARD_RR, &_sm2_::d_CARD_ONE_R
#endif // __CUDACC__
                      >;

} // namespace sm2

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_SM2_HPP

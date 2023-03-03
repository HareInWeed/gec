#pragma once
#ifndef GEC_CURVE_SECP256K1_HPP
#define GEC_CURVE_SECP256K1_HPP

#include <gec/bigint/preset.hpp>
#include <gec/curve/preset.hpp>

namespace gec {

namespace curve {

namespace secp256k1 {

// ----- Field -----
namespace _secp256k1_ {

using FT = uint64_t;
constexpr size_t FN = 4;

using FBase = bigint::ArrayBE<FT, FN>;

extern const FBase MOD;
constexpr FBase::LimbT MOD_P = 0xd838091dd2253531;
extern const FBase RR;
extern const FBase ONE_R;

#ifdef GEC_ENABLE_CUDA
#ifdef __CUDACC__
__constant__
#endif
    extern const FBase d_MOD;
#ifdef __CUDACC__
__constant__
#endif
    extern const FBase d_RR;
#ifdef __CUDACC__
__constant__
#endif
    extern const FBase d_ONE_R;
#endif // GEC_ENABLE_CUDA

} // namespace _secp256k1_
using Field =
    bigint::BaseField<_secp256k1_::FBase, &_secp256k1_::MOD, _secp256k1_::MOD_P,
                      &_secp256k1_::RR, &_secp256k1_::ONE_R
#ifdef GEC_ENABLE_CUDA
                      ,
                      &_secp256k1_::d_MOD, &_secp256k1_::d_RR,
                      &_secp256k1_::d_ONE_R
#endif // GEC_ENABLE_CUDA
                      >;

// ----- Scalar -----
namespace _secp256k1_ {
// same base between finite field and scalar
using SBase = FBase;

extern const SBase CARD;
constexpr SBase::LimbT CARD_P = 0x4b0dff665588b13full;
extern const SBase CARD_RR;
extern const SBase CARD_ONE_R;

#ifdef GEC_ENABLE_CUDA
#ifdef __CUDACC__
__constant__
#endif
    extern const SBase d_CARD;
#ifdef __CUDACC__
__constant__
#endif
    extern const SBase d_CARD_RR;
#ifdef __CUDACC__
__constant__
#endif
    extern const SBase d_CARD_ONE_R;
#endif // GEC_ENABLE_CUDA

} // namespace _secp256k1_
using Scalar = bigint::BaseField<
    _secp256k1_::SBase, &_secp256k1_::CARD, _secp256k1_::CARD_P,
    &_secp256k1_::CARD_RR, &_secp256k1_::CARD_ONE_R
#ifdef GEC_ENABLE_CUDA
    ,
    &_secp256k1_::d_CARD, &_secp256k1_::d_CARD_RR, &_secp256k1_::d_CARD_ONE_R
#endif // GEC_ENABLE_CUDA
    >;

// ----- Curve -----
namespace _secp256k1_ {
// A = 0
extern const Field B;

#ifdef GEC_ENABLE_CUDA
// d_A = 0
#ifdef __CUDACC__
__constant__
#endif
    extern const Field d_B;
#endif // GEC_ENABLE_CUDA

} // namespace _secp256k1_
template <template <typename FT, const FT *, const FT *, const FT *, const FT *,
                    bool InfY>
          class Coordinate = curve::JacobianCurve,
          bool InfYZero = true>
using Curve = Coordinate<Field, nullptr, &_secp256k1_::B, nullptr,
#ifdef GEC_ENABLE_CUDA
                         &_secp256k1_::d_B
#else
                         nullptr
#endif // GEC_ENABLE_CUDA
                         ,
                         InfYZero>;

extern const Curve<> Gen;
#ifdef __CUDACC__
__constant__ extern const Curve<> d_Gen;
#endif // __CUDACC__

} // namespace secp256k1

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_SECP256K1_HPP

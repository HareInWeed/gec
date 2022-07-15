#pragma once
#ifndef GEC_TEST_CURVE_HPP
#define GEC_TEST_CURVE_HPP

#include "common.hpp"
#include "field.hpp"

#include <gec/curve.hpp>

template <typename Core, typename FT, const FT *A, const FT *B, const FT *d_A,
          const FT *d_B>
class GEC_EMPTY_BASES CurveMixin
    : public gec::curve::CurveParams<Core, FT, A, B, d_A, d_B>,
      public gec::curve::ScalerMul<Core>,
      public gec::curve::WithPointContext<Core>,
      public gec::curve::PointOstream<Core>,
      public gec::curve::PointPrint<Core> {};

template <typename Core, typename FT, const FT *A, const FT *B, const FT *d_A,
          const FT *d_B>
class GEC_EMPTY_BASES AffineMixin : public CurveMixin<Core, FT, A, B, d_A, d_B>,
                                    public gec::curve::Affine<Core, FT>,
                                    public gec::curve::CompWiseEq<Core>,
                                    public gec::curve::WithPointHasher<Core> {};

template <typename FT, const FT *A, const FT *B, const FT *d_A = nullptr,
          const FT *d_B = nullptr>
struct GEC_EMPTY_BASES AffineC
    : public gec::curve::Point<FT, 2>,
      public AffineMixin<AffineC<FT, A, B, d_A, d_B>, FT, A, B, d_A, d_B> {
    using gec::curve::Point<FT, 2>::Point;
};

template <typename Core, typename FT, const FT *A, const FT *B, const FT *d_A,
          const FT *d_B>
class GEC_EMPTY_BASES JacobianMixin
    : public CurveMixin<Core, FT, A, B, d_A, d_B>,
      public gec::curve::Jacobain<Core, FT> {};

template <typename FT, const FT *A, const FT *B, const FT *d_A = nullptr,
          const FT *d_B = nullptr>
struct GEC_EMPTY_BASES JacobianC
    : public gec::curve::Point<FT, 3>,
      public JacobianMixin<JacobianC<FT, A, B, d_A, d_B>, FT, A, B, d_A, d_B> {
    using gec::curve::Point<FT, 3>::Point;
};

template <typename Core, typename FT, const FT *A, const FT *B, const FT *d_A,
          const FT *d_B>
class GEC_EMPTY_BASES ProjectiveMixin
    : public CurveMixin<Core, FT, A, B, d_A, d_B>,
      public gec::curve::Projective<Core, FT> {};

template <typename FT, const FT *A, const FT *B, const FT *d_A = nullptr,
          const FT *d_B = nullptr>
struct GEC_EMPTY_BASES ProjectiveC
    : public gec::curve::Point<FT, 3>,
      public ProjectiveMixin<ProjectiveC<FT, A, B, d_A, d_B>, FT, A, B, d_A,
                             d_B> {
    using gec::curve::Point<FT, 3>::Point;
};

#ifdef __CUDACC__
#define CURVE(coordinate, F, A, B) coordinate<F, &A, &B, &d_##A, &d_##B>
#define CURVE_A(coordinate, F, A) coordinate<F, &A, nullptr, &d_##A, nullptr>
#define CURVE_B(coordinate, F, B) coordinate<F, nullptr, &B, nullptr, &d_##B>
#else
#define CURVE(coordinate, F, A, B) coordinate<F, &A, &B>
#define CURVE_A(coordinate, F, A) coordinate<F, &A, nullptr>
#define CURVE_B(coordinate, F, B) coordinate<F, nullptr, &B>
#endif // __CUDACC__

decl_field(AR_160, Field160);
decl_field(BR_160, Field160);
using CurveA = CURVE(AffineC, Field160, AR_160, BR_160);
using CurveP = CURVE(ProjectiveC, Field160, AR_160, BR_160);
using CurveJ = CURVE(JacobianC, Field160, AR_160, BR_160);

decl_field(AR2_160, Field160_2);
decl_field(BR2_160, Field160_2);
using CurveA2 = CURVE(AffineC, Field160_2, AR2_160, BR2_160);
using CurveP2 = CURVE(ProjectiveC, Field160_2, AR2_160, BR2_160);
using CurveJ2 = CURVE(JacobianC, Field160_2, AR2_160, BR2_160);

decl_array(Dlp1P, LIMB_T, LN_160);
constexpr LIMB_T Dlp1P_P = 0x5afdc9d5u;
decl_array(Dlp1P_RR, LIMB_T, LN_160);
decl_array(Dlp1P_OneR, LIMB_T, LN_160);
using Dlp1Field = FIELD(LIMB_T, LN_160, alignof(LIMB_T), Dlp1P, Dlp1P_P,
                        Dlp1P_RR, Dlp1P_OneR);

// decl_field(Dlp1A, Dlp1Field);
decl_field(Dlp1B, Dlp1Field);

decl_array(Dlp1Card, LIMB_T, LN_160);
using Dlp1Scaler = ADD_GROUP(LIMB_T, LN_160, alignof(LIMB_T), Dlp1Card);

using Dlp1CurveA = CURVE_B(AffineC, Dlp1Field, Dlp1B);
using Dlp1CurveJ = CURVE_B(JacobianC, Dlp1Field, Dlp1B);

decl_aligned_array(Dlp1P2, LIMB2_T, LN2_160, 32);
constexpr LIMB2_T Dlp1P2_P = 0xdb83306e5afdc9d5llu;
decl_aligned_array(Dlp1P2_RR, LIMB2_T, LN2_160, 32);
decl_aligned_array(Dlp1P2_OneR, LIMB2_T, LN2_160, 32);
using Dlp1Field2 = FIELD(LIMB2_T, LN2_160, 32, Dlp1P2, Dlp1P2_P, Dlp1P2_RR,
                         Dlp1P2_OneR);

// decl_field(Dlp1A2, Dlp1Field2);
decl_field(Dlp1B2, Dlp1Field2);

decl_array(Dlp1Card2, LIMB2_T, LN2_160);
using Dlp1Scaler2 = ADD_GROUP(LIMB2_T, LN2_160, alignof(LIMB2_T), Dlp1Card2);

using Dlp1CurveJ2 = CURVE_B(JacobianC, Dlp1Field2, Dlp1B2);

decl_array(Dlp2P, LIMB_T, 1);
constexpr LIMB_T Dlp2P_P = 3105566705u;
decl_array(Dlp2P_RR, LIMB_T, 1);
decl_array(Dlp2P_OneR, LIMB_T, 1);
using Dlp2Field = FIELD(LIMB_T, 1, alignof(LIMB_T), Dlp2P, Dlp2P_P, Dlp2P_RR,
                        Dlp2P_OneR);

decl_field(Dlp2A, Dlp2Field);
decl_field(Dlp2B, Dlp2Field);

decl_array(Dlp2Card, LIMB_T, 1);
constexpr LIMB_T Dlp2Card_P = 0xfbd05cfu;
decl_array(Dlp2Card_RR, LIMB_T, 1);
decl_array(Dlp2Card_OneR, LIMB_T, 1);
using Dlp2Scaler = FIELD(LIMB_T, 1, alignof(LIMB_T), Dlp2Card, Dlp2Card_P,
                         Dlp2Card_RR, Dlp2Card_OneR);

using Dlp2CurveJ = CURVE(JacobianC, Dlp2Field, Dlp2A, Dlp2B);

constexpr size_t Dlp3N = 8;
decl_aligned_array(Dlp3P, LIMB_T, Dlp3N, 32);
constexpr LIMB_T Dlp3P_P = 0xd2253531u;
decl_aligned_array(Dlp3P_RR, LIMB_T, Dlp3N, 32);
decl_aligned_array(Dlp3P_OneR, LIMB_T, Dlp3N, 32);
using Dlp3Field = FIELD(LIMB_T, Dlp3N, 32, Dlp3P, Dlp3P_P, Dlp3P_RR,
                        Dlp3P_OneR);
#ifdef GEC_ENABLE_AVX2
using AVX2Dlp3Field = AVX2FIELD(LIMB_T, Dlp3N, 32, Dlp3P, Dlp3P_P, Dlp3P_RR,
                                Dlp3P_OneR);
#endif // GEC_ENABLE_AVX2

// decl_field(Dlp3A, Dlp3Field);
decl_field(Dlp3B, Dlp3Field);
using Dlp3CurveJ = CURVE_B(JacobianC, Dlp3Field, Dlp3B);
using Dlp3CurveA = CURVE_B(AffineC, Dlp3Field, Dlp3B);

#ifdef GEC_ENABLE_AVX2
// extern const AVX2Dlp3Field AVX2Dlp3A;
extern const AVX2Dlp3Field AVX2Dlp3B;
using AVX2Dlp3CurveA = AffineC<AVX2Dlp3Field, nullptr, &AVX2Dlp3B>;
#endif // GEC_ENABLE_AVX2

decl_field(Dlp3Gen1, Dlp3CurveA);
constexpr size_t Dlp3G1SN = 2;
decl_aligned_array(Dlp3G1Card, LIMB_T, Dlp3G1SN, 8);
constexpr LIMB_T Dlp3G1Card_P = 0x36a04ecdu;
decl_aligned_array(Dlp3G1Card_RR, LIMB_T, Dlp3G1SN, 8);
decl_aligned_array(Dlp3G1Card_OneR, LIMB_T, Dlp3G1SN, 8);
using Dlp3G1Scaler = FIELD(LIMB_T, Dlp3G1SN, 8, Dlp3G1Card, Dlp3G1Card_P,
                           Dlp3G1Card_RR, Dlp3G1Card_OneR);

decl_field(Dlp3Gen2, Dlp3CurveA);
constexpr size_t Dlp3G2SN = 2;
decl_aligned_array(Dlp3G2Card, LIMB_T, Dlp3G2SN, 8);
constexpr LIMB_T Dlp3G2Card_P = 0x9013b4b9u;
decl_aligned_array(Dlp3G2Card_RR, LIMB_T, Dlp3G2SN, 8);
decl_aligned_array(Dlp3G2Card_OneR, LIMB_T, Dlp3G2SN, 8);
using Dlp3G2Scaler = FIELD(LIMB_T, Dlp3G2SN, 8, Dlp3G2Card, Dlp3G2Card_P,
                           Dlp3G2Card_RR, Dlp3G2Card_OneR);

constexpr size_t Dlp3N2 = 4;
decl_aligned_array(Dlp3P2, LIMB2_T, Dlp3N2, 32);
constexpr LIMB2_T Dlp3P2_P = 0xd838091dd2253531llu;
decl_aligned_array(Dlp3P2_RR, LIMB2_T, Dlp3N2, 32);
decl_aligned_array(Dlp3P2_OneR, LIMB2_T, Dlp3N2, 32);
using Dlp3Field2 = FIELD(LIMB2_T, Dlp3N2, 32, Dlp3P2, Dlp3P2_P, Dlp3P2_RR,
                         Dlp3P2_OneR);

// decl_field(Dlp3A2, Dlp3Field2);
decl_field(Dlp3B2, Dlp3Field2);

constexpr size_t Dlp3SN2 = 1;
decl_aligned_array(Dlp3Card2, LIMB2_T, Dlp3SN2, 8);
constexpr LIMB2_T Dlp3Card2_P = 0x61edaaec36a04ecdllu;
decl_aligned_array(Dlp3Card2_RR, LIMB2_T, Dlp3SN2, 8);
decl_aligned_array(Dlp3Card2_OneR, LIMB2_T, Dlp3SN2, 8);
using Dlp3Scaler2 = FIELD(LIMB2_T, Dlp3SN2, 8, Dlp3Card2, Dlp3Card2_P,
                          Dlp3Card2_RR, Dlp3Card2_OneR);

using Dlp3CurveJ2 = CURVE_B(JacobianC, Dlp3Field2, Dlp3B2);

#endif // !GEC_TEST_CURVE_HPP
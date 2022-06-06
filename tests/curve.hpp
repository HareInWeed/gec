#pragma once
#ifndef GEC_TEST_CURVE_HPP
#define GEC_TEST_CURVE_HPP

#include "common.hpp"
#include "field.hpp"

#include <gec/curve.hpp>

template <typename Core>
class CurveMixin : public gec::curve::ScalerMul<Core>,
                   public gec::curve::WithPointContext<Core>,
                   public gec::curve::PointOstream<Core>,
                   public gec::curve::PointPrint<Core> {};

template <typename FT, const FT &A, const FT &B>
struct AffineC : public gec::curve::Point<FT, 2>,
                 public gec::curve::Affine<AffineC<FT, A, B>, FT, A, B>,
                 public gec::curve::CompWiseEq<AffineC<FT, A, B>>,
                 public gec::curve::WithPointHasher<AffineC<FT, A, B>>,
                 public CurveMixin<AffineC<FT, A, B>> {
    using gec::curve::Point<FT, 2>::Point;
};

template <typename FT, const FT &A, const FT &B>
struct JacobianC : public gec::curve::Point<FT, 3>,
                   public gec::curve::Jacobain<JacobianC<FT, A, B>, FT, A, B>,
                   public CurveMixin<JacobianC<FT, A, B>> {
    using gec::curve::Point<FT, 3>::Point;
};

template <typename FT, const FT &A, const FT &B>
struct ProjectiveC
    : public gec::curve::Point<FT, 3>,
      public gec::curve::Projective<ProjectiveC<FT, A, B>, FT, A, B>,
      public CurveMixin<ProjectiveC<FT, A, B>> {
    using gec::curve::Point<FT, 3>::Point;
};

extern const Field160 AR_160;
extern const Field160 BR_160;
using CurveA = AffineC<Field160, AR_160, BR_160>;
using CurveP = ProjectiveC<Field160, AR_160, BR_160>;
using CurveJ = JacobianC<Field160, AR_160, BR_160>;

extern const Field160_2 AR2_160;
extern const Field160_2 BR2_160;
using CurveA2 = AffineC<Field160_2, AR2_160, BR2_160>;
using CurveP2 = ProjectiveC<Field160_2, AR2_160, BR2_160>;
using CurveJ2 = JacobianC<Field160_2, AR2_160, BR2_160>;

decl_array(Dlp1P, LIMB_T, LN_160);
constexpr LIMB_T Dlp1P_P = 0x5afdc9d5u;
decl_array(Dlp1P_RR, LIMB_T, LN_160);
decl_array(Dlp1P_OneR, LIMB_T, LN_160);
using Dlp1Field = FIELD(LIMB_T, LN_160, alignof(LIMB_T), Dlp1P, Dlp1P_P,
                        Dlp1P_RR, Dlp1P_OneR);

extern const Dlp1Field Dlp1A;
extern const Dlp1Field Dlp1B;

decl_array(Dlp1Card, LIMB_T, LN_160);
using Dlp1Scaler = ADD_GROUP(LIMB_T, LN_160, alignof(LIMB_T), Dlp1Card);

using Dlp1CurveJ = JacobianC<Dlp1Field, Dlp1A, Dlp1B>;

decl_aligned_array(Dlp1P2, LIMB2_T, LN2_160, 32);
constexpr LIMB2_T Dlp1P2_P = 0xdb83306e5afdc9d5llu;
decl_aligned_array(Dlp1P2_RR, LIMB2_T, LN2_160, 32);
decl_aligned_array(Dlp1P2_OneR, LIMB2_T, LN2_160, 32);
using Dlp1Field2 = FIELD(LIMB2_T, LN2_160, 32, Dlp1P2, Dlp1P2_P, Dlp1P2_RR,
                         Dlp1P2_OneR);

extern const Dlp1Field2 Dlp1A2;
extern const Dlp1Field2 Dlp1B2;

decl_array(Dlp1Card2, LIMB2_T, LN2_160);
using Dlp1Scaler2 = ADD_GROUP(LIMB2_T, LN2_160, 0, Dlp1Card2);

using Dlp1CurveJ2 = JacobianC<Dlp1Field2, Dlp1A2, Dlp1B2>;

decl_array(Dlp2P, LIMB_T, 1);
constexpr LIMB_T Dlp2P_P = 3105566705u;
decl_array(Dlp2P_RR, LIMB_T, 1);
decl_array(Dlp2P_OneR, LIMB_T, 1);
using Dlp2Field = FIELD(LIMB_T, 1, alignof(LIMB_T), Dlp2P, Dlp2P_P, Dlp2P_RR,
                        Dlp2P_OneR);

extern const Dlp2Field Dlp2A;
extern const Dlp2Field Dlp2B;

decl_array(Dlp2Card, LIMB_T, 1);
constexpr LIMB_T Dlp2Card_P = 0xfbd05cfu;
decl_array(Dlp2Card_RR, LIMB_T, 1);
decl_array(Dlp2Card_OneR, LIMB_T, 1);
using Dlp2Scaler = FIELD(LIMB_T, 1, alignof(LIMB_T), Dlp2Card, Dlp2Card_P,
                         Dlp2Card_RR, Dlp2Card_OneR);

using Dlp2CurveJ = JacobianC<Dlp2Field, Dlp2A, Dlp2B>;

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

extern const Dlp3Field Dlp3A;
extern const Dlp3Field Dlp3B;
using Dlp3CurveJ = JacobianC<Dlp3Field, Dlp3A, Dlp3B>;
using Dlp3CurveA = AffineC<Dlp3Field, Dlp3A, Dlp3B>;

#ifdef GEC_ENABLE_AVX2
extern const AVX2Dlp3Field AVX2Dlp3A;
extern const AVX2Dlp3Field AVX2Dlp3B;
using AVX2Dlp3CurveA = AffineC<AVX2Dlp3Field, AVX2Dlp3A, AVX2Dlp3B>;
#endif // GEC_ENABLE_AVX2

extern const Dlp3CurveA Dlp3Gen1;
constexpr size_t Dlp3G1SN = 2;
decl_aligned_array(Dlp3G1Card, LIMB_T, Dlp3G1SN, 8);
constexpr LIMB_T Dlp3G1Card_P = 0x36a04ecdu;
decl_aligned_array(Dlp3G1Card_RR, LIMB_T, Dlp3G1SN, 8);
decl_aligned_array(Dlp3G1Card_OneR, LIMB_T, Dlp3G1SN, 8);
using Dlp3G1Scaler = FIELD(LIMB_T, Dlp3G1SN, 8, Dlp3G1Card, Dlp3G1Card_P,
                           Dlp3G1Card_RR, Dlp3G1Card_OneR);

extern const Dlp3CurveA Dlp3Gen2;
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

extern const Dlp3Field2 Dlp3A2;
extern const Dlp3Field2 Dlp3B2;

constexpr size_t Dlp3SN2 = 1;
decl_aligned_array(Dlp3Card2, LIMB2_T, Dlp3SN2, 8);
constexpr LIMB2_T Dlp3Card2_P = 0x61edaaec36a04ecdllu;
decl_aligned_array(Dlp3Card2_RR, LIMB2_T, Dlp3SN2, 8);
decl_aligned_array(Dlp3Card2_OneR, LIMB2_T, Dlp3SN2, 8);
using Dlp3Scaler2 = FIELD(LIMB2_T, Dlp3SN2, 8, Dlp3Card2, Dlp3Card2_P,
                          Dlp3Card2_RR, Dlp3Card2_OneR);

using Dlp3CurveJ2 = JacobianC<Dlp3Field2, Dlp3A2, Dlp3B2>;

#endif // !GEC_TEST_CURVE_HPP
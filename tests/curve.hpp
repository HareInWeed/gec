#pragma once
#ifndef GEC_TEST_CURVE_HPP
#define GEC_TEST_CURVE_HPP

#include "common.hpp"
#include "field.hpp"

#include <gec/curve.hpp>

extern const Field160 AR_160;
extern const Field160 BR_160;

extern const Field160_2 AR2_160;
extern const Field160_2 BR2_160;

class CurveA : public gec::curve::Point<Field160, 2>,
               public gec::curve::Affine<CurveA, Field160, AR_160, BR_160>,
               public gec::curve::ScalerMul<CurveA>,
               public gec::curve::WithPointContext<CurveA>,
               public gec::curve::PointOstream<CurveA>,
               public gec::curve::PointPrint<CurveA> {
    using Point::Point;
};

class CurveA2
    : public gec::curve::Point<Field160_2, 2>,
      public gec::curve::Affine<CurveA2, Field160_2, AR2_160, BR2_160>,
      public gec::curve::ScalerMul<CurveA2>,
      public gec::curve::WithPointContext<CurveA2>,
      public gec::curve::PointOstream<CurveA2>,
      public gec::curve::PointPrint<CurveA2> {
    using Point::Point;
};

class CurveP : public gec::curve::Point<Field160, 3>,
               public gec::curve::Jacobain<CurveP, Field160, AR_160, BR_160>,
               public gec::curve::ScalerMul<CurveP>,
               public gec::curve::WithPointContext<CurveP>,
               public gec::curve::PointOstream<CurveP>,
               public gec::curve::PointPrint<CurveP> {
    using Point::Point;
};

class CurveP2
    : public gec::curve::Point<Field160_2, 3>,
      public gec::curve::Jacobain<CurveP2, Field160_2, AR2_160, BR2_160>,
      public gec::curve::ScalerMul<CurveP2>,
      public gec::curve::WithPointContext<CurveP2>,
      public gec::curve::PointOstream<CurveP2>,
      public gec::curve::PointPrint<CurveP2> {
    using Point::Point;
};

class CurveJ : public gec::curve::Point<Field160, 3>,
               public gec::curve::Jacobain<CurveJ, Field160, AR_160, BR_160>,
               public gec::curve::ScalerMul<CurveJ>,
               public gec::curve::WithPointContext<CurveJ>,
               public gec::curve::PointOstream<CurveJ>,
               public gec::curve::PointPrint<CurveJ> {
    using Point::Point;
};

class CurveJ2
    : public gec::curve::Point<Field160_2, 3>,
      public gec::curve::Jacobain<CurveJ2, Field160_2, AR2_160, BR2_160>,
      public gec::curve::ScalerMul<CurveJ2>,
      public gec::curve::WithPointContext<CurveJ2>,
      public gec::curve::PointOstream<CurveJ2>,
      public gec::curve::PointPrint<CurveJ2> {
    using Point::Point;
};

extern const LIMB_T Dlp1P[LN_160];
constexpr LIMB_T Dlp1P_P = 0x5afdc9d5u;
extern const LIMB_T Dlp1P_RR[LN_160];
extern const LIMB_T Dlp1P_OneR[LN_160];
using Dlp1Field = Field<LIMB_T, LN_160, Dlp1P, Dlp1P_P, Dlp1P_RR, Dlp1P_OneR>;

extern const Dlp1Field Dlp1A;
extern const Dlp1Field Dlp1B;

extern const LIMB_T Dlp1Card[LN_160];
using Dlp1Scaler = AddGroup<LIMB_T, LN_160, Dlp1Card>;

class Dlp1CurveJ
    : public gec::curve::Point<Dlp1Field, 3>,
      public gec::curve::Jacobain<Dlp1CurveJ, Dlp1Field, Dlp1A, Dlp1B>,
      public gec::curve::ScalerMul<Dlp1CurveJ>,
      public gec::curve::WithPointContext<Dlp1CurveJ>,
      public gec::curve::PointOstream<Dlp1CurveJ>,
      public gec::curve::PointPrint<Dlp1CurveJ> {
    using Point::Point;
};

alignas(32) extern const LIMB2_T Dlp1P2[LN2_160];
constexpr LIMB2_T Dlp1P2_P = 0xdb83306e5afdc9d5llu;
alignas(32) extern const LIMB2_T Dlp1P2_RR[LN2_160];
alignas(32) extern const LIMB2_T Dlp1P2_OneR[LN2_160];
using Dlp1Field2 =
    Field<LIMB2_T, LN2_160, Dlp1P2, Dlp1P2_P, Dlp1P2_RR, Dlp1P2_OneR, 32>;

extern const Dlp1Field2 Dlp1A2;
extern const Dlp1Field2 Dlp1B2;

extern const LIMB2_T Dlp1Card2[LN2_160];
using Dlp1Scaler2 = AddGroup<LIMB2_T, LN2_160, Dlp1Card2>;

class Dlp1CurveJ2
    : public gec::curve::Point<Dlp1Field2, 3>,
      public gec::curve::Jacobain<Dlp1CurveJ2, Dlp1Field2, Dlp1A2, Dlp1B2>,
      public gec::curve::ScalerMul<Dlp1CurveJ2>,
      public gec::curve::WithPointContext<Dlp1CurveJ2>,
      public gec::curve::PointOstream<Dlp1CurveJ2>,
      public gec::curve::PointPrint<Dlp1CurveJ2> {
    using Point::Point;
};

extern const LIMB_T Dlp2P[1];
constexpr LIMB_T Dlp2P_P = 3105566705u;
extern const LIMB_T Dlp2P_RR[1];
extern const LIMB_T Dlp2P_OneR[1];
using Dlp2Field = Field<LIMB_T, 1, Dlp2P, Dlp2P_P, Dlp2P_RR, Dlp2P_OneR>;

extern const Dlp2Field Dlp2A;
extern const Dlp2Field Dlp2B;

extern const LIMB_T Dlp2Card[1];
constexpr LIMB_T Dlp2Card_P = 0xfbd05cfu;
extern const LIMB_T Dlp2Card_RR[1];
extern const LIMB_T Dlp2Card_OneR[1];
using Dlp2Scaler =
    Field<LIMB_T, 1, Dlp2Card, Dlp2Card_P, Dlp2Card_RR, Dlp2Card_OneR>;

class Dlp2CurveJ
    : public gec::curve::Point<Dlp2Field, 3>,
      public gec::curve::Jacobain<Dlp2CurveJ, Dlp2Field, Dlp2A, Dlp2B>,
      public gec::curve::ScalerMul<Dlp2CurveJ>,
      public gec::curve::WithPointContext<Dlp2CurveJ>,
      public gec::curve::PointOstream<Dlp2CurveJ>,
      public gec::curve::PointPrint<Dlp2CurveJ> {
    using Point::Point;
};

constexpr size_t Dlp3N = 8;
alignas(32) extern const LIMB_T Dlp3P[Dlp3N];
constexpr LIMB_T Dlp3P_P = 0xd2253531u;
alignas(32) extern const LIMB_T Dlp3P_RR[Dlp3N];
alignas(32) extern const LIMB_T Dlp3P_OneR[Dlp3N];
using Dlp3Field =
    Field<LIMB_T, Dlp3N, Dlp3P, Dlp3P_P, Dlp3P_RR, Dlp3P_OneR, 32>;

extern const Dlp3Field Dlp3A;
extern const Dlp3Field Dlp3B;

class Dlp3CurveJ
    : public gec::curve::Point<Dlp3Field, 3>,
      public gec::curve::Jacobain<Dlp3CurveJ, Dlp3Field, Dlp3A, Dlp3B>,
      public gec::curve::ScalerMul<Dlp3CurveJ>,
      public gec::curve::WithPointContext<Dlp3CurveJ>,
      public gec::curve::PointOstream<Dlp3CurveJ>,
      public gec::curve::PointPrint<Dlp3CurveJ> {
    using Point::Point;
};

class Dlp3CurveA
    : public gec::curve::Point<Dlp3Field, 2>,
      public gec::curve::Affine<Dlp3CurveA, Dlp3Field, Dlp3A, Dlp3B>,
      public gec::curve::ScalerMul<Dlp3CurveA>,
      public gec::curve::CompWiseEq<Dlp3CurveA>,
      public gec::curve::WithPointHasher<Dlp3CurveA>,
      public gec::curve::WithPointContext<Dlp3CurveA>,
      public gec::curve::PointOstream<Dlp3CurveA>,
      public gec::curve::PointPrint<Dlp3CurveA> {
    using Point::Point;
};

extern const Dlp3CurveA Dlp3Gen1;
constexpr size_t Dlp3G1SN = 2;
alignas(8) extern const LIMB_T Dlp3G1Card[Dlp3G1SN];
constexpr LIMB_T Dlp3G1Card_P = 0x36a04ecdu;
alignas(8) extern const LIMB_T Dlp3G1Card_RR[Dlp3G1SN];
alignas(8) extern const LIMB_T Dlp3G1Card_OneR[Dlp3G1SN];
using Dlp3G1Scaler = Field<LIMB_T, Dlp3G1SN, Dlp3G1Card, Dlp3G1Card_P,
                           Dlp3G1Card_RR, Dlp3G1Card_OneR, 8>;

extern const Dlp3CurveA Dlp3Gen2;
constexpr size_t Dlp3G2SN = 2;
alignas(8) extern const LIMB_T Dlp3G2Card[Dlp3G2SN];
constexpr LIMB_T Dlp3G2Card_P = 0x9013b4b9u;
alignas(8) extern const LIMB_T Dlp3G2Card_RR[Dlp3G2SN];
alignas(8) extern const LIMB_T Dlp3G2Card_OneR[Dlp3G2SN];
using Dlp3G2Scaler = Field<LIMB_T, Dlp3G2SN, Dlp3G2Card, Dlp3G2Card_P,
                           Dlp3G2Card_RR, Dlp3G2Card_OneR, 8>;

constexpr size_t Dlp3N2 = 4;
alignas(32) extern const LIMB2_T Dlp3P2[Dlp3N];
constexpr LIMB2_T Dlp3P2_P = 0xd838091dd2253531llu;
alignas(32) extern const LIMB2_T Dlp3P2_RR[Dlp3N];
alignas(32) extern const LIMB2_T Dlp3P2_OneR[Dlp3N];
using Dlp3Field2 =
    Field<LIMB2_T, Dlp3N2, Dlp3P2, Dlp3P2_P, Dlp3P2_RR, Dlp3P2_OneR, 32>;

extern const Dlp3Field2 Dlp3A2;
extern const Dlp3Field2 Dlp3B2;

constexpr size_t Dlp3SN2 = 1;
alignas(8) extern const LIMB2_T Dlp3Card2[Dlp3SN2];
constexpr LIMB2_T Dlp3Card2_P = 0x61edaaec36a04ecdllu;
alignas(8) extern const LIMB2_T Dlp3Card2_RR[Dlp3SN2];
alignas(8) extern const LIMB2_T Dlp3Card2_OneR[Dlp3SN2];
using Dlp3Scaler2 = Field<LIMB2_T, Dlp3SN2, Dlp3Card2, Dlp3Card2_P,
                          Dlp3Card2_RR, Dlp3Card2_OneR, 8>;

class Dlp3CurveJ2
    : public gec::curve::Point<Dlp3Field2, 3>,
      public gec::curve::Jacobain<Dlp3CurveJ2, Dlp3Field2, Dlp3A2, Dlp3B2>,
      public gec::curve::ScalerMul<Dlp3CurveJ2>,
      public gec::curve::WithPointContext<Dlp3CurveJ2>,
      public gec::curve::PointOstream<Dlp3CurveJ2>,
      public gec::curve::PointPrint<Dlp3CurveJ2> {
    using Point::Point;
};

#endif // !GEC_TEST_CURVE_HPP
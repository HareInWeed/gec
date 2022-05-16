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

alignas(8) extern const LIMB_T DlpP[LN_160];
constexpr LIMB_T DlpP_P = 0x5afdc9d5u;
alignas(8) extern const LIMB_T DlpP_RR[LN_160];
alignas(8) extern const LIMB_T DlpP_OneR[LN_160];
using DlpField = Field<LIMB_T, LN_160, DlpP, DlpP_P, DlpP_RR, DlpP_OneR>;

extern const DlpField DlpA;
extern const DlpField DlpB;

alignas(8) extern const LIMB_T DlpCard[LN_160];
using DlpScaler = AddGroup<LIMB_T, LN_160, DlpCard>;

class DlpCurveJ : public gec::curve::Point<DlpField, 3>,
                  public gec::curve::Jacobain<DlpCurveJ, DlpField, DlpA, DlpB>,
                  public gec::curve::ScalerMul<DlpCurveJ>,
                  public gec::curve::WithPointContext<DlpCurveJ>,
                  public gec::curve::PointOstream<DlpCurveJ>,
                  public gec::curve::PointPrint<DlpCurveJ> {
    using Point::Point;
};

alignas(32) extern const LIMB2_T DlpP2[LN2_160];
constexpr LIMB2_T DlpP2_P = 0xdb83306e5afdc9d5llu;
alignas(32) extern const LIMB2_T DlpP2_RR[LN2_160];
alignas(32) extern const LIMB2_T DlpP2_OneR[LN2_160];
using DlpField2 = Field<LIMB2_T, LN2_160, DlpP2, DlpP2_P, DlpP2_RR, DlpP2_OneR>;

extern const DlpField2 DlpA2;
extern const DlpField2 DlpB2;

alignas(32) extern const LIMB2_T DlpCard2[LN2_160];
using DlpScaler2 = AddGroup<LIMB2_T, LN2_160, DlpCard2>;

class DlpCurveJ2
    : public gec::curve::Point<DlpField2, 3>,
      public gec::curve::Jacobain<DlpCurveJ2, DlpField2, DlpA2, DlpB2>,
      public gec::curve::ScalerMul<DlpCurveJ2>,
      public gec::curve::WithPointContext<DlpCurveJ2>,
      public gec::curve::PointOstream<DlpCurveJ2>,
      public gec::curve::PointPrint<DlpCurveJ2> {
    using Point::Point;
};

#endif // !GEC_TEST_CURVE_HPP
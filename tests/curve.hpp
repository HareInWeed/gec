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
               public gec::curve::PointOstream<CurveA>,
               public gec::curve::PointPrint<CurveA> {
    using Point::Point;
};

class CurveA2
    : public gec::curve::Point<Field160_2, 2>,
      public gec::curve::Affine<CurveA2, Field160_2, AR2_160, BR2_160>,
      public gec::curve::PointOstream<CurveA2>,
      public gec::curve::PointPrint<CurveA2> {
    using Point::Point;
};

class CurveP : public gec::curve::Point<Field160, 3>,
               public gec::curve::Jacobain<CurveP, Field160, AR_160, BR_160>,
               public gec::curve::PointOstream<CurveP>,
               public gec::curve::PointPrint<CurveP> {
    using Point::Point;
};

class CurveP2
    : public gec::curve::Point<Field160_2, 3>,
      public gec::curve::Jacobain<CurveP2, Field160_2, AR2_160, BR2_160>,
      public gec::curve::PointOstream<CurveP2>,
      public gec::curve::PointPrint<CurveP2> {
    using Point::Point;
};

class CurveJ : public gec::curve::Point<Field160, 3>,
               public gec::curve::Jacobain<CurveJ, Field160, AR_160, BR_160>,
               public gec::curve::PointOstream<CurveJ>,
               public gec::curve::PointPrint<CurveJ> {
    using Point::Point;
};

class CurveJ2
    : public gec::curve::Point<Field160_2, 3>,
      public gec::curve::Jacobain<CurveJ2, Field160_2, AR2_160, BR2_160>,
      public gec::curve::PointOstream<CurveJ2>,
      public gec::curve::PointPrint<CurveJ2> {
    using Point::Point;
};

#endif // !GEC_TEST_CURVE_HPP
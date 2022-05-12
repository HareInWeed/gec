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

class CurveA : public gec::curve::Point2<Field160>,
               public gec::curve::Affine<CurveA, Field160, AR_160, BR_160>,
               public gec::curve::Point2Ostream<CurveA>,
               public gec::curve::Point2Print<CurveA> {
    using Point2::Point2;
};

class CurveA2
    : public gec::curve::Point2<Field160_2>,
      public gec::curve::Affine<CurveA2, Field160_2, AR2_160, BR2_160>,
      public gec::curve::Point2Ostream<CurveA2>,
      public gec::curve::Point2Print<CurveA2> {
    using Point2::Point2;
};

class CurveP : public gec::curve::Point3<Field160>,
               public gec::curve::Jacobain<CurveP, Field160, AR_160, BR_160>,
               public gec::curve::Point3Ostream<CurveP>,
               public gec::curve::Point3Print<CurveP> {
    using Point3::Point3;
};

class CurveP2
    : public gec::curve::Point3<Field160_2>,
      public gec::curve::Jacobain<CurveP2, Field160_2, AR2_160, BR2_160>,
      public gec::curve::Point3Ostream<CurveP2>,
      public gec::curve::Point3Print<CurveP2> {
    using Point3::Point3;
};

class CurveJ : public gec::curve::Point3<Field160>,
               public gec::curve::Jacobain<CurveJ, Field160, AR_160, BR_160>,
               public gec::curve::Point3Ostream<CurveJ>,
               public gec::curve::Point3Print<CurveJ> {
    using Point3::Point3;
};

class CurveJ2
    : public gec::curve::Point3<Field160_2>,
      public gec::curve::Jacobain<CurveJ2, Field160_2, AR2_160, BR2_160>,
      public gec::curve::Point3Ostream<CurveJ2>,
      public gec::curve::Point3Print<CurveJ2> {
    using Point3::Point3;
};

#endif // !GEC_TEST_CURVE_HPP
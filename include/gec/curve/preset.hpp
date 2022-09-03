#pragma once
#ifndef GEC_CURVE_PRESET_HPP
#define GEC_CURVE_PRESET_HPP

#include "data.hpp"
#include "mixin.hpp"

namespace gec {

namespace curve {

template <typename FT, const FT *A, const FT *B, const FT *d_A = nullptr,
          const FT *d_B = nullptr, bool InfYZero = true>
struct AffineCurve
    : public Point<FT, 2>,
      public AffineMixin<AffineCurve<FT, A, B, d_A, d_B, InfYZero>, FT, A, B,
                         d_A, d_B, InfYZero> {
    using Point<FT, 2>::Point;
};

template <typename FT, const FT *A, const FT *B, const FT *d_A = nullptr,
          const FT *d_B = nullptr, bool InfYZero = true>
struct ProjectiveCurve
    : public Point<FT, 3>,
      public ProjectiveMixin<ProjectiveCurve<FT, A, B, d_A, d_B, InfYZero>, FT,
                             A, B, d_A, d_B, InfYZero> {
    using Point<FT, 3>::Point;
};

template <typename FT, const FT *A, const FT *B, const FT *d_A = nullptr,
          const FT *d_B = nullptr, bool InfYZero = true>
struct JacobianCurve
    : public Point<FT, 3>,
      public JacobianMixin<JacobianCurve<FT, A, B, d_A, d_B, InfYZero>, FT, A,
                           B, d_A, d_B, InfYZero> {
    using Point<FT, 3>::Point;
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_PRESET_HPP

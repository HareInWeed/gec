#pragma once
#ifndef GEC_CURVE_MIXIN_HPP
#define GEC_CURVE_MIXIN_HPP

#include "mixin/arr_get_comp.hpp"
#include "mixin/compwise_eq.hpp"
#include "mixin/get_comp.hpp"
#include "mixin/hasher.hpp"
#include "mixin/named_comp.hpp"

#include "mixin/params.hpp"

#include "mixin/affine.hpp"
// #include "mixin/chudnovsky.hpp" // TODO
#include "mixin/jacobian.hpp"
// #include "mixin/modified_jacobian.hpp" // TODO
#include "mixin/projective.hpp"

#include "mixin/lift_x.hpp"
#include "mixin/scalar_mul.hpp"

#include "mixin/ostream.hpp"
#include "mixin/print.hpp"

namespace gec {

namespace curve {

template <typename Core, typename FT, const FT *A, const FT *B,
          const FT *d_A = nullptr, const FT *d_B = nullptr>
class BasicCurveMixin : public CurveParams<Core, FT, A, B, d_A, d_B>,
                        public ScalarMul<Core>,
                        public LiftX<Core, FT>,
                        public WithPointHasher<Core>,
                        public PointOstream<Core>,
                        public PointPrint<Core> {
  public:
};

template <class Core, typename FT, const FT *A, const FT *B,
          const FT *d_A = nullptr, const FT *d_B = nullptr,
          bool InfYZero = true>
class AffineMixin : public BasicCurveMixin<Core, FT, A, B, d_A, d_B>,
                    public AffineCoordinate<Core, FT, InfYZero>,
                    public CompWiseEq<Core> {};

template <class Core, typename FT, const FT *A, const FT *B,
          const FT *d_A = nullptr, const FT *d_B = nullptr,
          bool InfYZero = true>
class ProjectiveMixin : public BasicCurveMixin<Core, FT, A, B, d_A, d_B>,
                        public ProjectiveCoordinate<Core, FT, InfYZero> {};

template <class Core, typename FT, const FT *A, const FT *B,
          const FT *d_A = nullptr, const FT *d_B = nullptr,
          bool InfYZero = true>
class JacobianMixin : public BasicCurveMixin<Core, FT, A, B, d_A, d_B>,
                      public JacobianCoordinate<Core, FT, InfYZero> {};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_HPP

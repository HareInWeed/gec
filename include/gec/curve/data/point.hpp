#pragma once
#ifndef GEC_CURVE_DATA_POINT_HPP
#define GEC_CURVE_DATA_POINT_HPP

#include "../mixin/arr_get_comp.hpp"
#include "../mixin/named_comp.hpp"
#include <gec/bigint/data/array.hpp>

namespace gec {

namespace curve {

/** @brief N-dimensional point
 */
template <typename COMP_T, size_t N>
class Point : bigint::ArrayLE<COMP_T, N>,
              public ArrGetCompLE<Point<COMP_T, N>>,
              public NamedComp<Point<COMP_T, N>> {
    using Base = bigint::ArrayLE<COMP_T, N>;
    using Self = Point<COMP_T, N>;

    friend ArrGetCompLE<Self>;

  public:
    using CompT = COMP_T;
    const static size_t CompN = N;
    using Base::Base;
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_DATA_POINT_HPP

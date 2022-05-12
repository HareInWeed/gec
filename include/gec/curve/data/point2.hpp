#pragma once
#ifndef GEC_CURVE_DATA_POINT2_HPP
#define GEC_CURVE_DATA_POINT2_HPP

#include <gec/utils/basic.hpp>

namespace gec {

namespace curve {

/** @brief 2-dimensional point
 */
template <typename FIELD_T>
class Point2 {
  public:
    FIELD_T comp_x;
    FIELD_T comp_y;

    Point2() : comp_x(), comp_y() {}
    Point2(const FIELD_T &x, const FIELD_T &y) : comp_x(x), comp_y(y) {}
    Point2(const Point2 &other) : comp_x(other.comp_x), comp_y(other.comp_y) {}
    Point2 &operator=(const Point2 &other) {
        comp_x = other.comp_x;
        comp_y = other.comp_y;
        return *this;
    }

    __host__ __device__ GEC_INLINE const FIELD_T &x() const { return comp_x; }
    __host__ __device__ GEC_INLINE FIELD_T &x() { return comp_x; }
    __host__ __device__ GEC_INLINE const FIELD_T &y() const { return comp_y; }
    __host__ __device__ GEC_INLINE FIELD_T &y() { return comp_y; }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_DATA_POINT2_HPP

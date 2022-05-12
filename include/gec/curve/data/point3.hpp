#pragma once
#ifndef GEC_CURVE_DATA_POINT3_HPP
#define GEC_CURVE_DATA_POINT3_HPP

#include <gec/utils/basic.hpp>

namespace gec {

namespace curve {

/** @brief 3-dimensional point
 */
template <typename FIELD_T>
class Point3 {
  public:
    FIELD_T comp_x;
    FIELD_T comp_y;
    FIELD_T comp_z;

    Point3() : comp_x(), comp_y(), comp_z() {}
    Point3(const FIELD_T &x, const FIELD_T &y, const FIELD_T &z)
        : comp_x(x), comp_y(y), comp_z(z) {}
    Point3(const Point3 &other)
        : comp_x(other.comp_x), comp_y(other.comp_y), comp_z(other.comp_z) {}
    Point3 &operator=(const Point3 &other) {
        comp_x = other.comp_x;
        comp_y = other.comp_y;
        comp_z = other.comp_z;
        return *this;
    }

    __host__ __device__ GEC_INLINE const FIELD_T &x() const { return comp_x; }
    __host__ __device__ GEC_INLINE FIELD_T &x() { return comp_x; }
    __host__ __device__ GEC_INLINE const FIELD_T &y() const { return comp_y; }
    __host__ __device__ GEC_INLINE FIELD_T &y() { return comp_y; }
    __host__ __device__ GEC_INLINE const FIELD_T &z() const { return comp_z; }
    __host__ __device__ GEC_INLINE FIELD_T &z() { return comp_z; }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_DATA_POINT3_HPP

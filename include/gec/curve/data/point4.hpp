#pragma once
#ifndef GEC_CURVE_DATA_POINT4_HPP
#define GEC_CURVE_DATA_POINT4_HPP

#include <gec/utils/basic.hpp>

namespace gec {

namespace curve {

/** @brief 4-dimensional point
 */
template <typename FIELD_T>
class Point4 {
  public:
    FIELD_T comp_x;
    FIELD_T comp_y;
    FIELD_T comp_z;
    FIELD_T comp_z2;

    Point4() : comp_x(), comp_y(), comp_z(), comp_z2() {}
    Point4(const FIELD_T &x, const FIELD_T &y, const FIELD_T &z,
           const FIELD_T &z2)
        : comp_x(x), comp_y(y), comp_z(z), comp_z2(z2) {}
    Point4(const Point4 &other)
        : comp_x(other.comp_x), comp_y(other.comp_y), comp_z(other.comp_z),
          comp_z2(other.comp_z2) {}
    Point4 &operator=(const Point4 &other) {
        comp_x = other.comp_x;
        comp_y = other.comp_y;
        comp_z = other.comp_z;
        comp_z2 = other.comp_z2;
        return *this;
    }

    __host__ __device__ GEC_INLINE const FIELD_T &x() const { return comp_x; }
    __host__ __device__ GEC_INLINE FIELD_T &x() { return comp_x; }
    __host__ __device__ GEC_INLINE const FIELD_T &y() const { return comp_y; }
    __host__ __device__ GEC_INLINE FIELD_T &y() { return comp_y; }
    __host__ __device__ GEC_INLINE const FIELD_T &z() const { return comp_z; }
    __host__ __device__ GEC_INLINE FIELD_T &z() { return comp_z; }
    __host__ __device__ GEC_INLINE const FIELD_T &z2() const { return comp_z2; }
    __host__ __device__ GEC_INLINE FIELD_T &z2() { return comp_z2; }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_DATA_POINT4_HPP

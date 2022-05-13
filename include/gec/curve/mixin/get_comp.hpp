#pragma once
#ifndef GEC_CURVE_MIXIN_GET_COMP_HPP
#define GEC_CURVE_MIXIN_GET_COMP_HPP

#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

template <typename Point, size_t I>
struct GetCompHelper {};
template <typename Point>
struct GetCompHelper<Point, 0> {
    __host__ __device__ GEC_INLINE static typename Point::CompT &
    call(Point &p) {
        return p.x();
    }
    __host__ __device__ GEC_INLINE static const typename Point::CompT &
    call(const Point &p) {
        return p.x();
    }
};
template <typename Point>
struct GetCompHelper<Point, 1> {
    __host__ __device__ GEC_INLINE static typename Point::CompT &
    call(Point &p) {
        return p.y();
    }
    __host__ __device__ GEC_INLINE static const typename Point::CompT &
    call(const Point &p) {
        return p.y();
    }
};
template <typename Point>
struct GetCompHelper<Point, 2> {
    __host__ __device__ GEC_INLINE static typename Point::CompT &
    call(Point &p) {
        return p.z();
    }
    __host__ __device__ GEC_INLINE static const typename Point::CompT &
    call(const Point &p) {
        return p.z();
    }
};
template <typename Point>
struct GetCompHelper<Point, 3> {
    __host__ __device__ GEC_INLINE static typename Point::CompT &
    call(Point &p) {
        return p.z1();
    }
    __host__ __device__ GEC_INLINE static const typename Point::CompT &
    call(const Point &p) {
        return p.z1();
    }
};
template <typename Point>
struct GetCompHelper<Point, 4> {
    __host__ __device__ GEC_INLINE static typename Point::CompT &
    call(Point &p) {
        return p.z2();
    }
    __host__ __device__ GEC_INLINE static const typename Point::CompT &
    call(const Point &p) {
        return p.z2();
    }
};

/** @brief mixin that enables ...
 */
template <typename Core>
class GetComp : protected CRTP<Core, GetComp<Core>> {
    friend CRTP<Core, GetComp<Core>>;

  public:
    template <size_t I, typename P = Core,
              std::enable_if_t<(I < P::CompN && I < 5)> * = nullptr>
    __host__ __device__ GEC_INLINE const typename P::CompT &get() const {
        return GetCompHelper<Core, I>::call(this->core());
    }
    template <size_t I, typename P = Core,
              std::enable_if_t<(I < P::CompN && I < 5)> * = nullptr>
    __host__ __device__ GEC_INLINE typename P::CompT &get() {
        return GetCompHelper<Core, I>::call(this->core());
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_GET_COMP_HPP

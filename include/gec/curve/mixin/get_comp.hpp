#pragma once
#ifndef GEC_CURVE_MIXIN_GET_COMP_HPP
#define GEC_CURVE_MIXIN_GET_COMP_HPP

#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

template <typename Point, size_t I>
struct GEC_EMPTY_BASES GetCompHelper {};
template <typename Point>
struct GEC_EMPTY_BASES GetCompHelper<Point, 0> {
    GEC_HD GEC_INLINE static typename Point::CompT &call(Point &p) {
        return p.x();
    }
    GEC_HD GEC_INLINE static const typename Point::CompT &call(const Point &p) {
        return p.x();
    }
};
template <typename Point>
struct GEC_EMPTY_BASES GetCompHelper<Point, 1> {
    GEC_HD GEC_INLINE static typename Point::CompT &call(Point &p) {
        return p.y();
    }
    GEC_HD GEC_INLINE static const typename Point::CompT &call(const Point &p) {
        return p.y();
    }
};
template <typename Point>
struct GEC_EMPTY_BASES GetCompHelper<Point, 2> {
    GEC_HD GEC_INLINE static typename Point::CompT &call(Point &p) {
        return p.z();
    }
    GEC_HD GEC_INLINE static const typename Point::CompT &call(const Point &p) {
        return p.z();
    }
};
template <typename Point>
struct GEC_EMPTY_BASES GetCompHelper<Point, 3> {
    GEC_HD GEC_INLINE static typename Point::CompT &call(Point &p) {
        return p.z1();
    }
    GEC_HD GEC_INLINE static const typename Point::CompT &call(const Point &p) {
        return p.z1();
    }
};
template <typename Point>
struct GEC_EMPTY_BASES GetCompHelper<Point, 4> {
    GEC_HD GEC_INLINE static typename Point::CompT &call(Point &p) {
        return p.z2();
    }
    GEC_HD GEC_INLINE static const typename Point::CompT &call(const Point &p) {
        return p.z2();
    }
};

/** @brief mixin that enables ...
 */
template <typename Core>
class GEC_EMPTY_BASES GetComp : protected CRTP<Core, GetComp<Core>> {
    friend CRTP<Core, GetComp<Core>>;

  public:
    template <size_t I, typename P = Core,
              std::enable_if_t<(I < P::CompN && I < 5)> * = nullptr>
    GEC_HD GEC_INLINE const typename P::CompT &get() const {
        return GetCompHelper<Core, I>::call(this->core());
    }
    template <size_t I, typename P = Core,
              std::enable_if_t<(I < P::CompN && I < 5)> * = nullptr>
    GEC_HD GEC_INLINE typename P::CompT &get() {
        return GetCompHelper<Core, I>::call(this->core());
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_GET_COMP_HPP

#pragma once
#ifndef GEC_CURVE_MIXIN_OSTREAM_HPP
#define GEC_CURVE_MIXIN_OSTREAM_HPP

#include <gec/utils/crtp.hpp>
#include <iomanip>
#include <ostream>

namespace gec {

namespace curve {

template <typename Point, size_t I, size_t N>
struct GEC_EMPTY_BASES PointOstreamHelper {
    GEC_H GEC_INLINE static void call(std::ostream &o, const Point &point) {
        o << ',' << std::endl << ' ' << point.template get<I>();
        PointOstreamHelper<Point, I + 1, N>::call(o, point);
    }
};
template <typename Point, size_t N>
struct GEC_EMPTY_BASES PointOstreamHelper<Point, 0, N> {
    GEC_H GEC_INLINE static void call(std::ostream &o, const Point &point) {
        o << point.template get<0>();
        PointOstreamHelper<Point, 1, N>::call(o, point);
    }
};
template <typename Point, size_t N>
struct GEC_EMPTY_BASES PointOstreamHelper<Point, N, N> {
    GEC_H GEC_INLINE static void call(std::ostream &, const Point &) {}
};
template <typename Point>
struct GEC_EMPTY_BASES PointOstreamHelper<Point, 0, 0> {
    GEC_H GEC_INLINE static void call(std::ostream &, const Point &) {}
};

/** @brief mixin that enables output x() and y() with ostream
 */
template <typename Core>
class GEC_EMPTY_BASES PointOstream : protected CRTP<Core, PointOstream<Core>> {
    friend CRTP<Core, PointOstream<Core>>;

  public:
    friend std::ostream &operator<<(std::ostream &o, const Core &point) {
        o << '{';
        PointOstreamHelper<Core, 0, Core::CompN>::call(o, point);
        o << '}' << std::endl;
        return o;
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_OSTREAM_HPP

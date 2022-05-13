#pragma once
#ifndef GEC_CURVE_MIXIN_PRINT_HPP
#define GEC_CURVE_MIXIN_PRINT_HPP

#include <cstdio>
#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

template <typename Point, size_t I, size_t N>
struct PointPrintHelper {
    static void call(const Point &point) {
        printf(",\n ");
        point.template get<I>().print();
        PointPrintHelper<Point, I + 1, N>::call(point);
    }
};
template <typename Point, size_t N>
struct PointPrintHelper<Point, 0, N> {
    static void call(const Point &point) {
        point.template get<0>().print();
        PointPrintHelper<Point, 1, N>::call(point);
    }
};
template <typename Point, size_t N>
struct PointPrintHelper<Point, N, N> {
    static void call(const Point &point) {}
};
template <typename Point>
struct PointPrintHelper<Point, 0, 0> {
    static void call(const Point &point) {}
};

/** @brief mixin that enables output x() and y() with stdio
 */
template <typename Core>
class PointPrint : protected CRTP<Core, PointPrint<Core>> {
    friend CRTP<Core, PointPrint<Core>>;

  public:
    __host__ __device__ void print() {
        using namespace std;
        printf("{");
        PointPrintHelper<Core, 0, Core::CompN>::call(this->core());
        printf("}\n");
    }
    __host__ __device__ void println() const {
        print();
        printf("\n");
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_PRINT_HPP

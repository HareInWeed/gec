#pragma once
#ifndef GEC_CURVE_MIXIN_POINT2_PRINT_HPP
#define GEC_CURVE_MIXIN_POINT2_PRINT_HPP

#include <cstdio>
#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

/** @brief mixin that enables output x() and y() with stdio
 */
template <typename Core>
class Point2Print : protected CRTP<Core, Point2Print<Core>> {
    friend CRTP<Core, Point2Print<Core>>;

  public:
    __host__ __device__ void print() {
        using namespace std;
        printf("{");
        this->core().x().print();
        printf(",\n ");
        this->core().y().print();
        printf("}\n");
    }
    __host__ __device__ void println() const {
        print();
        printf("\n");
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_POINT2_PRINT_HPP

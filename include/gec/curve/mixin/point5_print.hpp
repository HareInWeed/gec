#pragma once
#ifndef GEC_CURVE_MIXIN_POINT5_PRINT_HPP
#define GEC_CURVE_MIXIN_POINT5_PRINT_HPP

#include <cstdio>
#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

/** @brief mixin that enables output x(), y(), z(), z1() and z2() with stdio
 */
template <typename Core>
class Point5Print : protected CRTP<Core, Point5Print<Core>> {
    friend CRTP<Core, Point5Print<Core>>;

  public:
    __host__ __device__ void print() {
        using namespace std;
        printf("{");
        this->core().x().print();
        printf(",\n ");
        this->core().y().print();
        printf(",\n ");
        this->core().z().print();
        printf(",\n ");
        this->core().z1().print();
        printf(",\n ");
        this->core().z2().print();
        printf("}\n");
    }
    __host__ __device__ void println() const {
        print();
        printf("\n");
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_POINT5_PRINT_HPP

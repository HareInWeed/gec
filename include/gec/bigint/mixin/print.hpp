#pragma once
#ifndef GEC_BIGINT_MIXIN_PRINT_HPP
#define GEC_BIGINT_MIXIN_PRINT_HPP

#include <gec/utils/crtp.hpp>

#include <cstdio>

namespace gec {

namespace print {
template <typename T>
__host__ __device__ void print(const T &data) {
    char *it = reinterpret_cast<char *>(&data);
    const char *end = it + sizeof(T);
    printf("Unknown {");
    if (it < end) {
        printf("%02x", *it++);
        for (; it < end; ++it) {
            printf(" %02x", *it);
        }
    }
    printf("}");
}

template <>
inline __host__ __device__ void print<uint32_t>(const uint32_t &data) {
    printf("%08x", data);
}

template <>
inline __host__ __device__ void print<uint64_t>(const uint64_t &data) {
    printf("%016llx", data);
}
} // namespace print

namespace bigint {

/** @brief mixin that enables output array with stdio
 */
template <class Core, class LIMB_T, size_t LIMB_N>
class ArrayPrintMixin
    : public CRTP<Core, ArrayPrintMixin<Core, LIMB_T, LIMB_N>> {
  public:
    __host__ __device__ void print() const {
        printf("0x");
        print::print(this->core().get_arr()[LIMB_N - 1]);
        for (size_t i = 1; i < LIMB_N; ++i) {
            printf(" ");
            print::print(this->core().get_arr()[LIMB_N - 1 - i]);
        }
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_PRINT_HPP
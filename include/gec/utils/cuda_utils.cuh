#pragma once
#ifndef GEC_UTILS_CUDA_UTILS_H
#define GEC_UTILS_CUDA_UTILS_H

#include "basic.hpp"

#ifdef __CUDACC__

namespace gec {

namespace utils {

/**
 * @brief set the `CC.CF` flag register
 *
 * @param flag: flag to set
 */
__device__ GEC_INLINE void set_cc_cf_(bool flag) {
    uint32_t x = uint32_t(flag);
    asm volatile("add.cc.u32 %0, %1, %0;" : "+r"(x) : "n"(0xffffffff));
}

/**
 * @brief get the `CC.CF` flag register
 *
 * note that calling the function `get_cc_cf_` again immediately after calling
 * `get_cc_cf_` will lead to undefined behavior
 *
 * @return value of `CC.CF` flag register
 */
__device__ GEC_INLINE bool get_cc_cf_() {
    uint32_t x;
    asm volatile("addc.u32 %0, %1, %1;" : "=r"(x) : "n"(0x0));
    return x;
}

/**
 * @brief `add.cc` ptx instruction
 *
 * carry will be written to flag `CC.CF`
 *
 * @tparam T: one of `uint32_t`, `uint64_t`, `int32_t` or `int64_t`
 * @param a: b + c
 * @param b: first addend
 * @param c: second addend
 */
template <typename T>
__device__ GEC_INLINE void add_cc_(T &a, const T &b, const T &c);
template <>
__device__ GEC_INLINE void add_cc_<uint32_t>(uint32_t &a, const uint32_t &b,
                                             const uint32_t &c) {
    asm("add.cc.u32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));
}
template <>
__device__ GEC_INLINE void add_cc_<uint64_t>(uint64_t &a, const uint64_t &b,
                                             const uint64_t &c) {
    asm("add.cc.u64 %0, %1, %2;" : "=l"(a) : "l"(b), "l"(c));
}
template <>
__device__ GEC_INLINE void add_cc_<int32_t>(int32_t &a, const int32_t &b,
                                            const int32_t &c) {
    asm("add.cc.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));
}
template <>
__device__ GEC_INLINE void add_cc_<int64_t>(int64_t &a, const int64_t &b,
                                            const int64_t &c) {
    asm("add.cc.s64 %0, %1, %2;" : "=l"(a) : "l"(b), "l"(c));
}

/**
 * @brief `addc` ptx instruction
 *
 * @tparam T: one of `uint32_t`, `uint64_t`, `int32_t` or `int64_t`
 * @param a: b + c + CC.CF
 * @param b: first addend
 * @param c: second addend
 */
template <typename T>
__device__ GEC_INLINE void addc_(T &a, const T &b, const T &c);
template <>
__device__ GEC_INLINE void addc_<uint32_t>(uint32_t &a, const uint32_t &b,
                                           const uint32_t &c) {
    asm("addc.u32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));
}
template <>
__device__ GEC_INLINE void addc_<uint64_t>(uint64_t &a, const uint64_t &b,
                                           const uint64_t &c) {
    asm("addc.u64 %0, %1, %2;" : "=l"(a) : "l"(b), "l"(c));
}
template <>
__device__ GEC_INLINE void addc_<int32_t>(int32_t &a, const int32_t &b,
                                          const int32_t &c) {
    asm("addc.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));
}
template <>
__device__ GEC_INLINE void addc_<int64_t>(int64_t &a, const int64_t &b,
                                          const int64_t &c) {
    asm("addc.s64 %0, %1, %2;" : "=l"(a) : "l"(b), "l"(c));
}

/**
 * @brief `addc.cc` ptx instruction
 *
 * carry will be written to flag `CC.CF`
 *
 * @tparam T: one of `uint32_t`, `uint64_t`, `int32_t` or `int64_t`
 * @param a: b + c + CC.CF
 * @param b: first addend
 * @param c: second addend
 */
template <typename T>
__device__ GEC_INLINE void addc_cc_(T &a, const T &b, const T &c);
template <>
__device__ GEC_INLINE void addc_cc_<uint32_t>(uint32_t &a, const uint32_t &b,
                                              const uint32_t &c) {
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));
}
template <>
__device__ GEC_INLINE void addc_cc_<uint64_t>(uint64_t &a, const uint64_t &b,
                                              const uint64_t &c) {
    asm("addc.cc.u64 %0, %1, %2;" : "=l"(a) : "l"(b), "l"(c));
}
template <>
__device__ GEC_INLINE void addc_cc_<int32_t>(int32_t &a, const int32_t &b,
                                             const int32_t &c) {
    asm("addc.cc.s32 %0, %1, %2;" : "=r"(a) : "r"(b), "r"(c));
}
template <>
__device__ GEC_INLINE void addc_cc_<int64_t>(int64_t &a, const int64_t &b,
                                             const int64_t &c) {
    asm("addc.cc.s64 %0, %1, %2;" : "=l"(a) : "l"(b), "l"(c));
}

} // namespace utils

} // namespace gec

#endif // __CUDACC__

#endif // !GEC_UTILS_CUDA_UTILS_H
#pragma once
#ifndef GEC_UTILS_MISC_HPP
#define GEC_UTILS_MISC_HPP

#include "basic.hpp"

#include <type_traits>

namespace gec {

namespace utils {

template <typename T, size_t I = std::numeric_limits<T>::digits,
          typename Enable = void>
struct LowerBitsMask {
    __host__ __device__ GEC_INLINE static T call(T x) {
        x = x | (x >> (std::numeric_limits<T>::digits / I));
        return LowerBitsMask<T, I / 2>::call(x);
    }
};

template <typename T, size_t I>
struct LowerBitsMask<T, I, typename std::enable_if_t<!std::is_unsigned_v<T>>> {
    __host__ __device__ GEC_INLINE static T call(T x) {
        return LowerBitsMask<std::make_unsigned_t<T>, I>::call(x);
    }
};

template <typename T>
struct LowerBitsMask<T, 1> {
    __host__ __device__ GEC_INLINE static T call(T x) { return x; }
};

template <typename T>
__host__ __device__ GEC_INLINE T lower_bit_mask(T x) {
    return LowerBitsMask<T>::call(x);
}

} // namespace utils

} // namespace gec

#endif // !GEC_UTILS_MISC_HPP

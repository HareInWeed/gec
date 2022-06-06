#pragma once
#ifndef GEC_UTILS_MISC_HPP
#define GEC_UTILS_MISC_HPP

#include "basic.hpp"

#include <type_traits>
#include <utility>

namespace gec {

namespace utils {

template <typename T>
__host__ __device__ GEC_INLINE void swap(T &a, T &b) {
    T tmp = std::move(a);
    a = std::move(b);
    b = std::move(tmp);
}

template <typename T, size_t I = std::numeric_limits<T>::digits,
          typename Enable = void>
struct SignificantMask {
    __host__ __device__ GEC_INLINE static T call(T x) {
        x = x | (x >> (std::numeric_limits<T>::digits / I));
        return SignificantMask<T, I / 2>::call(x);
    }
};
template <typename T, size_t I>
struct SignificantMask<T, I,
                       typename std::enable_if_t<!std::is_unsigned<T>::value>> {
    __host__ __device__ GEC_INLINE static T call(T x) {
        return SignificantMask<std::make_unsigned_t<T>, I>::call(x);
    }
};
template <typename T>
struct SignificantMask<T, 1> {
    __host__ __device__ GEC_INLINE static T call(T x) { return x; }
};
template <typename T>
__host__ __device__ GEC_INLINE T significant_mask(T x) {
    return SignificantMask<T>::call(x);
}

template <typename T, size_t K = 0, typename Enable = void>
struct LowerKMask {
    const static T value =
        K == std::numeric_limits<T>::digits ? ~T(0) : (T(1) << K) - 1;
};
template <typename T, size_t K>
struct LowerKMask<T, K, std::enable_if_t<!std::is_unsigned<T>::value>> {
    const static T value = T(LowerKMask<T, K, std::make_unsigned<T>>::value);
};

template <typename T, size_t K = 0>
struct HigherKMask {
    const static T value =
        T(~LowerKMask<T, std::numeric_limits<std::make_unsigned<T>>::digits - K,
                      std::make_unsigned<T>>::value);
};

template <typename T, size_t I = std::numeric_limits<T>::digits / 2,
          typename Enable = void>
struct TrailingZeros {
    __host__ __device__ GEC_INLINE static void call(T x, size_t &b) {
        if (!(x & LowerKMask<T, I>::value)) {
            b += I;
            x >>= I;
        }
        TrailingZeros<T, I / 2>::call(x, b);
    }
};
template <typename T>
struct TrailingZeros<T, 0> {
    __host__ __device__ GEC_INLINE static void call(T, size_t &) {}
};
template <typename T, size_t I>
struct TrailingZeros<T, I,
                     typename std::enable_if_t<!std::is_unsigned<T>::value>> {
    __host__ __device__ GEC_INLINE static void call(T x, size_t &b) {
        using UT = std::make_unsigned_t<T>;
        TrailingZeros<UT, std::numeric_limits<UT>::digits / 2>::call(x, b);
    }
};
template <typename T>
__host__ __device__ GEC_INLINE size_t trailing_zeros(T x) {
    size_t b = 0;
    TrailingZeros<T>::call(x, b);
    return b;
}

__host__ __device__ GEC_INLINE void hash_combine(size_t &seed, size_t h) {
    seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename LIMB_T, size_t LIMB_N, typename Hasher>
struct SeqHasher {
    __host__ __device__ GEC_INLINE static void
    call(size_t &seed, const LIMB_T *arr, const Hasher &hasher) {
        utils::hash_combine(seed, hasher(*arr));
        SeqHasher<LIMB_T, LIMB_N - 1, Hasher>::call(seed, arr + 1, hasher);
    }
};
template <typename LIMB_T, typename Hasher>
struct SeqHasher<LIMB_T, 0, Hasher> {
    __host__ __device__ GEC_INLINE static void call(size_t &, const LIMB_T *,
                                                    const Hasher &) {}
};

} // namespace utils

} // namespace gec

#endif // !GEC_UTILS_MISC_HPP

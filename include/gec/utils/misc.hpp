#pragma once
#ifndef GEC_UTILS_MISC_HPP
#define GEC_UTILS_MISC_HPP

#include "basic.hpp"

#include <type_traits>
#include <utility>

#if defined(_WIN32) &&                                                         \
    (defined(GEC_CLANG) || defined(GEC_MSVC) || defined(GEC_GCC))
#include <intrin.h>
#endif

namespace gec {

namespace utils {

template <typename T>
__host__ __device__ GEC_INLINE constexpr void swap(T &a, T &b) {
    T tmp = std::move(a);
    a = std::move(b);
    b = std::move(tmp);
}

template <typename T, T Part, size_t I, size_t N = utils::type_bits<T>::value,
          typename Enable = void>
struct RepeatingMask {
    static constexpr T value = Part;
};
template <typename T, T Part, size_t I, size_t N>
struct RepeatingMask<T, Part, I, N, typename std::enable_if_t<(I < N)>> {
    static constexpr T value =
        RepeatingMask<T, (Part | (Part << I)), 2 * I, N>::value;
};

template <typename T, size_t I = utils::type_bits<T>::value,
          typename Enable = void>
struct SignificantMask {
    __host__ __device__ GEC_INLINE static T call(T x) {
        x = x | (x >> (utils::type_bits<T>::value / I));
        return SignificantMask<T, I / 2>::call(x);
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

template <typename T, size_t K, typename Enable = void>
struct LowerKMask {
    const static T value =
        K == utils::type_bits<T>::value ? ~T(0) : (T(1) << K) - 1;
};

template <typename T, size_t K>
struct HigherKMask {
    const static T value =
        ~LowerKMask<T, utils::type_bits<T>::value - K>::value;
};

namespace _clz_ {

template <typename T, size_t L, size_t R>
struct BinSearchCLZHelper {
    __host__ __device__ GEC_INLINE static constexpr void call(size_t &n, T &x) {
        if ((x >> R) == 0) {
            n = n + L;
            x = x << L;
        }
        BinSearchCLZHelper<T, L / 2, R + L / 2>::call(n, x);
    }
};
template <typename T, size_t R>
struct BinSearchCLZHelper<T, 1, R> {
    __host__ __device__ GEC_INLINE static constexpr void call(size_t &, T &) {}
};
template <typename T>
struct BinSearchCLZ {
    __host__ __device__ static constexpr size_t call(T x) {
        constexpr size_t bits = utils::type_bits<T>::value;
        constexpr size_t bits_m_1 = bits - 1;
        constexpr size_t half_bits = bits / 2;
        size_t n = 1;
        BinSearchCLZHelper<T, half_bits, half_bits>::call(n, x);
        return n - (x >> bits_m_1);
    }
};

template <typename T, typename Enable = void>
struct HostCLZ {
    __host__ GEC_INLINE static size_t call(T x) {
        return BinSearchCLZ<T>::call(x);
    }
};

template <typename T, typename Enable = void>
struct DeviceCLZ {
    __device__ GEC_INLINE static size_t call(T x) {
        return BinSearchCLZ<T>::call(x);
    }
};

template <typename T, typename Enable = void>
struct CLZ {
    __host__ __device__ GEC_INLINE static size_t call(T x) {
#ifdef __CUDA_ARCH__
        return DeviceCLZ<T>::call(x);
#else
        return HostCLZ<T>::call(x);
#endif // __CUDA_ARCH__
    }
};

#if defined(GEC_GCC) || defined(GEC_CLANG)

template <>
struct HostCLZ<unsigned int> {
    __host__ GEC_INLINE static size_t call(unsigned int x) {
        return size_t(__builtin_clz(x));
    }
};
template <>
struct HostCLZ<unsigned long> {
    __host__ GEC_INLINE static size_t call(unsigned long x) {
        return size_t(__builtin_clzl(x));
    }
};
template <>
struct HostCLZ<unsigned long long> {
    __host__ GEC_INLINE static size_t call(unsigned long long x) {
        return size_t(__builtin_clzll(x));
    }
};

#endif

template <typename T, typename Enable = void>
struct CLZMostSignificantBit {
    __host__ __device__ GEC_INLINE static size_t call(T x) {
        constexpr size_t max_bit = utils::type_bits<T>::value - 1;
        return max_bit - CLZ<T>::call(x);
    }
};

template <typename T, typename Enable = void>
struct HostMostSignificantBit {
    __host__ GEC_INLINE static size_t call(T x) {
        return CLZMostSignificantBit<T>::call(x);
    }
};

template <typename T, typename Enable = void>
struct DeviceMostSignificantBit {
    __device__ GEC_INLINE static size_t call(T x) {
        return CLZMostSignificantBit<T>::call(x);
    }
};

template <typename T, typename Enable = void>
struct MostSignificantBit {
    __host__ __device__ GEC_INLINE static size_t call(T x) {
#ifdef __CUDA_ARCH__
        return DeviceMostSignificantBit<T>::call(x);
#else
        return HostMostSignificantBit<T>::call(x);
#endif // __CUDA_ARCH__
    }
};

#if defined(_MSC_VER)

#if defined(GEC_X86) || defined(GEC_AMD64)

#pragma intrinsic(_BitScanReverse)
template <typename T>
struct HostMostSignificantBit<
    T, std::enable_if_t<(utils::type_bits<T>::value <= 32)>> {
    __host__ GEC_INLINE static size_t call(T x) {
        unsigned long ret;
        _BitScanReverse(&ret, (unsigned long)x);
        return (size_t)ret;
    }
};

template <typename T>
struct HostCLZ<T, std::enable_if_t<(utils::type_bits<T>::value <= 32)>> {
    __host__ GEC_INLINE static size_t call(T x) {
        constexpr size_t max_bit = utils::type_bits<T>::value - 1;
        return max_bit - HostMostSignificantBit<T>::call(x);
    }
};

#endif

#if defined(GEC_AMD64)

#pragma intrinsic(_BitScanReverse64)
template <typename T>
struct HostMostSignificantBit<
    T, std::enable_if_t<(utils::type_bits<T>::value > 32 &&
                         utils::type_bits<T>::value <= 64)>> {
    __host__ GEC_INLINE static size_t call(T x) {
        unsigned long ret;
        _BitScanReverse64(&ret, (__int64)x);
        return (size_t)ret;
    }
};

template <typename T>
struct HostCLZ<T, std::enable_if_t<(utils::type_bits<T>::value <= 32 &&
                                    utils::type_bits<T>::value <= 64)>> {
    __host__ GEC_INLINE static size_t call(T x) {
        constexpr size_t max_bit = utils::type_bits<T>::value - 1;
        return max_bit - MostSignificantBit<T>::call(x);
    }
};

#endif

#endif

} // namespace _clz_

/**
 * @brief get the number of leading zeros
 *
 * note that `x == 0` may lead to undefined behaviour
 *
 * @tparam T any unsigned integral type
 * @param x the unsigned integer
 * @return size_t the number of leading zeros
 */
template <typename T>
__host__ __device__ GEC_INLINE size_t count_leading_zeros(T x) {
    return _clz_::CLZ<T>::call(x);
}

/**
 * @brief get the most significant bit
 *
 * note that `x == 0` may lead to undefined behaviour
 *
 * @tparam T any unsigned integral type
 * @param x the unsigned integer
 * @return size_t the position of the most significant bit
 */
template <typename T>
__host__ __device__ GEC_INLINE size_t most_significant_bit(T x) {
    return _clz_::MostSignificantBit<T>::call(x);
}

namespace _ctz_ {

template <typename T, size_t I, typename Enable = void>
struct BinSearchCTZHelper {
    __host__ __device__ GEC_INLINE static void call(size_t &n, T &x) {
        if (!(x & LowerKMask<T, I>::value)) {
            n += I;
            x >>= I;
        }
        BinSearchCTZHelper<T, I / 2>::call(n, x);
    }
};
template <typename T>
struct BinSearchCTZHelper<T, 1> {
    __host__ __device__ GEC_INLINE static void call(T, size_t &) {}
};
template <typename T>
struct BinSearchCTZ {
    __host__ __device__ static constexpr size_t call(T x) {
        constexpr size_t bits = utils::type_bits<T>::value;
        size_t n = 1;
        BinSearchCTZHelper<T, bits / 2>::call(n, x);
        return n - (x & 1);
    }
};

template <typename T, size_t I>
struct GaudetCTZHelper {
    __host__ __device__ GEC_INLINE static constexpr size_t call(T y) {
        constexpr T mask =
            RepeatingMask<T, LowerKMask<T, I>::value, 2 * I>::value;
        return ((y & mask) ? 0 : I) + GaudetCTZHelper<T, I / 2>::call(y);
    }
};
template <typename T>
struct GaudetCTZHelper<T, 0> {
    __host__ __device__ GEC_INLINE static constexpr size_t call(T) { return 0; }
};
template <typename T>
struct GaudetCTZ {
    __host__ __device__ static constexpr size_t call(T x) {
        constexpr size_t bits = utils::type_bits<T>::value;
        T y = x & -x;
        return (y ? 0 : 1) + GaudetCTZHelper<T, bits / 2>::call(y);
    }
};

template <typename T, typename Enable = void>
struct HostCTZ {
    __host__ GEC_INLINE static size_t call(T x) {
        return GaudetCTZ<T>::call(x);
    }
};
template <typename T, typename Enable = void>
struct DeviceCTZ {
    __device__ GEC_INLINE static size_t call(T x) {
        return GaudetCTZ<T>::call(x);
    }
};
template <typename T, typename Enable = void>
struct CTZ {
    __host__ __device__ GEC_INLINE static size_t call(T x) {
#ifdef __CUDA_ARCH__
        return DeviceCTZ<T>::call(x);
#else
        return HostCTZ<T>::call(x);
#endif // __CUDA_ARCH__
    }
};

#if defined(GEC_GCC) || defined(GEC_CLANG)

template <>
struct HostCTZ<unsigned int> {
    __host__ GEC_INLINE static size_t call(unsigned int x) {
        return size_t(__builtin_ctz(x));
    }
};
template <>
struct HostCTZ<unsigned long> {
    __host__ GEC_INLINE static size_t call(unsigned long x) {
        return size_t(__builtin_ctzl(x));
    }
};
template <>
struct HostCTZ<unsigned long long> {
    __host__ GEC_INLINE static size_t call(unsigned long long x) {
        return size_t(__builtin_ctzll(x));
    }
};

#endif

#if defined(_MSC_VER)

#if defined(GEC_X86) || defined(GEC_AMD64)

#pragma intrinsic(_BitScanForward)
template <typename T>
struct HostCTZ<T, std::enable_if_t<(utils::type_bits<T>::value <= 32)>> {
    __host__ GEC_INLINE static size_t call(T x) {
        unsigned long ret;
        _BitScanForward(&ret, (unsigned long)x);
        return (size_t)ret;
    }
};

#endif

#if defined(GEC_AMD64)

#pragma intrinsic(_BitScanForward64)
template <typename T>
struct HostCTZ<T, std::enable_if_t<(utils::type_bits<T>::value > 32 &&
                                    utils::type_bits<T>::value <= 64)>> {
    __host__ GEC_INLINE static size_t call(T x) {
        unsigned long ret;
        _BitScanForward64(&ret, (__int64)x);
        return (size_t)ret;
    }
};

#endif

#endif

template <typename T>
using LeastSignificantBit = CTZ<T>;

} // namespace _ctz_

/**
 * @brief get the number of trailing zeros
 *
 * note that `x == 0` may lead to undefined behaviour
 *
 * @tparam T any unsigned integral type
 * @param x the unsigned integer
 * @return size_t the number of trailing zeros
 */
template <typename T>
__host__ __device__ GEC_INLINE size_t count_trailing_zeros(T x) {
    return _ctz_::CTZ<T>::call(x);
}

/**
 * @brief get the least significant bit
 *
 * note that `x == 0` may lead to undefined behaviour
 *
 * @tparam T any unsigned integral type
 * @param x the unsigned integer
 * @return size_t the position of least most significant bit
 */
template <typename T>
__host__ __device__ GEC_INLINE size_t least_significant_bit(T x) {
    return _ctz_::LeastSignificantBit<T>::call(x);
}

} // namespace utils

} // namespace gec

#endif // !GEC_UTILS_MISC_HPP

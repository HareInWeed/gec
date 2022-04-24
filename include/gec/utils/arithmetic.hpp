#pragma once
#ifndef GEC_UTILS_ARITHMETIC_HPP
#define GEC_UTILS_ARITHMETIC_HPP

#include "basic.hpp"

#if defined(GEC_CLANG) || defined(GEC_MSVC) || defined(GEC_GCC)
#include <intrin.h>
#endif

namespace gec {

namespace utils {

/** @brief a + carry' = b + c + carry
 */
template <typename T>
__host__ __device__ GEC_INLINE bool
add_with_carry(T &GEC_RSTRCT a, const T &GEC_RSTRCT b, const T &GEC_RSTRCT c,
               bool carry) {
    // TODO: avoid variadic running time
    a = b + c + static_cast<T>(carry);
    return (a < b || a < c) || (carry && (a == b || a == c));
}

#if defined(__CUDA_ARCH__)

// Device Specialized `add_with_carry`
//
// Check out: <https://docs.nvidia.com/cuda/cuda-math-api/index.html>

// TODO: find suitible intrinsics to construct a specialized `add_with_carry`

#else

// Host Specialized `add_with_carry`
//
// In terms of Clang, see:
// <https://clang.llvm.org/docs/LanguageExtensions.html#multiprecision-arithmetic-builtins>
//
// In terms of GCC, see:
// <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=79173#c5>
// my guess is that gcc has already supported x86 instrinstics like
// `_addcarry_u...`, but without detailed documents?
//
// In terms of MSVC, see:
// <https://docs.microsoft.com/en-us/cpp/intrinsics/x86-intrinsics-list?view=msvc-170#x86-intrinsics>
// <https://docs.microsoft.com/en-us/cpp/intrinsics/x64-amd64-intrinsics-list?view=msvc-170#x64-intrinsics>
//
// | bit length \ Complier | GCC           | Clang         | MSVC           |
// | --------------------- | ------------- | ------------- | -------------- |
// | 8                     | -             | Builtin       | x86/64 Builtin |
// | 16                    | -             | Builtin       | x86/64 Builtin |
// | 32                    | x86 Intrinsic | Builtin & x86 | x86 Intrinsic  |
// | 64                    | x64 Intrinsic | Builtin & x64 | x64 Intrinsic  |

#if defined(__x86_64__)
template <>
__host__ GEC_INLINE bool
add_with_carry<uint64_t>(uint64_t &GEC_RSTRCT a, const uint64_t &GEC_RSTRCT b,
                         const uint64_t &GEC_RSTRCT c, bool carry) {
    return bool(_addcarry_u64((unsigned char)(carry), b, c, &a));
}
#endif

#if defined(__x86_64__) || defined(__i386__)
template <>
__host__ GEC_INLINE bool
add_with_carry<uint32_t>(uint32_t &GEC_RSTRCT a, const uint32_t &GEC_RSTRCT b,
                         const uint32_t &GEC_RSTRCT c, bool carry) {
    return bool(_addcarry_u32((unsigned char)(carry), b, c, &a));
}
#endif

#if defined(GEC_CLANG)
// TODO: check if `__builtin_add` has variadic running time
template <>
__host__ GEC_INLINE bool
add_with_carry<uint8_t>(uint8_t &GEC_RSTRCT a, const uint8_t &GEC_RSTRCT b,
                        const uint8_t &GEC_RSTRCT c, bool carry) {
    uint8_t new_carry;
    a = __builtin_addcb(b, c, carry, &new_carry);
    return new_carry;
}
template <>
__host__ GEC_INLINE bool
add_with_carry<uint16_t>(uint16_t &GEC_RSTRCT a, const uint16_t &GEC_RSTRCT b,
                         const uint16_t &GEC_RSTRCT c, bool carry) {
    uint16_t new_carry;
    a = __builtin_addcs(b, c, carry, &new_carry);
    return new_carry;
}

#if !defined(__x86_64__) && !defined(__i386__)
template <>
__host__ GEC_INLINE bool
add_with_carry<uint32_t>(uint32_t &GEC_RSTRCT a, const uint32_t &GEC_RSTRCT b,
                         const uint32_t &GEC_RSTRCT c, bool carry) {
    uint32_t new_carry;
    a = __builtin_addc(b, c, carry, &new_carry);
    return new_carry;
}
template <>
__host__ GEC_INLINE bool
add_with_carry<uint64_t>(uint64_t &GEC_RSTRCT a, const uint64_t &GEC_RSTRCT b,
                         const uint64_t &GEC_RSTRCT c, bool carry) {
    uint64_t new_carry;
    a = __builtin_addcll(b, c, carry, &new_carry);
    return new_carry;
}
#endif
#endif

#if defined(GEC_MSVC) && (defined(__i386__) || defined(__x86_64__))
template <>
__host__ GEC_INLINE bool
add_with_carry<uint8_t>(uint8_t &GEC_RSTRCT a, const uint8_t &GEC_RSTRCT b,
                        const uint8_t &GEC_RSTRCT c, bool carry) {
    return bool(_addcarry_u8((unsigned char)(carry), b, c, &a));
}
template <>
__host__ GEC_INLINE bool
add_with_carry<uint16_t>(uint16_t &GEC_RSTRCT a, const uint16_t &GEC_RSTRCT b,
                         const uint16_t &GEC_RSTRCT c, bool carry) {
    return bool(_addcarry_u16((unsigned char)(carry), b, c, &a));
}
#endif

#endif

/** @brief a + carry' = a + b + carry
 */
template <typename T>
__host__ __device__ GEC_INLINE bool
add_with_carry(T &GEC_RSTRCT a, const T &GEC_RSTRCT b, bool carry) {
    // TODO: specialized with intrinsics
    // TODO: avoid variadic running time
    const T a0 = a;
    a = b + a0 + static_cast<T>(carry);
    return (a < b || a < a0) || (carry && (a == b || a == a0));
}

/** @brief add sequence with a single bit carry
 */
template <size_t N, typename T>
struct SeqAdd {
    __host__ __device__ GEC_INLINE static bool call(T *GEC_RSTRCT a,
                                                    const T *GEC_RSTRCT b,
                                                    const T *GEC_RSTRCT c,
                                                    bool carry) {
        bool new_carry = add_with_carry(*a, *b, *c, carry);
        return SeqAdd<N - 1, T>::call(a + 1, b + 1, c + 1, new_carry);
    }
};
template <typename T>
struct SeqAdd<1, T> {
    __host__ __device__ GEC_INLINE static bool call(T *GEC_RSTRCT a,
                                                    const T *GEC_RSTRCT b,
                                                    const T *GEC_RSTRCT c,
                                                    bool carry) {
        return add_with_carry(*a, *b, *c, carry);
    }
};

/** @brief add sequence
 *
 * return a single bit carry
 */
template <size_t N, typename T>
__host__ __device__ GEC_INLINE bool
seq_add(T *GEC_RSTRCT a, const T *GEC_RSTRCT b, const T *GEC_RSTRCT c) {
    return SeqAdd<N, T>::call(a, b, c, false);
}

/** @brief add sequence inplace with a single bit carry
 */
template <size_t N, typename T>
struct SeqAddInplace {
    __host__ __device__ GEC_INLINE static bool
    call(T *GEC_RSTRCT a, const T *GEC_RSTRCT b, bool carry) {
        bool new_carry = add_with_carry(*a, *b, carry);
        return SeqAddInplace<N - 1, T>::call(a + 1, b + 1, new_carry);
    }
};
template <typename T>
struct SeqAddInplace<1, T> {
    __host__ __device__ GEC_INLINE static bool
    call(T *GEC_RSTRCT a, const T *GEC_RSTRCT b, bool carry) {
        return add_with_carry(*a, *b, carry);
    }
};

/** @brief add sequence inplace
 *
 * return a single bit carry
 */
template <size_t N, typename T>
__host__ __device__ GEC_INLINE bool seq_add(T *GEC_RSTRCT a,
                                            const T *GEC_RSTRCT b) {
    return SeqAddInplace<N, T>::call(a, b, false);
}

/** @brief c = a + borrow' - b - borrow
 *
 * returns a single bit borrow
 */
template <typename T>
__host__ __device__ GEC_INLINE bool
sub_with_borrow(T &GEC_RSTRCT a, const T &GEC_RSTRCT b, const T &GEC_RSTRCT c,
                bool borrow) {
    // TODO: specialized with intrinsics
    // TODO: avoid variadic running time
    a = b - c - static_cast<T>(borrow);
    return (a > b) || (borrow && a == b);
}

/** @brief c = a + borrow' - b - borrow
 *
 * returns a single bit borrow
 */
template <typename T>
__host__ __device__ GEC_INLINE bool
sub_with_borrow(T &GEC_RSTRCT a, const T &GEC_RSTRCT b, bool borrow) {
    // TODO: specialized with intrinsics
    // TODO: avoid variadic running time
    T a0 = a;
    a = a0 - b - static_cast<T>(borrow);
    return (a > a0) || (borrow && a == a0);
}

/** @brief subtract sequence with a single bit borrow
 */
template <size_t N, typename T>
struct SeqSub {
    __host__ __device__ GEC_INLINE static bool call(T *GEC_RSTRCT a,
                                                    const T *GEC_RSTRCT b,
                                                    const T *GEC_RSTRCT c,
                                                    bool borrow) {
        bool new_borrow = sub_with_borrow(*a, *b, *c, borrow);
        return SeqSub<N - 1, T>::call(a + 1, b + 1, c + 1, new_borrow);
    }
};
template <typename T>
struct SeqSub<1, T> {
    __host__ __device__ GEC_INLINE static bool call(T *GEC_RSTRCT a,
                                                    const T *GEC_RSTRCT b,
                                                    const T *GEC_RSTRCT c,
                                                    bool borrow) {
        return sub_with_borrow(*a, *b, *c, borrow);
    }
};

/** @brief subtract sequence with a single bit borrow
 *
 * return a single bit borrow
 */
template <size_t N, typename T>
__host__ __device__ GEC_INLINE bool
seq_sub(T *GEC_RSTRCT a, const T *GEC_RSTRCT b, const T *GEC_RSTRCT c) {
    return SeqSub<N, T>::call(a, b, c, false);
}

/** @brief subtract sequence with a single bit borrow
 */
template <size_t N, typename T>
struct SeqSubInplace {
    __host__ __device__ GEC_INLINE static bool
    call(T *GEC_RSTRCT a, const T *GEC_RSTRCT b, bool borrow) {
        bool new_borrow = sub_with_borrow(*a, *b, borrow);
        return SeqSubInplace<N - 1, T>::call(a + 1, b + 1, new_borrow);
    }
};
template <typename T>
struct SeqSubInplace<1, T> {
    __host__ __device__ GEC_INLINE static bool
    call(T *GEC_RSTRCT a, const T *GEC_RSTRCT b, bool borrow) {
        return sub_with_borrow(*a, *b, borrow);
    }
};

/** @brief subtract sequence inplace with a single bit borrow
 *
 * return a single bit borrow
 */
template <size_t N, typename T>
__host__ __device__ GEC_INLINE bool seq_sub(T *GEC_RSTRCT a,
                                            const T *GEC_RSTRCT b) {
    return SeqSubInplace<N, T>::call(a, b, false);
}

} // namespace utils

} // namespace gec

#endif // !GEC_UTILS_ARITHMETIC_HPP

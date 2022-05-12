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
uint_add_with_carry(T &GEC_RSTRCT a, const T &GEC_RSTRCT b,
                    const T &GEC_RSTRCT c, bool carry) {
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

#if defined(GEC_AMD64)
template <>
__host__ GEC_INLINE bool
uint_add_with_carry<uint64_t>(uint64_t &GEC_RSTRCT a,
                              const uint64_t &GEC_RSTRCT b,
                              const uint64_t &GEC_RSTRCT c, bool carry) {
    return bool(_addcarry_u64((unsigned char)(carry), b, c, &a));
}
#endif

#if defined(GEC_AMD64) || defined(GEC_X86)
template <>
__host__ GEC_INLINE bool
uint_add_with_carry<uint32_t>(uint32_t &GEC_RSTRCT a,
                              const uint32_t &GEC_RSTRCT b,
                              const uint32_t &GEC_RSTRCT c, bool carry) {
    return bool(_addcarry_u32((unsigned char)(carry), b, c, &a));
}
#endif

#if defined(GEC_CLANG)
// TODO: check if `__builtin_add` has variadic running time
template <>
__host__ GEC_INLINE bool
uint_add_with_carry<uint8_t>(uint8_t &GEC_RSTRCT a, const uint8_t &GEC_RSTRCT b,
                             const uint8_t &GEC_RSTRCT c, bool carry) {
    uint8_t new_carry;
    a = __builtin_addcb(b, c, carry, &new_carry);
    return new_carry;
}
template <>
__host__ GEC_INLINE bool
uint_add_with_carry<uint16_t>(uint16_t &GEC_RSTRCT a,
                              const uint16_t &GEC_RSTRCT b,
                              const uint16_t &GEC_RSTRCT c, bool carry) {
    uint16_t new_carry;
    a = __builtin_addcs(b, c, carry, &new_carry);
    return new_carry;
}

#if !defined(GEC_AMD64) && !defined(GEC_X86)
template <>
__host__ GEC_INLINE bool
uint_add_with_carry<uint32_t>(uint32_t &GEC_RSTRCT a,
                              const uint32_t &GEC_RSTRCT b,
                              const uint32_t &GEC_RSTRCT c, bool carry) {
    uint32_t new_carry;
    a = __builtin_addc(b, c, carry, &new_carry);
    return new_carry;
}
template <>
__host__ GEC_INLINE bool
uint_add_with_carry<uint64_t>(uint64_t &GEC_RSTRCT a,
                              const uint64_t &GEC_RSTRCT b,
                              const uint64_t &GEC_RSTRCT c, bool carry) {
    uint64_t new_carry;
    a = __builtin_addcll(b, c, carry, &new_carry);
    return new_carry;
}
#endif
#endif

#if defined(GEC_MSVC) && (defined(GEC_X86) || defined(GEC_AMD64))
template <>
__host__ GEC_INLINE bool
uint_add_with_carry<uint8_t>(uint8_t &GEC_RSTRCT a, const uint8_t &GEC_RSTRCT b,
                             const uint8_t &GEC_RSTRCT c, bool carry) {
    return bool(_addcarry_u8((unsigned char)(carry), b, c, &a));
}
template <>
__host__ GEC_INLINE bool
uint_add_with_carry<uint16_t>(uint16_t &GEC_RSTRCT a,
                              const uint16_t &GEC_RSTRCT b,
                              const uint16_t &GEC_RSTRCT c, bool carry) {
    return bool(_addcarry_u16((unsigned char)(carry), b, c, &a));
}
#endif

#endif

/** @brief a + carry' = a + b + carry
 *
 * optimized for inplace add rether than simply calling `uint_add_with_carry`
 */
template <typename T>
__host__ __device__ GEC_INLINE bool
uint_add_with_carry(T &GEC_RSTRCT a, const T &GEC_RSTRCT b, bool carry) {
    const T a0 = a;
    return uint_add_with_carry(a, a0, b, carry);
}

/** @brief add sequence with a single bit carry
 */
template <size_t N, typename T>
struct SeqAdd {
    __host__ __device__ GEC_INLINE static bool call(T *GEC_RSTRCT a,
                                                    const T *GEC_RSTRCT b,
                                                    const T *GEC_RSTRCT c,
                                                    bool carry) {
        bool new_carry = uint_add_with_carry(*a, *b, *c, carry);
        return SeqAdd<N - 1, T>::call(a + 1, b + 1, c + 1, new_carry);
    }
};
template <typename T>
struct SeqAdd<1, T> {
    __host__ __device__ GEC_INLINE static bool call(T *GEC_RSTRCT a,
                                                    const T *GEC_RSTRCT b,
                                                    const T *GEC_RSTRCT c,
                                                    bool carry) {
        return uint_add_with_carry(*a, *b, *c, carry);
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
        bool new_carry = uint_add_with_carry(*a, *b, carry);
        return SeqAddInplace<N - 1, T>::call(a + 1, b + 1, new_carry);
    }
};
template <typename T>
struct SeqAddInplace<1, T> {
    __host__ __device__ GEC_INLINE static bool
    call(T *GEC_RSTRCT a, const T *GEC_RSTRCT b, bool carry) {
        return uint_add_with_carry(*a, *b, carry);
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
uint_sub_with_borrow(T &GEC_RSTRCT a, const T &GEC_RSTRCT b,
                     const T &GEC_RSTRCT c, bool borrow) {
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
uint_sub_with_borrow(T &GEC_RSTRCT a, const T &GEC_RSTRCT b, bool borrow) {
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
        bool new_borrow = uint_sub_with_borrow(*a, *b, *c, borrow);
        return SeqSub<N - 1, T>::call(a + 1, b + 1, c + 1, new_borrow);
    }
};
template <typename T>
struct SeqSub<1, T> {
    __host__ __device__ GEC_INLINE static bool call(T *GEC_RSTRCT a,
                                                    const T *GEC_RSTRCT b,
                                                    const T *GEC_RSTRCT c,
                                                    bool borrow) {
        return uint_sub_with_borrow(*a, *b, *c, borrow);
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
        bool new_borrow = uint_sub_with_borrow(*a, *b, borrow);
        return SeqSubInplace<N - 1, T>::call(a + 1, b + 1, new_borrow);
    }
};
template <typename T>
struct SeqSubInplace<1, T> {
    __host__ __device__ GEC_INLINE static bool
    call(T *GEC_RSTRCT a, const T *GEC_RSTRCT b, bool borrow) {
        return uint_sub_with_borrow(*a, *b, borrow);
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

template <typename T>
__host__ __device__ GEC_INLINE void
uint_mul_lh(T &GEC_RSTRCT l, T &GEC_RSTRCT h, const T &GEC_RSTRCT a,
            const T &GEC_RSTRCT b) {
    constexpr size_t len = std::numeric_limits<T>::digits / 2;
    constexpr T lower_mask = (T(1) << len) - T(1);

    T al = a & lower_mask;
    T bl = b & lower_mask;
    T ah = a >> len;
    T bh = b >> len;

    T al_bl = al * bl;
    T ah_bl = ah * bl;
    T al_bh = al * bh;
    T ah_bh = ah * bh;

    T lh;
    T carry1 = T(uint_add_with_carry(lh, al_bh, ah_bl, false));
    T carry2 = T(uint_add_with_carry(l, al_bl, (lh << len), false));
    h = ah_bh + (lh >> len) + carry2 + (carry1 << len);
}

#if defined(GEC_MSVC) && defined(GEC_AMD64)
// see: <https://docs.microsoft.com/en-us/cpp/intrinsics/umul128?view=msvc-170>
template <>
__host__ GEC_INLINE void uint_mul_lh<uint64_t>(uint64_t &GEC_RSTRCT l,
                                               uint64_t &GEC_RSTRCT h,
                                               const uint64_t &GEC_RSTRCT a,
                                               const uint64_t &GEC_RSTRCT b) {
    l = _umul128(a, b, &h);
}
#endif // GEC_AMD64

#define GEC_specialized_uint_mul_lh(U, DU)                                     \
    template <>                                                                \
    __host__ __device__ GEC_INLINE void uint_mul_lh<U>(                        \
        U & GEC_RSTRCT l, U & GEC_RSTRCT h, const U &GEC_RSTRCT a,             \
        const U &GEC_RSTRCT b) {                                               \
        DU product = DU(a) * DU(b);                                            \
        l = U(product);                                                        \
        h = product >> std::numeric_limits<U>::digits;                         \
    }
GEC_specialized_uint_mul_lh(uint32_t, uint64_t);
GEC_specialized_uint_mul_lh(uint16_t, uint32_t);
GEC_specialized_uint_mul_lh(uint8_t, uint16_t);
#undef GEC_specialized_uint_mul_lh

/** @brief a = a + b * x
 *
 * return the last limb in the product
 */
template <size_t N, typename T>
__host__ __device__ GEC_INLINE T seq_add_mul_limb(T *GEC_RSTRCT a,
                                                  const T *GEC_RSTRCT b,
                                                  const T &x) {
    // x * b[4] - -
    // x * b[3]   - -
    // x * b[2]     - -
    // x * b[1]       - -
    // x * b[0]         - -
    // b          - - - - -
    // x                  -

    T l0, h0, l1, h1;
    bool carry0 = false, carry1 = false;
    // TODO: refactor the loop condition
    for (int i = 0; i < int(N) - 2; i += 2) { // deal with case N < 3, not ideal
        uint_mul_lh(l0, h0, b[i], x);
        carry0 = uint_add_with_carry(a[i + 1], h0,
                                     uint_add_with_carry(a[i], l0, carry0));

        uint_mul_lh(l1, h1, b[i + 1], x);
        carry1 = uint_add_with_carry(a[i + 2], h1,
                                     uint_add_with_carry(a[i + 1], l1, carry1));
    }

    T last_limb;
    if (N & 0x1) { // N is odd
        uint_mul_lh(l0, h0, b[N - 1], x);
        last_limb = h0 +
                    static_cast<T>(uint_add_with_carry(a[N - 1], l0, carry0)) +
                    static_cast<T>(carry1);
    } else { // N is even
        uint_mul_lh(l0, h0, b[N - 2], x);
        carry0 = uint_add_with_carry(a[N - 1], h0,
                                     uint_add_with_carry(a[N - 2], l0, carry0));

        uint_mul_lh(l1, h1, b[N - 1], x);
        last_limb = h1 +
                    static_cast<T>(uint_add_with_carry(a[N - 1], l1, carry1)) +
                    static_cast<T>(carry0);
    }

    return last_limb;
}

} // namespace utils

} // namespace gec

#endif // !GEC_UTILS_ARITHMETIC_HPP

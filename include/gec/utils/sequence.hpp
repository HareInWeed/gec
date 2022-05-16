#pragma once
#ifndef GEC_UTILS_SEQUENCE_HPP
#define GEC_UTILS_SEQUENCE_HPP

#include <type_traits>

#include "basic.hpp"
#include "operators.hpp"

namespace gec {

namespace utils {

template <typename T>
__host__ __device__ GEC_INLINE void fill_le(T *dst) {}
/** @brief fill `dst` with the rest of arguments, from lower to higher (big
 * endian)
 */
template <typename T, typename... S>
__host__ __device__ GEC_INLINE void fill_le(T *dst, T elem, S... seq) {
    *dst = elem;
    fill_le<T>(dst + 1, seq...);
}

/** @brief fill `dst` with the rest of arguments, from higher to lower (big
 * endian)
 */
template <typename T>
__host__ __device__ GEC_INLINE void fill_be(T *dst, T elem) {
    *dst = elem;
}
template <typename T, typename... S>
__host__ __device__ GEC_INLINE void fill_be(T *dst, T elem, S... seq) {
    *(dst + sizeof...(seq)) = elem;
    fill_be<T>(dst, seq...);
}

/** @brief fill `dst` with a single limb
 */
template <size_t LEN, typename T>
struct FillSeqLimb {
    __host__ __device__ GEC_INLINE static void call(T *GEC_RSTRCT dst,
                                                    const T &GEC_RSTRCT elem) {
        *(dst + LEN - 1) = elem;
        FillSeqLimb<LEN - 1, T>::call(dst, elem);
    }
};
template <typename T>
struct FillSeqLimb<1, T> {
    __host__ __device__ GEC_INLINE static void call(T *GEC_RSTRCT dst,
                                                    const T &GEC_RSTRCT elem) {
        *dst = elem;
    }
};
template <typename T>
struct FillSeqLimb<0, T> {
    __host__ __device__ GEC_INLINE static void call(T *GEC_RSTRCT dst,
                                                    const T &GEC_RSTRCT elem) {
        // don't do anything
    }
};

/** @brief fill `dst` with a single limb
 */
template <size_t LEN, typename T>
__host__ __device__ GEC_INLINE void fill_seq_limb(T *GEC_RSTRCT dst,
                                                  const T &GEC_RSTRCT elem) {
    FillSeqLimb<LEN, T>::call(dst, elem);
}

/** @brief test if sequence is fill with same limb
 * TODO: rename with `VtSeqEqLimb`
 */
template <size_t LEN, typename T>
struct SeqEqLimb {
    __host__ __device__ GEC_INLINE static bool call(const T *GEC_RSTRCT a,
                                                    const T &GEC_RSTRCT elem) {
        return *(a + LEN - 1) == elem && SeqEqLimb<LEN - 1, T>::call(a, elem);
    }
};
template <typename T>
struct SeqEqLimb<1, T> {
    __host__ __device__ GEC_INLINE static bool call(const T *GEC_RSTRCT a,
                                                    const T &GEC_RSTRCT elem) {
        return *a == elem;
    }
};
/** @brief test if sequence is fill with the same limb
 */
template <size_t LEN, typename T>
__host__ __device__ GEC_INLINE bool seq_all_limb(const T *GEC_RSTRCT a,
                                                 const T &GEC_RSTRCT elem) {
    return SeqEqLimb<LEN, T>::call(a, elem);
}

/** @brief apply unary operator on limbs respectively
 */
template <size_t LEN, typename T, typename F>
struct SeqUnaryOp {
    __host__ __device__ GEC_INLINE static void call(T *GEC_RSTRCT dst,
                                                    const T *GEC_RSTRCT src) {
        F::call(*(dst + LEN - 1), *(src + LEN - 1));
        SeqUnaryOp<LEN - 1, T, F>::call(dst, src);
    }
};
template <typename T, typename F>
struct SeqUnaryOp<1, T, F> {
    __host__ __device__ GEC_INLINE static void call(T *GEC_RSTRCT dst,
                                                    const T *GEC_RSTRCT src) {
        F::call(*dst, *src);
    }
};

/** @brief fill `dst` with anthor sequence `src`
 */
template <size_t LEN, typename T>
__host__ __device__ GEC_INLINE void fill_seq(T *GEC_RSTRCT dst,
                                             const T *GEC_RSTRCT src) {
    SeqUnaryOp<LEN, T, ops::Let<T>>::call(dst, src);
}

/** @brief variadic time test if all limbs yields true under operator
 */
template <size_t LEN, typename T, typename F>
struct VtSeqAll {
    __host__ __device__ GEC_INLINE static bool call(const T *GEC_RSTRCT a,
                                                    const T *GEC_RSTRCT b) {
        return F::call(*(a + LEN - 1), *(b + LEN - 1)) &&
               VtSeqAll<LEN - 1, T, F>::call(a, b);
    }
};
template <typename T, typename F>
struct VtSeqAll<1, T, F> {
    __host__ __device__ GEC_INLINE static bool call(const T *GEC_RSTRCT a,
                                                    const T *GEC_RSTRCT b) {
        return F::call(*a, *b);
    }
};

/** @brief variadic time dictionary comparsion of two sequence
 *
 * return value is consistant with constants in `CmpEnum`
 */
template <size_t LEN, typename T>
struct VtSeqCmp {
    __host__ __device__ GEC_INLINE static CmpEnum call(const T *GEC_RSTRCT a,
                                                       const T *GEC_RSTRCT b) {
        if (*(a + LEN - 1) != *(b + LEN - 1)) {
            return *(a + LEN - 1) < *(b + LEN - 1) ? CmpEnum::Lt : CmpEnum::Gt;
        }
        return VtSeqCmp<LEN - 1, T>::call(a, b);
    }
};
template <typename T>
struct VtSeqCmp<0, T> {
    __host__ __device__ GEC_INLINE static CmpEnum call(const T *GEC_RSTRCT a,
                                                       const T *GEC_RSTRCT b) {
        return CmpEnum::Eq;
    }
};

/** @brief apply binary operator on limbs respectively
 */
template <size_t LEN, typename T, typename F>
struct SeqBinOp {
    __host__ __device__ GEC_INLINE static void
    call(T *GEC_RSTRCT c, const T *GEC_RSTRCT a, const T *GEC_RSTRCT b) {
        F::call(*(c + LEN - 1), *(a + LEN - 1), *(b + LEN - 1));
        SeqBinOp<LEN - 1, T, F>::call(c, a, b);
    }
};
template <typename T, typename F>
struct SeqBinOp<1, T, F> {
    __host__ __device__ GEC_INLINE static void call(T *c, const T *a,
                                                    const T *b) {
        F::call(*c, *a, *b);
    }
};

/** @brief inplace right shift
 */
template <size_t N, size_t LS, size_t BS, typename T>
struct SeqShiftRightInplace {
    __host__ __device__ GEC_INLINE static void call(T *a) {
        *a = (*(a + LS) >> BS) |
             (*(a + LS + 1) << (std::numeric_limits<T>::digits - BS));
        SeqShiftRightInplace<N - 1, LS, BS, T>::call(a + 1);
    }
};
template <size_t LS, size_t BS, typename T>
struct SeqShiftRightInplace<1, LS, BS, T> {
    __host__ __device__ GEC_INLINE static void call(T *a) {
        *a = *(a + LS) >> BS;
    }
};
template <size_t N, size_t LS, typename T>
struct SeqShiftRightInplace<N, LS, 0, T> {
    __host__ __device__ GEC_INLINE static void call(T *a) {
        *a = *(a + LS);
        SeqShiftRightInplace<N - 1, LS, 0, T>::call(a + 1);
    }
};
template <size_t LS, typename T>
struct SeqShiftRightInplace<1, LS, 0, T> {
    __host__ __device__ GEC_INLINE static void call(T *a) { *a = *(a + LS); }
};
template <size_t LS, size_t BS, typename T>
struct SeqShiftRightInplace<0, LS, BS, T> {
    __host__ __device__ GEC_INLINE static void call(T *a) {
        // do nothing
    }
};
template <size_t LS, typename T>
struct SeqShiftRightInplace<0, LS, 0, T> {
    __host__ __device__ GEC_INLINE static void call(T *a) {
        // do nothing
    }
};
/** @brief inplace right shift
 *
 * @param LEN limb number
 * @param B   the bit length to shift
 * @param a   the sequence to be shifted
 *
 * Beware, if `LEN * limb_bits < B`, the compiler may hang forever
 */
template <size_t LEN, size_t B, typename T>
__host__ __device__ GEC_INLINE void seq_shift_right(T *a) {
    constexpr size_t LS = B / std::numeric_limits<T>::digits;
    constexpr size_t BS = B % std::numeric_limits<T>::digits;
    constexpr size_t N = LEN - LS;
    SeqShiftRightInplace<N, LS, BS, T>::call(a);
    fill_seq_limb<LS, T>(a + N, 0);
}

/** @brief inplace left shift
 */
template <size_t N, size_t LS, size_t BS, typename T>
struct SeqShiftLeftInplace {
    __host__ __device__ GEC_INLINE static void call(T *a) {
        *(a + LS + N - 1) =
            (*(a + N - 1) << BS) |
            (*(a + N - 2) >> (std::numeric_limits<T>::digits - BS));
        SeqShiftLeftInplace<N - 1, LS, BS, T>::call(a);
    }
};
template <size_t LS, size_t BS, typename T>
struct SeqShiftLeftInplace<1, LS, BS, T> {
    __host__ __device__ GEC_INLINE static void call(T *a) {
        *(a + LS) = (*a << BS);
    }
};
template <size_t N, size_t LS, typename T>
struct SeqShiftLeftInplace<N, LS, 0, T> {
    __host__ __device__ GEC_INLINE static void call(T *a) {
        *(a + LS + N - 1) = *(a + N - 1);
        SeqShiftLeftInplace<N - 1, LS, 0, T>::call(a);
    }
};
template <size_t LS, typename T>
struct SeqShiftLeftInplace<1, LS, 0, T> {
    __host__ __device__ GEC_INLINE static void call(T *a) { *(a + LS) = *a; }
};
template <size_t LS, size_t BS, typename T>
struct SeqShiftLeftInplace<0, LS, BS, T> {
    __host__ __device__ GEC_INLINE static void call(T *a) {
        // do nothing
    }
};
template <size_t LS, typename T>
struct SeqShiftLeftInplace<0, LS, 0, T> {
    __host__ __device__ GEC_INLINE static void call(T *a) {
        // do nothing
    }
};
/** @brief inplace left shift
 *
 * @param LEN limb number
 * @param B   the bit length to shift
 * @param a   the sequence to be shifted
 *
 * Beware, if `LEN * limb_bits < B`, the compiler may hang forever
 */
template <size_t LEN, size_t B, typename T>
__host__ __device__ GEC_INLINE void seq_shift_left(T *a) {
    constexpr size_t LS = B / std::numeric_limits<T>::digits;
    constexpr size_t BS = B % std::numeric_limits<T>::digits;
    constexpr size_t N = LEN - LS;
    SeqShiftLeftInplace<N, LS, BS, T>::call(a);
    fill_seq_limb<LS, T>(a, 0);
}

} // namespace utils

} // namespace gec

#endif // !GEC_UTILS_SEQUENCE_HPP
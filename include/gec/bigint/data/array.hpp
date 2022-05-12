#pragma once
#ifndef GEC_BIGINT_DATA_ARRAY_HPP
#define GEC_BIGINT_DATA_ARRAY_HPP

#include <gec/utils/sequence.hpp>

namespace gec {

namespace bigint {

/** @brief TODO:
 */
template <class LIMB_T, size_t LIMB_N>
class Array {
  public:
    using LimbT = LIMB_T;
    const static size_t LimbN = LIMB_N;
    LIMB_T arr[LIMB_N];

    __host__ __device__ GEC_INLINE Array() {
        utils::fill_seq_limb<LIMB_N, LIMB_T>(arr, 0);
    }
    template <typename... LIMBS,
              std::enable_if_t<(sizeof...(LIMBS) == LIMB_N)> * = nullptr>
    __host__ __device__ GEC_INLINE Array(LIMBS... limbs) {
        utils::fill_be<LIMB_T>(arr, limbs...);
    }
    template <std::enable_if_t<(LIMB_N > 1)> * = nullptr>
    __host__ __device__ GEC_INLINE Array(LIMB_T limb) {
        *arr = limb;
        utils::fill_seq_limb<LIMB_N - 1, LIMB_T>(arr + 1, 0);
    }

    template <size_t LIMB_M>
    __host__ __device__ GEC_INLINE Array(const Array<LIMB_T, LIMB_M> &other) {
        utils::fill_seq<(LIMB_N > LIMB_M ? LIMB_M : LIMB_N)>(arr, other.arr);
        utils::fill_seq_limb<(LIMB_N > LIMB_M ? LIMB_N - LIMB_M : 0), LIMB_T>(
            arr + LIMB_M, 0);
    }
    template <size_t LIMB_M>
    __host__ __device__ GEC_INLINE Array &
    operator=(const Array &GEC_RSTRCT other) {
        if (this != &other) {
            utils::fill_seq<(LIMB_N > LIMB_M ? LIMB_M : LIMB_N)>(arr,
                                                                 other.arr);
            utils::fill_seq_limb<(LIMB_N > LIMB_M ? LIMB_N - LIMB_M : 0),
                                 LIMB_T>(arr + LIMB_M, 0);
        }
        return *this;
    }

    __host__ __device__ GEC_INLINE const LIMB_T *array() const { return arr; }
    __host__ __device__ GEC_INLINE LIMB_T *array() { return arr; }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_DATA_ARRAY_HPP

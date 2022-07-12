#pragma once
#ifndef GEC_BIGINT_MIXIN_CONSTANTS_HPP
#define GEC_BIGINT_MIXIN_CONSTANTS_HPP

#include <gec/utils/arithmetic.hpp>
#include <gec/utils/crtp.hpp>
#include <gec/utils/sequence.hpp>

namespace gec {

namespace bigint {

/** @brief mixin that enables test and assignment of common constants
 */
template <class Core, typename LIMB_T, size_t LIMB_N>
class GEC_EMPTY_BASES Constants
    : protected CRTP<Core, Constants<Core, LIMB_T, LIMB_N>> {
    friend CRTP<Core, Constants<Core, LIMB_T, LIMB_N>>;

  public:
    __host__ __device__ GEC_INLINE bool is_zero() const {
        return utils::seq_all_limb<LIMB_N, LIMB_T>(this->core().array(),
                                                   LIMB_T(0));
    }

    __host__ __device__ GEC_INLINE void set_zero() {
        utils::fill_seq_limb<LIMB_N, LIMB_T>(this->core().array(), LIMB_T(0));
    }

    __host__ __device__ GEC_INLINE bool is_one() const {
        return this->core().array()[0] == LIMB_T(1) &&
               utils::seq_all_limb<LIMB_N - 1, LIMB_T>(this->core().array() + 1,
                                                       0);
    }

    __host__ __device__ GEC_INLINE void set_one() {
        this->core().array()[0] = LIMB_T(1);
        utils::fill_seq_limb<LIMB_N - 1, LIMB_T>(this->core().array() + 1, 0);
    }

    template <size_t K>
    __host__ __device__ GEC_INLINE void set_pow2() {
        constexpr size_t LimbBit = utils::type_bits<LIMB_T>::value;
        constexpr size_t LimbIdx = K / LimbBit;
        constexpr LIMB_T LimbVal = LIMB_T(1) << (K % LimbBit);
        this->set_zero();
        this->core().array()[LimbIdx] = LimbVal;
    }

    __host__ __device__ GEC_INLINE void set_pow2(size_t k) {
        constexpr size_t LimbBit = utils::type_bits<LIMB_T>::value;
        size_t LimbIdx = k / LimbBit;
        LIMB_T LimbVal = LIMB_T(1) << (k % LimbBit);
        this->set_zero();
        this->core().array()[LimbIdx] = LimbVal;
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_CONSTANTS_HPP

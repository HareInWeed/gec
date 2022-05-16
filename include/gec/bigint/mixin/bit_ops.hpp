#pragma once
#ifndef GEC_BIGINT_MIXIN_BIT_OPS_HPP
#define GEC_BIGINT_MIXIN_BIT_OPS_HPP

#include <gec/utils/basic.hpp>
#include <gec/utils/crtp.hpp>
#include <gec/utils/sequence.hpp>

namespace gec {

namespace bigint {

/** @brief mixin that enables bit operations
 */
template <class Core, class LIMB_T, size_t LIMB_N>
class BitOps : protected CRTP<Core, BitOps<Core, LIMB_T, LIMB_N>> {
    friend CRTP<Core, BitOps<Core, LIMB_T, LIMB_N>>;

  public:
    __host__ __device__ GEC_INLINE static void
    bit_and(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
            const Core &GEC_RSTRCT c) {
        utils::SeqBinOp<LIMB_N, LIMB_T, utils::ops::BitAnd<LIMB_T>>::call(
            a.array(), b.array(), c.array());
    }
    __host__ __device__ GEC_INLINE static void
    bit_or(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
           const Core &GEC_RSTRCT c) {
        utils::SeqBinOp<LIMB_N, LIMB_T, utils::ops::BitOr<LIMB_T>>::call(
            a.array(), b.array(), c.array());
    }
    __host__ __device__ GEC_INLINE static void
    bit_not(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b) {
        utils::SeqUnaryOp<LIMB_N, LIMB_T, utils::ops::BitNot<LIMB_T>>::call(
            a.array(), b.array());
    }
    __host__ __device__ GEC_INLINE static void
    bit_xor(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
            const Core &GEC_RSTRCT c) {
        utils::SeqBinOp<LIMB_N, LIMB_T, utils::ops::BitXor<LIMB_T>>::call(
            a.array(), b.array(), c.array());
    }

    __host__ __device__ GEC_INLINE Core &bit_and(const Core &GEC_RSTRCT a,
                                                 const Core &GEC_RSTRCT b) {
        bit_and(this->core(), a, b);
        return this->core();
    }
    __host__ __device__ GEC_INLINE Core &bit_or(const Core &GEC_RSTRCT a,
                                                const Core &GEC_RSTRCT b) {
        bit_or(this->core(), a, b);
        return this->core();
    }
    __host__ __device__ GEC_INLINE Core &bit_not(const Core &GEC_RSTRCT a) {
        bit_not(this->core(), a);
        return this->core();
    }
    __host__ __device__ GEC_INLINE Core &bit_xor(const Core &GEC_RSTRCT a,
                                                 const Core &GEC_RSTRCT b) {
        bit_xor(this->core(), a, b);
        return this->core();
    }

    /** @brief shift element by `B` bit
     *
     * Beware, `B` should be less then the bit length of the `data` field,
     * otherwise the complier might hang forever during compilation.
     */
    template <size_t B>
    __host__ __device__ GEC_INLINE void shift_right() {
        utils::seq_shift_right<LIMB_N, B>(this->core().array());
    }

    // TODO: runtime shift_right
    // void shift_right(size_t n) {}

    /** @brief shift element by `B` bit
     *
     * Beware, `B` should be less then the bit length of the `data` field,
     * otherwise the complier might hang forever during compilation.
     *
     * This method does not check whether the shifted element is still in range.
     * User should check the value of the element before left shifting
     */
    template <size_t B>
    __host__ __device__ GEC_INLINE void shift_left() {
        utils::seq_shift_left<LIMB_N, B>(this->core().array());
    }

    // TODO: runtime shift_right
    // void shift_left(size_t n) {}

    __host__ __device__ GEC_INLINE size_t trailing_zeros() {
        // TODO
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_BIT_OPS_HPP

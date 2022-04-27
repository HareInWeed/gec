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
class BitOpsMixin : public CRTP<Core, BitOpsMixin<Core, LIMB_T, LIMB_N>> {
  public:
    __host__ __device__ GEC_INLINE static void
    bit_and(BitOpsMixin &GEC_RSTRCT a, const BitOpsMixin &GEC_RSTRCT b,
            const BitOpsMixin &GEC_RSTRCT c) {
        utils::SeqBinOp<LIMB_N, LIMB_T, utils::ops::BitAnd<LIMB_T>>::call(
            a.core().get_arr(), b.core().get_arr(), c.core().get_arr());
    }
    __host__ __device__ GEC_INLINE static void
    bit_or(BitOpsMixin &GEC_RSTRCT a, const BitOpsMixin &GEC_RSTRCT b,
           const BitOpsMixin &GEC_RSTRCT c) {
        utils::SeqBinOp<LIMB_N, LIMB_T, utils::ops::BitOr<LIMB_T>>::call(
            a.core().get_arr(), b.core().get_arr(), c.core().get_arr());
    }
    __host__ __device__ GEC_INLINE static void
    bit_not(BitOpsMixin &GEC_RSTRCT a, const BitOpsMixin &GEC_RSTRCT b) {
        utils::SeqUnaryOp<LIMB_N, LIMB_T, utils::ops::BitNot<LIMB_T>>::call(
            a.core().get_arr(), b.core().get_arr());
    }
    __host__ __device__ GEC_INLINE static void
    bit_xor(BitOpsMixin &GEC_RSTRCT a, const BitOpsMixin &GEC_RSTRCT b,
            const BitOpsMixin &GEC_RSTRCT c) {
        utils::SeqBinOp<LIMB_N, LIMB_T, utils::ops::BitXor<LIMB_T>>::call(
            a.core().get_arr(), b.core().get_arr(), c.core().get_arr());
    }

    __host__ __device__ GEC_INLINE BitOpsMixin &
    bit_and(const BitOpsMixin &GEC_RSTRCT a, const BitOpsMixin &GEC_RSTRCT b) {
        bit_and(*this, a, b);
        return *this;
    }
    __host__ __device__ GEC_INLINE BitOpsMixin &
    bit_or(const BitOpsMixin &GEC_RSTRCT a, const BitOpsMixin &GEC_RSTRCT b) {
        bit_or(*this, a, b);
        return *this;
    }
    __host__ __device__ GEC_INLINE BitOpsMixin &
    bit_not(const BitOpsMixin &GEC_RSTRCT a) {
        bit_not(*this, a);
        return *this;
    }
    __host__ __device__ GEC_INLINE BitOpsMixin &
    bit_xor(const BitOpsMixin &GEC_RSTRCT a, const BitOpsMixin &GEC_RSTRCT b) {
        bit_xor(*this, a, b);
        return *this;
    }

    /** @brief shift element by `B` bit
     *
     * Beware, `B` should be less then the bit length of the `data` field,
     * otherwise the complier might hang forever during compilation.
     */
    template <size_t B>
    __host__ __device__ GEC_INLINE void shift_right() {
        utils::seq_shift_right<LIMB_N, B>(this->core().get_arr());
    }

    // TODO: shift_right
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
        utils::seq_shift_left<LIMB_N, B>(this->core().get_arr());
    }

    // TODO: shift_right
    // void shift_left(size_t n) {}
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_BIT_OPS_HPP

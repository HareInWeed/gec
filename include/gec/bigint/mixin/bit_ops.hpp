#pragma once
#ifndef GEC_BIGINT_MIXIN_BIT_OPS_HPP
#define GEC_BIGINT_MIXIN_BIT_OPS_HPP

#include <gec/utils/basic.hpp>
#include <gec/utils/crtp.hpp>
#include <gec/utils/misc.hpp>
#include <gec/utils/sequence.hpp>

namespace gec {

namespace bigint {

/** @brief mixin that enables bit operations
 */
template <class Core, class LIMB_T, size_t LIMB_N>
class GEC_EMPTY_BASES BitOps
    : protected CRTP<Core, BitOps<Core, LIMB_T, LIMB_N>> {
    friend CRTP<Core, BitOps<Core, LIMB_T, LIMB_N>>;

  public:
    GEC_HD GEC_INLINE static void bit_and(Core &GEC_RSTRCT a,
                                          const Core &GEC_RSTRCT b,
                                          const Core &GEC_RSTRCT c) {
        utils::SeqBinOp<LIMB_N, LIMB_T, utils::ops::BitAnd<LIMB_T>>::call(
            a.array(), b.array(), c.array());
    }
    GEC_HD GEC_INLINE static void bit_or(Core &GEC_RSTRCT a,
                                         const Core &GEC_RSTRCT b,
                                         const Core &GEC_RSTRCT c) {
        utils::SeqBinOp<LIMB_N, LIMB_T, utils::ops::BitOr<LIMB_T>>::call(
            a.array(), b.array(), c.array());
    }
    GEC_HD GEC_INLINE static void bit_not(Core &GEC_RSTRCT a,
                                          const Core &GEC_RSTRCT b) {
        utils::SeqUnaryOp<LIMB_N, LIMB_T, utils::ops::BitNot<LIMB_T>>::call(
            a.array(), b.array());
    }
    GEC_HD GEC_INLINE static void bit_xor(Core &GEC_RSTRCT a,
                                          const Core &GEC_RSTRCT b,
                                          const Core &GEC_RSTRCT c) {
        utils::SeqBinOp<LIMB_N, LIMB_T, utils::ops::BitXor<LIMB_T>>::call(
            a.array(), b.array(), c.array());
    }

    GEC_HD GEC_INLINE Core &bit_and(const Core &GEC_RSTRCT a,
                                    const Core &GEC_RSTRCT b) {
        bit_and(this->core(), a, b);
        return this->core();
    }
    GEC_HD GEC_INLINE Core &bit_or(const Core &GEC_RSTRCT a,
                                   const Core &GEC_RSTRCT b) {
        bit_or(this->core(), a, b);
        return this->core();
    }
    GEC_HD GEC_INLINE Core &bit_not(const Core &GEC_RSTRCT a) {
        bit_not(this->core(), a);
        return this->core();
    }
    GEC_HD GEC_INLINE Core &bit_xor(const Core &GEC_RSTRCT a,
                                    const Core &GEC_RSTRCT b) {
        bit_xor(this->core(), a, b);
        return this->core();
    }

    /** @brief shift element by `B` bit */
    template <size_t B,
              std::enable_if_t<(B <= LIMB_N * utils::type_bits<LIMB_T>::value)>
                  * = nullptr>
    GEC_HD GEC_INLINE void shift_right() {
        utils::seq_shift_right<LIMB_N, B>(this->core().array());
    }

    GEC_HD void shift_right(size_t n) {
        utils::seq_shift_right<LIMB_N>(this->core().array(), n);
    }

    /** @brief shift element by `B` bit */
    template <size_t B,
              std::enable_if_t<(B <= LIMB_N * utils::type_bits<LIMB_T>::value)>
                  * = nullptr>
    GEC_HD GEC_INLINE void shift_left() {
        utils::seq_shift_left<LIMB_N, B>(this->core().array());
    }

    GEC_HD void shift_left(size_t n) {
        utils::seq_shift_left<LIMB_N>(this->core().array(), n);
    }

    GEC_HD size_t most_significant_bit() {
        constexpr size_t limb_digits = utils::type_bits<LIMB_T>::value;
        constexpr size_t is_zero = limb_digits * LIMB_N;

        size_t i = LIMB_N;
        do {
            --i;
            LIMB_T limb = this->core().array()[i];
            if (limb) {
                return i * limb_digits + utils::most_significant_bit(limb);
            }
        } while (i != 0);

        return is_zero;
    }

    GEC_HD size_t leading_zeros() {
        constexpr size_t limb_n_m_1 = LIMB_N - 1;
        constexpr size_t limb_digits = utils::type_bits<LIMB_T>::value;
        constexpr size_t is_zero = limb_digits * LIMB_N;

        size_t i = LIMB_N;
        do {
            --i;
            LIMB_T limb = this->core().array()[i];
            if (limb) {
                return (limb_n_m_1 - i) * limb_digits +
                       utils::count_leading_zeros(limb);
            }
        } while (i != 0);

        return is_zero;
    }

    GEC_HD size_t least_significant_bit() {
        constexpr size_t limb_digits = utils::type_bits<LIMB_T>::value;
        constexpr size_t is_zero = limb_digits * LIMB_N;

        for (size_t i = 0; i < LIMB_N; ++i) {
            LIMB_T limb = this->core().array()[i];
            if (limb) {
                return i * limb_digits + utils::least_significant_bit(limb);
            }
        }

        return is_zero;
    }

    GEC_HD size_t trailing_zeros() {
        constexpr size_t limb_digits = utils::type_bits<LIMB_T>::value;
        constexpr size_t is_zero = limb_digits * LIMB_N;

        for (size_t i = 0; i < LIMB_N; ++i) {
            LIMB_T limb = this->core().array()[i];
            if (limb) {
                return i * limb_digits + utils::count_trailing_zeros(limb);
            }
        }

        return is_zero;
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_BIT_OPS_HPP

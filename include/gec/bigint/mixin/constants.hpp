#pragma once
#ifndef GEC_BIGINT_MIXIN_CONSTANTS_HPP
#define GEC_BIGINT_MIXIN_CONSTANTS_HPP

#include <gec/utils/arithmetic.hpp>
#include <gec/utils/crtp.hpp>
#include <gec/utils/sequence.hpp>

namespace gec {

namespace bigint {

/// @brief base mixin that enables test and assignment of common constants
template <class Core, typename LIMB_T, size_t LIMB_N>
class GEC_EMPTY_BASES ConstantsBase
    : protected CRTP<Core, ConstantsBase<Core, LIMB_T, LIMB_N>> {
    friend CRTP<Core, ConstantsBase<Core, LIMB_T, LIMB_N>>;

  public:
    GEC_HD GEC_INLINE bool is_zero() const {
        return utils::seq_all_limb<LIMB_N, LIMB_T>(this->core().array(),
                                                   LIMB_T(0));
    }

    GEC_HD GEC_INLINE void set_zero() {
        utils::fill_seq_limb<LIMB_N, LIMB_T>(this->core().array(), LIMB_T(0));
    }

    GEC_HD GEC_INLINE bool is_one() const {
        return this->core().array()[0] == LIMB_T(1) &&
               utils::seq_all_limb<LIMB_N - 1, LIMB_T>(this->core().array() + 1,
                                                       0);
    }

    GEC_HD GEC_INLINE void set_one() {
        this->core().array()[0] = LIMB_T(1);
        utils::fill_seq_limb<LIMB_N - 1, LIMB_T>(this->core().array() + 1, 0);
    }

    template <size_t K>
    GEC_HD GEC_INLINE void set_pow2() {
        constexpr size_t LimbBit = utils::type_bits<LIMB_T>::value;
        constexpr size_t LimbIdx = K / LimbBit;
        constexpr LIMB_T LimbVal = LIMB_T(1) << (K % LimbBit);
        this->set_zero();
        this->core().array()[LimbIdx] = LimbVal;
    }

    GEC_HD GEC_INLINE void set_pow2(size_t k) {
        constexpr size_t LimbBit = utils::type_bits<LIMB_T>::value;
        size_t LimbIdx = k / LimbBit;
        LIMB_T LimbVal = LIMB_T(1) << (k % LimbBit);
        this->set_zero();
        this->core().array()[LimbIdx] = LimbVal;
    }
};

/// @brief mixin that enables test and assignment of common constants
template <class Core, typename LIMB_T, size_t LIMB_N>
class GEC_EMPTY_BASES Constants : public ConstantsBase<Core, LIMB_T, LIMB_N> {
  public:
    GEC_HD GEC_INLINE constexpr static LIMB_T mul_id() { return 1; }
    GEC_HD GEC_INLINE constexpr bool is_mul_id() const {
        return this->is_one();
    }
    GEC_HD GEC_INLINE constexpr void set_mul_id() { this->set_one(); }
};

/// @brief mixin that enables test and assignment of common constants from
/// montgomery form
template <class Core, typename LIMB_T, size_t LIMB_N>
class GEC_EMPTY_BASES MonConstants
    : public ConstantsBase<Core, LIMB_T, LIMB_N> {
  public:
    GEC_HD GEC_INLINE constexpr static const Core &mul_id() {
        return Core::one_r();
    }
    GEC_HD GEC_INLINE constexpr bool is_mul_id() const {
        return utils::VtSeqAll<LIMB_N, LIMB_T, utils::ops::Eq<LIMB_T>>::call(
            this->core().array(), Core::one_r().array());
    }
    GEC_HD GEC_INLINE constexpr void set_mul_id() {
        utils::fill_seq<LIMB_N>(this->core().array(), Core::one_r().array());
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_CONSTANTS_HPP

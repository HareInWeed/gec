#pragma once
#ifndef GEC_BIGINT_DATA_ARRAY_HPP
#define GEC_BIGINT_DATA_ARRAY_HPP

#include <gec/utils/sequence.hpp>

namespace gec {

namespace bigint {

/** @brief TODO:
 */
template <class LIMB_T, size_t LIMB_N, size_t align = alignof(LIMB_T)>
class GEC_EMPTY_BASES Array {
  public:
    using LimbT = LIMB_T;
    const static size_t LimbN = LIMB_N;
    alignas(align) LIMB_T arr[LIMB_N];

    GEC_HD GEC_INLINE constexpr Array() noexcept : arr() {}
    GEC_HD
    GEC_INLINE constexpr explicit Array(const LIMB_T &limb) noexcept : arr() {
        *arr = limb;
    }

    template <size_t LIMB_M>
    GEC_HD GEC_INLINE constexpr Array(
        const Array<LIMB_T, LIMB_M> &GEC_RSTRCT other) noexcept
        : arr() {
        utils::fill_seq<(LIMB_N > LIMB_M ? LIMB_M : LIMB_N)>(arr, other.arr);
    }

    template <size_t LIMB_M>
    GEC_HD GEC_INLINE constexpr Array<LIMB_T, LIMB_M> &
    operator=(const Array<LIMB_T, LIMB_M> &GEC_RSTRCT other) {
        if (this != &other) {
            utils::fill_seq<(LIMB_N > LIMB_M ? LIMB_M : LIMB_N)>(arr,
                                                                 other.arr);
            utils::fill_seq_limb<(LIMB_N > LIMB_M ? LIMB_N - LIMB_M : 0),
                                 LIMB_T>(arr + LIMB_M, LIMB_T());
        }
        return other;
    }

    GEC_HD GEC_INLINE constexpr const LIMB_T *array() const { return arr; }
    GEC_HD GEC_INLINE constexpr LIMB_T *array() { return arr; }
};

/** @brief TODO:
 */
template <class LIMB_T, size_t LIMB_N, size_t align = alignof(LIMB_T)>
class GEC_EMPTY_BASES ArrayBE : public Array<LIMB_T, LIMB_N, align> {
    using Base = Array<LIMB_T, LIMB_N, align>;

  public:
    using Base::Base;

    template <typename... LIMBS,
              std::enable_if_t<(sizeof...(LIMBS) == LIMB_N &&
                                sizeof...(LIMBS) > 1)> * = nullptr>
    GEC_HD GEC_INLINE constexpr ArrayBE(const LIMBS &...limbs) noexcept {
        utils::fill_be<LIMB_T>(this->arr, limbs...);
    }
};

/** @brief TODO:
 */
template <class LIMB_T, size_t LIMB_N, size_t align = alignof(LIMB_T)>
class GEC_EMPTY_BASES ArrayLE : public Array<LIMB_T, LIMB_N, align> {
    using Base = Array<LIMB_T, LIMB_N, align>;

  public:
    using Base::Base;

    GEC_HD GEC_INLINE constexpr ArrayLE() noexcept : Base() {}

    template <typename... LIMBS,
              std::enable_if_t<(sizeof...(LIMBS) == LIMB_N &&
                                sizeof...(LIMBS) > 1)> * = nullptr>
    GEC_HD GEC_INLINE constexpr ArrayLE(const LIMBS &...limbs) noexcept {
        utils::fill_le<LIMB_T>(this->arr, limbs...);
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_DATA_ARRAY_HPP

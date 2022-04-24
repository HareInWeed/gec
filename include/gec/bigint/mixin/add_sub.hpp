#pragma once
#ifndef GEC_BIGINT_MIXIN_ADD_SUB_HPP
#define GEC_BIGINT_MIXIN_ADD_SUB_HPP

#include <gec/utils/arithmetic.hpp>
#include <gec/utils/crtp.hpp>
#include <gec/utils/sequence.hpp>

namespace gec {

namespace bigint {

/** @brief mixin that enables addition and substrcation operation
 */
template <class Core, typename LIMB_T, size_t LIMB_N>
class AddSubMixin : public CRTP<Core, AddSubMixin<Core, LIMB_T, LIMB_N>> {
  public:
    __host__ __device__ GEC_INLINE bool is_zero() const {
        return utils::seq_all_limb<LIMB_N, LIMB_T>(this->core().get_arr(), 0);
    }

    __host__ __device__ GEC_INLINE void set_zero() {
        utils::fill_seq_limb<LIMB_N>(this->core().get_arr(), 0);
    }

    /** @brief a + carry = b + c
     *
     * return the carry bit
     */
    static bool add(AddSubMixin &GEC_RSTRCT a, const AddSubMixin &GEC_RSTRCT b,
                    const AddSubMixin &GEC_RSTRCT c) {
        return utils::seq_add<LIMB_N>(a.core().get_arr(), b.core().get_arr(),
                                      c.core().get_arr());
    }

    /** @brief a + carry = a + b
     *
     * return the carry bit
     */
    static bool add(AddSubMixin &GEC_RSTRCT a,
                    const AddSubMixin &GEC_RSTRCT b) {
        return utils::seq_add<LIMB_N>(a.core().get_arr(), b.core().get_arr());
    }

    /** @brief a + borrow = b - c
     *
     * return the borrow bit
     */
    static bool sub(AddSubMixin &GEC_RSTRCT a, const AddSubMixin &GEC_RSTRCT b,
                    const AddSubMixin &GEC_RSTRCT c) {
        return utils::seq_sub<LIMB_N>(a.core().get_arr(), b.core().get_arr(),
                                      c.core().get_arr());
    }

    /** @brief a + borrow = a - b
     *
     * return the borrow bit
     */
    static bool sub(AddSubMixin &GEC_RSTRCT a,
                    const AddSubMixin &GEC_RSTRCT b) {
        return utils::seq_sub<LIMB_N>(a.core().get_arr(), b.core().get_arr());
    }

    AddSubMixin &add(const AddSubMixin &GEC_RSTRCT a,
                     const AddSubMixin &GEC_RSTRCT b) {
        add(*this, a, b);
        return *this;
    }
    AddSubMixin &add(const AddSubMixin &GEC_RSTRCT a) {
        add(*this, a);
        return *this;
    }

    AddSubMixin &sub(const AddSubMixin &GEC_RSTRCT a,
                     const AddSubMixin &GEC_RSTRCT b) {
        sub(*this, a, b);
        return *this;
    }
    AddSubMixin &sub(const AddSubMixin &GEC_RSTRCT a) {
        sub(*this, a);
        return *this;
    }

    AddSubMixin &operator+=(const AddSubMixin &GEC_RSTRCT a) {
        add(a);
        return *this;
    }

    AddSubMixin &operator-=(const AddSubMixin &GEC_RSTRCT a) {
        sub(a);
        return *this;
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_ADD_SUB_HPP
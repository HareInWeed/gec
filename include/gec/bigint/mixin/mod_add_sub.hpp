#pragma once
#ifndef GEC_BIGINT_MIXIN_MOD_ADD_SUB_HPP
#define GEC_BIGINT_MIXIN_MOD_ADD_SUB_HPP

#include <gec/utils/arithmetic.hpp>
#include <gec/utils/crtp.hpp>
#include <gec/utils/sequence.hpp>

namespace gec {

namespace bigint {

/** @brief mixin that enables addition and substrcation operation
 */
template <class Core, typename LIMB_T, size_t LIMB_N, const LIMB_T *MOD>
class ModAddSubMixin
    : public CRTP<Core, ModAddSubMixin<Core, LIMB_T, LIMB_N, MOD>> {
  public:
    __host__ __device__ GEC_INLINE bool is_zero() const {
        return utils::seq_all_limb<LIMB_N, LIMB_T>(this->core().get_arr(), 0);
    }

    __host__ __device__ GEC_INLINE void set_zero() {
        utils::fill_seq_limb<LIMB_N, LIMB_T>(this->core().get_arr(), 0);
    }

    /** @brief a = b + c (mod MOD)
     */
    static void add(ModAddSubMixin &GEC_RSTRCT a,
                    const ModAddSubMixin &GEC_RSTRCT b,
                    const ModAddSubMixin &GEC_RSTRCT c) {
        bool carry = utils::seq_add<LIMB_N>(
            a.core().get_arr(), b.core().get_arr(), c.core().get_arr());
        if (carry || utils::VtSeqCmp<LIMB_N, LIMB_T>::call(
                         a.core().get_arr(), MOD) != utils::CmpEnum::Lt) {
            utils::seq_sub<LIMB_N>(a.core().get_arr(), MOD);
        }
    }

    /** @brief a = - b (mod MOD)
     */
    static void neg(ModAddSubMixin &GEC_RSTRCT a,
                    const ModAddSubMixin &GEC_RSTRCT b) {
        if (b.is_zero()) {
            a.set_zero();
        } else {
            utils::seq_sub<LIMB_N>(a.core().get_arr(), MOD, b.core().get_arr());
        }
    }

    /** @brief a = b - c (mod MOD)
     */
    static void sub(ModAddSubMixin &GEC_RSTRCT a,
                    const ModAddSubMixin &GEC_RSTRCT b,
                    const ModAddSubMixin &GEC_RSTRCT c) {
        bool borrow = utils::seq_sub<LIMB_N>(
            a.core().get_arr(), b.core().get_arr(), c.core().get_arr());
        if (borrow) {
            utils::seq_add<LIMB_N>(a.core().get_arr(), MOD);
        }
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_MOD_ADD_SUB_HPP

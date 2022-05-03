#pragma once
#ifndef GEC_BIGINT_MIXIN_MOD_ADD_SUB_HPP
#define GEC_BIGINT_MIXIN_MOD_ADD_SUB_HPP

#include <gec/utils/arithmetic.hpp>
#include <gec/utils/crtp.hpp>
#include <gec/utils/sequence.hpp>

namespace gec {

namespace bigint {

/** @brief mixin that enables addition and substrcation operation
 *
 * TODO: detailed description
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
    static void add(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                    const Core &GEC_RSTRCT c) {
        bool carry =
            utils::seq_add<LIMB_N>(a.get_arr(), b.get_arr(), c.get_arr());
        if (carry || utils::VtSeqCmp<LIMB_N, LIMB_T>::call(a.get_arr(), MOD) !=
                         utils::CmpEnum::Lt) {
            utils::seq_sub<LIMB_N>(a.get_arr(), MOD);
        }
    }

    /** @brief a = - b (mod MOD)
     */
    static void neg(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b) {
        if (b.is_zero()) {
            a.set_zero();
        } else {
            utils::seq_sub<LIMB_N>(a.get_arr(), MOD, b.get_arr());
        }
    }

    /** @brief a = b - c (mod MOD)
     */
    static void sub(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                    const Core &GEC_RSTRCT c) {
        bool borrow =
            utils::seq_sub<LIMB_N>(a.get_arr(), b.get_arr(), c.get_arr());
        if (borrow) {
            utils::seq_add<LIMB_N>(a.get_arr(), MOD);
        }
    }
};

/** @brief mixin that enables addition and substrcation operation without
 * checking for carry bit
 *
 * TODO: detailed description
 */
template <class Core, typename LIMB_T, size_t LIMB_N, const LIMB_T *MOD>
class ModAddSubMixinCarryFree
    : public CRTP<Core, ModAddSubMixinCarryFree<Core, LIMB_T, LIMB_N, MOD>> {
  public:
    __host__ __device__ GEC_INLINE bool is_zero() const {
        return utils::seq_all_limb<LIMB_N, LIMB_T>(this->core().get_arr(), 0);
    }

    __host__ __device__ GEC_INLINE void set_zero() {
        utils::fill_seq_limb<LIMB_N, LIMB_T>(this->core().get_arr(), 0);
    }

    /** @brief a = b + c (mod MOD)
     */
    static void add(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                    const Core &GEC_RSTRCT c) {
        utils::seq_add<LIMB_N>(a.get_arr(), b.get_arr(), c.get_arr());
        if (utils::VtSeqCmp<LIMB_N, LIMB_T>::call(a.get_arr(), MOD) !=
            utils::CmpEnum::Lt) {
            utils::seq_sub<LIMB_N>(a.get_arr(), MOD);
        }
    }

    /** @brief a = - b (mod MOD)
     */
    static void neg(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b) {
        if (b.is_zero()) {
            a.set_zero();
        } else {
            utils::seq_sub<LIMB_N>(a.get_arr(), MOD, b.get_arr());
        }
    }

    /** @brief a = b - c (mod MOD)
     */
    static void sub(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                    const Core &GEC_RSTRCT c) {
        bool borrow =
            utils::seq_sub<LIMB_N>(a.get_arr(), b.get_arr(), c.get_arr());
        if (borrow) {
            utils::seq_add<LIMB_N>(a.get_arr(), MOD);
        }
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_MOD_ADD_SUB_HPP

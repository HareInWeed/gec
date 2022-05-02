#pragma once
#ifndef GEC_BIGINT_MIXIN_MONTGOMERY_HPP
#define GEC_BIGINT_MIXIN_MONTGOMERY_HPP

#include <gec/utils/arithmetic.hpp>
#include <gec/utils/crtp.hpp>
#include <gec/utils/sequence.hpp>

namespace gec {

namespace bigint {

/** @brief mixin that enables Montgomery Multiplication
 */
template <class Core, typename LIMB_T, size_t LIMB_N, const LIMB_T *MOD,
          LIMB_T MOD_P>
class Montgomery
    : public CRTP<Core, Montgomery<Core, LIMB_T, LIMB_N, MOD, MOD_P>> {
  public:
    __host__ __device__ static void add_mul(Montgomery &GEC_RSTRCT a,
                                            const Montgomery &GEC_RSTRCT b,
                                            const Montgomery &GEC_RSTRCT c) {
        using namespace utils;
        LIMB_T *a_arr = a.core().get_arr();
        const LIMB_T *b_arr = b.core().get_arr();
        const LIMB_T *c_arr = c.core().get_arr();

        bool carry = false;
        for (int i = 0; i < LIMB_N; ++i) {
            LIMB_T m = (a_arr[0] + b_arr[i] * c_arr[0]) * MOD_P;
            LIMB_T last0 = seq_add_mul_limb<LIMB_N>(a_arr, c_arr, b_arr[i]);
            LIMB_T last1 = seq_add_mul_limb<LIMB_N>(a_arr, MOD, m);
            carry = uint_add_with_carry(last0, last1, carry);

            seq_shift_right<LIMB_N, std::numeric_limits<LIMB_T>::digits>(a_arr);
            a_arr[LIMB_N - 1] = last0;
        }

        if (VtSeqCmp<LIMB_N, LIMB_T>::call(a_arr, MOD) != CmpEnum::Lt) {
            seq_sub<LIMB_N>(a_arr, MOD);
        }
    }

    __host__ __device__ static void mul(Montgomery &GEC_RSTRCT a,
                                        const Montgomery &GEC_RSTRCT b,
                                        const Montgomery &GEC_RSTRCT c) {
        utils::fill_seq_limb<LIMB_N>(a.core().get_arr(), LIMB_T(0));
        add_mul(a, b, c);
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_MONTGOMERY_HPP

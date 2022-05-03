#pragma once
#ifndef GEC_BIGINT_MIXIN_MONTGOMERY_HPP
#define GEC_BIGINT_MIXIN_MONTGOMERY_HPP

#include <gec/utils/arithmetic.hpp>
#include <gec/utils/crtp.hpp>
#include <gec/utils/sequence.hpp>

namespace gec {

namespace bigint {

/** @brief mixin that enables Montgomery multiplication
 *
 * TODO: detailed description
 */
template <class Core, typename LIMB_T, size_t LIMB_N, const LIMB_T *MOD,
          LIMB_T MOD_P, const LIMB_T *RR>
class Montgomery
    : public CRTP<Core, Montgomery<Core, LIMB_T, LIMB_N, MOD, MOD_P, RR>> {
  public:
    __host__ __device__ static void add_mul(Core &GEC_RSTRCT a,
                                            const Core &GEC_RSTRCT b,
                                            const Core &GEC_RSTRCT c) {
        using namespace utils;
        LIMB_T *a_arr = a.get_arr();
        const LIMB_T *b_arr = b.get_arr();
        const LIMB_T *c_arr = c.get_arr();

        bool carry = false;
        for (int i = 0; i < LIMB_N; ++i) {
            LIMB_T m = (a_arr[0] + b_arr[i] * c_arr[0]) * MOD_P;
            LIMB_T last0 = seq_add_mul_limb<LIMB_N>(a_arr, c_arr, b_arr[i]);
            LIMB_T last1 = seq_add_mul_limb<LIMB_N>(a_arr, MOD, m);
            carry = uint_add_with_carry(last0, last1, carry);

            seq_shift_right<LIMB_N, std::numeric_limits<LIMB_T>::digits>(a_arr);
            a_arr[LIMB_N - 1] = last0;
        }

        if (carry ||
            VtSeqCmp<LIMB_N, LIMB_T>::call(a_arr, MOD) != CmpEnum::Lt) {
            seq_sub<LIMB_N>(a_arr, MOD);
        }
    }

    __host__ __device__ static void mul(Core &GEC_RSTRCT a,
                                        const Core &GEC_RSTRCT b,
                                        const Core &GEC_RSTRCT c) {
        utils::fill_seq_limb<LIMB_N>(a.get_arr(), LIMB_T(0));
        add_mul(a, b, c);
    }

    __host__ __device__ static void inv(Core &GEC_RSTRCT a,
                                        const Core &GEC_RSTRCT b,
                                        Core &GEC_RSTRCT r, Core &GEC_RSTRCT s,
                                        Core &GEC_RSTRCT t) {
        utils::fill_seq<LIMB_N>(a.get_arr(), b.get_arr());
        inv(a, r, s, t);
    }

    __host__ __device__ static void inv(Core &GEC_RSTRCT a, Core &GEC_RSTRCT r,
                                        Core &GEC_RSTRCT s,
                                        Core &GEC_RSTRCT t) {
        using utils::CmpEnum;
        const auto &rr = *reinterpret_cast<const Core *>(RR);
        constexpr size_t LimbBit = std::numeric_limits<LIMB_T>::digits;
        constexpr size_t Bits = LimbBit * LIMB_N;
        constexpr LIMB_T mask = LIMB_T(1) << (LimbBit - 1);

        LIMB_T *a_arr = a.get_arr();
        LIMB_T *r_arr = r.get_arr();
        LIMB_T *s_arr = s.get_arr();
        LIMB_T *t_arr = t.get_arr();

        utils::fill_seq<LIMB_N>(r_arr, a_arr);
        s_arr[0] = LIMB_T(1);
        utils::fill_seq_limb<LIMB_N - 1>(s_arr + 1, LIMB_T(0));
        utils::fill_seq<LIMB_N>(t_arr, MOD);
        utils::fill_seq_limb<LIMB_N>(a_arr, LIMB_T(0));
        int k = 0;
        bool a_carry = false, s_carry = false;
        while (!utils::SeqEqLimb<LIMB_N, LIMB_T>::call(r_arr, 0)) {
            if (!(t_arr[0] & 0x1)) {
                utils::seq_shift_right<LIMB_N, 1>(t_arr);
                utils::seq_shift_left<LIMB_N, 1>(s_arr);
            } else if (!(r_arr[0] & 0x1)) {
                utils::seq_shift_right<LIMB_N, 1>(r_arr);
                utils::seq_shift_left<LIMB_N, 1>(a_arr);
            } else if (utils::VtSeqCmp<LIMB_N, LIMB_T>::call(t_arr, r_arr) ==
                       CmpEnum::Gt) {
                utils::seq_sub<LIMB_N>(t_arr, r_arr);
                utils::seq_shift_right<LIMB_N, 1>(t_arr);
                bool carry = utils::seq_add<LIMB_N>(a_arr, s_arr);
                a_carry = a_carry || s_carry || carry;
                s_carry = s_carry || bool(mask & s_arr[LIMB_N - 1]);
                utils::seq_shift_left<LIMB_N, 1>(s_arr);
            } else {
                utils::seq_sub<LIMB_N>(r_arr, t_arr);
                utils::seq_shift_right<LIMB_N, 1>(r_arr);
                bool carry = utils::seq_add<LIMB_N>(s_arr, a_arr);
                s_carry = a_carry || s_carry || carry;
                a_carry = a_carry || bool(mask & a_arr[LIMB_N - 1]);
                utils::seq_shift_left<LIMB_N, 1>(a_arr);
            }
            ++k;
        }
        if (a_carry ||
            utils::VtSeqCmp<LIMB_N, LIMB_T>::call(a_arr, MOD) != CmpEnum::Lt) {
            utils::seq_sub<LIMB_N>(a_arr, MOD);
        }
        utils::seq_sub<LIMB_N>(s_arr, MOD, a_arr);
        if (k < Bits) {
            mul(t, s, rr);
            k += Bits;

            mul(s, t, rr);

            utils::fill_seq_limb<LIMB_N>(r_arr, LIMB_T(0));
            int bit = 2 * Bits - k;
            r_arr[bit / LimbBit] = LIMB_T(1) << (bit % LimbBit);
            mul(a, s, r);
        } else {
            mul(t, s, rr);

            utils::fill_seq_limb<LIMB_N>(r_arr, LIMB_T(0));
            int bit = 2 * Bits - k;
            r_arr[bit / LimbBit] = LIMB_T(1) << (bit % LimbBit);
            mul(a, t, r);
        }
    }
};

/** @brief mixin that enables Montgomery multiplication without checking carry
 * bit
 *
 * TODO: detailed description
 */
template <class Core, typename LIMB_T, size_t LIMB_N, const LIMB_T *MOD,
          LIMB_T MOD_P, const LIMB_T *RR>
class MontgomeryCarryFree
    : public CRTP<Core,
                  MontgomeryCarryFree<Core, LIMB_T, LIMB_N, MOD, MOD_P, RR>> {
  public:
    __host__ __device__ static void add_mul(Core &GEC_RSTRCT a,
                                            const Core &GEC_RSTRCT b,
                                            const Core &GEC_RSTRCT c) {
        using namespace utils;
        LIMB_T *a_arr = a.get_arr();
        const LIMB_T *b_arr = b.get_arr();
        const LIMB_T *c_arr = c.get_arr();

        for (int i = 0; i < LIMB_N; ++i) {
            LIMB_T m = (a_arr[0] + b_arr[i] * c_arr[0]) * MOD_P;
            LIMB_T last(0);
            last += seq_add_mul_limb<LIMB_N>(a_arr, c_arr, b_arr[i]);
            last += seq_add_mul_limb<LIMB_N>(a_arr, MOD, m);

            seq_shift_right<LIMB_N, std::numeric_limits<LIMB_T>::digits>(a_arr);
            a_arr[LIMB_N - 1] = last;
        }

        if (VtSeqCmp<LIMB_N, LIMB_T>::call(a_arr, MOD) != CmpEnum::Lt) {
            seq_sub<LIMB_N>(a_arr, MOD);
        }
    }

    __host__ __device__ static void mul(Core &GEC_RSTRCT a,
                                        const Core &GEC_RSTRCT b,
                                        const Core &GEC_RSTRCT c) {
        utils::fill_seq_limb<LIMB_N>(a.get_arr(), LIMB_T(0));
        add_mul(a, b, c);
    }

    __host__ __device__ static void inv(Core &GEC_RSTRCT a,
                                        const Core &GEC_RSTRCT b,
                                        Core &GEC_RSTRCT r, Core &GEC_RSTRCT s,
                                        Core &GEC_RSTRCT t) {
        utils::fill_seq<LIMB_N>(a.get_arr(), b.get_arr());
        inv(a, r, s, t);
    }

    __host__ __device__ static void inv(Core &GEC_RSTRCT a, Core &GEC_RSTRCT r,
                                        Core &GEC_RSTRCT s,
                                        Core &GEC_RSTRCT t) {
        using utils::CmpEnum;
        const auto &rr = *reinterpret_cast<const Core *>(RR);
        constexpr size_t LimbBit = std::numeric_limits<LIMB_T>::digits;
        constexpr size_t Bits = LimbBit * LIMB_N;
        constexpr LIMB_T mask = LIMB_T(1) << (LimbBit - 1);

        LIMB_T *a_arr = a.get_arr();
        LIMB_T *r_arr = r.get_arr();
        LIMB_T *s_arr = s.get_arr();
        LIMB_T *t_arr = t.get_arr();

        utils::fill_seq<LIMB_N>(r_arr, a_arr);
        s_arr[0] = LIMB_T(1);
        utils::fill_seq_limb<LIMB_N - 1>(s_arr + 1, LIMB_T(0));
        utils::fill_seq<LIMB_N>(t_arr, MOD);
        utils::fill_seq_limb<LIMB_N>(a_arr, LIMB_T(0));
        int k = 0;
        while (!utils::SeqEqLimb<LIMB_N, LIMB_T>::call(r_arr, 0)) {
            if (!(t_arr[0] & 0x1)) {
                utils::seq_shift_right<LIMB_N, 1>(t_arr);
                utils::seq_shift_left<LIMB_N, 1>(s_arr);
            } else if (!(r_arr[0] & 0x1)) {
                utils::seq_shift_right<LIMB_N, 1>(r_arr);
                utils::seq_shift_left<LIMB_N, 1>(a_arr);
            } else if (utils::VtSeqCmp<LIMB_N, LIMB_T>::call(t_arr, r_arr) ==
                       CmpEnum::Gt) {
                utils::seq_sub<LIMB_N>(t_arr, r_arr);
                utils::seq_shift_right<LIMB_N, 1>(t_arr);
                utils::seq_add<LIMB_N>(a_arr, s_arr);
                utils::seq_shift_left<LIMB_N, 1>(s_arr);
            } else {
                utils::seq_sub<LIMB_N>(r_arr, t_arr);
                utils::seq_shift_right<LIMB_N, 1>(r_arr);
                utils::seq_add<LIMB_N>(s_arr, a_arr);
                utils::seq_shift_left<LIMB_N, 1>(a_arr);
            }
            ++k;
        }
        if (utils::VtSeqCmp<LIMB_N, LIMB_T>::call(a_arr, MOD) != CmpEnum::Lt) {
            utils::seq_sub<LIMB_N>(a_arr, MOD);
        }
        utils::seq_sub<LIMB_N>(s_arr, MOD, a_arr);
        if (k < Bits) {
            mul(t, s, rr);
            k += Bits;

            mul(s, t, rr);

            utils::fill_seq_limb<LIMB_N>(r_arr, LIMB_T(0));
            int bit = 2 * Bits - k;
            r_arr[bit / LimbBit] = LIMB_T(1) << (bit % LimbBit);
            mul(a, s, r);
        } else {
            mul(t, s, rr);

            utils::fill_seq_limb<LIMB_N>(r_arr, LIMB_T(0));
            int bit = 2 * Bits - k;
            r_arr[bit / LimbBit] = LIMB_T(1) << (bit % LimbBit);
            mul(a, t, r);
        }
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_MONTGOMERY_HPP

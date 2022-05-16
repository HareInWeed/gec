#pragma once
#ifndef GEC_BIGINT_MIXIN_MONTGOMERY_HPP
#define GEC_BIGINT_MIXIN_MONTGOMERY_HPP

#include <gec/utils/arithmetic.hpp>
#include <gec/utils/context_check.hpp>
#include <gec/utils/crtp.hpp>
#include <gec/utils/sequence.hpp>

namespace gec {

namespace bigint {

/** @brief mixin that enables Montgomery multiplication
 *
 * require `Core::set_zero`, `Core::set_one`, `Core::set_pow2` methods
 */
template <class Core, typename LIMB_T, size_t LIMB_N, const LIMB_T *MOD,
          LIMB_T MOD_P, const LIMB_T *RR, const LIMB_T *OneR>
class Montgomery
    : public CRTP<Core,
                  Montgomery<Core, LIMB_T, LIMB_N, MOD, MOD_P, RR, OneR>> {
  public:
    __host__ __device__ GEC_INLINE static const Core &r_sqr() {
        return *reinterpret_cast<const Core *>(RR);
    }
    __host__ __device__ GEC_INLINE static const Core &one_r() {
        return *reinterpret_cast<const Core *>(OneR);
    }

    bool is_mul_id() const {
        return utils::VtSeqAll<LIMB_N, LIMB_T, utils::ops::Eq<LIMB_T>>::call(
            this->core().array(), OneR);
    }
    void set_mul_id() { utils::fill_seq<LIMB_N>(this->core().array(), OneR); }

    __host__ __device__ static void add_mul(Core &GEC_RSTRCT a,
                                            const Core &GEC_RSTRCT b,
                                            const Core &GEC_RSTRCT c) {
        using namespace utils;
        LIMB_T *a_arr = a.array();
        const LIMB_T *b_arr = b.array();
        const LIMB_T *c_arr = c.array();

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

    __host__ __device__ GEC_INLINE static void mul(Core &GEC_RSTRCT a,
                                                   const Core &GEC_RSTRCT b,
                                                   const Core &GEC_RSTRCT c) {
        a.set_zero();
        add_mul(a, b, c);
    }

    template <typename CTX>
    __host__ __device__ GEC_INLINE static void
    inv(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b, CTX &GEC_RSTRCT ctx) {
        a = b;
        inv(a, ctx);
    }

    template <typename CTX>
    __host__ __device__ static void inv(Core &GEC_RSTRCT a,
                                        CTX &GEC_RSTRCT ctx) {
        GEC_CTX_CAP(CTX, 3);

        using utils::CmpEnum;
        const auto &rr = *reinterpret_cast<const Core *>(RR);
        constexpr size_t LimbBit = std::numeric_limits<LIMB_T>::digits;
        constexpr size_t Bits = LimbBit * LIMB_N;
        constexpr LIMB_T mask = LIMB_T(1) << (LimbBit - 1);

        Core &r = ctx.template get<0>();
        Core &s = ctx.template get<1>();
        Core &t = ctx.template get<2>();

        LIMB_T *a_arr = a.array();
        LIMB_T *r_arr = r.array();
        LIMB_T *s_arr = s.array();
        LIMB_T *t_arr = t.array();

        r = a;
        s.set_one();
        utils::fill_seq<LIMB_N>(t_arr, MOD);
        a.set_zero();
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

            r.set_pow2(2 * Bits - k);
            mul(a, s, r);
        } else {
            mul(t, s, rr);

            r.set_pow2(2 * Bits - k);
            mul(a, t, r);
        }
    }
};

/** @brief mixin that enables Montgomery multiplication without checking carry
 * bit
 *
 * Note this mixin does not check overflow during calculation.
 *
 * If `Core` can hold twice as `MOD`, than replacing `ModAddSubMixin` with this
 * mixin might have a performance boost. Otherwise, the mixin could lead to
 * incorrect result.
 *
 * require `Core::set_zero`, `Core::set_one`, `Core::set_pow2` methods
 */
template <class Core, typename LIMB_T, size_t LIMB_N, const LIMB_T *MOD,
          LIMB_T MOD_P, const LIMB_T *RR, const LIMB_T *OneR>
class MontgomeryCarryFree
    : public CRTP<Core, MontgomeryCarryFree<Core, LIMB_T, LIMB_N, MOD, MOD_P,
                                            RR, OneR>> {
  public:
    bool is_mul_id() const {
        return utils::VtSeqAll<LIMB_N, LIMB_T, utils::ops::Eq<LIMB_T>>::call(
            this->core().array(), OneR);
    }
    void set_mul_id() { utils::fill_seq<LIMB_N>(this->core().array(), OneR); }

    __host__ __device__ static void add_mul(Core &GEC_RSTRCT a,
                                            const Core &GEC_RSTRCT b,
                                            const Core &GEC_RSTRCT c) {
        using namespace utils;
        LIMB_T *a_arr = a.array();
        const LIMB_T *b_arr = b.array();
        const LIMB_T *c_arr = c.array();

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

    __host__ __device__ GEC_INLINE static void mul(Core &GEC_RSTRCT a,
                                                   const Core &GEC_RSTRCT b,
                                                   const Core &GEC_RSTRCT c) {
        a.set_zero();
        add_mul(a, b, c);
    }

    template <typename CTX>
    __host__ __device__ GEC_INLINE static void
    inv(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b, CTX &GEC_RSTRCT ctx) {
        a = b;
        inv(a, ctx);
    }

    template <typename CTX>
    __host__ __device__ static void inv(Core &GEC_RSTRCT a,
                                        CTX &GEC_RSTRCT ctx) {
        GEC_CTX_CAP(CTX, 3);

        using utils::CmpEnum;
        const auto &rr = *reinterpret_cast<const Core *>(RR);
        constexpr size_t LimbBit = std::numeric_limits<LIMB_T>::digits;
        constexpr size_t Bits = LimbBit * LIMB_N;
        constexpr LIMB_T mask = LIMB_T(1) << (LimbBit - 1);

        Core &r = ctx.template get<0>();
        Core &s = ctx.template get<1>();
        Core &t = ctx.template get<2>();

        LIMB_T *a_arr = a.array();
        LIMB_T *r_arr = r.array();
        LIMB_T *s_arr = s.array();
        LIMB_T *t_arr = t.array();

        r = a;
        s.set_one();
        utils::fill_seq<LIMB_N>(t_arr, MOD);
        a.set_zero();
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

            r.set_pow2(2 * Bits - k);
            mul(a, s, r);
        } else {
            mul(t, s, rr);

            r.set_pow2(2 * Bits - k);
            mul(a, t, r);
        }
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_MONTGOMERY_HPP

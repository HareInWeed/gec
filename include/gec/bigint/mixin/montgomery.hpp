#pragma once
#ifndef GEC_BIGINT_MIXIN_MONTGOMERY_HPP
#define GEC_BIGINT_MIXIN_MONTGOMERY_HPP

#include <immintrin.h>

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

    __host__ __device__ GEC_INLINE static void
    to_montgomery(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b) {
        a.set_zero();
        add_mul(a, b, r_sqr());
    }
    __host__ __device__ GEC_INLINE static void
    from_montgomery(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b) {
        a.set_zero();
        using namespace utils;

        LIMB_T *a_arr = a.array();
        const LIMB_T *b_arr = b.array();

        fill_seq<LIMB_N>(a_arr, b_arr);

        for (int i = 0; i < LIMB_N; ++i) {
            LIMB_T m = a_arr[0] * MOD_P;
            LIMB_T last = seq_add_mul_limb<LIMB_N>(a_arr, MOD, m);

            seq_shift_right<LIMB_N, std::numeric_limits<LIMB_T>::digits>(a_arr);
            a_arr[LIMB_N - 1] = last;
        }

        if (VtSeqCmp<LIMB_N, LIMB_T>::call(a_arr, MOD) != CmpEnum::Lt) {
            seq_sub<LIMB_N>(a_arr, MOD);
        }
    }

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

    __host__ __device__ GEC_INLINE static void
    to_montgomery(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b) {
        a.set_zero();
        add_mul(a, b, r_sqr());
    }
    __host__ __device__ GEC_INLINE static void
    from_montgomery(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b) {
        a.set_zero();
        using namespace utils;

        LIMB_T *a_arr = a.array();
        const LIMB_T *b_arr = b.array();

        fill_seq<LIMB_N>(a_arr, b_arr);

        for (int i = 0; i < LIMB_N; ++i) {
            LIMB_T m = a_arr[0] * MOD_P;
            LIMB_T last = seq_add_mul_limb<LIMB_N>(a_arr, MOD, m);

            seq_shift_right<LIMB_N, std::numeric_limits<LIMB_T>::digits>(a_arr);
            a_arr[LIMB_N - 1] = last;
        }

        if (VtSeqCmp<LIMB_N, LIMB_T>::call(a_arr, MOD) != CmpEnum::Lt) {
            seq_sub<LIMB_N>(a_arr, MOD);
        }
    }

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

/** @brief mixin that enables Montgomery Multiplication with AVX2
 */
template <class Core, typename LIMB_T, size_t LIMB_N,
          const LIMB_T (&MOD)[LIMB_N], LIMB_T MOD_P>
class AVX2Montgomery
    : public CRTP<Core, AVX2Montgomery<Core, LIMB_T, LIMB_N, MOD, MOD_P>> {
  public:
    __host__ static void add_mul(AVX2Montgomery &GEC_RSTRCT a,
                                 const AVX2Montgomery &GEC_RSTRCT b,
                                 const AVX2Montgomery &GEC_RSTRCT c);

    __host__ static void mul(AVX2Montgomery &GEC_RSTRCT a,
                             const AVX2Montgomery &GEC_RSTRCT b,
                             const AVX2Montgomery &GEC_RSTRCT c);
};

/** @brief mixin that enables Montgomery Multiplication with AVX2
 */
template <class Core, const uint32_t (&MOD)[8], uint32_t MOD_P>
class AVX2Montgomery<Core, uint32_t, 8, MOD, MOD_P>
    : public CRTP<Core, AVX2Montgomery<Core, uint32_t, 8, MOD, MOD_P>> {
    using LIMB_T = uint32_t;
    constexpr static size_t LIMB_N = 8;

    __host__ GEC_INLINE static __m256i add_limbs(__m256i &a, const __m256i &b,
                                                 const __m256i &c,
                                                 const __m256i &least_mask) {
        __m256i m = _mm256_max_epu32(b, c);
        a = _mm256_add_epi32(b, c);
        return _mm256_andnot_si256(
            _mm256_cmpeq_epi32(_mm256_max_epu32(a, m), a), least_mask);
    }

    __host__ GEC_INLINE static void
    mul_limbs(__m256i &l, __m256i &h, const __m256i &a, const __m256i &b) {
        __m256i a_odd = _mm256_shuffle_epi32(a, 0xf5);
        __m256i b_odd = _mm256_shuffle_epi32(b, 0xf5);
        __m256i p_even = _mm256_mul_epu32(a, b);
        __m256i p_odd = _mm256_mul_epu32(a_odd, b_odd);
        __m256i lo = _mm256_unpacklo_epi32(p_even, p_odd);
        __m256i hi = _mm256_unpackhi_epi32(p_even, p_odd);
        l = _mm256_unpacklo_epi64(lo, hi);
        h = _mm256_unpackhi_epi64(lo, hi);
    }

  public:
    __host__ static void add_mul(AVX2Montgomery &GEC_RSTRCT a,
                                 const AVX2Montgomery &GEC_RSTRCT b,
                                 const AVX2Montgomery &GEC_RSTRCT c) {
        using namespace utils;
        using V = __m256i *;
        using CV = const __m256i *;

        constexpr static uint32_t cir_right[8] = {1, 2, 3, 4, 5, 6, 7, 0};
        constexpr static uint32_t least_mask[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        uint32_t carries[8];

        LIMB_T *a_arr = a.core().get_arr();
        const LIMB_T *b_arr = b.core().get_arr();
        const LIMB_T *c_arr = c.core().get_arr();

        __m256i lm = _mm256_loadu_si256(reinterpret_cast<CV>(least_mask));
        __m256i cr = _mm256_loadu_si256(reinterpret_cast<CV>(cir_right));
        __m256i vm = _mm256_loadu_si256(reinterpret_cast<CV>(MOD));
        __m256i va = _mm256_loadu_si256(reinterpret_cast<CV>(a_arr));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<CV>(b_arr));
        __m256i vc = _mm256_loadu_si256(reinterpret_cast<CV>(c_arr));
        __m256i c0 = _mm256_broadcastd_epi32(_mm256_castsi256_si128(vc));
        __m256i mp = _mm256_set1_epi32(static_cast<int>(MOD_P));
        __m256i carry = _mm256_setzero_si256();

        for (int i = 0; i < LIMB_N; ++i) {
            __m256i vl, vh1, vh2, new_carry;
            __m256i bi = _mm256_broadcastd_epi32(_mm256_castsi256_si128(vb));
            __m256i m = _mm256_mullo_epi32(
                _mm256_add_epi32(
                    _mm256_broadcastd_epi32(_mm256_castsi256_si128(va)),
                    _mm256_mullo_epi32(bi, c0)),
                mp);

            mul_limbs(vl, vh1, bi, vc);
            new_carry = add_limbs(va, va, vl, lm);
            carry = _mm256_add_epi32(carry, new_carry);

            mul_limbs(vl, vh2, m, vm);
            new_carry = add_limbs(va, va, vl, lm);
            carry = _mm256_add_epi32(carry, new_carry);

            va = _mm256_permutevar8x32_epi32(va, cr);
            carry = add_limbs(va, va, carry, lm);

            new_carry = add_limbs(va, va, vh1, lm);
            carry = _mm256_add_epi32(carry, new_carry);
            new_carry = add_limbs(va, va, vh2, lm);
            carry = _mm256_add_epi32(carry, new_carry);

            vb = _mm256_permutevar8x32_epi32(vb, cr);
        }

        _mm256_storeu_si256(reinterpret_cast<V>(carries), carry);
        _mm256_storeu_si256(reinterpret_cast<V>(a_arr), va);

        seq_add<LIMB_N - 1>(a_arr + 1, carries);

        if (VtSeqCmp<LIMB_N, LIMB_T>::call(a_arr, MOD) != CmpEnum::Lt) {
            seq_sub<LIMB_N>(a_arr, MOD);
        }
    }

    __host__ static void mul(AVX2Montgomery &GEC_RSTRCT a,
                             const AVX2Montgomery &GEC_RSTRCT b,
                             const AVX2Montgomery &GEC_RSTRCT c) {
        utils::fill_seq_limb<LIMB_N>(a.core().get_arr(), LIMB_T(0));
        add_mul(a, b, c);
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_MONTGOMERY_HPP

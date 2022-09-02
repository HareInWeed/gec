#pragma once
#ifndef GEC_BIGINT_MIXIN_DIVISION_HPP
#define GEC_BIGINT_MIXIN_DIVISION_HPP

#include <gec/utils/arithmetic.hpp>
#include <gec/utils/basic.hpp>
#include <gec/utils/crtp.hpp>
#include <gec/utils/misc.hpp>
#include <gec/utils/sequence.hpp>

namespace gec {

namespace bigint {

namespace _division_ {

GEC_HD GEC_INLINE bool multi_gt() { return false; }
template <typename T, typename... Args>
GEC_HD GEC_INLINE bool multi_gt(const T &GEC_RSTRCT a, const T &GEC_RSTRCT b,
                                const Args &GEC_RSTRCT... args) {
    return a > b || (a == b && multi_gt(args...));
}

template <typename T>
GEC_HD GEC_INLINE bool seq_add(T *GEC_RSTRCT a, const T *GEC_RSTRCT b,
                               size_t n) {
    using namespace ::gec::utils;
    bool carry = false;
    for (size_t k = 0; k < n; ++k) {
        carry = uint_add_with_carry(a[k], b[k], carry);
    }
    return carry;
}

template <typename T>
GEC_HD GEC_INLINE bool seq_sub(T *GEC_RSTRCT a, const T *GEC_RSTRCT b,
                               size_t n) {
    using namespace ::gec::utils;
    bool borrow = false;
    for (size_t k = 0; k < n; ++k) {
        borrow = uint_sub_with_borrow(a[k], b[k], borrow);
    }
    return borrow;
}

template <typename T>
GEC_HD GEC_INLINE T seq_add_mul_limb(T *GEC_RSTRCT a, const T *GEC_RSTRCT b,
                                     size_t n, const T &x) {
    using namespace ::gec::utils;
    T l0, h0, l1, h1;
    bool carry0 = false, carry1 = false;
    // TODO: refactor the loop condition
    for (int i = 0; i < int(n) - 2; i += 2) { // deal with case N < 3, not ideal
        uint_mul_lh(l0, h0, b[i], x);
        carry0 = uint_add_with_carry(a[i + 1], h0,
                                     uint_add_with_carry(a[i], l0, carry0));

        uint_mul_lh(l1, h1, b[i + 1], x);
        carry1 = uint_add_with_carry(a[i + 2], h1,
                                     uint_add_with_carry(a[i + 1], l1, carry1));
    }

    T last_limb;
    if (n & 0x1) { // N is odd
        uint_mul_lh(l0, h0, b[n - 1], x);
        last_limb = h0 +
                    static_cast<T>(uint_add_with_carry(a[n - 1], l0, carry0)) +
                    static_cast<T>(carry1);
    } else { // N is even
        uint_mul_lh(l0, h0, b[n - 2], x);
        carry0 = uint_add_with_carry(a[n - 1], h0,
                                     uint_add_with_carry(a[n - 2], l0, carry0));

        uint_mul_lh(l1, h1, b[n - 1], x);
        last_limb = h1 +
                    static_cast<T>(uint_add_with_carry(a[n - 1], l1, carry1)) +
                    static_cast<T>(carry0);
    }

    return last_limb;
}

template <bool NeedR, typename TT, typename T, size_t I>
struct CastSingleDivRemHelper {
    GEC_HD GEC_INLINE static void call(T *GEC_RSTRCT q, T &GEC_RSTRCT r,
                                       const T *GEC_RSTRCT a,
                                       const T &GEC_RSTRCT b) {
        using namespace gec::utils;
        constexpr size_t bits = type_bits<T>::value;
        TT t = (TT(r) << bits) | TT(a[I]);
        q[I] = t / b;
        r = t % b;
        CastSingleDivRemHelper<NeedR, TT, T, I - 1>::call(q, r, a, b);
    }
};
template <bool NeedR, typename TT, typename T>
struct CastSingleDivRemHelper<NeedR, TT, T, 0> {
    GEC_HD GEC_INLINE static void call(T *GEC_RSTRCT q, T &GEC_RSTRCT r,
                                       const T *GEC_RSTRCT a,
                                       const T &GEC_RSTRCT b) {
        using namespace gec::utils;
        constexpr size_t bits = type_bits<T>::value;
        TT t = (TT(r) << bits) | TT(a[0]);
        q[0] = t / b;
        if (NeedR) {
            r = t % b;
        }
    }
};

/**
 * @brief single limb division with remainder, casting method
 *
 * limb divisions are performed by casting two limbs into a larger type
 *
 * @tparam NeedR flag for computing remainder, if false, the remainder
 *               computation step will be skipped
 * @tparam TT type used in limb casting, whose bit capacity must be at least
 *            twice the size of `T`
 * @tparam N limb length
 * @param q quotient, \f$\lfoor a / b \rfloor\f$
 * @param r remainder, single limb, \f$a - q b\f$, if `NeedR == false`, then the
 *          result value of `r` is undefined
 * @param a dividend
 * @param b divisor, single limb
 * @param ctx context
 */
template <bool NeedR, typename TT, size_t N, typename T>
GEC_HD void cast_single_div_rem(T *GEC_RSTRCT q, T &GEC_RSTRCT r,
                                const T *GEC_RSTRCT a, const T &GEC_RSTRCT b) {
    r = 0;
    CastSingleDivRemHelper<NeedR, TT, T, N - 1>::call(q, r, a, b);
}

/**
 * @brief workaround for "calling a constexpr __host__ function" warning
 */
template <typename T>
struct TypeMax {
    static constexpr T value = std::numeric_limits<T>::max();
};

/**
 * @brief division with remainder, casting method
 *
 * limb divisions are performed by casting two limbs into a larger type
 *
 * @tparam NeedQ flag for computing quotient, if false, the quotient
 *               computation step will be skipped
 * @tparam NeedR flag for computing remainder, if false, the remainder
 *               computation step will be skipped
 * @tparam TT type used in limb casting, whose bit capacity must be at least
 *            twice the size of `T`
 * @tparam N limb length
 * @tparam T limb type
 * @param[out] q quotient, \f$\lfoor a / b \rfloor\f$
 * @param[in,out] a dividend, will hold the remainder, \f$a - q b\f$,
 *                  if `NeedR == true`
 * @param[in] b divisor
 * @param qb temporary buffer for holding intermediate result
 */
template <bool NeedQ, bool NeedR, typename TT, size_t N, typename T>
GEC_HD static void cast_div_rem(T *GEC_RSTRCT q, T *GEC_RSTRCT a,
                                T *GEC_RSTRCT b, T *GEC_RSTRCT qb) {
    using namespace gec::utils;

    constexpr size_t bits = type_bits<T>::value;
    constexpr T max_limb = TypeMax<T>::value;

    bool flag;

    size_t n = 0;
    while (n < N && b[n]) {
        ++n;
    }

    if (n == 1) {
        T r0;
        cast_single_div_rem<NeedR, TT, N>(q, r0, a, b[0]);
        if (NeedR) {
            fill_seq_limb<N>(a, T(0));
            a[0] = r0;
        }
        return;
    }

    // ----- normalization -----

    size_t shift = count_leading_zeros(b[n - 1]);
    T a_last = a[N - 1] >> (bits - shift);
    a_last = shift ? a_last : T(0);
    seq_shift_left<N>(a, shift);
    seq_shift_left<N>(b, shift);

    T q_est, b2, b1, b0;
    TT q_est2;
    size_t i = N - n;

    if (NeedQ) {
        fill_seq_limb<N>(q, T(0));
    }

    // ----- i = m -----

    // min((a[n + m] * base + a[n + m - 1]) / b[n - 1], base - 1)
    q_est2 = ((TT(a_last) << bits) | TT(a[N - 1])) / b[n - 1];
    q_est = (q_est2 > max_limb) ? max_limb : q_est2;

    // q_est * (b[n - 1] * base + b[n - 2])
    uint_mul_lh(b0, b1, q_est, b[n - 2]);
    T b1p;
    uint_mul_lh(b1p, b2, q_est, b[n - 1]);
    b2 += T(uint_add_with_carry(b1, b1p, false));

    while (multi_gt(b2, a_last,      // 2
                    b1, a[N - 1],    // 1
                    b0, a[N - 2])) { // 0
        --q_est;
        b2 -= T(uint_sub_with_borrow(
            b1, b[n - 1], uint_sub_with_borrow(b0, b[n - 2], false)));
    }

    // q_est * b[...]
    fill_seq_limb<N>(qb, T(0));
    T qb_last = seq_add_mul_limb(qb, b, n, q_est);
    // a[...] -= q_est * b[...]
    flag = uint_sub_with_borrow(a_last, qb_last, seq_sub(a + i, qb, n));

    // a[...] < 0
    if (flag) {
        if (NeedQ) {
            --q_est;
        }
        // a[...] += b[...]
        uint_sub_with_borrow(a_last, T(0), seq_add(a + i, b, n));
    }

    if (NeedQ) {
        q[i] = q_est;
    }

    // ----- i = m - 1 to 0 -----
    if (i > 0) {
        size_t in = i + n;
        do {
            --i;
            --in;

            // min((a[n + m] * base + a[n + m - 1]) / b[n - 1], base - 1)
            q_est2 = ((TT(a[in]) << bits) | TT(a[in - 1])) / b[n - 1];
            q_est = (q_est2 > max_limb) ? max_limb : q_est2;

            // q_est * (b[n - 1] * base + b[n - 2])
            uint_mul_lh(b0, b1, q_est, b[n - 2]);
            T b1p;
            uint_mul_lh(b1p, b2, q_est, b[n - 1]);
            b2 += T(uint_add_with_carry(b1, b1p, false));

            while (multi_gt(b2, a[in],        // 2
                            b1, a[in - 1],    // 1
                            b0, a[in - 2])) { // 0
                --q_est;
                b2 -= T(uint_sub_with_borrow(
                    b1, b[n - 1], uint_sub_with_borrow(b0, b[n - 2], false)));
            }

            // q_est * b[...]
            fill_seq_limb<N>(qb, T(0));
            T qb_last = seq_add_mul_limb(qb, b, n, q_est);
            // a[...] -= q_est * b[...]
            flag = uint_sub_with_borrow(a[in], qb_last, seq_sub(a + i, qb, n));

            // a[...] < 0
            if (flag) {
                if (NeedQ) {
                    --q_est;
                }
                // a[...] += b[...]
                uint_sub_with_borrow(a[in], T(0), seq_add(a + i, b, n));
            }

            if (NeedQ) {
                q[i] = q_est;
            }
        } while (i > 0);
    }

    // ----- denormalization -----
    if (NeedR) {
        seq_shift_right<N>(a, shift);
        T r_last = a[N - 1] | (a_last << (bits - shift));
        a[N - 1] = shift ? r_last : a[N - 1];
    }
}

template <bool NeedR, typename T, size_t I>
struct SplitSingleDivRemHelper {
    GEC_HD GEC_INLINE static void call(T *GEC_RSTRCT q, T &GEC_RSTRCT r,
                                       const T *GEC_RSTRCT a,
                                       const T &GEC_RSTRCT b) {
        using namespace gec::utils;
        constexpr size_t bits = type_bits<T>::value;
        constexpr size_t half_bits = bits / 2;
        constexpr T mask_l = LowerKMask<T, half_bits>::value;
        T t;

        t = (r << half_bits) | (a[I] >> half_bits);
        q[I] = t / b;
        r = t % b;

        t = (r << half_bits) | (a[I] & mask_l);
        q[I] = t / b;
        r = t % b;

        SplitSingleDivRemHelper<NeedR, T, I - 1>::call(q, r, a, b);
    }
};
template <bool NeedR, typename T>
struct SplitSingleDivRemHelper<NeedR, T, 0> {
    GEC_HD GEC_INLINE static void call(T *GEC_RSTRCT q, T &GEC_RSTRCT r,
                                       const T *GEC_RSTRCT a,
                                       const T &GEC_RSTRCT b) {
        using namespace gec::utils;
        constexpr size_t bits = type_bits<T>::value;
        constexpr size_t half_bits = bits / 2;
        constexpr T mask_l = LowerKMask<T, half_bits>::value;
        T t;

        t = (r << half_bits) | (a[0] >> half_bits);
        q[0] = t / b;
        r = t % b;

        t = (r << half_bits) | (a[0] & mask_l);
        q[0] = t / b;
        if (NeedR) {
            r = t % b;
        }
    }
};

/**
 * @brief single limb division with remainder, spliting method
 *
 * limb divisions are performed by spliting a limb into two half limb
 *
 * @tparam NeedR flag for computing remainder, if false, the remainder
 *               computation step will be skipped
 * @tparam N limb length
 * @tparam T limb type
 * @param q quotient, \f$\lfoor a / b \rfloor\f$
 * @param r remainder, single limb, \f$a - q b\f$, if `NeedR == false`, then the
 *          result value of `r` is undefined
 * @param a dividend
 * @param b divisor, single limb, if `b > `
 * @param ctx context
 */
template <bool NeedR, size_t N, typename T>
GEC_HD void split_single_div_rem(T *GEC_RSTRCT q, T &GEC_RSTRCT r,
                                 const T *GEC_RSTRCT a, const T &GEC_RSTRCT b) {
    r = 0;
    SplitSingleDivRemHelper<NeedR, T, N - 1>::call(q, r, a, b);
}

template <size_t I, typename T, typename HT>
struct SplitDivRemHelper {
    GEC_HD GEC_INLINE static void split(HT *h_arr, const T *arr) {
        using namespace ::gec::utils;
        constexpr size_t shift = type_bits<T>::value / 2;
        h_arr[2 * I + 1] = arr[I] >> shift;
        h_arr[2 * I] = arr[I];
        SplitDivRemHelper<I - 1, T, HT>::split(h_arr, arr);
    }

    GEC_HD GEC_INLINE static void merge(T *arr, const HT *h_arr) {
        using namespace ::gec::utils;
        constexpr size_t shift = type_bits<T>::value / 2;
        arr[I] = (T(h_arr[2 * I + 1]) << shift) | T(h_arr[2 * I]);
        SplitDivRemHelper<I - 1, T, HT>::merge(arr, h_arr);
    }
};
template <typename T, typename HT>
struct SplitDivRemHelper<0, T, HT> {
    GEC_HD GEC_INLINE static void split(HT *h_arr, const T *arr) {
        using namespace ::gec::utils;
        constexpr size_t shift = type_bits<T>::value / 2;
        h_arr[1] = arr[0] >> shift;
        h_arr[0] = arr[0];
    }

    GEC_HD GEC_INLINE static void merge(T *arr, const HT *h_arr) {
        using namespace ::gec::utils;
        constexpr size_t shift = type_bits<T>::value / 2;
        arr[0] = (T(h_arr[1]) << shift) | T(h_arr[0]);
    }
};

/**
 * @brief division with remainder, spliting method
 *
 * limb divisions are performed by spliting a limb into two half limb
 *
 * @tparam NeedQ flag for computing quotient, if false, the quotient
 *               computation step will be skipped
 * @tparam NeedR flag for computing remainder, if false, the remainder
 *               computation step will be skipped
 * @tparam HT type used in limb casting, whose bit capacity must be exactly half
 *            the size of `T`
 * @tparam N limb length
 * @tparam T limb type
 * @param[out] q quotient, \f$\lfoor a / b \rfloor\f$
 * @param[out] r remainder, \f$a - q b\f$, if `NeedR == false`, then the result
 *               value of `r` is undefined
 * @param[in] a dividend
 * @param[in] b divisor
 * @param hq temporary buffer, whose length must be no less then `2 * N`
 * @param ha temporary buffer, whose length must be no less then `2 * N`
 * @param hb temporary buffer, whose length must be no less then `2 * N`
 * @param hqb temporary buffer, whose length must be no less then `2 * N`
 */
template <bool NeedQ, bool NeedR, typename HT, size_t N, typename T>
GEC_HD GEC_INLINE static void split_div_rem(T *q, T *r, const T *a, const T *b,
                                            HT *hq, HT *ha, HT *hb, HT *hqb) {
    using namespace gec::utils;
    using Helper = SplitDivRemHelper<N - 1, T, HT>;
    Helper::split(ha, a);
    Helper::split(hb, b);
    cast_div_rem<NeedQ, NeedR, T, 2 * N>(hq, ha, hb, hqb);
    if (NeedQ) {
        Helper::merge(q, hq);
    }
    if (NeedR) {
        Helper::merge(r, ha);
    }
}

enum Method { Split, Cast };

template <size_t N, typename T, Method M>
struct DivRemHelper;

template <size_t N, typename T>
struct DivRemHelper<N, T, Method::Cast> {
    template <typename CTX>
    GEC_HD GEC_INLINE static void
    div_rem(T *GEC_RSTRCT q, T *GEC_RSTRCT r, const T *GEC_RSTRCT a,
            const T *GEC_RSTRCT b, CTX &GEC_RSTRCT ctx) {
        using namespace ::gec::utils;
#ifdef __CUDA_ARCH__
        using TT = typename DeviceCast2Uint<T>::type;
#else
        using TT = typename HostCast2Uint<T>::type;
#endif // __CUDA_ARCH__
        auto &ctx_view = ctx.template view_as<T[N], T[N]>();
        auto &nb = ctx_view.template get<0>();
        auto &qb = ctx_view.template get<1>();
        fill_seq<N>(r, a);
        fill_seq<N>(nb, b);
        cast_div_rem<true, true, TT, N>(q, r, nb, qb);
    }

    template <typename CTX>
    GEC_HD GEC_INLINE static void div(T *GEC_RSTRCT q, const T *GEC_RSTRCT a,
                                      const T *GEC_RSTRCT b,
                                      CTX &GEC_RSTRCT ctx) {
        using namespace ::gec::utils;
#ifdef __CUDA_ARCH__
        using TT = typename DeviceCast2Uint<T>::type;
#else
        using TT = typename HostCast2Uint<T>::type;
#endif // __CUDA_ARCH__
        auto &ctx_view = ctx.template view_as<T[N], T[N], T[N]>();
        auto &na = ctx_view.template get<0>();
        auto &nb = ctx_view.template get<1>();
        auto &qb = ctx_view.template get<2>();
        fill_seq<N>(na, a);
        fill_seq<N>(nb, b);
        cast_div_rem<true, false, TT, N>(q, na, nb, qb);
    }

    template <typename CTX>
    GEC_HD GEC_INLINE static void rem(T *GEC_RSTRCT r, const T *GEC_RSTRCT a,
                                      const T *GEC_RSTRCT b,
                                      CTX &GEC_RSTRCT ctx) {
        using namespace ::gec::utils;
#ifdef __CUDA_ARCH__
        using TT = typename DeviceCast2Uint<T>::type;
#else
        using TT = typename HostCast2Uint<T>::type;
#endif // __CUDA_ARCH__
        auto &ctx_view = ctx.template view_as<T[N], T[N], T[N]>();
        auto &q = ctx_view.template get<0>();
        auto &nb = ctx_view.template get<1>();
        auto &qb = ctx_view.template get<2>();
        fill_seq<N>(r, a);
        fill_seq<N>(nb, b);
        cast_div_rem<false, true, TT, N>(q, r, nb, qb);
    }
};

template <size_t N, typename T>
struct DivRemHelper<N, T, Method::Split> {
    template <typename CTX>
    GEC_HD GEC_INLINE static void
    div_rem(T *GEC_RSTRCT q, T *GEC_RSTRCT r, const T *GEC_RSTRCT a,
            const T *GEC_RSTRCT b, CTX &GEC_RSTRCT ctx) {
        using namespace ::gec::utils;
#ifdef __CUDA_ARCH__
        using HT = typename DeviceCastHalfUint<T>::type;
#else
        using HT = typename HostCastHalfUint<T>::type;
#endif // __CUDA_ARCH__
        auto &ctx_view =
            ctx.template view_as<HT[2 * N], HT[2 * N], HT[2 * N], HT[2 * N]>();
        auto &hq = ctx_view.template get<0>();
        auto &hr = ctx_view.template get<1>();
        auto &ha = ctx_view.template get<2>();
        auto &hb = ctx_view.template get<3>();
        split_div_rem<true, true, HT, N>(q, r, a, b, hq, hr, ha, hb);
    }

    template <typename CTX>
    GEC_HD GEC_INLINE static void div(T *GEC_RSTRCT q, const T *GEC_RSTRCT a,
                                      const T *GEC_RSTRCT b,
                                      CTX &GEC_RSTRCT ctx) {
        using namespace ::gec::utils;
#ifdef __CUDA_ARCH__
        using HT = typename DeviceCastHalfUint<T>::type;
#else
        using HT = typename HostCastHalfUint<T>::type;
#endif // __CUDA_ARCH__
        auto &ctx_view =
            ctx.template view_as<HT[2 * N], HT[2 * N], HT[2 * N], HT[2 * N]>();
        auto &hq = ctx_view.template get<0>();
        auto &hr = ctx_view.template get<1>();
        auto &ha = ctx_view.template get<2>();
        auto &hb = ctx_view.template get<3>();
        split_div_rem<true, false, HT, N>(q, (T *)(nullptr), a, b, hq, hr, ha,
                                          hb);
    }

    template <typename CTX>
    GEC_HD GEC_INLINE static void rem(T *GEC_RSTRCT r, const T *GEC_RSTRCT a,
                                      const T *GEC_RSTRCT b,
                                      CTX &GEC_RSTRCT ctx) {
        using namespace ::gec::utils;
#ifdef __CUDA_ARCH__
        using HT = typename DeviceCastHalfUint<T>::type;
#else
        using HT = typename HostCastHalfUint<T>::type;
#endif // __CUDA_ARCH__
        auto &ctx_view =
            ctx.template view_as<HT[2 * N], HT[2 * N], HT[2 * N], HT[2 * N]>();
        auto &hq = ctx_view.template get<0>();
        auto &hr = ctx_view.template get<1>();
        auto &ha = ctx_view.template get<2>();
        auto &hb = ctx_view.template get<3>();
        split_div_rem<false, true, HT, N>((T *)(nullptr), r, a, b, hq, hr, ha,
                                          hb);
    }
};

} // namespace _division_

/**
 * @brief mixin that enables division, based on Cast2Uint
 *
 * division is performed by casting LIMB_T to a unsigned integer type with
 * double bit length
 *
 * @tparam Core
 * @tparam LIMB_T
 * @tparam LIMB_N
 */
template <class Core, typename LIMB_T, size_t LIMB_N>
class GEC_EMPTY_BASES CastDivision
    : protected CRTP<Core, CastDivision<Core, LIMB_T, LIMB_N>> {
    friend CRTP<Core, CastDivision<Core, LIMB_T, LIMB_N>>;

  public:
    /**
     * @brief return the quotient and remainder of division
     *
     * the behaviour is undefined if b == 0
     *
     * @tparam CTX context type
     * @param q quotient, \f$\lfoor a / b \rfloor\f$
     * @param r remainder, \f$a - q b\f$
     * @param a dividend
     * @param b divisor
     * @param ctx context
     */
    template <typename CTX>
    GEC_HD static void div_rem(Core &GEC_RSTRCT q, Core &GEC_RSTRCT r,
                               const Core &GEC_RSTRCT a,
                               const Core &GEC_RSTRCT b, CTX &GEC_RSTRCT ctx) {
        using namespace _division_;
        DivRemHelper<LIMB_N, LIMB_T, Method::Cast>::div_rem(
            q.array(), r.array(), a.array(), b.array(), ctx);
    }

    /**
     * @brief return the quotient of division
     *
     * the behaviour is undefined if b == 0
     *
     * @tparam CTX context type
     * @param q quotient, \f$\lfoor a / b \rfloor\f$
     * @param a dividend
     * @param b divisor
     * @param ctx context
     */
    template <typename CTX>
    GEC_HD GEC_INLINE static void
    div(Core &GEC_RSTRCT q, const Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
        CTX &GEC_RSTRCT ctx) {
        using namespace _division_;
        DivRemHelper<LIMB_N, LIMB_T, Method::Cast>::div(q.array(), a.array(),
                                                        b.array(), ctx);
    }

    /**
     * @brief return the remainder of division
     *
     * the behaviour is undefined if b == 0
     *
     * @tparam CTX context type
     * @param r remainder, \f$a - \lfoor a / b \rfloor b\f$
     * @param a dividend
     * @param b divisor
     * @param ctx context
     */
    template <typename CTX>
    GEC_HD GEC_INLINE static void
    rem(const Core &GEC_RSTRCT r, const Core &GEC_RSTRCT a,
        const Core &GEC_RSTRCT b, CTX &GEC_RSTRCT ctx) {
        using namespace _division_;
        DivRemHelper<LIMB_N, LIMB_T, Method::Cast>::rem(r.array(), a.array(),
                                                        b.array(), ctx);
    }
};

/**
 * @brief mixin that enables division, based on Cast2Uint
 *
 * division is performed by spliting a single LIMB_T into 2 smaller limbs
 *
 * @tparam Core
 * @tparam LIMB_T
 * @tparam LIMB_N
 */
template <class Core, typename LIMB_T, size_t LIMB_N>
class GEC_EMPTY_BASES SplitDivision
    : protected CRTP<Core, SplitDivision<Core, LIMB_T, LIMB_N>> {
    friend CRTP<Core, SplitDivision<Core, LIMB_T, LIMB_N>>;

  public:
    /**
     * @brief return the quotient and remainder of division
     *
     * the behaviour is undefined if b == 0
     *
     * @tparam CTX context type
     * @param q quotient, \f$\lfoor a / b \rfloor\f$
     * @param r remainder, \f$a - q b\f$
     * @param a dividend
     * @param b divisor
     * @param ctx context
     */
    template <typename CTX>
    GEC_HD static void div_rem(Core &GEC_RSTRCT q, Core &GEC_RSTRCT r,
                               const Core &GEC_RSTRCT a,
                               const Core &GEC_RSTRCT b, CTX &GEC_RSTRCT ctx) {
        using namespace _division_;
        DivRemHelper<LIMB_N, LIMB_T, Method::Split>::div_rem(
            q.array(), r.array(), a.array(), b.array(), ctx);
    }

    /**
     * @brief return the quotient of division
     *
     * the behaviour is undefined if b == 0
     *
     * @tparam CTX context type
     * @param q quotient, \f$\lfoor a / b \rfloor\f$
     * @param a dividend
     * @param b divisor
     * @param ctx context
     */
    template <typename CTX>
    GEC_HD GEC_INLINE static void
    div(Core &GEC_RSTRCT q, const Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
        CTX &GEC_RSTRCT ctx) {
        using namespace _division_;
        DivRemHelper<LIMB_N, LIMB_T, Method::Split>::div(q.array(), a.array(),
                                                         b.array(), ctx);
    }

    /**
     * @brief return the remainder of division
     *
     * the behaviour is undefined if b == 0
     *
     * @tparam CTX context type
     * @param r remainder, \f$a - \lfoor a / b \rfloor b\f$
     * @param a dividend
     * @param b divisor
     * @param ctx context
     */
    template <typename CTX>
    GEC_HD GEC_INLINE static void
    rem(const Core &GEC_RSTRCT r, const Core &GEC_RSTRCT a,
        const Core &GEC_RSTRCT b, CTX &GEC_RSTRCT ctx) {
        using namespace _division_;
        DivRemHelper<LIMB_N, LIMB_T, Method::Split>::rem(r.array(), a.array(),
                                                         b.array(), ctx);
    }
};

/**
 * @brief mixin that enables division, based on Cast2Uint
 *
 * the division algorithm is automatically chosen between split method and cast
 * method
 *
 * @tparam Core
 * @tparam LIMB_T
 * @tparam LIMB_N
 */
template <class Core, typename LIMB_T, size_t LIMB_N>
class GEC_EMPTY_BASES Division
    : protected CRTP<Core, Division<Core, LIMB_T, LIMB_N>> {
    friend CRTP<Core, Division<Core, LIMB_T, LIMB_N>>;

  public:
    /**
     * @brief return the quotient and remainder of division
     *
     * the behaviour is undefined if b == 0
     *
     * @tparam CTX context type
     * @param q quotient, \f$\lfoor a / b \rfloor\f$
     * @param r remainder, \f$a - q b\f$
     * @param a dividend
     * @param b divisor
     * @param ctx context
     */
    template <typename CTX>
    GEC_HD static void div_rem(Core &GEC_RSTRCT q, Core &GEC_RSTRCT r,
                               const Core &GEC_RSTRCT a,
                               const Core &GEC_RSTRCT b, CTX &GEC_RSTRCT ctx) {
        using namespace ::gec::utils;
        using namespace _division_;
        DivRemHelper<LIMB_N, LIMB_T,
#ifdef __CUDA_ARCH__
                     DeviceCast2Uint<LIMB_T>::value
#else
                     HostCast2Uint<LIMB_T>::value
#endif // __CUDA_ARCH__
                         ? Method::Cast
                         : Method::Split>::div_rem(q.array(), r.array(),
                                                   a.array(), b.array(), ctx);
    }

    /**
     * @brief return the quotient of division
     *
     * the behaviour is undefined if b == 0
     *
     * @tparam CTX context type
     * @param q quotient, \f$\lfoor a / b \rfloor\f$
     * @param a dividend
     * @param b divisor
     * @param ctx context
     */
    template <typename CTX>
    GEC_HD GEC_INLINE static void
    div(Core &GEC_RSTRCT q, const Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
        CTX &GEC_RSTRCT ctx) {
        using namespace ::gec::utils;
        using namespace _division_;
        DivRemHelper<LIMB_N, LIMB_T,
#ifdef __CUDA_ARCH__
                     DeviceCast2Uint<LIMB_T>::value
#else
                     HostCast2Uint<LIMB_T>::value
#endif // __CUDA_ARCH__
                         ? Method::Cast
                         : Method::Split>::div(q.array(), a.array(), b.array(),
                                               ctx);
    }

    /**
     * @brief return the remainder of division
     *
     * the behaviour is undefined if b == 0
     *
     * @tparam CTX context type
     * @param r remainder, \f$a - \lfoor a / b \rfloor b\f$
     * @param a dividend
     * @param b divisor
     * @param ctx context
     */
    template <typename CTX>
    GEC_HD GEC_INLINE static void
    rem(Core &GEC_RSTRCT r, const Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
        CTX &GEC_RSTRCT ctx) {
        using namespace ::gec::utils;
        using namespace _division_;
        DivRemHelper<LIMB_N, LIMB_T,
#ifdef __CUDA_ARCH__
                     DeviceCast2Uint<LIMB_T>::value
#else
                     HostCast2Uint<LIMB_T>::value
#endif // __CUDA_ARCH__
                         ? Method::Cast
                         : Method::Split>::rem(r.array(), a.array(), b.array(),
                                               ctx);
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_DIVISION_HPP
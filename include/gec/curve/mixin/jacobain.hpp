#pragma once
#ifndef GEC_CURVE_MIXIN_JACOBAIN_HPP
#define GEC_CURVE_MIXIN_JACOBAIN_HPP

#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

/** @brief mixin that enables elliptic curve arithmetic with Jacobian coordinate
 */
template <typename Core, typename FIELD_T>
class GEC_EMPTY_BASES Jacobain : protected CRTP<Core, Jacobain<Core, FIELD_T>> {
    friend CRTP<Core, Jacobain<Core, FIELD_T>>;

    using F = FIELD_T;

  public:
    using Field = FIELD_T;

    GEC_HD GEC_INLINE bool is_inf() const { return this->core().z().is_zero(); }
    GEC_HD GEC_INLINE void set_inf() {
        this->core().x().set_zero();
        this->core().y().set_zero();
        this->core().z().set_zero();
    }

    template <typename F_CTX>
    GEC_HD static bool on_curve(const Core &GEC_RSTRCT a,
                                F_CTX &GEC_RSTRCT ctx) {
        auto &ctx_view = ctx.template view_as<F, F, F, F>();

        auto &l = ctx_view.template get<0>();
        auto &r = ctx_view.template get<1>();
        auto &t1 = ctx_view.template get<2>();
        auto &t2 = ctx_view.template get<3>();

#ifdef __CUDACC__
        // suppress false positive NULL reference warning in nvcc
        GEC_NV_DIAGNOSTIC_PUSH
        GEC_NV_DIAG_SUPPRESS(284)
#endif // __CUDACC__

        if (a.b() != nullptr || a.a() != nullptr) {
            F::mul(t1, a.z(), a.z()); // z^2
            F::mul(t2, t1, t1);       // z^4
        }                             //
        if (a.b() != nullptr) {       //
            F::mul(r, t1, t2);        // z^6
        }                             //
        if (a.a() != nullptr) {       //
            F::mul(l, a.x(), t2);     // x z^4
            F::mul(t2, *a.a(), l);    // a x z^4
        }                             //
        if (a.b() != nullptr) {       //
            F::mul(t1, *a.b(), r);    // b z^6
        }                             //
        F::mul(l, a.x(), a.x());      // x^2
        F::mul(r, l, a.x());          // x^3
        if (a.a() != nullptr) {       //
            F::add(r, t2);            // x^3 + a x z^4
        }                             //
        if (a.b() != nullptr) {       //
            F::add(r, t1);            // right = x^3 + a x z^4 + b z^6
        }                             //
        F::mul(l, a.y(), a.y());      // left = y^2
        return l == r;

#ifdef __CUDACC__
        GEC_NV_DIAGNOSTIC_POP
#endif // __CUDACC__
    }

    template <typename F_CTX>
    GEC_HD static void to_affine(Core &GEC_RSTRCT a, F_CTX &GEC_RSTRCT ctx) {
        auto &ctx_view = ctx.template view_as<F, F>();

        if (a.z().is_mul_id()) {
            return;
        } else if (a.is_inf()) {
            a.x().set_zero();
            a.y().set_zero();
        } else {
            auto &t1 = ctx_view.template get<0>();
            auto &t2 = ctx_view.template get<1>();

            F::inv(a.z(), ctx);       // z^-1
            F::mul(t1, a.z(), a.z()); // z^-2
            F::mul(t2, a.x(), t1);    // x z^-2
            a.x() = t2;               //
            F::mul(t2, t1, a.z());    // z^-3
            F::mul(t1, a.y(), t2);    // y z^-3
            a.y() = t1;               //
            // we don't assign z = 1 here, so `to_affine` and `from_affine`
            // should be paired
        }
    }

    GEC_HD static void from_affine(Core &GEC_RSTRCT a) {
        if (a.x().is_zero() && a.y().is_zero()) {
            a.z().set_zero();
        } else {
            a.z().set_mul_id();
        }
    }

    template <typename F_CTX>
    GEC_HD static bool eq(const Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                          F_CTX &GEC_RSTRCT ctx) {
        auto &ctx_view = ctx.template view_as<F, F, F, F>();

        bool a_inf = a.is_inf();
        bool b_inf = b.is_inf();
        if (a_inf && b_inf) { // both infinity
            return true;
        } else if (a_inf || b_inf) { // only one infinity
            return false;
        } else if (a.z() == b.z()) { // z1 == z2
            return a.x() == b.x() && a.y() == b.y();
        } else { // z1 != z2
            auto &ta = ctx_view.template get<0>();
            auto &tb = ctx_view.template get<1>();
            auto &tc = ctx_view.template get<2>();
            auto &td = ctx_view.template get<3>();

            F::mul(tc, a.z(), a.z()); // z1^2
            F::mul(td, b.z(), b.z()); // z2^2
            F::mul(ta, a.x(), td);    // x1 z2^2
            F::mul(tb, b.x(), tc);    // x2 z1^2
            // check x1 z2^2 == x2 z1^2
            if (ta != tb) {
                return false;
            }
            F::mul(ta, tc, a.z()); // z1^3
            F::mul(tb, td, b.z()); // z2^3
            F::mul(tc, a.y(), tb); // y1 z2^3
            F::mul(td, b.y(), ta); // y2 z1^3
            // check y1 z2^3 == y2 z1^3
            if (tc != td) {
                return false;
            }
            return true;
        }
    }

    /** @brief add distinct point with some precomputed value
     *
     * ctx.get<0>() == a == x1 z2^2
     * ctx.get<1>() == b == x2 z1^2
     * ctx.get<2>() == c == y1 z2^3
     * ctx.get<3>() == d == y2 z1^3
     */
    template <typename F_CTX>
    GEC_HD static void
    add_distinct_inner(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                       const Core &GEC_RSTRCT c, F_CTX &GEC_RSTRCT ctx) {
        auto &ctx_view = ctx.template view_as<F, F, F, F>();

        auto &t1 = ctx_view.template get<0>();
        auto &t2 = ctx_view.template get<1>();
        auto &t3 = ctx_view.template get<2>();
        auto &t4 = ctx_view.template get<3>();

        F::sub(t2, t1);           // e = b - a
        F::sub(t4, t3);           // f = d - c
        F::mul(a.z(), t2, t2);    // e^2
        F::mul(a.y(), t1, a.z()); // a e^2
        F::mul(t1, a.z(), t2);    // e^3
        F::mul(a.z(), t3, t1);    // c e^3
        F::add(t3, a.y(), a.y()); // 2 a e^2
        F::mul(a.x(), t4, t4);    // f^2
        F::sub(a.x(), t3);        // f^2 - 2 a e^2
        F::sub(a.x(), t1);        // x = f^2 - 2 a e^2 - e^3
        F::sub(t1, a.y(), a.x()); // a e^2 - x
        F::mul(a.y(), t4, t1);    // f (a e^2 - x)
        F::sub(a.y(), a.z());     // y = f (a e^2 - x) - c e^3
        F::mul(t1, b.z(), c.z()); // z1 z2
        F::mul(a.z(), t1, t2);    // z = z1 z2 e
    }

    template <typename F_CTX>
    GEC_HD static void
    add_distinct(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                 const Core &GEC_RSTRCT c, F_CTX &GEC_RSTRCT ctx) {
        auto &ctx_view = ctx.template view_as<F, F, F, F, F>();

        auto &ta = ctx_view.template get<0>();
        auto &tb = ctx_view.template get<1>();
        auto &tc = ctx_view.template get<2>();
        auto &td = ctx_view.template get<3>();
        auto &t = ctx_view.template get<4>();

        F::mul(tc, c.z(), c.z()); // z2^2
        F::mul(t, tc, c.z());     // z2^3
        F::mul(ta, tc, b.x());    // a = x1 z2^2
        F::mul(tc, t, b.y());     // c = y1 z2^3

        F::mul(td, b.z(), b.z()); // z1^2
        F::mul(t, td, b.z());     // z1^3
        F::mul(tb, td, c.x());    // b = x2 z1^2
        F::mul(td, t, c.y());     // d = y2 z1^3

        add_distinct_inner(a, b, c, ctx);
    }

    template <typename F_CTX>
    GEC_HD static void add_self(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                                F_CTX &GEC_RSTRCT ctx) {
        auto &ctx_view = ctx.template view_as<F, F, F>();

        auto &t4 = ctx_view.template get<0>();
        auto &t5 = ctx_view.template get<1>();

#ifdef __CUDACC__
        // suppress false positive NULL reference warning in nvcc
        GEC_NV_DIAGNOSTIC_PUSH
        GEC_NV_DIAG_SUPPRESS(284)
#endif // __CUDACC__

        F::mul(t4, b.x(), b.x());        // x1^2
        F::add(t5, t4, t4);              // 2 x1^2
        F::add(t5, t4);                  // 3 x1^2
        if (a.a() != nullptr) {          //
            F::mul(a.z(), b.z(), b.z()); // z1^2
            F::mul(t4, a.z(), a.z());    // z1^4
            F::mul(a.z(), *a.a(), t4);   // A z1^4
            F::add(t5, a.z());           // b = 3 x1^2 + A z1^4
        }                                //
        F::mul(a.z(), b.y(), b.y());     // y1^2
        F::mul(t4, b.x(), a.z());        // x1 y1^2
        F::template mul_pow2<2>(t4);     // a = 4 x1 y1^2
        F::add(a.y(), t4, t4);           // 2 a
        F::mul(a.x(), t5, t5);           // b^2
        F::sub(a.x(), a.y());            // x = b^2 - 2 a
        F::sub(t4, a.x());               // a - x
        F::mul(a.y(), t5, t4);           // b (a - x)
        F::mul(t4, a.z(), a.z());        // y1^4
        F::template mul_pow2<3>(t4);     // 8 y1^4
        F::sub(a.y(), t4);               // y = b(a - x) -8 y1^4
        F::mul(a.z(), b.y(), b.z());     // y1 z1
        F::template mul_pow2<1>(a.z());  // z = 2 y1 z1

#ifdef __CUDACC__
        GEC_NV_DIAGNOSTIC_POP
#endif // __CUDACC__
    }

    template <typename F_CTX>
    GEC_HD static void add(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                           const Core &GEC_RSTRCT c, F_CTX &GEC_RSTRCT ctx) {
        auto &ctx_view = ctx.template view_as<F, F, F, F, F>();

        // checking for infinity here is not necessary
        if (b.is_inf()) {
            a = c;
        } else if (c.is_inf()) {
            a = b;
        } else {
            auto &ta = ctx_view.template get<0>();
            auto &tb = ctx_view.template get<1>();
            auto &tc = ctx_view.template get<2>();
            auto &td = ctx_view.template get<3>();
            auto &t = ctx_view.template get<4>();

            F::mul(tc, c.z(), c.z()); // z2^2
            F::mul(t, tc, c.z());     // z2^3
            F::mul(ta, tc, b.x());    // a = x1 z2^2
            F::mul(tc, t, b.y());     // c = y1 z2^3

            F::mul(td, b.z(), b.z()); // z1^2
            F::mul(t, td, b.z());     // z1^3
            F::mul(tb, td, c.x());    // b = x2 z1^2
            F::mul(td, t, c.y());     // d = y2 z1^3

            if (ta == tb && tc == td) {
                add_self(a, b, ctx);
            } else {
                add_distinct_inner(a, b, c, ctx);
            }
        }
    }

    GEC_HD GEC_INLINE static void neg(Core &GEC_RSTRCT a,
                                      const Core &GEC_RSTRCT b) {
        a.x() = b.y();
        F::neg(a.y(), b.y());
        a.z() = b.z();
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_JACOBAIN_HPP

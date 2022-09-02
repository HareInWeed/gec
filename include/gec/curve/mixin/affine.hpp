#pragma once
#ifndef GEC_CURVE_MIXIN_AFFINE_HPP
#define GEC_CURVE_MIXIN_AFFINE_HPP

#include <gec/utils/context.hpp>
#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

/** @brief mixin that enables elliptic curve arithmetic with affine coordinate
 *
 * TODO: list `FIELD_T` requirement
 */
template <typename Core, typename FIELD_T>
class GEC_EMPTY_BASES Affine : protected CRTP<Core, Affine<Core, FIELD_T>> {
    friend CRTP<Core, Affine<Core, FIELD_T>>;

    using F = FIELD_T;

  public:
    using Field = FIELD_T;

    GEC_HD GEC_INLINE bool is_inf() const {
        return this->core().x().is_zero() && this->core().y().is_zero();
    }

    GEC_HD GEC_INLINE void set_inf() {
        this->core().x().set_zero();
        this->core().y().set_zero();
    }

    template <typename F_CTX>
    GEC_HD GEC_INLINE static bool eq(const Core &GEC_RSTRCT a,
                                     const Core &GEC_RSTRCT b, F_CTX &) {
        return eq(a, b);
    }

    GEC_HD GEC_INLINE static bool eq(const Core &GEC_RSTRCT a,
                                     const Core &GEC_RSTRCT b) {
        return a.x() == b.x() && a.y() == b.y();
    }

    template <typename F_CTX>
    GEC_HD static bool on_curve(const Core &GEC_RSTRCT a,
                                F_CTX &GEC_RSTRCT ctx) {
        auto &ctx_view = ctx.template view_as<F, F, F>();

        if (a.is_inf()) {
            return true;
        }
        auto &l = ctx_view.template get<0>();
        auto &r = ctx_view.template get<1>();
        auto &t = ctx_view.template get<2>();

#ifdef __CUDACC__
        // suppress false positive NULL reference warning
        GEC_NV_DIAGNOSTIC_PUSH
        GEC_NV_DIAG_SUPPRESS(284)
#endif // __CUDACC__

        F::mul(l, a.y(), a.y());      // left = y^2
        F::mul(t, a.x(), a.x());      // x^2
        F::mul(r, t, a.x());          // x^3
        if (a.a() != nullptr) {       //
            F::mul(t, *a.a(), a.x()); // A x
            F::add(r, t);             // x^3 + A x
        }                             //
        if (a.b() != nullptr) {       //
            F::add(r, *a.b());        // right = x^3 + A x + B
        }
        return l == r;

#ifdef __CUDACC__
        GEC_NV_DIAGNOSTIC_POP
#endif // __CUDACC__
    }

    template <typename F_CTX>
    GEC_HD static void
    add_distinct(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                 const Core &GEC_RSTRCT c, F_CTX &GEC_RSTRCT ctx) {
        auto &ctx_view = ctx.template view_as<F, F, F, F>();
        auto &inv_ctx = ctx.template view_as<F>().rest();

        auto &d = ctx_view.template get<0>();
        auto &l = ctx_view.template get<1>();

        F::sub(d, b.x(), c.x());     // x1 - x2
        F::inv(d, inv_ctx);          // 1 / (x1 - x2)
        F::sub(a.y(), b.y(), c.y()); // y1 - y2
        F::mul(l, a.y(), d);         // l = (y1 - y2) / (x1 - x2)
        F::mul(a.y(), l, l);         // l^2
        F::sub(a.y(), b.x());        // l^2 - x1
        F::sub(a.x(), a.y(), c.x()); // x = l^2 - x1 - x2
        F::sub(d, b.x(), a.x());     // x1 - x
        F::mul(a.y(), l, d);         // l (x1 - x)
        F::sub(a.y(), b.y());        // y = l (x1 - x) - y1
    }

    template <typename F_CTX>
    GEC_HD static void add_self(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                                F_CTX &GEC_RSTRCT ctx) {
        if (b.y().is_zero()) {
            a.set_inf();
            return;
        }
        auto &ctx_view = ctx.template view_as<F>();
        auto &d = ctx_view.template get<0>();
        auto &inv_ctx = ctx_view.rest();
        auto &l = inv_ctx.template view_as<F>().template get<0>();

#ifdef __CUDACC__
        // suppress false positive NULL reference warning
        GEC_NV_DIAGNOSTIC_PUSH
        GEC_NV_DIAG_SUPPRESS(284)
#endif // __CUDACC__

        F::add(d, b.y(), b.y());     // 2 y1
        F::inv(d, inv_ctx);          // (2 y1)^-1
        F::mul(a.y(), b.x(), b.x()); // x1^2
        F::add(a.x(), a.y(), a.y()); // 2 x1^2
        F::add(a.x(), a.y());        // 3 x1^2
        if (a.a() != nullptr) {      //
            F::add(a.x(), *a.a());   // 3 x1^2 + A
        }                            //
        F::mul(l, a.x(), d);         // l = (3 x1^2 + A) / (2 y1)
        F::mul(a.x(), l, l);         // l^2
        F::add(a.y(), b.x(), b.x()); // 2 x1
        F::sub(a.x(), a.y());        // x = l^2 - 2 x1
        F::sub(a.y(), b.x(), a.x()); // x1 - x
        F::mul(d, l, a.y());         // l (x1 - x)
        F::sub(a.y(), d, b.y());     // y = l (x1 - x) - y1

#ifdef __CUDACC__
        GEC_NV_DIAGNOSTIC_POP
#endif // __CUDACC__
    }

    template <typename F_CTX>
    GEC_HD static void add(Core &GEC_RSTRCT a, const Core &b, const Core &c,
                           F_CTX &GEC_RSTRCT ctx) {
        if (b.is_inf()) {
            a = c;
        } else if (c.is_inf()) {
            a = b;
        } else if (b.x() != c.x()) {
            add_distinct(a, b, c, ctx);
        } else if (b.y() == c.y()) {
            add_self(a, b, ctx);
        } else {
            a.set_inf();
        }
    }

    GEC_HD GEC_INLINE static void neg(Core &GEC_RSTRCT a,
                                      const Core &GEC_RSTRCT b) {
        a.x() = b.x();
        F::neg(a.y(), b.y());
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_AFFINE_HPP

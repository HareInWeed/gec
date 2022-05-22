#pragma once
#ifndef GEC_CURVE_MIXIN_PROJECTIVE_HPP
#define GEC_CURVE_MIXIN_PROJECTIVE_HPP

#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

/** @brief mixin that enables elliptic curve arithmetic with projective
 * coordinate
 */
template <typename Core, typename FIELD_T, const FIELD_T &A, const FIELD_T &B>
class Projective : protected CRTP<Core, Projective<Core, FIELD_T, A, B>> {
    friend CRTP<Core, Projective<Core, FIELD_T, A, B>>;
    using F = FIELD_T;

  public:
    using Field = FIELD_T;

    __host__ __device__ GEC_INLINE bool is_inf() const {
        return this->core().z().is_zero();
    }
    __host__ __device__ GEC_INLINE void set_inf() {
        this->core().x().set_zero();
        this->core().y().set_zero();
        this->core().z().set_zero();
    }

    template <typename F_CTX>
    __host__ __device__ static void to_affine(Core &GEC_RSTRCT a,
                                              F_CTX &GEC_RSTRCT ctx) {
        auto &ctx_view = ctx.template view_as<F>();

        if (a.z().is_mul_id()) {
            return;
        } else if (a.is_inf()) {
            a.x().set_zero();
            a.y().set_zero();
        } else {
            auto &t1 = ctx_view.template get<0>();

            F::inv(a.z(), ctx);
            F::mul(t1, a.x(), a.z());
            a.x() = t1;
            F::mul(t1, a.y(), a.z());
            a.y() = t1;
            // we don't assign z = 1 here, so `to_affine` and `from_affine`
            // should be paired
        }
    }

    __host__ __device__ static void from_affine(Core &GEC_RSTRCT a) {
        a.z().set_mul_id();
    }

    template <typename F_CTX>
    __host__ __device__ static bool on_curve(const Core &GEC_RSTRCT a,
                                             F_CTX &ctx) {
        auto &ctx_view = ctx.template view_as<F, F, F, F>();

        auto &l = ctx_view.template get<0>();
        auto &r = ctx_view.template get<1>();
        auto &t1 = ctx_view.template get<2>();
        auto &t2 = ctx_view.template get<3>();

        F::mul(t2, a.z(), a.z()); // z^2
        F::mul(r, a.x(), t2);     // x z^2
        F::mul(l, a.x(), a.x());  // x^2
        F::mul(t1, A, r);         // A x z^2
        F::mul(r, l, a.x());      // x^3
        F::add(r, t1);            // x^3 + A x z^2
        F::mul(t1, t2, a.z());    // z^3
        F::mul(t2, t1, B);        // B z^3
        F::add(r, t2);            // right = x^3 + A x z^2 + B z^3
        F::mul(t1, a.y(), a.y()); // y^2
        F::mul(l, t1, a.z());     // left = y^2 z
        return l == r;
    }

    template <typename F_CTX>
    __host__ __device__ static bool eq(const Core &GEC_RSTRCT a,
                                       const Core &GEC_RSTRCT b,
                                       F_CTX &GEC_RSTRCT ctx) {
        auto &ctx_view = ctx.template view_as<F, F>();

        bool a_inf = a.is_inf();
        bool b_inf = b.is_inf();
        if (a_inf && b_inf) { // both infinity
            return true;
        } else if (a_inf || b_inf) { // only one infinity
            return false;
        } else if (a.z() == b.z()) { // z1 == z2
            return a.x() == b.x() && a.y() == b.y();
        } else { // z1 != z2
            auto &p1 = ctx_view.template get<0>();
            auto &p2 = ctx_view.template get<1>();
            // check x1 z2 == x2 z1
            F::mul(p1, a.x(), b.z());
            F::mul(p2, b.x(), a.z());
            if (p1 != p2) {
                return false;
            }
            // check y1 z2 == y2 z1
            F::mul(p1, a.y(), b.z());
            F::mul(p2, b.y(), a.z());
            if (p1 != p2) {
                return false;
            }
            return true;
        }
    }

    template <typename F_CTX>
    __host__ __device__ static void
    add_distinct(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                 const Core &GEC_RSTRCT c, F_CTX &GEC_RSTRCT ctx) {
        auto &ctx_view = ctx.template view_as<F, F, F, F>();

        auto &x1z2 = ctx_view.template get<0>();
        auto &x2z1 = ctx_view.template get<1>();
        auto &y1z2 = ctx_view.template get<2>();
        auto &y2z1 = ctx_view.template get<3>();

        F::mul(x1z2, b.x(), c.z()); // x1 z2
        F::mul(x2z1, c.x(), b.z()); // x2 z1
        F::mul(y1z2, b.y(), c.z()); // y1 z2
        F::mul(y2z1, c.y(), b.z()); // y2 z1
        add_distinct_inner(a, b, c, ctx);
    }

    template <typename F_CTX>
    __host__ __device__ static void
    add_distinct_inner(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                       const Core &GEC_RSTRCT c, F_CTX &GEC_RSTRCT ctx) {
        auto &ctx_view = ctx.template view_as<F, F, F, F, F>();

        auto &x1z2 = ctx_view.template get<0>();
        auto &x2z1 = ctx_view.template get<1>();
        auto &y1z2 = ctx_view.template get<2>();
        auto &y2z1 = ctx_view.template get<3>();
        auto &t = ctx_view.template get<4>();

        F::sub(y2z1, y1z2);          // a = y2 z1 - y1 z2
        F::sub(x2z1, x1z2);          // b = x2 z1 - x1 z2
        F::mul(a.y(), x2z1, x2z1);   // b^2
        F::mul(t, a.y(), x1z2);      // b^2 x1 z2
        F::mul(a.x(), a.y(), x2z1);  // b^3
        F::mul(x1z2, a.x(), y1z2);   // b^3 y1 z2
        F::mul(a.z(), y2z1, y2z1);   // a^2
        F::mul(a.y(), b.z(), c.z()); // z1 z2
        F::mul(y1z2, a.y(), a.z());  // a^2 z1 z2
        F::mul(a.z(), a.x(), a.y()); // z = b^3 z1 z2
        F::add(a.y(), t, t);         // 2 b^2 x1 z2
        F::sub(y1z2, a.y());         // a^2 z1 z2 - 2 b^2 x1 z2
        F::sub(y1z2, a.x());         // c = a^2 z1 z2 - 2 b^2 x1 z2 - b^3
        F::sub(t, y1z2);             // b^2 x1 z2 - c
        F::mul(a.y(), t, y2z1);      // a (b^2 x1 z2 - c)
        F::sub(a.y(), x1z2);         // y = a (b^2 x1 z2 - c) - b^3 y1 z2
        F::mul(a.x(), y1z2, x2z1);   // x = b c
    }

    template <typename F_CTX>
    __host__ __device__ static void
    add_self(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b, F_CTX &ctx) {
        auto &ctx_view = ctx.template view_as<F, F, F>();

        auto &t1 = ctx_view.template get<0>();
        auto &t2 = ctx_view.template get<1>();
        auto &t3 = ctx_view.template get<2>();

        F::mul(t3, b.z(), b.z());       // z1^2
        F::mul(t2, A, t3);              // A z1^2
        F::mul(t3, b.x(), b.x());       // x1^2
        F::add(a.z(), t3, t3);          // 2 x1^2
        F::add(a.z(), t3);              // 3 x1^2
        F::add(t2, a.z());              // a = A z1^2 + 3 x1^2
        F::mul(t1, b.y(), b.z());       // b = y1 z1
        F::mul(t3, b.x(), b.y());       // x1 y1
        F::mul(a.z(), t3, t1);          // c = x1 y1 b
        F::mul(t3, t2, t2);             // a^2
        F::template mul_pow2<2>(a.z()); // 4 c
        F::add(a.x(), a.z(), a.z());    // 8 c
        F::sub(t3, a.x());              // d = a^2 - 8 c
        F::mul(a.x(), t1, t3);          // b d
        F::template mul_pow2<1>(a.x()); // X = 2 b d
        F::sub(a.z(), t3);              // 4 c - d
        F::mul(a.y(), t2, a.z());       // a(4 c - d)
        F::mul(t2, t1, t1);             // b^2
        F::template mul_pow2<3>(t2);    // 8 b^2
        F::mul(a.z(), b.y(), b.y());    // y1^2
        F::mul(t3, t2, a.z());          // 8 y1^2 b^2
        F::sub(a.y(), t3);              // y = a(4 c - d) - 8 y1^2 b^2
        F::mul(a.z(), t2, t1);          // z = 8 b^3
    }

    template <typename F_CTX>
    __host__ __device__ static void add(Core &GEC_RSTRCT a,
                                        const Core &GEC_RSTRCT b,
                                        const Core &GEC_RSTRCT c, F_CTX &ctx) {
        auto &ctx_view = ctx.template view_as<F, F, F, F>();

        // checking for infinity here is not necessary
        if (b.is_inf()) {
            a = c;
        } else if (c.is_inf()) {
            a = b;
        } else {
            auto &x1z2 = ctx_view.template get<0>();
            auto &x2z1 = ctx_view.template get<1>();
            auto &y1z2 = ctx_view.template get<2>();
            auto &y2z1 = ctx_view.template get<3>();

            F::mul(x1z2, b.x(), c.z()); // x1 z2
            F::mul(x2z1, c.x(), b.z()); // x2 z1
            F::mul(y1z2, b.y(), c.z()); // y1 z2
            F::mul(y2z1, c.y(), b.z()); // y2 z1
            if (x1z2 == x2z1 && y1z2 == y2z1) {
                add_self(a, b, ctx);
            } else {
                add_distinct_inner(a, b, c, ctx);
            }
        }
    }

    __host__ __device__ GEC_INLINE static void neg(Core &GEC_RSTRCT a,
                                                   const Core &GEC_RSTRCT b) {
        a.x() = b.x();
        F::neg(a.y(), b.y());
        a.z() = b.z();
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_PROJECTIVE_HPP

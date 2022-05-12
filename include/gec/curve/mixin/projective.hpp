#pragma once
#ifndef GEC_CURVE_MIXIN_PROJECTIVE_HPP
#define GEC_CURVE_MIXIN_PROJECTIVE_HPP

#include <gec/utils/context_check.hpp>
#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

/** @brief mixin that enables elliptic curve arithmetic with projective
 * coordinate
 */
template <typename Core, typename FIELD_T, const FIELD_T &A, const FIELD_T &B>
class Projective : protected CRTP<Core, Projective<Core, FIELD_T, A, B>> {
    friend CRTP<Core, Projective<Core, FIELD_T, A, B>>;

  public:
    template <typename F_CTX>
    __host__ __device__ static void
    add_distinct_inner(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                       const Core &GEC_RSTRCT c, F_CTX &GEC_RSTRCT ctx) {
        GEC_CTX_CAP(F_CTX, 5);

        FIELD_T &x1z2 = ctx.template get<0>();
        FIELD_T &x2z1 = ctx.template get<1>();
        FIELD_T &y1z2 = ctx.template get<2>();
        FIELD_T &y2z1 = ctx.template get<3>();
        FIELD_T &t = ctx.template get<4>();

        FIELD_T::sub(y2z1, y1z2);          // a = y2 z1 - y1 z2
        FIELD_T::sub(x2z1, x1z2);          // b = x2 z1 - x1 z2
        FIELD_T::mul(a.y(), x2z1, x2z1);   // b^2
        FIELD_T::mul(t, a.y(), x1z2);      // b^2 x1 z2
        FIELD_T::mul(a.x(), a.y(), x2z1);  // b^3
        FIELD_T::mul(x1z2, a.x(), y1z2);   // b^3 y1 z2
        FIELD_T::mul(a.z(), y2z1, y2z1);   // a^2
        FIELD_T::mul(a.y(), b.z(), c.z()); // z1 z2
        FIELD_T::mul(y1z2, a.y(), a.z());  // a^2 z1 z2
        FIELD_T::mul(a.z(), a.x(), a.y()); // z = b^3 z1 z2
        FIELD_T::add(a.y(), t, t);         // 2 b^2 x1 z2
        FIELD_T::sub(y1z2, a.y());         // a^2 z1 z2 - 2 b^2 x1 z2
        FIELD_T::sub(y1z2, a.x());         // c = a^2 z1 z2 - 2 b^2 x1 z2 - b^3
        FIELD_T::sub(t, y1z2);             // b^2 x1 z2 - c
        FIELD_T::mul(a.y(), t, y2z1);      // a (b^2 x1 z2 - c)
        FIELD_T::sub(a.y(), x1z2);         // y = a (b^2 x1 z2 - c) - b^3 y1 z2
        FIELD_T::mul(a.x(), y1z2, x2z1);   // x = b c
    }

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
        GEC_CTX_CAP(F_CTX, 1);

        if (a.is_inf() || a.z().is_mul_id())
            return;

        FIELD_T &t1 = ctx.template get<0>();

        FIELD_T::inv(a.z(), ctx);
        FIELD_T::mul(t1, a.x(), a.z());
        a.x() = t1;
        FIELD_T::mul(t1, a.y(), a.z());
        a.y() = t1;
        // we don't assign z = 1 here, so `to_affine` and `from_affine` should
        // be paired
    }

    __host__ __device__ static void from_affine(Core &GEC_RSTRCT a) {
        a.z().set_mul_id();
    }

    template <typename F_CTX>
    __host__ __device__ static bool on_curve(const Core &GEC_RSTRCT a,
                                             F_CTX &ctx) {
        GEC_CTX_CAP(F_CTX, 4);

        FIELD_T &l = ctx.template get<0>();
        FIELD_T &r = ctx.template get<1>();
        FIELD_T &t1 = ctx.template get<2>();
        FIELD_T &t2 = ctx.template get<3>();

        FIELD_T::mul(t2, a.z(), a.z()); // z^2
        FIELD_T::mul(r, a.x(), t2);     // x z^2
        FIELD_T::mul(l, a.x(), a.x());  // x^2
        FIELD_T::mul(t1, A, r);         // A x z^2
        FIELD_T::mul(r, l, a.x());      // x^3
        FIELD_T::add(r, t1);            // x^3 + A x z^2
        FIELD_T::mul(t1, t2, a.z());    // z^3
        FIELD_T::mul(t2, t1, B);        // B z^3
        FIELD_T::add(r, t2);            // right = x^3 + A x z^2 + B z^3
        FIELD_T::mul(t1, a.y(), a.y()); // y^2
        FIELD_T::mul(l, t1, a.z());     // left = y^2 z
        return l == r;
    }

    template <typename F_CTX>
    __host__ __device__ static bool eq(const Core &GEC_RSTRCT a,
                                       const Core &GEC_RSTRCT b,
                                       F_CTX &GEC_RSTRCT ctx) {
        GEC_CTX_CAP(F_CTX, 2);

        bool a_inf = a.is_inf();
        bool b_inf = b.is_inf();
        if (a_inf && b_inf) { // both infinity
            return true;
        } else if (a_inf || b_inf) { // only one infinity
            return false;
        } else if (a.z() == b.z()) { // z1 == z2
            return a.x() == b.x() && a.y() == b.y();
        } else { // z1 != z2
            FIELD_T &p1 = ctx.template get<0>();
            FIELD_T &p2 = ctx.template get<1>();
            // check x1 z2 == x2 z1
            FIELD_T::mul(p1, a.x(), b.z());
            FIELD_T::mul(p2, b.x(), a.z());
            if (p1 != p2) {
                return false;
            }
            // check y1 z2 == y2 z1
            FIELD_T::mul(p1, a.y(), b.z());
            FIELD_T::mul(p2, b.y(), a.z());
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
        GEC_CTX_CAP(F_CTX, 4);

        FIELD_T::mul(ctx.template get<0>(), b.x(), c.z()); // x1 z2
        FIELD_T::mul(ctx.template get<1>(), c.x(), b.z()); // x2 z1
        FIELD_T::mul(ctx.template get<2>(), b.y(), c.z()); // y1 z2
        FIELD_T::mul(ctx.template get<3>(), c.y(), b.z()); // y2 z1
        add_distinct_inner(a, b, c, ctx);
    }

    template <typename F_CTX>
    __host__ __device__ static void
    add_self(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b, F_CTX &ctx) {
        GEC_CTX_CAP(F_CTX, 3);

        FIELD_T &t1 = ctx.template get<0>();
        FIELD_T &t2 = ctx.template get<1>();
        FIELD_T &t3 = ctx.template get<2>();

        FIELD_T::mul(t3, b.z(), b.z());       // z1^2
        FIELD_T::mul(t2, A, t3);              // A z1^2
        FIELD_T::mul(t3, b.x(), b.x());       // x1^2
        FIELD_T::add(a.z(), t3, t3);          // 2 x1^2
        FIELD_T::add(a.z(), t3);              // 3 x1^2
        FIELD_T::add(t2, a.z());              // a = A z1^2 + 3 x1^2
        FIELD_T::mul(t1, b.y(), b.z());       // b = y1 z1
        FIELD_T::mul(t3, b.x(), b.y());       // x1 y1
        FIELD_T::mul(a.z(), t3, t1);          // c = x1 y1 b
        FIELD_T::mul(t3, t2, t2);             // a^2
        FIELD_T::template mul_pow2<2>(a.z()); // 4 c
        FIELD_T::add(a.x(), a.z(), a.z());    // 8 c
        FIELD_T::sub(t3, a.x());              // d = a^2 - 8 c
        FIELD_T::mul(a.x(), t1, t3);          // b d
        FIELD_T::template mul_pow2<1>(a.x()); // X = 2 b d
        FIELD_T::sub(a.z(), t3);              // 4 c - d
        FIELD_T::mul(a.y(), t2, a.z());       // a(4 c - d)
        FIELD_T::mul(t2, t1, t1);             // b^2
        FIELD_T::template mul_pow2<3>(t2);    // 8 b^2
        FIELD_T::mul(a.z(), b.y(), b.y());    // y1^2
        FIELD_T::mul(t3, t2, a.z());          // 8 y1^2 b^2
        FIELD_T::sub(a.y(), t3);              // y = a(4 c - d) - 8 y1^2 b^2
        FIELD_T::mul(a.z(), t2, t1);          // z = 8 b^3
    }

    template <typename F_CTX>
    __host__ __device__ static void add(Core &GEC_RSTRCT a,
                                        const Core &GEC_RSTRCT b,
                                        const Core &GEC_RSTRCT c, F_CTX &ctx) {
        GEC_CTX_CAP(F_CTX, 4);

        if (b.is_inf()) {
            a = c;
        } else if (c.is_inf()) {
            a = b;
        } else {
            FIELD_T &x1z2 = ctx.template get<0>();
            FIELD_T &x2z1 = ctx.template get<1>();
            FIELD_T &y1z2 = ctx.template get<2>();
            FIELD_T &y2z1 = ctx.template get<3>();

            FIELD_T::mul(x1z2, b.x(), c.z()); // x1 z2
            FIELD_T::mul(x2z1, c.x(), b.z()); // x2 z1
            FIELD_T::mul(y1z2, b.y(), c.z()); // y1 z2
            FIELD_T::mul(y2z1, c.y(), b.z()); // y2 z1
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
        FIELD_T::neg(a.y(), b.y());
        a.z() = b.z();
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_PROJECTIVE_HPP

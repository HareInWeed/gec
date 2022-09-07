#pragma once
#ifndef GEC_CURVE_MIXIN_PROJECTIVE_HPP
#define GEC_CURVE_MIXIN_PROJECTIVE_HPP

#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

/** @brief mixin that enables elliptic curve arithmetic with projective
 * coordinate
 */
template <typename Core, typename FIELD_T, bool InfYZero = true>
class GEC_EMPTY_BASES ProjectiveCoordinate
    : protected CRTP<Core, ProjectiveCoordinate<Core, FIELD_T, InfYZero>> {
    friend CRTP<Core, ProjectiveCoordinate<Core, FIELD_T, InfYZero>>;
    using F = FIELD_T;

    GEC_HD GEC_INLINE static void
    add_distinct_inner(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                       const Core &GEC_RSTRCT c, FIELD_T &GEC_RSTRCT x1z2,
                       FIELD_T &GEC_RSTRCT x2z1, FIELD_T &GEC_RSTRCT y1z2,
                       FIELD_T &GEC_RSTRCT y2z1) {
        F t;

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

  public:
    using Field = FIELD_T;

    GEC_HD GEC_INLINE bool is_affine_inf() {
        return this->core().x().is_zero() &&
               (InfYZero ? this->core().y().is_zero()
                         : this->core().y().is_mul_id());
    }
    GEC_HD GEC_INLINE void set_affine_inf() {
        this->core().x().set_zero();
        if (InfYZero) {
            this->core().y().set_zero();
        } else {
            this->core().y().set_mul_id();
        }
    }

    GEC_HD GEC_INLINE bool is_inf() const { return this->core().z().is_zero(); }
    GEC_HD GEC_INLINE void set_inf() {
        this->set_affine_inf();
        this->core().z().set_zero();
    }

    GEC_HD static void to_affine(Core &GEC_RSTRCT a) {
        if (a.z().is_mul_id()) {
            return;
        } else if (a.is_inf()) {
            a.set_affine_inf();
        } else {
            F::inv(a.z());
            F t1;
            F::mul(t1, a.x(), a.z());
            a.x() = t1;
            F::mul(t1, a.y(), a.z());
            a.y() = t1;
            // we don't assign z = 1 here, so `to_affine` and `from_affine`
            // should be paired
        }
    }

    GEC_HD GEC_INLINE static void from_non_inf_affine(Core &a) {
        a.z().set_mul_id();
    }
    GEC_HD GEC_INLINE static void from_affine(Core &a) {
        if (a.is_affine_inf()) {
            a.z().set_zero();
        } else {
            a.z().set_mul_id();
        }
    }

    GEC_HD static bool on_curve(const Core &GEC_RSTRCT a) {
#ifdef __CUDACC__
        // suppress false positive NULL reference warning
        GEC_NV_DIAGNOSTIC_PUSH
        GEC_NV_DIAG_SUPPRESS(284)
#endif // __CUDACC__

        F l, r, t1, t2;
        if (a.a() != nullptr || a.b() != nullptr) {
            F::mul(t2, a.z(), a.z()); // z^2
        }                             //
        F::mul(l, a.x(), a.x());      // x^2
        if (a.a() != nullptr) {       //
            F::mul(r, a.x(), t2);     // x z^2
            F::mul(t1, *a.a(), r);    // A x z^2
        }                             //
        F::mul(r, l, a.x());          // x^3
        if (a.a() != nullptr) {       //
            F::add(r, t1);            // x^3 + A x z^2
        }                             //
        if (a.b() != nullptr) {       //
            F::mul(t1, t2, a.z());    // z^3
            F::mul(t2, t1, *a.b());   // B z^3
            F::add(r, t2);            // right = x^3 + A x z^2 + B z^3
        }                             //
        F::mul(t1, a.y(), a.y());     // y^2
        F::mul(l, t1, a.z());         // left = y^2 z
        return l == r;

#ifdef __CUDACC__
        GEC_NV_DIAGNOSTIC_POP
#endif // __CUDACC__
    }

    GEC_HD static bool eq(const Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b) {
        bool a_inf = a.is_inf();
        bool b_inf = b.is_inf();
        if (a_inf && b_inf) { // both infinity
            return true;
        } else if (a_inf || b_inf) { // only one infinity
            return false;
        } else if (a.z() == b.z()) { // z1 == z2
            return a.x() == b.x() && a.y() == b.y();
        } else { // z1 != z2
            F p1, p2;
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

    GEC_HD static void add_distinct(Core &GEC_RSTRCT a,
                                    const Core &GEC_RSTRCT b,
                                    const Core &GEC_RSTRCT c) {
        F x1z2, x2z1, y1z2, y2z1;

        F::mul(x1z2, b.x(), c.z());
        F::mul(x2z1, c.x(), b.z());
        F::mul(y1z2, b.y(), c.z());
        F::mul(y2z1, c.y(), b.z());
        add_distinct_inner(a, b, c, x1z2, x2z1, y1z2, y2z1);
    }

    GEC_HD static void add_self(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b) {
#ifdef __CUDACC__
        // suppress false positive NULL reference warning
        GEC_NV_DIAGNOSTIC_PUSH
        GEC_NV_DIAG_SUPPRESS(284)
#endif // __CUDACC__

        F t1, t2, t3;

        F::mul(t3, b.x(), b.x());       // x1^2
        F::add(t2, t3, t3);             // 2 x1^2
        F::add(t2, t3);                 // 3 x1^2
        if (a.a() != nullptr) {         //
            F::mul(t3, b.z(), b.z());   // z1^2
            F::mul(a.z(), *a.a(), t3);  // A z1^2
            F::add(t2, a.z());          // a = 3 x1^2 + A z1^2
        }                               //
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

#ifdef __CUDACC__
        GEC_NV_DIAGNOSTIC_POP
#endif // __CUDACC__
    }

    GEC_HD static void add(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                           const Core &GEC_RSTRCT c) {
        // checking for infinity here is not necessary
        if (b.is_inf()) {
            a.set_inf();
        } else if (c.is_inf()) {
            a.set_inf();
        } else {
            {
                F x1z2, x2z1, y1z2, y2z1;
                F::mul(x1z2, b.x(), c.z()); // x1 z2
                F::mul(x2z1, c.x(), b.z()); // x2 z1
                F::mul(y1z2, b.y(), c.z()); // y1 z2
                F::mul(y2z1, c.y(), b.z()); // y2 z1

                if (x1z2 != x2z1 || y1z2 != y2z1) {
                    add_distinct_inner(a, b, c, x1z2, x2z1, y1z2, y2z1);
                    return;
                }
            }
            add_self(a, b);
        }
    }

    GEC_HD GEC_INLINE static void neg(Core &GEC_RSTRCT a,
                                      const Core &GEC_RSTRCT b) {
        a.x() = b.x();
        F::neg(a.y(), b.y());
        a.z() = b.z();
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_PROJECTIVE_HPP

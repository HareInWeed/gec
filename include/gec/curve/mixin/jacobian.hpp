#pragma once
#ifndef GEC_CURVE_MIXIN_JACOBAIN_HPP
#define GEC_CURVE_MIXIN_JACOBAIN_HPP

#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

/** @brief mixin that enables elliptic curve arithmetic with Jacobian coordinate
 */
template <typename Core, typename FIELD_T, bool InfYZero = true>
class GEC_EMPTY_BASES JacobianCoordinate
    : protected CRTP<Core, JacobianCoordinate<Core, FIELD_T, InfYZero>> {
    friend CRTP<Core, JacobianCoordinate<Core, FIELD_T, InfYZero>>;

    using F = FIELD_T;

    /** @brief add distinct point with some precomputed value
     *
     * ta == x1 z2^2
     * tb == x2 z1^2
     * tc == y1 z2^3
     * td == y2 z1^3
     */
    GEC_INLINE GEC_HD static void
    add_distinct_inner(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                       const Core &GEC_RSTRCT c, FIELD_T &GEC_RSTRCT ta,
                       FIELD_T &GEC_RSTRCT tb, FIELD_T &GEC_RSTRCT tc,
                       FIELD_T &GEC_RSTRCT td) {
        F::sub(tb, ta);           // e = b - a
        F::sub(td, tc);           // f = d - c
        F::mul(a.z(), tb, tb);    // e^2
        F::mul(a.y(), ta, a.z()); // a e^2
        F::mul(ta, a.z(), tb);    // e^3
        F::mul(a.z(), tc, ta);    // c e^3
        F::add(tc, a.y(), a.y()); // 2 a e^2
        F::mul(a.x(), td, td);    // f^2
        F::sub(a.x(), tc);        // f^2 - 2 a e^2
        F::sub(a.x(), ta);        // x = f^2 - 2 a e^2 - e^3
        F::sub(ta, a.y(), a.x()); // a e^2 - x
        F::mul(a.y(), td, ta);    // f (a e^2 - x)
        F::sub(a.y(), a.z());     // y = f (a e^2 - x) - c e^3
        F::mul(ta, b.z(), c.z()); // z1 z2
        F::mul(a.z(), ta, tb);    // z = z1 z2 e
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

    GEC_HD static bool on_curve(const Core &GEC_RSTRCT a) {
#ifdef __CUDACC__
        // suppress false positive NULL reference warning in nvcc
        GEC_NV_DIAGNOSTIC_PUSH
        GEC_NV_DIAG_SUPPRESS(284)
#endif // __CUDACC__

        F l, r, t1, t2;
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

    GEC_HD static void to_affine(Core &GEC_RSTRCT a) {
        if (a.z().is_mul_id()) {
            return;
        } else if (a.is_inf()) {
            a.set_affine_inf();
        } else {

            F::inv(a.z());            // z^-1
            F t1, t2;                 //
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
            F ta, tb, tc, td;

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

    GEC_HD static void add_distinct(Core &GEC_RSTRCT a,
                                    const Core &GEC_RSTRCT b,
                                    const Core &GEC_RSTRCT c) {
        F ta, tb, tc, td;
        {
            F t;
            F::mul(tc, c.z(), c.z()); // z2^2
            F::mul(t, tc, c.z());     // z2^3
            F::mul(ta, tc, b.x());    // a = x1 z2^2
            F::mul(tc, t, b.y());     // c = y1 z2^3

            F::mul(td, b.z(), b.z()); // z1^2
            F::mul(t, td, b.z());     // z1^3
            F::mul(tb, td, c.x());    // b = x2 z1^2
            F::mul(td, t, c.y());     // d = y2 z1^3
        }
        add_distinct_inner(a, b, c, ta, tb, tc, td);
    }

    GEC_HD static void add_self(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b) {
#ifdef __CUDACC__
        // suppress false positive NULL reference warning in nvcc
        GEC_NV_DIAGNOSTIC_PUSH
        GEC_NV_DIAG_SUPPRESS(284)
#endif // __CUDACC__

        F t4, t5;
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

    GEC_HD static void add(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                           const Core &GEC_RSTRCT c) {
        // checking for infinity here is not necessary
        if (b.is_inf()) {
            a = c;
        } else if (c.is_inf()) {
            a = b;
        } else {
            {
                F ta, tb, tc, td;
                {
                    F t;
                    F::mul(tc, c.z(), c.z()); // z2^2
                    F::mul(t, tc, c.z());     // z2^3
                    F::mul(ta, tc, b.x());    // a = x1 z2^2
                    F::mul(tc, t, b.y());     // c = y1 z2^3

                    F::mul(td, b.z(), b.z()); // z1^2
                    F::mul(t, td, b.z());     // z1^3
                    F::mul(tb, td, c.x());    // b = x2 z1^2
                    F::mul(td, t, c.y());     // d = y2 z1^3
                }
                if (ta != tb || tc != td) {
                    add_distinct_inner(a, b, c, ta, tb, tc, td);
                    return;
                }
            }
            add_self(a, b);
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

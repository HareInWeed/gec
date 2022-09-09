#pragma once
#ifndef GEC_CURVE_MIXIN_AFFINE_HPP
#define GEC_CURVE_MIXIN_AFFINE_HPP

#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

/** @brief mixin that enables elliptic curve arithmetic with affine coordinate
 *
 * TODO: list `FIELD_T` requirement
 */
template <typename Core, typename FIELD_T, bool InfYZero = true>
class GEC_EMPTY_BASES AffineCoordinate
    : protected CRTP<Core, AffineCoordinate<Core, FIELD_T, InfYZero>> {
    friend CRTP<Core, AffineCoordinate<Core, FIELD_T, InfYZero>>;

    using F = FIELD_T;

  public:
    using Field = FIELD_T;

    GEC_HD GEC_INLINE bool is_inf() const {
        return this->core().x().is_zero() &&
               (InfYZero ? this->core().y().is_zero()
                         : this->core().y().is_mul_id());
    }

    GEC_HD GEC_INLINE void set_inf() {
        this->core().x().set_zero();
        if (InfYZero) {
            this->core().y().set_zero();
        } else {
            this->core().y().set_mul_id();
        }
    }

    GEC_HD GEC_INLINE static bool eq(const Core &GEC_RSTRCT a,
                                     const Core &GEC_RSTRCT b) {
        return a.x() == b.x() && a.y() == b.y();
    }

    GEC_HD static bool on_curve(const Core &GEC_RSTRCT a) {
        if (a.is_inf()) {
            return true;
        }

#ifdef GEC_NVCC
        // suppress false positive NULL reference warning
        GEC_NV_DIAGNOSTIC_PUSH
        GEC_NV_DIAG_SUPPRESS(284)
#endif // GEC_NVCC

        F t, l, r;
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

#ifdef GEC_NVCC
        GEC_NV_DIAGNOSTIC_POP
#endif // GEC_NVCC
    }

    GEC_HD static void add_distinct(Core &GEC_RSTRCT a,
                                    const Core &GEC_RSTRCT b,
                                    const Core &GEC_RSTRCT c) {

        F d;                         //
        F::sub(d, b.x(), c.x());     // x1 - x2
        F::inv(d);                   // 1 / (x1 - x2)
        F l;                         //
        F::sub(a.y(), b.y(), c.y()); // y1 - y2
        F::mul(l, a.y(), d);         // l = (y1 - y2) / (x1 - x2)
        F::mul(a.y(), l, l);         // l^2
        F::sub(a.y(), b.x());        // l^2 - x1
        F::sub(a.x(), a.y(), c.x()); // x = l^2 - x1 - x2
        F::sub(d, b.x(), a.x());     // x1 - x
        F::mul(a.y(), l, d);         // l (x1 - x)
        F::sub(a.y(), b.y());        // y = l (x1 - x) - y1
    }

    GEC_HD static void add_self(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b) {
        if (b.y().is_zero()) {
            a.set_inf();
            return;
        }

#ifdef GEC_NVCC
        // suppress false positive NULL reference warning
        GEC_NV_DIAGNOSTIC_PUSH
        GEC_NV_DIAG_SUPPRESS(284)
#endif // GEC_NVCC

        F d;
        F::add(d, b.y(), b.y());     // 2 y1
        F::inv(d);                   // (2 y1)^-1
        F l;                         //
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

#ifdef GEC_NVCC
        GEC_NV_DIAGNOSTIC_POP
#endif // GEC_NVCC
    }

    GEC_HD static void add(Core &GEC_RSTRCT a, const Core &b, const Core &c) {
        if (b.is_inf()) {
            a = c;
        } else if (c.is_inf()) {
            a = b;
        } else if (b.x() != c.x()) {
            add_distinct(a, b, c);
        } else if (b.y() == c.y()) {
            add_self(a, b);
        } else {
            a.set_inf();
        }
    }

    GEC_HD GEC_INLINE static void neg(Core &GEC_RSTRCT a,
                                      const Core &GEC_RSTRCT b) {
        a.x() = b.x();
        F::neg(a.y(), b.y());
    }

    GEC_HD GEC_INLINE static void to_affine(Core &) {
        // added for consistant interface across different coordinate, no op
    }
    GEC_HD GEC_INLINE static void from_affine(Core &) {
        // added for consistant interface across different coordinate, no op
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_AFFINE_HPP

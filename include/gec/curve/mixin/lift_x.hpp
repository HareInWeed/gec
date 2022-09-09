#pragma once
#ifndef GEC_CURVE_MIXIN_LIFT_X_HPP
#define GEC_CURVE_MIXIN_LIFT_X_HPP

#include <gec/bigint/mixin/random.hpp>
#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

template <class Core, typename FIELD_T>
class LiftX : protected CRTP<Core, LiftX<Core, FIELD_T>> {
    friend CRTP<Core, LiftX<Core, FIELD_T>>;

    using F = FIELD_T;

  public:
    /**
     * @brief match last bit of y coordinate of point with given flag bit
     *
     * @param a point to edit
     * @param last_bit bit flag
     */
    static void match_y_bit(Core &a, bool last_bit) {
        using LimbT = typename F::LimbT;

        if ((a.y().array()[0] & LimbT(1)) != LimbT(last_bit)) {
            F ny;
            F::neg(ny, a.y());
            a.y() = ny;
        }
    }

    /**
     * @brief lift a field element to a point on curve
     *
     * @param a result point on curve, if returned flag is false, the value of
     *          `a` is undefined
     * @param x x coordinate of the point
     * @param rng random number generator
     * @return true lifting succeeded
     * @return false lifting failed
     */
    template <typename Rng>
    static bool lift_x(Core &a, const FIELD_T &x, GecRng<Rng> &rng) {
        F x2, x3;

#ifdef GEC_NVCC
        // suppress false positive NULL reference warning
        GEC_NV_DIAGNOSTIC_PUSH
        GEC_NV_DIAG_SUPPRESS(284)
#endif // GEC_NVCC

        F::mul(x2, x, x);          // x^2
        F::mul(x3, x2, x);         // x^3
        if (a.a() != nullptr) {    //
            F::mul(x2, *a.a(), x); // a x
            F::add(x3, x2);        // x^3 + a x
        }                          //
        if (a.b() != nullptr) {    //
            F::add(x3, *a.b());    // x^3 + a x + b
        }                          //

#ifdef GEC_NVCC
        GEC_NV_DIAGNOSTIC_POP
#endif // GEC_NVCC

        bool flag = F::mod_sqrt(a.y(), x3, rng);

        if (flag) {
            a.x() = x;
            Core::from_affine(a);
        }
        return flag;
    }

    /**
     * @brief lift a field element to a point on curve
     *
     * @param a result point on curve, if returned flag is false, the value of
     *          `a` is undefined
     * @param x x coordinate of the point
     * @param y_bit last bit of y coordinate
     * @return true lifting succeeded
     * @return false lifting failed
     */
    template <typename Rng>
    static bool lift_x(Core &a, const FIELD_T &x, bool y_bit,
                       GecRng<Rng> &rng) {
        if (lift_x(a, x, rng)) {
            match_y_bit(a, y_bit);
            return true;
        } else {
            return false;
        }
    }

    /**
     * @brief lift a field element `x` to a point on curve. if failed, keeping
     * increasing `x` until it succeeds
     *
     * @param a result point on curve
     * @param x x coordinate of the point
     */
    template <typename Rng>
    static void lift_x_with_inc(Core &a, const FIELD_T &x, GecRng<Rng> &rng) {
        if (lift_x(a, x, rng)) {
            return;
        }

        F tmp;

        F::add(tmp, x, F::mul_id());
        while (!lift_x(a, tmp, rng)) {
            F::add(tmp, F::mul_id());
        }
    }

    /**
     * @brief lift a field element `x` to a point on curve. if failed, keeping
     * increasing `x` until it succeeds
     *
     * @param a result point on curve
     * @param x x coordinate of the point
     */
    template <typename Rng>
    static void lift_x_with_inc(Core &a, const FIELD_T &x, bool y_bit,
                                GecRng<Rng> &rng) {
        lift_x_with_inc(a, x, rng);
        match_y_bit(a, y_bit);
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_LIFT_X_HPP

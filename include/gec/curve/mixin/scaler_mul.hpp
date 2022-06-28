#pragma once
#ifndef GEC_CURVE_MIXIN_SCALER_MUL_HPP
#define GEC_CURVE_MIXIN_SCALER_MUL_HPP

#include <gec/utils/crtp.hpp>
#include <gec/utils/misc.hpp>
#include <gec/utils/sequence.hpp>

namespace gec {

namespace curve {

/** @brief mixin that enables exponentiation
 *
 * require `Core::set_mul_id`, `Core::mul` methods
 */
template <class Core>
class ScalerMul : protected CRTP<Core, ScalerMul<Core>> {
    friend CRTP<Core, ScalerMul<Core>>;

  public:
    template <typename CTX, size_t N = 1, typename IntT = uint32_t,
              std::enable_if_t<std::is_integral<IntT>::value> * = nullptr>
    __host__ __device__ static void
    mul(Core &GEC_RSTRCT a, const IntT *GEC_RSTRCT e, const Core &GEC_RSTRCT b,
        CTX &GEC_RSTRCT ctx) {
        auto &ctx_view = ctx.template view_as<Core>();

        auto &ap = ctx_view.template get<0>();
        bool in_dist = true;
#define GEC_DIST_ (in_dist ? ap : a)
#define GEC_SRC_ (in_dist ? a : ap)
        a.set_inf();
        constexpr size_t Bits = utils::type_bits<IntT>::value;
        int i = N - 1, j;
        for (; i >= 0; --i) {
            for (j = Bits - 1; j >= 0; --j) {
                if ((IntT(1) << j) & e[i]) {
                    goto mul;
                }
            }
        }
    mul:
        for (; i >= 0; --i) {
            for (; j >= 0; --j) {
                Core::add(GEC_DIST_, GEC_SRC_, GEC_SRC_, ctx_view.rest());
                in_dist = !in_dist;
                if ((IntT(1) << j) & e[i]) {
                    Core::add(GEC_DIST_, GEC_SRC_, b, ctx_view.rest());
                    in_dist = !in_dist;
                }
            }
            j = Bits - 1;
        }
        if (!in_dist) {
            a = ap;
        }
#undef GEC_DIST_
#undef GEC_SRC_
    }

    template <typename CTX, typename IntT,
              std::enable_if_t<std::is_integral<IntT>::value> * = nullptr>
    __host__ __device__ GEC_INLINE static void
    mul(Core &GEC_RSTRCT a, const IntT &GEC_RSTRCT e, const Core &GEC_RSTRCT b,
        CTX &GEC_RSTRCT ctx) {
        mul(a, &e, b, ctx);
    }

    template <typename CTX, typename IntT,
              std::enable_if_t<!std::is_integral<IntT>::value> * = nullptr>
    __host__ __device__ GEC_INLINE static void
    mul(Core &GEC_RSTRCT a, const IntT &GEC_RSTRCT e, const Core &GEC_RSTRCT b,
        CTX &GEC_RSTRCT ctx) {
        mul<CTX, IntT::LimbN>(a, e.array(), b, ctx);
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_SCALER_MUL_HPP

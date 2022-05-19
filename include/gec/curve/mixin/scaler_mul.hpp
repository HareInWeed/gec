#pragma once
#ifndef GEC_CURVE_MIXIN_SCALER_MUL_HPP
#define GEC_CURVE_MIXIN_SCALER_MUL_HPP

#include <gec/utils/context_check.hpp>
#include <gec/utils/crtp.hpp>
#include <gec/utils/sequence.hpp>
#include <utility>

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
        // TODO: check context capacity

        Core &ap = ctx.template get_p<0>();
        bool need_copy = false;
        Core *p1 = &a, *p2 = &ap;
        p1->set_inf();
        constexpr size_t Bits = std::numeric_limits<IntT>::digits;
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
                Core::add(*p2, *p1, *p1, ctx.template rest<0, 1>());
                std::swap(p1, p2);
                need_copy = !need_copy;
                if ((IntT(1) << j) & e[i]) {
                    Core::add(*p2, *p1, b, ctx.template rest<0, 1>());
                    std::swap(p1, p2);
                    need_copy = !need_copy;
                }
            }
            j = Bits - 1;
        }
        if (need_copy) {
            a = ap;
        }
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

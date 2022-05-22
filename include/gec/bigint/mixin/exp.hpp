#pragma once
#ifndef GEC_BIGINT_MIXIN_EXPONENTIATION_HPP
#define GEC_BIGINT_MIXIN_EXPONENTIATION_HPP

#include <gec/utils/crtp.hpp>
#include <utility>

namespace gec {

namespace bigint {

/** @brief mixin that enables exponentiation
 *
 * require `Core::set_mul_id`, `Core::mul` methods
 */
template <class Core>
class Exponentiation : protected CRTP<Core, Exponentiation<Core>> {
    friend CRTP<Core, Exponentiation<Core>>;

  public:
    template <typename CTX, size_t N = 1, typename IntT = uint32_t,
              std::enable_if_t<std::is_integral<IntT>::value> * = nullptr>
    __host__ __device__ static void
    pow(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b, const IntT *GEC_RSTRCT e,
        CTX &GEC_RSTRCT ctx) {
        auto ctx_view = ctx.template view_as<Core>();

        auto &ap = ctx_view.template get<0>();

        bool need_copy = false;
        Core *p1 = &a, *p2 = &ap;
        p1->set_mul_id();
        constexpr size_t Bits = std::numeric_limits<IntT>::digits;
        int i = N - 1, j;
        for (; i >= 0; --i) {
            for (j = Bits - 1; j >= 0; --j) {
                if ((IntT(1) << j) & e[i]) {
                    goto exp;
                }
            }
        }
    exp:
        for (; i >= 0; --i) {
            for (; j >= 0; --j) {
                Core::mul(*p2, *p1, *p1);
                std::swap(p1, p2);
                need_copy = !need_copy;
                if ((IntT(1) << j) & e[i]) {
                    Core::mul(*p2, *p1, b);
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
    pow(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b, const IntT &GEC_RSTRCT e,
        CTX &GEC_RSTRCT ctx) {
        pow(a, b, &e, ctx);
    }

    template <typename CTX, typename IntT,
              std::enable_if_t<!std::is_integral<IntT>::value> * = nullptr>
    __host__ __device__ GEC_INLINE static void
    pow(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b, const IntT &GEC_RSTRCT e,
        CTX &GEC_RSTRCT ctx) {
        pow<CTX, IntT::LimbN>(a, b, e.array(), ctx);
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_EXPONENTIATION_HPP

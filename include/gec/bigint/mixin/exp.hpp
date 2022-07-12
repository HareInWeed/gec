#pragma once
#ifndef GEC_BIGINT_MIXIN_EXPONENTIATION_HPP
#define GEC_BIGINT_MIXIN_EXPONENTIATION_HPP

#include <gec/utils/crtp.hpp>
#include <gec/utils/misc.hpp>

namespace gec {

namespace bigint {

/** @brief mixin that enables exponentiation
 *
 * require `Core::set_mul_id`, `Core::mul` methods
 */
template <class Core>
class GEC_EMPTY_BASES Exponentiation
    : protected CRTP<Core, Exponentiation<Core>> {
    friend CRTP<Core, Exponentiation<Core>>;

  public:
    template <typename CTX, size_t N = 1, typename IntT = uint32_t,
              std::enable_if_t<std::is_integral<IntT>::value> * = nullptr>
    __host__ __device__ static void
    pow(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b, const IntT *GEC_RSTRCT e,
        CTX &GEC_RSTRCT ctx) {
        auto &ctx_view = ctx.template view_as<Core>();

        auto &ap = ctx_view.template get<0>();

        bool in_dist = true;
#define GEC_DIST_ (in_dist ? ap : a)
#define GEC_SRC_ (in_dist ? a : ap)
        a.set_mul_id();
        constexpr size_t Bits = utils::type_bits<IntT>::value;
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
                Core::mul(GEC_DIST_, GEC_SRC_, GEC_SRC_);
                in_dist = !in_dist;
                if ((IntT(1) << j) & e[i]) {
                    Core::mul(GEC_DIST_, GEC_SRC_, b);
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

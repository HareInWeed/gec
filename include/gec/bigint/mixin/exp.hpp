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
    template <
        size_t N, typename IntT,
        std::enable_if_t<std::numeric_limits<IntT>::is_integer> * = nullptr>
    GEC_HD static void pow(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                           const IntT *GEC_RSTRCT e) {
        Core ap;

        bool in_dest = true;
#define GEC_DEST_ (in_dest ? ap : a)
#define GEC_SRC_ (in_dest ? a : ap)
#define GEC_RELOAD_ (in_dest = !in_dest)
        a.set_mul_id();
        constexpr size_t Bits = utils::type_bits<IntT>::value;
        int i = N - 1, j;
        for (; i >= 0; --i) {
            if (e[i]) {
                j = utils::most_significant_bit(e[i]);
                break;
            }
        }
        for (; i >= 0; --i) {
            for (; j >= 0; --j) {
                Core::mul(GEC_DEST_, GEC_SRC_, GEC_SRC_);
                GEC_RELOAD_;
                if ((IntT(1) << j) & e[i]) {
                    Core::mul(GEC_DEST_, GEC_SRC_, b);
                    GEC_RELOAD_;
                }
            }
            j = Bits - 1;
        }
        if (!in_dest) {
            a = ap;
        }
#undef GEC_DEST_
#undef GEC_SRC_
#undef GEC_RELOAD_
    }

    template <
        typename IntT,
        std::enable_if_t<std::numeric_limits<IntT>::is_integer> * = nullptr>
    GEC_HD GEC_INLINE static void pow(Core &GEC_RSTRCT a,
                                      const Core &GEC_RSTRCT b,
                                      const IntT &GEC_RSTRCT e) {
        pow<1>(a, b, &e);
    }

    template <
        typename IntT,
        std::enable_if_t<!std::numeric_limits<IntT>::is_integer> * = nullptr>
    GEC_HD GEC_INLINE static void pow(Core &GEC_RSTRCT a,
                                      const Core &GEC_RSTRCT b,
                                      const IntT &GEC_RSTRCT e) {
        pow<IntT::LimbN>(a, b, e.array());
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_EXPONENTIATION_HPP

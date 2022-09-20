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
class GEC_EMPTY_BASES ScalarMul : protected CRTP<Core, ScalarMul<Core>> {
    friend CRTP<Core, ScalarMul<Core>>;

  public:
    template <
        size_t N, typename IntT,
        std::enable_if_t<std::numeric_limits<IntT>::is_integer> * = nullptr>
    GEC_HD static void mul(Core &GEC_RSTRCT a, const IntT *GEC_RSTRCT s,
                           const Core &GEC_RSTRCT b) {
        Core ap;

        bool in_dest = true;
#define GEC_DEST_ (in_dest ? ap : a)
#define GEC_SRC_ (in_dest ? a : ap)
#define GEC_RELOAD_ (in_dest = !in_dest)

        a.set_inf();
        constexpr size_t Bits = utils::type_bits<IntT>::value;
        int i = N - 1, j = -1;
        for (; i >= 0; --i) {
            if (s[i]) {
                j = int(utils::most_significant_bit(s[i]));
                break;
            }
        }
        for (; i >= 0; --i) {
            for (; j >= 0; --j) {
                Core::add(GEC_DEST_, GEC_SRC_, GEC_SRC_);
                GEC_RELOAD_;
                if ((IntT(1) << j) & s[i]) {
                    Core::add(GEC_DEST_, GEC_SRC_, b);
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
    GEC_HD GEC_INLINE static void mul(Core &GEC_RSTRCT a,
                                      const IntT &GEC_RSTRCT e,
                                      const Core &GEC_RSTRCT b) {
        mul<1>(a, &e, b);
    }

    template <
        typename IntT,
        std::enable_if_t<!std::numeric_limits<IntT>::is_integer> * = nullptr>
    GEC_HD GEC_INLINE static void mul(Core &GEC_RSTRCT a,
                                      const IntT &GEC_RSTRCT e,
                                      const Core &GEC_RSTRCT b) {
        mul<IntT::LimbN>(a, e.array(), b);
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_SCALER_MUL_HPP

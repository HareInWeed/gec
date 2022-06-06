#pragma once
#ifndef GEC_CURVE_MIXIN_CONTEXT_HPP
#define GEC_CURVE_MIXIN_CONTEXT_HPP

#include <gec/bigint/mixin/context.hpp>
#include <gec/utils/context.hpp>
#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

/// @brief Minimum context size for any function in GEC to work
constexpr size_t MIN_POINT_NUM = 7;

/** @brief mixin that add context type
 *
 * TODO
 */
template <typename Core, size_t FN = bigint::MIN_BIGINT_NUM,
          size_t PN = MIN_POINT_NUM>
class WithPointContext : protected CRTP<Core, WithPointContext<Core, FN, PN>> {
    friend CRTP<Core, WithPointContext<Core, FN, PN>>;

  public:
    template <typename P = Core>
    using Context = utils::Context<
        PN * utils::AlignTo<sizeof(P), alignof(P)>::value +
            FN * utils::AlignTo<sizeof(typename P::Field),
                                alignof(typename P::Field)>::value,
        alignof(P), 0>;
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_CONTEXT_HPP

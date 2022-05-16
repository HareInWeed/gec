#pragma once
#ifndef GEC_CURVE_MIXIN_CONTEXT_HPP
#define GEC_CURVE_MIXIN_CONTEXT_HPP

#include "../data/point_context.hpp"
#include <gec/bigint/mixin/context.hpp>
#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

/// @brief Minimum context size for any function in GEC to work
constexpr size_t MIN_POINT_CONTEXT_SIZE = 7;

/** @brief mixin that add context type
 *
 * TODO
 */
template <typename Core, size_t FN = bigint::MIN_CONTEXT_SIZE,
          size_t PN = MIN_POINT_CONTEXT_SIZE>
class WithPointContext : protected CRTP<Core, WithPointContext<Core, FN, PN>> {
    friend CRTP<Core, WithPointContext<Core, FN, PN>>;

  public:
    template <typename P = Core>
    using Context = PointContext<P, FN, PN>;
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_CONTEXT_HPP

#pragma once
#ifndef GEC_CURVE_MIXIN_CONTEXT_HPP
#define GEC_CURVE_MIXIN_CONTEXT_HPP

#include "../data/point_context.hpp"
#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

/// @brief Minimum context size for any function in GEC to work
constexpr size_t MIN_COMPOUND_CONTEXT_SIZE = 5;

/** @brief mixin that add context type
 *
 * TODO
 */
template <typename Core, size_t N = MIN_COMPOUND_CONTEXT_SIZE>
class WithPointContext : protected CRTP<Core, WithPointContext<Core, N>> {
    friend CRTP<Core, WithPointContext<Core, N>>;

  public:
    template <typename P = Core>
    using Context = CompoundContext<P, N>;
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_CONTEXT_HPP

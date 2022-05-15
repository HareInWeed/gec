#pragma once
#ifndef GEC_BIGINT_MIXIN_CONTEXT_HPP
#define GEC_BIGINT_MIXIN_CONTEXT_HPP

#include <gec/utils/context.hpp>
#include <gec/utils/crtp.hpp>

namespace gec {

namespace bigint {

/// @brief Minimum context size for any function in GEC to work
constexpr size_t MIN_CONTEXT_SIZE = 5;

/** @brief mixin that add context type
 *
 * The mixin will add `Core::Context` type, a `gec::utils::Context` able to hold
 * up tp `N` `Core`, in `Core`.
 *
 * The default value of `N` is `gec::bigint::MIN_CONTEXT_SIZE`. Note that not
 * every function requires such a context size. You may want to choose a smaller
 * size if only a subset of function in GEC is used.
 */
template <typename Core, size_t N = MIN_CONTEXT_SIZE>
class WithBigintContext : protected CRTP<Core, WithBigintContext<Core, N>> {
    friend CRTP<Core, WithBigintContext<Core, N>>;

  public:
    using Context = utils::Context<Core, N>;
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_CONTEXT_HPP

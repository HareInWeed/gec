#pragma once
#ifndef GEC_BIGINT_MIXIN_HASHER_HPP
#define GEC_BIGINT_MIXIN_HASHER_HPP

#include "../data/hasher.hpp"
#include <gec/utils/crtp.hpp>

namespace gec {

namespace bigint {

/** @brief mixin that add a hasher
 */
template <typename Core>
class WithArrayHasher : protected CRTP<Core, WithArrayHasher<Core>> {
    friend CRTP<Core, WithArrayHasher<Core>>;

  public:
    using Hasher = ArrayHasher<Core>;
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_HASHER_HPP

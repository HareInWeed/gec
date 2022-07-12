#pragma once
#ifndef GEC_CURVE_MIXIN_HASHER_HPP
#define GEC_CURVE_MIXIN_HASHER_HPP

#include "../data/hasher.hpp"
#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

/** @brief mixin that add a hasher
 */
template <typename Core>
class GEC_EMPTY_BASES WithPointHasher
    : protected CRTP<Core, WithPointHasher<Core>> {
    friend CRTP<Core, WithPointHasher<Core>>;

  public:
    using Hasher = PointHasher<Core>;
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_HASHER_HPP

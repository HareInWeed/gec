#pragma once
#ifndef GEC_CURVE_MIXIN_COMPWISE_EQ_HPP
#define GEC_CURVE_MIXIN_COMPWISE_EQ_HPP

#include <cstdio>
#include <gec/utils/crtp.hpp>
#include <gec/utils/sequence.hpp>

namespace gec {

namespace curve {

/** @brief test if two points are equal by components
 */
template <typename Core>
class GEC_EMPTY_BASES CompWiseEq : protected CRTP<Core, CompWiseEq<Core>> {
    friend CRTP<Core, CompWiseEq<Core>>;

  public:
    __host__ __device__ bool operator==(const Core &other) const {
        return utils::VtSeqAll<
            Core::CompN, typename Core::CompT,
            utils::ops::Eq<typename Core::CompT>>::call(this->core().array(),
                                                        other.array());
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_COMPWISE_EQ_HPP

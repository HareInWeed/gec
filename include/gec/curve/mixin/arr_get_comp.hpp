#pragma once
#ifndef GEC_CURVE_MIXIN_ARR_GET_COMP_HPP
#define GEC_CURVE_MIXIN_ARR_GET_COMP_HPP

#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

/** @brief mixin that enables ...
 */
template <typename Core>
class GEC_EMPTY_BASES ArrGetCompLE : protected CRTP<Core, ArrGetCompLE<Core>> {
    friend CRTP<Core, ArrGetCompLE<Core>>;

  public:
    template <size_t I, typename P = Core,
              std::enable_if_t<(I < P::CompN)> * = nullptr>
    __host__ __device__ GEC_INLINE const typename P::CompT &get() const {
        return this->core().array()[I];
    }

    template <size_t I, typename P = Core,
              std::enable_if_t<(I < P::CompN)> * = nullptr>
    __host__ __device__ GEC_INLINE typename P::CompT &get() {
        return this->core().array()[I];
    }
};

/** @brief mixin that enables ...
 */
template <typename Core>
class GEC_EMPTY_BASES ArrGetCompBE : protected CRTP<Core, ArrGetCompBE<Core>> {
    friend CRTP<Core, ArrGetCompBE<Core>>;

  public:
    template <size_t I, typename P = Core,
              std::enable_if_t<(I < P::CompN)> * = nullptr>
    __host__ __device__ GEC_INLINE const typename P::CompT &get() const {
        return this->core().array()[Core::CompN - 1 - I];
    }

    template <size_t I, typename P = Core,
              std::enable_if_t<(I < P::CompN)> * = nullptr>
    __host__ __device__ GEC_INLINE typename P::CompT &get() {
        return this->core().array()[Core::CompN - 1 - I];
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_ARR_GET_COMP_HPP

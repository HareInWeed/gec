#pragma once
#ifndef GEC_CURVE_MIXIN_NAMED_COMP_HPP
#define GEC_CURVE_MIXIN_NAMED_COMP_HPP

#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

/** @brief mixin that enables ...
 */
template <typename Core>
class GEC_EMPTY_BASES NamedComp : protected CRTP<Core, NamedComp<Core>> {
    friend CRTP<Core, NamedComp<Core>>;

  public:
    template <typename P = Core, std::enable_if_t<(P::CompN > 0)> * = nullptr>
    __host__ __device__ GEC_INLINE const typename P::CompT &x() const {
        return this->core().template get<0>();
    }
    template <typename P = Core, std::enable_if_t<(P::CompN > 0)> * = nullptr>
    __host__ __device__ GEC_INLINE typename P::CompT &x() {
        return this->core().template get<0>();
    }

    template <typename P = Core, std::enable_if_t<(P::CompN > 1)> * = nullptr>
    __host__ __device__ GEC_INLINE const typename P::CompT &y() const {
        return this->core().template get<1>();
    }
    template <typename P = Core, std::enable_if_t<(P::CompN > 1)> * = nullptr>
    __host__ __device__ GEC_INLINE typename P::CompT &y() {
        return this->core().template get<1>();
    }

    template <typename P = Core, std::enable_if_t<(P::CompN > 2)> * = nullptr>
    __host__ __device__ GEC_INLINE const typename P::CompT &z() const {
        return this->core().template get<2>();
    }
    template <typename P = Core, std::enable_if_t<(P::CompN > 2)> * = nullptr>
    __host__ __device__ GEC_INLINE typename P::CompT &z() {
        return this->core().template get<2>();
    }

    template <typename P = Core, std::enable_if_t<(P::CompN > 3)> * = nullptr>
    __host__ __device__ GEC_INLINE const typename P::CompT &z1() const {
        return this->core().template get<3>();
    }
    template <typename P = Core, std::enable_if_t<(P::CompN > 3)> * = nullptr>
    __host__ __device__ GEC_INLINE typename P::CompT &z1() {
        return this->core().template get<3>();
    }

    template <typename P = Core, std::enable_if_t<(P::CompN > 4)> * = nullptr>
    __host__ __device__ GEC_INLINE const typename P::CompT &z2() const {
        return this->core().template get<4>();
    }
    template <typename P = Core, std::enable_if_t<(P::CompN > 4)> * = nullptr>
    __host__ __device__ GEC_INLINE typename P::CompT &z2() {
        return this->core().template get<4>();
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_NAMED_COMP_HPP

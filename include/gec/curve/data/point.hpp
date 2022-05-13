#pragma once
#ifndef GEC_CURVE_DATA_POINT_HPP
#define GEC_CURVE_DATA_POINT_HPP

#include <gec/bigint/data/array.hpp>

namespace gec {

namespace curve {

/** @brief N-dimensional point
 */
template <typename COMP_T, size_t N>
class Point : bigint::ArrayLE<COMP_T, N> {
    using Base = bigint::ArrayLE<COMP_T, N>;

  public:
    using CompT = COMP_T;
    const static size_t CompN = N;

    using Base::Base;

    template <size_t I, std::enable_if_t<(I < N)> * = nullptr>
    __host__ __device__ GEC_INLINE const CompT &get() const {
        return this->array()[I];
    }
    template <size_t I, std::enable_if_t<(I < N)> * = nullptr>
    __host__ __device__ GEC_INLINE CompT &get() {
        return this->array()[I];
    }

    template <size_t N_ = N, std::enable_if_t<(N_ > 0)> * = nullptr>
    __host__ __device__ GEC_INLINE const CompT &x() const {
        return this->template get<0>();
    }
    template <size_t N_ = N, std::enable_if_t<(N_ > 0)> * = nullptr>
    __host__ __device__ GEC_INLINE CompT &x() {
        return this->template get<0>();
    }

    template <size_t N_ = N, std::enable_if_t<(N_ > 1)> * = nullptr>
    __host__ __device__ GEC_INLINE const CompT &y() const {
        return this->template get<1>();
    }
    template <size_t N_ = N, std::enable_if_t<(N_ > 1)> * = nullptr>
    __host__ __device__ GEC_INLINE CompT &y() {
        return this->template get<1>();
    }

    template <size_t N_ = N, std::enable_if_t<(N_ > 2)> * = nullptr>
    __host__ __device__ GEC_INLINE const CompT &z() const {
        return this->template get<2>();
    }
    template <size_t N_ = N, std::enable_if_t<(N_ > 2)> * = nullptr>
    __host__ __device__ GEC_INLINE CompT &z() {
        return this->template get<2>();
    }

    template <size_t N_ = N, std::enable_if_t<(N_ > 3)> * = nullptr>
    __host__ __device__ GEC_INLINE const CompT &z1() const {
        return this->template get<3>();
    }
    template <size_t N_ = N, std::enable_if_t<(N_ > 3)> * = nullptr>
    __host__ __device__ GEC_INLINE CompT &z1() {
        return this->template get<3>();
    }

    template <size_t N_ = N, std::enable_if_t<(N_ > 4)> * = nullptr>
    __host__ __device__ GEC_INLINE const CompT &z2() const {
        return this->template get<4>();
    }
    template <size_t N_ = N, std::enable_if_t<(N_ > 4)> * = nullptr>
    __host__ __device__ GEC_INLINE CompT &z2() {
        return this->template get<4>();
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_DATA_POINT_HPP

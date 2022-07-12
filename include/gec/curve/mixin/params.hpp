#pragma once
#ifndef GEC_CURVE_MIXIN_PARAMS_HPP
#define GEC_CURVE_MIXIN_PARAMS_HPP

#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

template <class Core, typename FIELD_T, const FIELD_T *A, const FIELD_T *B,
          const FIELD_T *d_A = nullptr, const FIELD_T *d_B = nullptr>
class GEC_EMPTY_BASES CurveParams
    : protected CRTP<Core, CurveParams<Core, FIELD_T, A, B, d_A, d_B>> {
    friend CRTP<Core, CurveParams<Core, FIELD_T, A, B, d_A, d_B>>;

  public:
    __host__ __device__ GEC_INLINE static const FIELD_T &a() {
#ifdef __CUDA_ARCH__
        return *d_A;
#else
        return *A;
#endif // __CUDA_ARCH__
    }

    __host__ __device__ GEC_INLINE static const FIELD_T &b() {
#ifdef __CUDA_ARCH__
        return *d_B;
#else
        return *B;
#endif // __CUDA_ARCH__
    }
};

} // namespace curve

} // namespace gec

#endif // GEC_CURVE_MIXIN_PARAMS_HPP

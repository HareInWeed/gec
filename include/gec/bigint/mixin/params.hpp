#pragma once
#ifndef GEC_BIGINT_MIXIN_PARAMS_HPP
#define GEC_BIGINT_MIXIN_PARAMS_HPP

#include <gec/utils/crtp.hpp>

namespace gec {

namespace bigint {

template <class Core, typename LIMB_T, size_t LIMB_N,
          const LIMB_T (*MOD)[LIMB_N], const LIMB_T (*d_MOD)[LIMB_N] = nullptr>
class GEC_EMPTY_BASES AddGroupParams
    : protected CRTP<Core, AddGroupParams<Core, LIMB_T, LIMB_N, MOD, d_MOD>> {
    friend CRTP<Core, AddGroupParams<Core, LIMB_T, LIMB_N, MOD, d_MOD>>;

  public:
    __host__ __device__ GEC_INLINE static constexpr const Core &mod() {
#ifdef __CUDA_ARCH__
        return *reinterpret_cast<const Core *>(d_MOD);
#else
        return *reinterpret_cast<const Core *>(MOD);
#endif
    }
};

template <class Core, typename LIMB_T, size_t LIMB_N, LIMB_T MOD_P,
          const LIMB_T (*RR)[LIMB_N], const LIMB_T (*OneR)[LIMB_N],
          const LIMB_T (*d_RR)[LIMB_N] = nullptr,
          const LIMB_T (*d_OneR)[LIMB_N] = nullptr>
class GEC_EMPTY_BASES MontgomeryParams
    : protected CRTP<Core, MontgomeryParams<Core, LIMB_T, LIMB_N, MOD_P, RR,
                                            OneR, d_RR, d_OneR>> {
    friend CRTP<Core, MontgomeryParams<Core, LIMB_T, LIMB_N, MOD_P, RR, OneR,
                                       d_RR, d_OneR>>;

  public:
    __host__ __device__ GEC_INLINE static constexpr LIMB_T mod_p() {
        return MOD_P;
    }
    __host__ __device__ GEC_INLINE static constexpr const Core &r_sqr() {
#ifdef __CUDA_ARCH__
        return *reinterpret_cast<const Core *>(d_RR);
#else
        return *reinterpret_cast<const Core *>(RR);
#endif
    }
    __host__ __device__ GEC_INLINE static constexpr const Core &one_r() {
#ifdef __CUDA_ARCH__
        return *reinterpret_cast<const Core *>(d_OneR);
#else
        return *reinterpret_cast<const Core *>(OneR);
#endif
    }
};

} // namespace bigint

} // namespace gec

#endif // GEC_BIGINT_MIXIN_PARAMS_HPP

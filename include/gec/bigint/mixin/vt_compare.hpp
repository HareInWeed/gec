#pragma once
#ifndef GEC_BIGINT_MIXIN_VT_COMPARE_HPP
#define GEC_BIGINT_MIXIN_VT_COMPARE_HPP

#include <gec/utils/crtp.hpp>
#include <gec/utils/sequence.hpp>

namespace gec {

namespace bigint {

/** @brief TODO:
 */
template <class Core, class LIMB_T, size_t LIMB_N>
struct VtCompare : protected CRTP<Core, VtCompare<Core, LIMB_T, LIMB_N>> {
    friend CRTP<Core, VtCompare<Core, LIMB_T, LIMB_N>>;

    __host__ __device__ GEC_INLINE utils::CmpEnum cmp(const Core &other) const {
        return utils::VtSeqCmp<LIMB_N, LIMB_T>::call(this->core().array(),
                                                     other.array());
    }
    __host__ __device__ GEC_INLINE bool operator==(const Core &other) const {
        return utils::VtSeqAll<LIMB_N, LIMB_T, utils::ops::Eq<LIMB_T>>::call(
            this->core().array(), other.array());
    }
    __host__ __device__ GEC_INLINE bool operator!=(const Core &other) const {
        return !(this->core() == other);
    }
    __host__ __device__ GEC_INLINE bool operator<(const Core &other) const {
        return this->cmp(other) == utils::CmpEnum::Lt;
    }
    __host__ __device__ GEC_INLINE bool operator>=(const Core &other) const {
        return !(this->core() < other);
    }
    __host__ __device__ GEC_INLINE bool operator>(const Core &other) const {
        return this->cmp(other) == utils::CmpEnum::Gt;
    }
    __host__ __device__ GEC_INLINE bool operator<=(const Core &other) const {
        return !(this->core() > other);
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_VT_COMPARE_HPP

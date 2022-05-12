#pragma once
#ifndef GEC_UTILS_CRTP_HPP
#define GEC_UTILS_CRTP_HPP

#include "basic.hpp"

#include <type_traits>

namespace gec {

template <typename Core, typename Mixin>
struct CRTP {
  public:
    __host__ __device__ GEC_INLINE Core const &core() const {
        return static_cast<const Core &>(*this);
    }
    __host__ __device__ GEC_INLINE Core &core() {
        return static_cast<Core &>(*this);
    }

  private:
    CRTP() {}
    friend Mixin;
};

} // namespace gec

#endif // !GEC_UTILS_CRTP_HPP

#pragma once
#ifndef GEC_UTILS_CRTP_HPP
#define GEC_UTILS_CRTP_HPP

#include "basic.hpp"

#include <type_traits>

namespace gec {

template <typename Core, typename Mixin>
struct GEC_EMPTY_BASES CRTP {
  public:
    GEC_HD GEC_INLINE constexpr Core const &core() const {
        return static_cast<const Core &>(*this);
    }
    GEC_HD GEC_INLINE constexpr Core &core() {
        return static_cast<Core &>(*this);
    }

  private:
    GEC_HD GEC_INLINE constexpr CRTP() noexcept {}
    friend Mixin;
};

} // namespace gec

#endif // !GEC_UTILS_CRTP_HPP

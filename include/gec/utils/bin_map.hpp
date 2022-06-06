#pragma once
#ifndef GEC_UTILS_BIN_MAP_HPP
#define GEC_UTILS_BIN_MAP_HPP

#include "basic.hpp"

namespace gec {

namespace utils {

template <size_t N, typename Key, typename Value>
struct BinMap {
    size_t left[2 * N];
    size_t right[2 * N];
    Value vals[N];
};

} // namespace utils

} // namespace gec

#endif // !GEC_UTILS_BIN_MAP_HPP
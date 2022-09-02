#pragma once
#ifndef GEC_CURVE_DATA_HASHER_HPP
#define GEC_CURVE_DATA_HASHER_HPP

#include <gec/bigint/data/hasher.hpp>
#include <gec/utils/basic.hpp>
#include <gec/utils/hash.hpp>

#include <functional>

namespace gec {

namespace curve {

template <typename POINT>
struct GEC_EMPTY_BASES PointHasher {
    using argument_type = POINT;
    using result_type = size_t;

    GEC_HD GEC_INLINE constexpr PointHasher() {}
    GEC_HD GEC_INLINE constexpr PointHasher(const PointHasher &) {}
    GEC_HD GEC_INLINE PointHasher &operator=(const PointHasher &) {
        return *this;
    }

    GEC_HD GEC_INLINE result_type operator()(const argument_type &point) const {
        using Hasher = gec::bigint::ArrayHasher<typename POINT::CompT>;
        size_t seed = 0;
        Hasher hasher;
        hash::SeqHasher<typename POINT::CompT, POINT::CompN, Hasher>::call(
            seed, point.array(), hasher);
        return seed;
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_DATA_HASHER_HPP

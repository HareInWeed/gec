#pragma once
#ifndef GEC_BIGINT_DATA_HASHER_HPP
#define GEC_BIGINT_DATA_HASHER_HPP

#include <gec/utils/basic.hpp>
#include <gec/utils/hash.hpp>
#include <gec/utils/operators.hpp>

namespace gec {

namespace bigint {

template <typename ARRAY>
struct GEC_EMPTY_BASES ArrayHasher {
    using argument_type = ARRAY;
    using result_type = size_t;

    GEC_HD GEC_INLINE constexpr ArrayHasher() {}
    GEC_HD GEC_INLINE constexpr ArrayHasher(const ArrayHasher &) {}
    GEC_HD GEC_INLINE ArrayHasher &operator=(const ArrayHasher &) {
        return *this;
    }

    GEC_HD GEC_INLINE result_type operator()(const argument_type &array) const {
        using H = hash::Hash<typename argument_type::LimbT>;
        size_t seed = 0;
        H hasher;
        hash::SeqHasher<typename ARRAY::LimbT, ARRAY::LimbN, H>::call(
            seed, array.array(), hasher);
        return seed;
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_DATA_HASHER_HPP

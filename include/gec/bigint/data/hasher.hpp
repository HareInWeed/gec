#pragma once
#ifndef GEC_BIGINT_DATA_HASHER_HPP
#define GEC_BIGINT_DATA_HASHER_HPP

#include <gec/utils/basic.hpp>
#include <gec/utils/hash.hpp>
#include <gec/utils/operators.hpp>

namespace gec {

namespace bigint {

template <typename ARRAY>
struct ArrayHasher {
    using argument_type = ARRAY;
    using result_type = size_t;

    __host__ __device__ GEC_INLINE constexpr ArrayHasher() {}
    __host__ __device__ GEC_INLINE constexpr ArrayHasher(const ArrayHasher &) {}
    __host__ __device__ GEC_INLINE ArrayHasher &operator=(const ArrayHasher &) {
        return *this;
    }

    __host__ __device__ GEC_INLINE result_type
    operator()(const argument_type &array) const {
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

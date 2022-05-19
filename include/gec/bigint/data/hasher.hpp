#pragma once
#ifndef GEC_BIGINT_DATA_HASHER_HPP
#define GEC_BIGINT_DATA_HASHER_HPP

#include <gec/utils/basic.hpp>
#include <gec/utils/misc.hpp>

#include <functional>

namespace gec {

namespace bigint {

template <typename ARRAY>
struct ArrayHasher {
    using argument_type = ARRAY;
    using result_type = size_t;

    __host__ __device__ GEC_INLINE ArrayHasher() {}
    __host__ __device__ GEC_INLINE ArrayHasher(const ArrayHasher &) {}
    __host__ __device__ GEC_INLINE ArrayHasher &operator=(const ArrayHasher &) {
        return *this;
    }

    __host__ __device__ GEC_INLINE result_type
    operator()(const argument_type &array) const {
        using Hasher = std::hash<typename ARRAY::LimbT>;
        size_t seed = 0;
        Hasher hasher;
        utils::SeqHasher<typename ARRAY::LimbT, ARRAY::LimbN, Hasher>::call(
            seed, array.array(), hasher);
        return seed;
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_DATA_HASHER_HPP

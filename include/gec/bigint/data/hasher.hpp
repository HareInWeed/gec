#pragma once
#ifndef GEC_BIGINT_DATA_HASHER_HPP
#define GEC_BIGINT_DATA_HASHER_HPP

#include <gec/utils/basic.hpp>
#include <gec/utils/misc.hpp>
#include <gec/utils/operators.hpp>

namespace gec {

namespace bigint {

template <typename T, typename Enable = void>
struct Hasher;

template <typename T>
struct Hasher<T, std::enable_if_t<(sizeof(T) <= sizeof(size_t))>> {
    using argument_type = T;
    using result_type = size_t;

    __host__ __device__ GEC_INLINE result_type
    operator()(const argument_type &x) const {
        // FIXME: portable hash
        return x;
    }
};

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
        using H = Hasher<typename argument_type::LimbT>;
        size_t seed = 0;
        H hasher;
        utils::SeqHasher<typename ARRAY::LimbT, ARRAY::LimbN, H>::call(
            seed, array.array(), hasher);
        return seed;
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_DATA_HASHER_HPP

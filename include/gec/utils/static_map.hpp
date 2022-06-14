#pragma once
#ifndef GEC_UTILS_STATIC_MAP_HPP
#define GEC_UTILS_STATIC_MAP_HPP

#include "basic.hpp"
#include "hash.hpp"
#include "misc.hpp"
#include "prime.hpp"

#include <algorithm>
#include <cstring>

namespace gec {

namespace utils {

namespace _CHD_ {

template <size_t B>
struct BucketHash {
    __host__ __device__ GEC_INLINE static size_t call(size_t hash) {
        return hash % B;
    }
};

template <size_t N>
struct HashFamily {
    __host__ __device__ GEC_INLINE static size_t call(size_t hash_id,
                                                      size_t hash) {
        hash::hash_combine(hash_id, hash);
        return hash_id % N;
    }
};

} // namespace _CHD_

template <size_t N, size_t B = next_prime(N / 10),
          typename BucketHash = _CHD_::BucketHash<B>,
          typename HashFamily = _CHD_::HashFamily<N>>
struct CHD {
    size_t buckets[B];
    __host__ __device__ GEC_INLINE constexpr CHD() noexcept : buckets() {}
    __host__ __device__ GEC_INLINE constexpr CHD(size_t *hashes,
                                                 size_t n) noexcept
        : buckets() {
        build(hashes, n);
    }
    __host__ __device__ GEC_INLINE void clear() {
        memset(buckets, 0, sizeof(size_t) * B);
    }
    __host__ __device__ void build(size_t *hashes, size_t n) {
        size_t indices[B];
        size_t count[B] = {};
        size_t *hash_buckets[B] = {};
        bool flags[N] = {};

        for (size_t k = 0; k < n; ++k) {
            ++count[BucketHash::call(hashes[k])];
        }
        for (size_t k = 0; k < B; ++k) {
            if (count[k] != 0) {
                hash_buckets[k] = new size_t[count[k]];
                count[k] = 0;
            }
            indices[k] = k;
        }
        for (size_t k = 0; k < n; ++k) {
            size_t bucket_id = BucketHash::call(hashes[k]);
            hash_buckets[bucket_id][count[bucket_id]] = hashes[k];
            ++count[bucket_id];
        }
        // for small bucket number, insertion sorting should be enough.
        for (size_t k = 0; k < B; ++k) {
            size_t cdt = k;
            for (size_t j = cdt + 1; j < B; ++j) {
                if (count[indices[cdt]] < count[indices[j]]) {
                    cdt = j;
                }
            }
            if (cdt != k) {
                utils::swap(indices[cdt], indices[k]);
            }
        }
        for (size_t i = 0; i < B; ++i) {
            size_t bucket_id = indices[i];
            size_t bucket_len = count[bucket_id];
            if (bucket_len == 0) {
                break;
            }
            size_t *bucket = hash_buckets[bucket_id];
            for (size_t hash_id = 0;; ++hash_id) {
                size_t j = 0;
                for (; j < bucket_len; ++j) {
                    size_t idx = HashFamily::call(hash_id, bucket[j]);
                    if (flags[idx]) {
                        goto failed;
                    } else {
                        flags[idx] = true;
                    }
                }
                // success
                this->buckets[bucket_id] = hash_id;
                break;
            failed:
                for (size_t k = 0; k < j; ++k) {
                    flags[HashFamily::call(hash_id, bucket[k])] = false;
                }
            }
        }
        for (size_t k = 0; k < B; ++k) {
            if (hash_buckets[k]) {
                delete hash_buckets[k];
            }
        }
    }
    size_t get(size_t hash) {
        return HashFamily::call(buckets[BucketHash::call(hash)], hash);
    }
};

} // namespace utils

} // namespace gec

#endif // !GEC_UTILS_STATIC_MAP_HPP
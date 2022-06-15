#pragma once
#ifndef GEC_UTILS_STATIC_MAP_HPP
#define GEC_UTILS_STATIC_MAP_HPP

#include "basic.hpp"
#include "hash.hpp"
#include "misc.hpp"
#include "prime.hpp"

#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

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

__host__ __device__ GEC_INLINE void multi_swap(size_t, size_t) {}

template <typename T, typename... Args>
__host__ __device__ GEC_INLINE void multi_swap(size_t i, size_t j, T *arr,
                                               Args *...args) {
    utils::swap(arr[i], arr[j]);
    multi_swap(i, j, args...);
}

} // namespace _CHD_

template <size_t N, size_t B = next_prime(N / 10),
          typename BucketHash = _CHD_::BucketHash<B>,
          typename HashFamily = _CHD_::HashFamily<N>>
class CHD {
    size_t buckets[B];

  public:
    __host__ __device__ GEC_INLINE constexpr CHD() noexcept : buckets() {}
    __host__ __device__ GEC_INLINE void clear() {
        memset(buckets, 0, sizeof(size_t) * B);
    }
    __host__ std::vector<std::pair<size_t, size_t>> build(const size_t *hashes,
                                                          size_t n) {
        // TODO: make build function device compatible
        // 1. device compatible sort
        // 2. device compatible deduplication
        //    - device compatible vector
        //    - device algorithm

        size_t indices[B];
        size_t count[B] = {};
        size_t *hash_buckets[B] = {};
        bool flags[N] = {};

        // count elements
        for (size_t k = 0; k < n; ++k) {
            ++count[BucketHash::call(hashes[k])];
        }

        // allocating resources
        for (size_t k = 0; k < B; ++k) {
            if (count[k] != 0) {
                hash_buckets[k] = new size_t[count[k]];
                count[k] = 0;
            }
            indices[k] = k;
        }

        // construct buckets & deduplicating
        std::vector<std::pair<size_t, size_t>> duplicates;
        for (size_t k = 0; k < n; ++k) {
            const size_t hash = hashes[k];
            const size_t bucket_id = BucketHash::call(hash);
            auto &bucket = hash_buckets[bucket_id];
            auto &len = count[bucket_id];
            for (size_t j = 0; j < count[bucket_id]; ++j) {
                if (hashes[bucket[j]] == hash) {
                    duplicates.push_back(std::make_pair(bucket[j], k));
                    goto skip;
                }
            }
            bucket[len] = k;
            ++len;
        skip:;
        }
        for (size_t k = 0; k < B; ++k) {
            auto &bucket = hash_buckets[k];
            const size_t len = count[k];
            for (size_t j = 0; j < len; ++j) {
                bucket[j] = hashes[bucket[j]];
            }
        }

        // sort buckets
        std::sort(indices, indices + B, [&](const size_t &a, const size_t &b) {
            return count[a] > count[b];
        });

        // build phf
        for (size_t i = 0; i < B; ++i) {
            const size_t bucket_id = indices[i];
            const size_t bucket_len = count[bucket_id];
            if (bucket_len == 0) {
                break;
            }
            const size_t *bucket = hash_buckets[bucket_id];
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

        // free resources
        for (size_t k = 0; k < B; ++k) {
            if (hash_buckets[k]) {
                delete hash_buckets[k];
            }
        }

        return duplicates;
    }
    __host__ __device__ GEC_INLINE size_t get(size_t hash) {
        return HashFamily::call(buckets[BucketHash::call(hash)], hash);
    }
    __host__ __device__ static size_t fill_placeholder(size_t *hashes,
                                                       size_t n) {
        size_t placeholder = 0;
        for (size_t k = 0; k < n; ++k) {
            if (placeholder == hashes[k]) {
                ++placeholder;
                k = 0;
            }
        }
        for (size_t k = n; k < N; ++k) {
            hashes[k] = placeholder;
        }
        return placeholder;
    }
    template <typename... Args>
    __host__ __device__ void rearrange(size_t *hashes, size_t n,
                                       size_t placeholder, Args *...args) {
        for (size_t k = 0; k < n; ++k) {
            if (hashes[k] == placeholder)
                continue;
            size_t j = get(hashes[k]);
            for (;;) {
                if (hashes[j] == hashes[k])
                    break;
                _CHD_::multi_swap(k, j, hashes, args...);
                if (hashes[k] == placeholder)
                    break;
                j = get(hashes[k]);
            }
        }
    }
};

} // namespace utils

} // namespace gec

#endif // !GEC_UTILS_STATIC_MAP_HPP
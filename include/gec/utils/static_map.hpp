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

struct BucketHash {
    __host__ __device__ GEC_INLINE static size_t call(size_t bucket_num,
                                                      size_t hash) {
        return hash % bucket_num;
    }
};

struct HashFamily {
    __host__ __device__ GEC_INLINE static size_t
    call(size_t hash_id, size_t hash_range, size_t hash) {
        hash::hash_combine(hash_id, hash);
        return hash_id % hash_range;
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

template <typename BucketHash = _CHD_::BucketHash,
          typename HashFamily = _CHD_::HashFamily>
class CHD {
  public:
    size_t *buckets;
    size_t M;
    size_t N;
    size_t B;

    __host__ __device__ GEC_INLINE constexpr CHD(size_t *buckets,
                                                 size_t M) noexcept
        : CHD(buckets, M, next_prime(M * 123 / 100)) {}
    __host__ __device__ GEC_INLINE constexpr CHD(size_t *buckets, size_t M,
                                                 size_t N) noexcept
        : CHD(buckets, M, N, next_prime(N / 10)) {}
    __host__ __device__ GEC_INLINE constexpr CHD(size_t *buckets, size_t M,
                                                 size_t N, size_t B) noexcept
        : buckets(buckets), M(M), N(N), B(B) {}

    __host__ std::vector<std::pair<size_t, size_t>>
    build(const size_t *hashes) {
        // TODO: make build function device compatible
        // 1. device compatible sort
        // 2. device compatible deduplication
        //    - device compatible vector
        //    - device algorithm

        std::vector<size_t> indices(B, 0);
        std::vector<std::vector<size_t>> hash_buckets(B);
        std::vector<char> flags(N, 0);

        // count elements
        for (size_t k = 0; k < M; ++k) {
            ++indices[BucketHash::call(B, hashes[k])];
        }

        // allocating resources
        for (size_t k = 0; k < B; ++k) {
            if (indices[k] != 0) {
                hash_buckets[k].reserve(indices[k]);
                indices[k] = 0;
            }
            indices[k] = k;
        }

        // construct buckets & deduplicating
        std::vector<std::pair<size_t, size_t>> duplicates;
        for (size_t k = 0; k < M; ++k) {
            const size_t hash = hashes[k];
            const size_t bucket_id = BucketHash::call(B, hash);
            auto &bucket = hash_buckets[bucket_id];
            for (auto &idx : bucket) {
                if (hashes[idx] == hash) {
                    duplicates.push_back(std::make_pair(idx, k));
                    goto skip;
                }
            }
            bucket.push_back(k);
        skip:;
        }
        for (size_t k = 0; k < B; ++k) {
            auto &bucket = hash_buckets[k];
            for (auto &idx : bucket) {
                idx = hashes[idx];
            }
        }

        // sort buckets
        std::sort(indices.begin(), indices.end(),
                  [&](const size_t &a, const size_t &b) {
                      return hash_buckets[a].size() > hash_buckets[b].size();
                  });

        // build phf
        for (size_t i = 0; i < B; ++i) {
            const size_t bucket_id = indices[i];
            const auto &bucket = hash_buckets[bucket_id];
            if (bucket.size() == 0) {
                break;
            }
            for (size_t hash_id = 0;; ++hash_id) {
                size_t j = 0;
                for (; j < bucket.size(); ++j) {
                    size_t idx = HashFamily::call(hash_id, N, bucket[j]);
                    if (flags[idx]) {
                        goto failed;
                    } else {
                        flags[idx] = true;
                    }
                }
                // success
                buckets[bucket_id] = hash_id;
                break;
            failed:
                for (size_t k = 0; k < j; ++k) {
                    flags[HashFamily::call(hash_id, N, bucket[k])] = false;
                }
            }
        }

        return duplicates;
    }
    __host__ __device__ GEC_INLINE size_t get(size_t hash) {
        return HashFamily::call(buckets[BucketHash::call(B, hash)], N, hash);
    }
    __host__ __device__ size_t fill_placeholder(size_t *hashes) {
        size_t placeholder = 0;
        for (size_t k = 0; k < M; ++k) {
            if (placeholder == hashes[k]) {
                ++placeholder;
                k = 0;
            }
        }
        for (size_t k = M; k < N; ++k) {
            hashes[k] = placeholder;
        }
        return placeholder;
    }
    template <typename... Args>
    __host__ __device__ void rearrange(size_t *hashes, size_t placeholder,
                                       Args *...args) {
        for (size_t k = 0; k < M; ++k) {
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
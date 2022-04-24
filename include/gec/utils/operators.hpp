#pragma once
#ifndef GEC_UTILS_MISC_HPP
#define GEC_UTILS_MISC_HPP

#include "basic.hpp"

namespace gec {

namespace utils {

namespace ops {

template <typename T>
struct Let {
    __host__ __device__ GEC_INLINE static void call(T &GEC_RSTRCT a,
                                                    const T &GEC_RSTRCT b) {
        a = b;
    }
};

template <typename T>
struct Eq {
    __host__ __device__ GEC_INLINE static bool call(const T &GEC_RSTRCT a,
                                                    const T &GEC_RSTRCT b) {
        return a == b;
    }
};

template <typename T>
struct Lt {
    __host__ __device__ GEC_INLINE static bool call(const T &GEC_RSTRCT a,
                                                    const T &GEC_RSTRCT b) {
        return a < b;
    }
};

template <typename T>
struct Gt {
    __host__ __device__ GEC_INLINE static bool call(const T &GEC_RSTRCT a,
                                                    const T &GEC_RSTRCT b) {
        return a > b;
    }
};

template <typename T>
struct BitAnd {
    __host__ __device__ GEC_INLINE static void
    call(T &GEC_RSTRCT a, const T &GEC_RSTRCT b, const T &GEC_RSTRCT c) {
        a = b & c;
    }
};

template <typename T>
struct BitOr {
    __host__ __device__ GEC_INLINE static void
    call(T &GEC_RSTRCT a, const T &GEC_RSTRCT b, const T &GEC_RSTRCT c) {
        a = b | c;
    }
};

template <typename T>
struct BitNot {
    __host__ __device__ GEC_INLINE static void call(T &GEC_RSTRCT a,
                                                    const T &GEC_RSTRCT b) {
        a = ~b;
    }
};

template <typename T>
struct BitXor {
    __host__ __device__ GEC_INLINE static void
    call(T &GEC_RSTRCT a, const T &GEC_RSTRCT b, const T &GEC_RSTRCT c) {
        a = b ^ c;
    }
};

template <typename T>
struct BitAndInplace {
    __host__ __device__ GEC_INLINE static void call(T &GEC_RSTRCT a,
                                                    const T &GEC_RSTRCT b) {
        a &= b;
    }
};

template <typename T>
struct BitOrInplace {
    __host__ __device__ GEC_INLINE static void call(T &GEC_RSTRCT a,
                                                    const T &GEC_RSTRCT b) {
        a |= b;
    }
};

template <typename T>
struct BitNotInplace {
    __host__ __device__ GEC_INLINE static void call(T &a) { a = ~a; }
};

template <typename T>
struct BitXorInplace {
    __host__ __device__ GEC_INLINE static void call(T &GEC_RSTRCT a,
                                                    const T &GEC_RSTRCT b) {
        a ^= b;
    }
};

} // namespace ops

} // namespace utils

} // namespace gec

#endif // !GEC_UTILS_MISC_HPP
#pragma once
#ifndef GEC_UTILS_OPERATOR_HPP
#define GEC_UTILS_OPERATOR_HPP

#include "basic.hpp"

namespace gec {

namespace utils {

namespace ops {

template <typename T>
struct Let {
    GEC_HD GEC_INLINE static void call(T &GEC_RSTRCT a, const T &GEC_RSTRCT b) {
        a = b;
    }
};

template <typename T>
struct Eq {
    GEC_HD GEC_INLINE static bool call(const T &GEC_RSTRCT a,
                                       const T &GEC_RSTRCT b) {
        return a == b;
    }
};

template <typename T>
struct Lt {
    GEC_HD GEC_INLINE static bool call(const T &GEC_RSTRCT a,
                                       const T &GEC_RSTRCT b) {
        return a < b;
    }
};

template <typename T>
struct Gt {
    GEC_HD GEC_INLINE static bool call(const T &GEC_RSTRCT a,
                                       const T &GEC_RSTRCT b) {
        return a > b;
    }
};

template <typename T>
struct BitAnd {
    GEC_HD GEC_INLINE static void call(T &GEC_RSTRCT a, const T &GEC_RSTRCT b,
                                       const T &GEC_RSTRCT c) {
        a = b & c;
    }
};

template <typename T>
struct BitOr {
    GEC_HD GEC_INLINE static void call(T &GEC_RSTRCT a, const T &GEC_RSTRCT b,
                                       const T &GEC_RSTRCT c) {
        a = b | c;
    }
};

template <typename T>
struct BitNot {
    GEC_HD GEC_INLINE static void call(T &GEC_RSTRCT a, const T &GEC_RSTRCT b) {
        a = ~b;
    }
};

template <typename T>
struct BitXor {
    GEC_HD GEC_INLINE static void call(T &GEC_RSTRCT a, const T &GEC_RSTRCT b,
                                       const T &GEC_RSTRCT c) {
        a = b ^ c;
    }
};

template <typename T>
struct BitAndInplace {
    GEC_HD GEC_INLINE static void call(T &GEC_RSTRCT a, const T &GEC_RSTRCT b) {
        a &= b;
    }
};

template <typename T>
struct BitOrInplace {
    GEC_HD GEC_INLINE static void call(T &GEC_RSTRCT a, const T &GEC_RSTRCT b) {
        a |= b;
    }
};

template <typename T>
struct BitNotInplace {
    GEC_HD GEC_INLINE static void call(T &a) { a = ~a; }
};

template <typename T>
struct BitXorInplace {
    GEC_HD GEC_INLINE static void call(T &GEC_RSTRCT a, const T &GEC_RSTRCT b) {
        a ^= b;
    }
};

} // namespace ops

} // namespace utils

} // namespace gec

#endif // !GEC_UTILS_OPERATOR_HPP
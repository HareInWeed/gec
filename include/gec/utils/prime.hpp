#pragma once
#ifndef GEC_UTILS_PRIME_HPP
#define GEC_UTILS_PRIME_HPP

#include "basic.hpp"

namespace gec {

namespace utils {

template <typename T>
GEC_HD constexpr T mod_exp(T x, T n, T m) {
    T result = T(1);
    for (;;) {
        if (n & T(1)) {
            result = (result * x) % m;
        }
        n = n >> 1;
        if (n == T(0)) {
            break;
        }
        x = (x * x) % m;
    }
    return result;
}

template <typename T>
GEC_HD constexpr bool miller_rabin(T tester, T x) {
    int r = 0;
    T xm1 = x - 1;
    T d = xm1;
    while ((d & 1) == 0) {
        d = d >> 1;
        ++r;
    }
    T y = mod_exp(tester, d, x);
    if (y == 1 || y == xm1) {
        return true;
    }
    for (int k = 1; k < r; ++k) {
        y = (y * y) % x;
        if (y == xm1) {
            return true;
        }
    }
    return false;
}

template <typename T>
GEC_HD constexpr bool mod_miller_rabin(T tester, T x) {
    tester = tester % x;
    return tester == 0 || miller_rabin(tester, x);
}

namespace _is_prime_ {

template <typename T, size_t bytes>
struct IsPrime {
    GEC_HD GEC_INLINE static constexpr bool call(T x) {
        // fallback to 8-bytes
        return IsPrime<T, 8>::call(x);
    }
};

template <typename T>
struct IsPrime<T, 1> {
    GEC_HD static constexpr bool call(T x) { return miller_rabin(T(2), x); }
};

template <typename T>
struct IsPrime<T, 2> {
    GEC_HD static constexpr bool call(T x) {
        T xm1 = x - 1;
        if (!miller_rabin(T(2), x))
            return false;
        if (T(7) >= xm1)
            return true;
        return miller_rabin(T(7), x);
    }
};

template <typename T>
struct IsPrime<T, 4> {
    GEC_HD static constexpr bool call(T x) {
        T xm1 = x - 1;
        if (!miller_rabin(T(2), x))
            return false;
        if (T(7) >= xm1)
            return true;
        if (!miller_rabin(T(7), x))
            return false;
        if (T(61) >= xm1)
            return true;
        return miller_rabin(T(61), x);
    }
};

template <typename T>
struct IsPrime<T, 8> {
    GEC_HD static constexpr bool call(T x) {
        // TODO: better test with segmentation
        if (x <= T(0xFFFFFFFF)) {
            return IsPrime<T, 4>::call(x);
        } else {
            // clang-format off
            return mod_miller_rabin(T(2),          x) &&
                   mod_miller_rabin(T(325),        x) &&
                   mod_miller_rabin(T(9375),       x) &&
                   mod_miller_rabin(T(28178),      x) &&
                   mod_miller_rabin(T(450775),     x) &&
                   mod_miller_rabin(T(9780504),    x) &&
                   mod_miller_rabin(T(1795265022), x);
            // clang-format on
        }
    }
};

} // namespace _is_prime_

template <typename T>
GEC_HD constexpr bool is_prime(T x) {
    return x == T(2) || (x > T(1) && (x & T(1)) &&
                         _is_prime_::IsPrime<T, sizeof(T)>::call(x));
}

template <typename T>
GEC_HD constexpr T next_prime(T x) {
    ++x;
    while (!is_prime(x)) {
        ++x;
    }
    return x;
}

} // namespace utils

} // namespace gec

#endif // GEC_UTILS_PRIME_HPP
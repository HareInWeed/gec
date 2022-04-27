#pragma once
#ifndef GEC_TEST_UTILS_H
#define GEC_TEST_UTILS_H

#include "common.hpp"

template <typename T>
struct OpaqueNum {
    T num;

    constexpr OpaqueNum() : num() {}
    constexpr OpaqueNum(T num) : num(num) {}
    constexpr operator T() { return num; }

    OpaqueNum &operator=(const OpaqueNum &others) {
        num = others.num;
        return *this;
    }

    constexpr friend OpaqueNum operator+(const OpaqueNum &a,
                                         const OpaqueNum &b) {
        return OpaqueNum(a.num + b.num);
    }
    OpaqueNum &operator+=(const OpaqueNum &others) {
        num += others.num;
        return (*this);
    }

    constexpr friend OpaqueNum operator-(const OpaqueNum &a,
                                         const OpaqueNum &b) {
        return OpaqueNum(a.num - b.num);
    }
    OpaqueNum &operator-=(const OpaqueNum &others) {
        num -= others.num;
        return (*this);
    }

    constexpr friend OpaqueNum operator*(const OpaqueNum &a,
                                         const OpaqueNum &b) {
        return OpaqueNum(a.num * b.num);
    }
    OpaqueNum &operator*=(const OpaqueNum &others) {
        num *= others.num;
        return (*this);
    }

    constexpr friend OpaqueNum operator/(const OpaqueNum &a,
                                         const OpaqueNum &b) {
        return OpaqueNum(a.num / b.num);
    }
    OpaqueNum &operator/=(const OpaqueNum &others) {
        num /= others.num;
        return (*this);
    }

    constexpr friend OpaqueNum operator&(const OpaqueNum &a,
                                         const OpaqueNum &b) {
        return OpaqueNum(a.num & b.num);
    }
    OpaqueNum &operator&=(const OpaqueNum &others) {
        num &= others.num;
        return (*this);
    }

    constexpr friend OpaqueNum operator|(const OpaqueNum &a,
                                         const OpaqueNum &b) {
        return OpaqueNum(a.num | b.num);
    }
    OpaqueNum &operator|=(const OpaqueNum &others) {
        num |= others.num;
        return (*this);
    }

    constexpr friend OpaqueNum operator^(const OpaqueNum &a,
                                         const OpaqueNum &b) {
        return OpaqueNum(a.num ^ b.num);
    }
    OpaqueNum &operator^=(const OpaqueNum &others) {
        num ^= others.num;
        return (*this);
    }

    constexpr friend OpaqueNum operator>>(const OpaqueNum &a, size_t n) {
        return OpaqueNum(a.num >> n);
    }
    OpaqueNum &operator>>=(size_t n) {
        num >>= n;
        return (*this);
    }

    constexpr friend OpaqueNum operator<<(const OpaqueNum &a, size_t n) {
        return OpaqueNum(a.num << n);
    }
    OpaqueNum &operator<<=(size_t n) {
        num <<= n;
        return (*this);
    }

    constexpr friend bool operator==(const OpaqueNum &a, const OpaqueNum &b) {
        return a.num == b.num;
    }
    constexpr friend bool operator!=(const OpaqueNum &a, const OpaqueNum &b) {
        return a.num != b.num;
    }
    constexpr friend bool operator<(const OpaqueNum &a, const OpaqueNum &b) {
        return a.num < b.num;
    }
    constexpr friend bool operator<=(const OpaqueNum &a, const OpaqueNum &b) {
        return a.num <= b.num;
    }
    constexpr friend bool operator>(const OpaqueNum &a, const OpaqueNum &b) {
        return a.num > b.num;
    }
    constexpr friend bool operator>=(const OpaqueNum &a, const OpaqueNum &b) {
        return a.num >= b.num;
    }
};

template <typename T>
class std::numeric_limits<OpaqueNum<T>> : public std::numeric_limits<T> {};

#endif // !GEC_TEST_UTILS_H

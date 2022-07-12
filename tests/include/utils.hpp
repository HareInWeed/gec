#pragma once
#ifndef GEC_TEST_UTILS_H
#define GEC_TEST_UTILS_H

#include "common.hpp"

template <typename T>
struct OpaqueNum {
    T num;

    __host__ __device__ inline constexpr OpaqueNum() : num() {}
    __host__ __device__ inline constexpr OpaqueNum(T num) : num(num) {}

    __host__ __device__ constexpr OpaqueNum(const OpaqueNum &others)
        : num(others.num) {}

    __host__ __device__ OpaqueNum &operator=(const OpaqueNum &others) {
        num = others.num;
        return *this;
    }

    __host__ __device__ constexpr friend OpaqueNum
    operator+(const OpaqueNum &a, const OpaqueNum &b) {
        return OpaqueNum(a.num + b.num);
    }
    __host__ __device__ OpaqueNum &operator+=(const OpaqueNum &others) {
        num += others.num;
        return (*this);
    }

    __host__ __device__ constexpr friend OpaqueNum
    operator-(const OpaqueNum &a, const OpaqueNum &b) {
        return OpaqueNum(a.num - b.num);
    }
    __host__ __device__ OpaqueNum &operator-=(const OpaqueNum &others) {
        num -= others.num;
        return (*this);
    }

    __host__ __device__ constexpr friend OpaqueNum
    operator*(const OpaqueNum &a, const OpaqueNum &b) {
        return OpaqueNum(a.num * b.num);
    }
    __host__ __device__ OpaqueNum &operator*=(const OpaqueNum &others) {
        num *= others.num;
        return (*this);
    }

    __host__ __device__ constexpr friend OpaqueNum
    operator/(const OpaqueNum &a, const OpaqueNum &b) {
        return OpaqueNum(a.num / b.num);
    }
    __host__ __device__ OpaqueNum &operator/=(const OpaqueNum &others) {
        num /= others.num;
        return (*this);
    }

    __host__ __device__ constexpr friend OpaqueNum
    operator&(const OpaqueNum &a, const OpaqueNum &b) {
        return OpaqueNum(a.num & b.num);
    }
    __host__ __device__ OpaqueNum &operator&=(const OpaqueNum &others) {
        num &= others.num;
        return (*this);
    }

    __host__ __device__ constexpr friend OpaqueNum
    operator|(const OpaqueNum &a, const OpaqueNum &b) {
        return OpaqueNum(a.num | b.num);
    }
    __host__ __device__ OpaqueNum &operator|=(const OpaqueNum &others) {
        num |= others.num;
        return (*this);
    }

    __host__ __device__ constexpr friend OpaqueNum
    operator^(const OpaqueNum &a, const OpaqueNum &b) {
        return OpaqueNum(a.num ^ b.num);
    }
    __host__ __device__ OpaqueNum &operator^=(const OpaqueNum &others) {
        num ^= others.num;
        return (*this);
    }

    __host__ __device__ constexpr friend OpaqueNum
    operator>>(const OpaqueNum &a, size_t n) {
        return OpaqueNum(a.num >> n);
    }
    __host__ __device__ OpaqueNum &operator>>=(size_t n) {
        num >>= n;
        return (*this);
    }

    __host__ __device__ constexpr friend OpaqueNum
    operator<<(const OpaqueNum &a, size_t n) {
        return OpaqueNum(a.num << n);
    }
    __host__ __device__ OpaqueNum &operator<<=(size_t n) {
        num <<= n;
        return (*this);
    }

    __host__ __device__ constexpr friend bool operator==(const OpaqueNum &a,
                                                         const OpaqueNum &b) {
        return a.num == b.num;
    }
    __host__ __device__ constexpr friend bool operator!=(const OpaqueNum &a,
                                                         const OpaqueNum &b) {
        return a.num != b.num;
    }
    __host__ __device__ constexpr friend bool operator<(const OpaqueNum &a,
                                                        const OpaqueNum &b) {
        return a.num < b.num;
    }
    __host__ __device__ constexpr friend bool operator<=(const OpaqueNum &a,
                                                         const OpaqueNum &b) {
        return a.num <= b.num;
    }
    __host__ __device__ constexpr friend bool operator>(const OpaqueNum &a,
                                                        const OpaqueNum &b) {
        return a.num > b.num;
    }
    __host__ __device__ constexpr friend bool operator>=(const OpaqueNum &a,
                                                         const OpaqueNum &b) {
        return a.num >= b.num;
    }

    template <typename U>
    constexpr operator U() {
        return U(this->num);
    }
};

template <typename T>
class std::numeric_limits<OpaqueNum<T>> : public std::numeric_limits<T> {};

#endif // !GEC_TEST_UTILS_H

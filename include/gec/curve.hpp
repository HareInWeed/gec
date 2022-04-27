#pragma once
#ifndef GEC_CURVE_HPP
#define GEC_CURVE_HPP

#include "utils.hpp"

namespace gec {

namespace curve {

template <class FIELD_T, class SCALER_T, const FIELD_T &A, const FIELD_T &B>
struct APoint {
    FIELD_T x;
    FIELD_T y;

    using FIELD = FIELD_T;
    using SCALER = SCALER_T;

    __host__ __device__ GEC_INLINE APoint(FIELD_T x, FIELD_T y) : x(x), y(y) {}

    __host__ __device__ GEC_INLINE APoint(const APoint &p) : APoint(p.x, p.y) {}
    __host__ __device__ GEC_INLINE APoint &operator=(const APoint &other) {
        x = other.x;
        y = other.y;
        return *this;
    }
};

template <class FIELD_T, class SCALER_T, const FIELD_T &A, const FIELD_T &B>
struct JPoint {
    FIELD_T x;
    FIELD_T y;
    FIELD_T z;

    using Field = FIELD_T;
    using Scaler = SCALER_T;

    __host__ __device__ GEC_INLINE JPoint(FIELD_T x, FIELD_T y, FIELD_T z)
        : x(x), y(y), z(z) {}
    __host__ __device__ GEC_INLINE JPoint(FIELD_T x, FIELD_T y) : x(x), y(y) {
        from_APoint();
    }

    __host__ __device__ GEC_INLINE JPoint(const JPoint &p)
        : JPoint(p.x, p.y, p.z) {}
    __host__ __device__ GEC_INLINE JPoint &operator=(const JPoint &other) {
        x = other.x;
        y = other.y;
        z = other.z;
        return *this;
    }

    __host__ __device__ GEC_INLINE void into_APoint() {
        z.inv();
        FIELD_T tmp = z;
        z.square();
        x.mul(z);
        // tmp.cube();
        z.mul(&tmp);
        y.mul(z);
    }
    __host__ __device__ GEC_INLINE void from_APoint() { z = z.one(); }

    __host__ __device__ GEC_INLINE JPoint &operator+=(const JPoint &other) {
        add(*this, other);
        return *this;
    }

    __host__ __device__ JPoint &operator-() {
        // TODO
    }

    __host__ __device__ JPoint &operator*=(const Scaler &scaler) {
        // TODO
    }

    static __host__ __device__ void add(JPoint &a, const JPoint &b,
                                        const JPoint &c) {
        // TODO
    }

    static __host__ __device__ void add(JPoint &a, const JPoint &b) {
        // TODO
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_HPP

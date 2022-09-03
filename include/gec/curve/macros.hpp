#pragma once
#ifndef GEC_CURVE_MACROS_HPP
#define GEC_CURVE_MACROS_HPP

#ifdef __CUDACC__
#define GEC_CURVE_INF(coordinate, F, A, B, InfYZero)                           \
    coordinate<F, &A, &B, &d_##A, &d_##B, InfYZero>
#define GEC_CURVE_A_INF(coordinate, F, A, InfYZero)                            \
    coordinate<F, &A, nullptr, &d_##A, nullptr, InfYZero>
#define GEC_CURVE_B_INF(coordinate, F, B, InfYZero)                            \
    coordinate<F, nullptr, &B, nullptr, &d_##B, InfYZero>
#define GEC_CURVE(coordinate, F, A, B) GEC_CURVE_INF(coordinate, F, A, B, true)
#define GEC_CURVE_A(coordinate, F, A) GEC_CURVE_A_INF(coordinate, F, A, true)
#define GEC_CURVE_B(coordinate, F, B) GEC_CURVE_B_INF(coordinate, F, B, true)
#else
#define GEC_CURVE_INF(coordinate, F, A, B, InfYZero)                           \
    coordinate<F, &A, &B, nullptr, nullptr, InfYZero>
#define GEC_CURVE_A_INF(coordinate, F, A, InfYZero)                            \
    coordinate<F, &A, nullptr, nullptr, nullptr, InfYZero>
#define GEC_CURVE_B_INF(coordinate, F, B, InfYZero)                            \
    coordinate<F, nullptr, &B, nullptr, nullptr, InfYZero>
#define GEC_CURVE(coordinate, F, A, B) GEC_CURVE_INF(coordinate, F, A, B, true)
#define GEC_CURVE_A(coordinate, F, A) GEC_CURVE_A_INF(coordinate, F, A, true)
#define GEC_CURVE_B(coordinate, F, B) GEC_CURVE_B_INF(coordinate, F, B, true)
#endif // __CUDACC__

#endif // !GEC_CURVE_MACROS_HPP

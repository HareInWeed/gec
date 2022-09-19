#pragma once
#ifndef GEC_TEST_CURVE_HPP
#define GEC_TEST_CURVE_HPP

#include "common.hpp"
#include "field.hpp"

#include <gec/curve.hpp>

GEC_DECL_GLOBAL(AR_160, Field160);
GEC_DECL_GLOBAL(BR_160, Field160);
using CurveA = GEC_CURVE(gec::curve::AffineCurve, Field160, AR_160, BR_160);
using CurveP = GEC_CURVE(gec::curve::ProjectiveCurve, Field160, AR_160, BR_160);
using CurveJ = GEC_CURVE(gec::curve::JacobianCurve, Field160, AR_160, BR_160);

GEC_DECL_GLOBAL(AR2_160, Field160_2);
GEC_DECL_GLOBAL(BR2_160, Field160_2);
using CurveA2 = GEC_CURVE(gec::curve::AffineCurve, Field160_2, AR2_160,
                          BR2_160);
using CurveP2 = GEC_CURVE(gec::curve::ProjectiveCurve, Field160_2, AR2_160,
                          BR2_160);
using CurveJ2 = GEC_CURVE(gec::curve::JacobianCurve, Field160_2, AR2_160,
                          BR2_160);

// -------------------- dlp 1 with 32-bit limb --------------------

using Dlp1Array = gec::bigint::ArrayBE<LIMB_T, LN_160>;
GEC_DECL_GLOBAL(Dlp1P, Dlp1Array);
constexpr LIMB_T Dlp1P_P = 0x5afdc9d5u;
GEC_DECL_GLOBAL(Dlp1P_RR, Dlp1Array);
GEC_DECL_GLOBAL(Dlp1P_OneR, Dlp1Array);
using Dlp1Field = GEC_BASE_FIELD(Dlp1Array, Dlp1P, Dlp1P_P, Dlp1P_RR,
                                 Dlp1P_OneR);

// GEC_DECL_GLOBAL(Dlp1A, Dlp1Field);
GEC_DECL_GLOBAL(Dlp1B, Dlp1Field);

using Dlp1SArray = gec::bigint::ArrayBE<LIMB_T, LN_160>;
GEC_DECL_GLOBAL(Dlp1Card, Dlp1SArray);
using Dlp1Scalar = GEC_BASE_ADD_GROUP(Dlp1SArray, Dlp1Card);

using Dlp1CurveA = GEC_CURVE_B(gec::curve::AffineCurve, Dlp1Field, Dlp1B);
using Dlp1CurveJ = GEC_CURVE_B(gec::curve::JacobianCurve, Dlp1Field, Dlp1B);

// -------------------- dlp 1 with 64-bit limb --------------------

using Dlp1Array_2 = gec::bigint::ArrayBE<LIMB2_T, LN2_160, 32>;
GEC_DECL_GLOBAL(Dlp1P2, Dlp1Array_2);
constexpr LIMB2_T Dlp1P2_P = 0xdb83306e5afdc9d5llu;
GEC_DECL_GLOBAL(Dlp1P2_RR, Dlp1Array_2);
GEC_DECL_GLOBAL(Dlp1P2_OneR, Dlp1Array_2);
using Dlp1Field2 = GEC_BASE_FIELD(Dlp1Array_2, Dlp1P2, Dlp1P2_P, Dlp1P2_RR,
                                  Dlp1P2_OneR);

// GEC_DECL_GLOBAL(Dlp1A2, Dlp1Field2);
GEC_DECL_GLOBAL(Dlp1B2, Dlp1Field2);

using Dlp1SArray_2 = gec::bigint::ArrayBE<LIMB2_T, LN2_160>;
GEC_DECL_GLOBAL(Dlp1Card2, Dlp1SArray_2);
using Dlp1Scalar2 = GEC_BASE_ADD_GROUP(Dlp1SArray_2, Dlp1Card2);

using Dlp1CurveJ2 = GEC_CURVE_B(gec::curve::JacobianCurve, Dlp1Field2, Dlp1B2);

// -------------------- dlp 2 with 32-bit limb --------------------

using Dlp2Array = gec::bigint::ArrayBE<LIMB_T, 1>;
GEC_DECL_GLOBAL(Dlp2P, Dlp2Array);
constexpr LIMB_T Dlp2P_P = 3105566705u;
GEC_DECL_GLOBAL(Dlp2P_RR, Dlp2Array);
GEC_DECL_GLOBAL(Dlp2P_OneR, Dlp2Array);
using Dlp2Field = GEC_BASE_FIELD(Dlp2Array, Dlp2P, Dlp2P_P, Dlp2P_RR,
                                 Dlp2P_OneR);

GEC_DECL_GLOBAL(Dlp2A, Dlp2Field);
GEC_DECL_GLOBAL(Dlp2B, Dlp2Field);

using Dlp2SArray = gec::bigint::ArrayBE<LIMB_T, 1>;
GEC_DECL_GLOBAL(Dlp2Card, Dlp2SArray);
constexpr LIMB_T Dlp2Card_P = 0xfbd05cfu;
GEC_DECL_GLOBAL(Dlp2Card_RR, Dlp2SArray);
GEC_DECL_GLOBAL(Dlp2Card_OneR, Dlp2SArray);
using Dlp2Scalar = GEC_BASE_FIELD(Dlp2SArray, Dlp2Card, Dlp2Card_P, Dlp2Card_RR,
                                  Dlp2Card_OneR);

using Dlp2CurveJ = GEC_CURVE(gec::curve::JacobianCurve, Dlp2Field, Dlp2A,
                             Dlp2B);

// -------------------- dlp 3 with 32-bit limb --------------------

constexpr size_t Dlp3N = 8;
using Dlp3Array = gec::bigint::ArrayBE<LIMB_T, Dlp3N, 32>;
GEC_DECL_GLOBAL(Dlp3P, Dlp3Array);
constexpr LIMB_T Dlp3P_P = 0xd2253531u;
GEC_DECL_GLOBAL(Dlp3P_RR, Dlp3Array);
GEC_DECL_GLOBAL(Dlp3P_OneR, Dlp3Array);
using Dlp3Field = GEC_BASE_FIELD(Dlp3Array, Dlp3P, Dlp3P_P, Dlp3P_RR,
                                 Dlp3P_OneR);
#ifdef GEC_ENABLE_AVX2
using AVX2Dlp3Field = GEC_BASE_AVX2FIELD(Dlp3Array, Dlp3P, Dlp3P_P, Dlp3P_RR,
                                         Dlp3P_OneR);
#endif // GEC_ENABLE_AVX2

// GEC_DECL_GLOBAL(Dlp3A, Dlp3Field);
GEC_DECL_GLOBAL(Dlp3B, Dlp3Field);
using Dlp3CurveJ = GEC_CURVE_B(gec::curve::JacobianCurve, Dlp3Field, Dlp3B);
using Dlp3CurveA = GEC_CURVE_B(gec::curve::AffineCurve, Dlp3Field, Dlp3B);

#ifdef GEC_ENABLE_AVX2
// extern const AVX2Dlp3Field AVX2Dlp3A;
extern const AVX2Dlp3Field AVX2Dlp3B;
using AVX2Dlp3CurveA =
    gec::curve::AffineCurve<AVX2Dlp3Field, nullptr, &AVX2Dlp3B>;
#endif // GEC_ENABLE_AVX2

GEC_DECL_GLOBAL(Dlp3Gen1, Dlp3CurveA);
constexpr size_t Dlp3G1SN = 2;
using Dlp3G1SArray = gec::bigint::ArrayBE<LIMB_T, Dlp3G1SN, 8>;
GEC_DECL_GLOBAL(Dlp3G1Card, Dlp3G1SArray);
constexpr LIMB_T Dlp3G1Card_P = 0x36a04ecdu;
GEC_DECL_GLOBAL(Dlp3G1Card_RR, Dlp3G1SArray);
GEC_DECL_GLOBAL(Dlp3G1Card_OneR, Dlp3G1SArray);
using Dlp3G1Scalar = GEC_BASE_FIELD(Dlp3G1SArray, Dlp3G1Card, Dlp3G1Card_P,
                                    Dlp3G1Card_RR, Dlp3G1Card_OneR);

GEC_DECL_GLOBAL(Dlp3Gen2, Dlp3CurveA);
constexpr size_t Dlp3G2SN = 2;
using Dlp3G2SArray = gec::bigint::ArrayBE<LIMB_T, Dlp3G2SN, 8>;
GEC_DECL_GLOBAL(Dlp3G2Card, Dlp3G2SArray);
constexpr LIMB_T Dlp3G2Card_P = 0x9013b4b9u;
GEC_DECL_GLOBAL(Dlp3G2Card_RR, Dlp3G2SArray);
GEC_DECL_GLOBAL(Dlp3G2Card_OneR, Dlp3G2SArray);
using Dlp3G2Scalar = GEC_BASE_FIELD(Dlp3G2SArray, Dlp3G2Card, Dlp3G2Card_P,
                                    Dlp3G2Card_RR, Dlp3G2Card_OneR);

// -------------------- dlp 3 with 64-bit limb --------------------

constexpr size_t Dlp3N2 = 4;
using Dlp3Array_2 = gec::bigint::ArrayBE<LIMB2_T, Dlp3N2, 32>;
GEC_DECL_GLOBAL(Dlp3P2, Dlp3Array_2);
constexpr LIMB2_T Dlp3P2_P = 0xd838091dd2253531llu;
GEC_DECL_GLOBAL(Dlp3P2_RR, Dlp3Array_2);
GEC_DECL_GLOBAL(Dlp3P2_OneR, Dlp3Array_2);
using Dlp3Field2 = GEC_BASE_FIELD(Dlp3Array_2, Dlp3P2, Dlp3P2_P, Dlp3P2_RR,
                                  Dlp3P2_OneR);

// GEC_DECL_GLOBAL(Dlp3A2, Dlp3Field2);
GEC_DECL_GLOBAL(Dlp3B2, Dlp3Field2);

constexpr size_t Dlp3SN2 = 1;
using Dlp3SArray_2 = gec::bigint::ArrayBE<LIMB2_T, Dlp3SN2, 8>;
GEC_DECL_GLOBAL(Dlp3Card2, Dlp3SArray_2);
constexpr LIMB2_T Dlp3Card2_P = 0x61edaaec36a04ecdllu;
GEC_DECL_GLOBAL(Dlp3Card2_RR, Dlp3SArray_2);
GEC_DECL_GLOBAL(Dlp3Card2_OneR, Dlp3SArray_2);
using Dlp3Scalar2 = GEC_BASE_FIELD(Dlp3SArray_2, Dlp3Card2, Dlp3Card2_P,
                                   Dlp3Card2_RR, Dlp3Card2_OneR);

using Dlp3CurveJ2 = GEC_CURVE_B(gec::curve::JacobianCurve, Dlp3Field2, Dlp3B2);

#endif // !GEC_TEST_CURVE_HPP
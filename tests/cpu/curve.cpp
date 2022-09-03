#include <curve.hpp>

GEC_DEF_GLOBAL(AR_160, Field160, //
               0x821006a8u, 0x420792f5u, 0x3a8009a5u, 0xd1ddf060u, 0x5d2d2098u);
GEC_DEF_GLOBAL(BR_160, Field160, //
               0x60c3a8e6u, 0x5781342fu, 0x242d1db1u, 0x2583f9d6u, 0xcac79e59u);

GEC_DEF_GLOBAL(AR2_160, Field160_2, //
               0x821006a8llu, 0x3a8009a5420792f5llu, 0x5d2d2098d1ddf060llu);
GEC_DEF_GLOBAL(BR2_160, Field160_2, //
               0x60c3a8e6llu, 0x242d1db15781342fllu, 0xcac79e592583f9d6llu);

GEC_DEF_GLOBAL(Dlp1P, Dlp1Array, //
               0x8049a325u, 0xd5a0ed72u, 0x448756f6u, 0x1ddf5414u, 0x9b7ed883u);
GEC_DEF_GLOBAL(Dlp1P_RR, Dlp1Array, //
               0x4a86f4adu, 0xa64010c6u, 0x6c613523u, 0xfd7ddbbdu, 0x9166fb51u);
GEC_DEF_GLOBAL(Dlp1P_OneR, Dlp1Array, //
               0x7fb65cdau, 0x2a5f128du, 0xbb78a909u, 0xe220abebu, 0x6481277du);

// GEC_DEF_GLOBAL(Dlp1A, Dlp1Field, //
//           0);
GEC_DEF_GLOBAL(Dlp1B, Dlp1Field, //
               0x7baf70c8u, 0x7b92164du, 0xfc11e794u, 0x3fea12cau, 0xe3915053u);

GEC_DEF_GLOBAL(Dlp1Card, Dlp1SArray, //
               0x4024d192u, 0xead076b9u, 0x2244486cu, 0x9101e4f5u, 0xfa89e3f2u);

GEC_DEF_GLOBAL(Dlp1P2, Dlp1Array_2, //
               0x8049a325llu, 0xd5a0ed72448756f6llu, 0x1ddf54149b7ed883llu);
GEC_DEF_GLOBAL(Dlp1P2_RR, Dlp1Array_2, //
               0x564b44fallu, 0x880eba1d4858bd67llu, 0xf6d8bed667130c77llu);
GEC_DEF_GLOBAL(Dlp1P2_OneR, Dlp1Array_2, //
               0x00f7f560llu, 0xe9e1701be85df2ccllu, 0x912d3bbe10d1a50fllu);

// GEC_DEF_GLOBAL(Dlp1A2, Dlp1Field2, //
//           0);
GEC_DEF_GLOBAL(Dlp1B2, Dlp1Field2, //
               0x07bfab07llu, 0x4f0b80df42ef9664llu, 0x8969ddf0868d2878llu);

GEC_DEF_GLOBAL(Dlp1Card2, Dlp1SArray_2, //
               0x4024d192llu, 0xead076b92244486cllu, 0x9101e4f5fa89e3f2llu);

GEC_DEF_GLOBAL(Dlp2P, Dlp2Array, //
               7919);
GEC_DEF_GLOBAL(Dlp2P_RR, Dlp2Array, //
               3989);
GEC_DEF_GLOBAL(Dlp2P_OneR, Dlp2Array, //
               2618);

GEC_DEF_GLOBAL(Dlp2A, Dlp2Field, //
               7348);
GEC_DEF_GLOBAL(Dlp2B, Dlp2Field, //
               157);

GEC_DEF_GLOBAL(Dlp2Card, Dlp2SArray, //
               7889);
GEC_DEF_GLOBAL(Dlp2Card_RR, Dlp2SArray, //
               2697);
GEC_DEF_GLOBAL(Dlp2Card_OneR, Dlp2SArray, //
               6360);

GEC_DEF_GLOBAL(Dlp3P, Dlp3Array, //
               0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
               0xffffffffu, 0xfffffffeu, 0xfffffc2fu);
GEC_DEF_GLOBAL(Dlp3P_RR, Dlp3Array, //
               0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
               0x00000001u, 0x000007a2u, 0x000e90a1u);
GEC_DEF_GLOBAL(Dlp3P_OneR, Dlp3Array, //
               0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
               0x00000000u, 0x00000001u, 0x000003d1u);

// GEC_DEF_GLOBAL(Dlp3A, Dlp3Field, //
//           0, 0, 0, 0, 0, 0, 0, 0);
GEC_DEF_GLOBAL(Dlp3B, Dlp3Field, //
               0, 0, 0, 0, 0, 0, 0x0000000du, 0x0000319du);
#ifdef GEC_ENABLE_AVX2
// const AVX2Dlp3Field AVX2Dlp3A(0, 0, 0, 0, 0, 0, 0, 0);
const AVX2Dlp3Field AVX2Dlp3B(0, 0, 0, 0, 0, 0, 0x0000000du, 0x0000319du);
#endif // GEC_ENABLE_AVX2

const Dlp3CurveA
    Dlp3Gen1(Dlp3Field(0x6e06edecu, 0xefa32aa4u, 0xacf634cbu, 0x55003db1u,
                       0x89c0b917u, 0x486c0257u, 0xb6cfd7e4u, 0x4a292897u),
             Dlp3Field(0xf4b3a4a6u, 0x2c380515u, 0x6932c226u, 0x5f0340d5u,
                       0xaed2e0f3u, 0x584483c0u, 0x1639af71u, 0xa493edbeu));
GEC_DEF_GLOBAL(Dlp3G1Card, Dlp3G1SArray, //
               0x00000002u, 0x85b3b1fbu);
GEC_DEF_GLOBAL(Dlp3G1Card_RR, Dlp3G1SArray, //
               0x00000000u, 0x1a7f0f15u);
GEC_DEF_GLOBAL(Dlp3G1Card_OneR, Dlp3G1SArray, //
               0x00000001u, 0xdfd85c45u);

const Dlp3CurveA
    Dlp3Gen2(Dlp3Field(0xba90058bu, 0x535fa443u, 0x7418d8a0u, 0xcb3be8bcu,
                       0xb3ab8171u, 0xd25c1af2u, 0xada23380u, 0x84f96137u),
             Dlp3Field(0x36fb2623u, 0x484b9e96u, 0x2f1e027cu, 0x7a96bc35u,
                       0x644fd629u, 0xb48707edu, 0x7226bc75u, 0x4628e702u));
GEC_DEF_GLOBAL(Dlp3G2Card, Dlp3G2SArray, //
               0x000014d8u, 0xc7ebee77u);
GEC_DEF_GLOBAL(Dlp3G2Card_RR, Dlp3G2SArray, //
               0x00000ca3u, 0x38d33a63u);
GEC_DEF_GLOBAL(Dlp3G2Card_OneR, Dlp3G2SArray, //
               0x000002c3u, 0x613d3042u);

GEC_DEF_GLOBAL(Dlp3P2, Dlp3Array_2, //
               0xffffffffffffffffllu, 0xffffffffffffffffllu,
               0xffffffffffffffffllu, 0xfffffffefffffc2fllu);
GEC_DEF_GLOBAL(Dlp3P2_RR, Dlp3Array_2, //
               0x0000000000000000llu, 0x0000000000000000llu,
               0x0000000000000001llu, 0x000007a2000e90a1llu);
GEC_DEF_GLOBAL(Dlp3P2_OneR, Dlp3Array_2, //
               0x0000000000000000llu, 0x0000000000000000llu,
               0x0000000000000000llu, 0x00000001000003d1llu);

// GEC_DEF_GLOBAL(Dlp3A2, Dlp3Field2, //
//           0, 0, 0, 0);
GEC_DEF_GLOBAL(Dlp3B2, Dlp3Field2, //
               0, 0, 0, 0x0000000d0000319dllu);

GEC_DEF_GLOBAL(Dlp3Card2, Dlp3SArray_2, //
               0x0000000285b3b1fbllu);
GEC_DEF_GLOBAL(Dlp3Card2_RR, Dlp3SArray_2, //
               0x000000001a7f0f15llu);
GEC_DEF_GLOBAL(Dlp3Card2_OneR, Dlp3SArray_2, //
               0x00000001dfd85c45llu);

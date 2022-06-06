#include "curve.hpp"

const Field160 AR_160(0x821006a8u, 0x420792f5u, 0x3a8009a5u, 0xd1ddf060u,
                      0x5d2d2098u);
const Field160 BR_160(0x60c3a8e6u, 0x5781342fu, 0x242d1db1u, 0x2583f9d6u,
                      0xcac79e59u);

const Field160_2 AR2_160(0x821006a8llu, 0x3a8009a5420792f5llu,
                         0x5d2d2098d1ddf060llu);
const Field160_2 BR2_160(0x60c3a8e6llu, 0x242d1db15781342fllu,
                         0xcac79e592583f9d6llu);

def_array(Dlp1P, LIMB_T, LN_160, //
          0x9b7ed883u, 0x1ddf5414u, 0x448756f6u, 0xd5a0ed72u, 0x8049a325u);
def_array(Dlp1P_RR, LIMB_T, LN_160, //
          0x9166fb51u, 0xfd7ddbbdu, 0x6c613523u, 0xa64010c6u, 0x4a86f4adu);
def_array(Dlp1P_OneR, LIMB_T, LN_160, //
          0x6481277du, 0xe220abebu, 0xbb78a909u, 0x2a5f128du, 0x7fb65cdau);

const Dlp1Field Dlp1A(0);
const Dlp1Field Dlp1B(0x7baf70c8u, 0x7b92164du, 0xfc11e794u, 0x3fea12cau,
                      0xe3915053u);

def_array(Dlp1Card, LIMB_T, LN_160, //
          0xfa89e3f2u, 0x9101e4f5u, 0x2244486cu, 0xead076b9u, 0x4024d192u);

def_aligned_array(Dlp1P2, LIMB2_T, LN2_160, 32, //
                  0x1ddf54149b7ed883llu, 0xd5a0ed72448756f6llu, 0x8049a325llu);
def_aligned_array(Dlp1P2_RR, LIMB2_T, LN2_160, 32, //
                  0xf6d8bed667130c77llu, 0x880eba1d4858bd67llu, 0x564b44fallu);
def_aligned_array(Dlp1P2_OneR, LIMB2_T, LN2_160, 32, //
                  0x912d3bbe10d1a50fllu, 0xe9e1701be85df2ccllu, 0x00f7f560llu);

const Dlp1Field2 Dlp1A2(0);
const Dlp1Field2 Dlp1B2(0x07bfab07llu, 0x4f0b80df42ef9664llu,
                        0x8969ddf0868d2878llu);

def_array(Dlp1Card2, LIMB2_T, LN2_160, //
          0x9101e4f5fa89e3f2llu, 0xead076b92244486cllu, 0x4024d192llu);

def_array(Dlp2P, LIMB_T, 1, //
          7919);
def_array(Dlp2P_RR, LIMB_T, 1, //
          3989);
def_array(Dlp2P_OneR, LIMB_T, 1, //
          2618);

const Dlp2Field Dlp2A(7348);
const Dlp2Field Dlp2B(157);

def_array(Dlp2Card, LIMB_T, 1, //
          7889);
def_array(Dlp2Card_RR, LIMB_T, 1, //
          2697);
def_array(Dlp2Card_OneR, LIMB_T, 1, //
          6360);

def_aligned_array(Dlp3P, LIMB_T, Dlp3N, 32, //
                  0xfffffc2fu, 0xfffffffeu, 0xffffffffu, 0xffffffffu,
                  0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
def_aligned_array(Dlp3P_RR, LIMB_T, Dlp3N, 32, //
                  0x000e90a1u, 0x000007a2u, 0x00000001u, 0x00000000u,
                  0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u);
def_aligned_array(Dlp3P_OneR, LIMB_T, Dlp3N, 32, //
                  0x000003d1u, 0x00000001u, 0x00000000u, 0x00000000u,
                  0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u);

const Dlp3Field Dlp3A(0, 0, 0, 0, 0, 0, 0, 0);
const Dlp3Field Dlp3B(0, 0, 0, 0, 0, 0, 0x0000000du, 0x0000319du);
#ifdef GEC_ENABLE_AVX2
const AVX2Dlp3Field AVX2Dlp3A(0, 0, 0, 0, 0, 0, 0, 0);
const AVX2Dlp3Field AVX2Dlp3B(0, 0, 0, 0, 0, 0, 0x0000000du, 0x0000319du);
#endif // GEC_ENABLE_AVX2

const Dlp3CurveA
    Dlp3Gen1(Dlp3Field(0x6e06edecu, 0xefa32aa4u, 0xacf634cbu, 0x55003db1u,
                       0x89c0b917u, 0x486c0257u, 0xb6cfd7e4u, 0x4a292897u),
             Dlp3Field(0xf4b3a4a6u, 0x2c380515u, 0x6932c226u, 0x5f0340d5u,
                       0xaed2e0f3u, 0x584483c0u, 0x1639af71u, 0xa493edbeu));
def_aligned_array(Dlp3G1Card, LIMB_T, Dlp3G1SN, 8, //
                  0x85b3b1fbu, 0x00000002u);
def_aligned_array(Dlp3G1Card_RR, LIMB_T, Dlp3G1SN, 8, //
                  0x1a7f0f15u, 0x00000000u);
def_aligned_array(Dlp3G1Card_OneR, LIMB_T, Dlp3G1SN, 8, //
                  0xdfd85c45u, 0x00000001u);

const Dlp3CurveA
    Dlp3Gen2(Dlp3Field(0xba90058bu, 0x535fa443u, 0x7418d8a0u, 0xcb3be8bcu,
                       0xb3ab8171u, 0xd25c1af2u, 0xada23380u, 0x84f96137u),
             Dlp3Field(0x36fb2623u, 0x484b9e96u, 0x2f1e027cu, 0x7a96bc35u,
                       0x644fd629u, 0xb48707edu, 0x7226bc75u, 0x4628e702u));
def_aligned_array(Dlp3G2Card, LIMB_T, Dlp3G2SN, 8, //
                  0xc7ebee77u, 0x000014d8u);
def_aligned_array(Dlp3G2Card_RR, LIMB_T, Dlp3G2SN, 8, //
                  0x38d33a63u, 0x00000ca3u);
def_aligned_array(Dlp3G2Card_OneR, LIMB_T, Dlp3G2SN, 8, //
                  0x613d3042u, 0x000002c3u);

def_aligned_array(Dlp3P2, LIMB2_T, Dlp3N2, 32, //
                  0xfffffffefffffc2fllu, 0xffffffffffffffffllu,
                  0xffffffffffffffffllu, 0xffffffffffffffffllu);
def_aligned_array(Dlp3P2_RR, LIMB2_T, Dlp3N2, 32, //
                  0x000007a2000e90a1llu, 0x0000000000000001llu,
                  0x0000000000000000llu, 0x0000000000000000llu);
def_aligned_array(Dlp3P2_OneR, LIMB2_T, Dlp3N2, 32, //
                  0x00000001000003d1llu, 0x0000000000000000llu,
                  0x0000000000000000llu, 0x0000000000000000llu);

const Dlp3Field2 Dlp3A2(0, 0, 0, 0);
const Dlp3Field2 Dlp3B2(0, 0, 0, 0x0000000d0000319dllu);

def_aligned_array(Dlp3Card2, LIMB2_T, Dlp3SN2, 8, //
                  0x0000000285b3b1fbllu);
def_aligned_array(Dlp3Card2_RR, LIMB2_T, Dlp3SN2, 8, //
                  0x000000001a7f0f15llu);
def_aligned_array(Dlp3Card2_OneR, LIMB2_T, Dlp3SN2, 8, //
                  0x00000001dfd85c45llu);

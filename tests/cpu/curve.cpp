#include <curve.hpp>

#ifdef GEC_NVCC
GEC_INT_TOO_LARGE
#endif // GEC_NVCC

using namespace gec::bigint::literal;

GEC_DEF_GLOBAL(AR_160, Field160, //
               0x821006a8'420792f5'3a8009a5'd1ddf060'5d2d2098_int);
GEC_DEF_GLOBAL(BR_160, Field160, //
               0x60c3a8e6'5781342f'242d1db1'2583f9d6'cac79e59_int);

GEC_DEF_GLOBAL(AR2_160, Field160_2, //
               0x821006a8'3a8009a5'420792f5'5d2d2098'd1ddf060_int);
GEC_DEF_GLOBAL(BR2_160, Field160_2, //
               0x60c3a8e6'242d1db1'5781342f'cac79e59'2583f9d6_int);

GEC_DEF_GLOBAL(Dlp1P, Dlp1Array, //
               0x8049a325'd5a0ed72'448756f6'1ddf5414'9b7ed883_int);
GEC_DEF_GLOBAL(Dlp1P_RR, Dlp1Array, //
               0x4a86f4ad'a64010c6'6c613523'fd7ddbbd'9166fb51_int);
GEC_DEF_GLOBAL(Dlp1P_OneR, Dlp1Array, //
               0x7fb65cda'2a5f128d'bb78a909'e220abeb'6481277d_int);

// GEC_DEF_GLOBAL(Dlp1A, Dlp1Field, //
//           0);
GEC_DEF_GLOBAL(Dlp1B, Dlp1Field, //
               0x7baf70c8'7b92164d'fc11e794'3fea12ca'e3915053_int);

GEC_DEF_GLOBAL(Dlp1Card, Dlp1SArray, //
               0x4024d192'ead076b9'2244486c'9101e4f5'fa89e3f2_int);

GEC_DEF_GLOBAL(Dlp1P2, Dlp1Array_2, //
               0x8049a325'd5a0ed72'448756f6'1ddf5414'9b7ed883_int);
GEC_DEF_GLOBAL(Dlp1P2_RR, Dlp1Array_2, //
               0x564b44fa'880eba1d'4858bd67'f6d8bed6'67130c77_int);
GEC_DEF_GLOBAL(Dlp1P2_OneR, Dlp1Array_2, //
               0x00f7f560'e9e1701b'e85df2cc'912d3bbe'10d1a50f_int);

// GEC_DEF_GLOBAL(Dlp1A2, Dlp1Field2, //
//           0);
GEC_DEF_GLOBAL(Dlp1B2, Dlp1Field2, //
               0x07bfab07'4f0b80df'42ef9664'8969ddf0'868d2878_int);

GEC_DEF_GLOBAL(Dlp1Card2, Dlp1SArray_2, //
               0x4024d192'ead076b9'2244486c'9101e4f5'fa89e3f2_int);

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

GEC_DEF_GLOBAL(
    Dlp3P, Dlp3Array, //
    0xffffffff'ffffffff'ffffffff'ffffffff'ffffffff'ffffffff'fffffffe'fffffc2f_int);
GEC_DEF_GLOBAL(
    Dlp3P_RR, Dlp3Array, //
    0x00000000'00000000'00000000'00000000'00000000'00000001'000007a2'000e90a1_int);
GEC_DEF_GLOBAL(
    Dlp3P_OneR, Dlp3Array, //
    0x00000000'00000000'00000000'00000000'00000000'00000000'00000001'000003d1_int);

// GEC_DEF_GLOBAL(Dlp3A, Dlp3Field, //
//                0);
GEC_DEF_GLOBAL(Dlp3B, Dlp3Field, //
               0xd'0000319d_int);
#ifdef GEC_ENABLE_AVX2
// const AVX2Dlp3Field AVX2Dlp3A(0, 0, 0, 0, 0, 0, 0, 0);
const AVX2Dlp3Field AVX2Dlp3B(0x0000000d'0000319d_int);
#endif // GEC_ENABLE_AVX2

const Dlp3CurveA Dlp3Gen1(
    Dlp3Field(
        0x6e06edec'efa32aa4'acf634cb'55003db1'89c0b917'486c0257'b6cfd7e4'4a292897_int),
    Dlp3Field(
        0xf4b3a4a6'2c380515'6932c226'5f0340d5'aed2e0f3'584483c0'1639af71'a493edbe_int));
GEC_DEF_GLOBAL(Dlp3G1Card, Dlp3G1SArray, //
               0x00000002'85b3b1fb_int);
GEC_DEF_GLOBAL(Dlp3G1Card_RR, Dlp3G1SArray, //
               0x00000000'1a7f0f15_int);
GEC_DEF_GLOBAL(Dlp3G1Card_OneR, Dlp3G1SArray, //
               0x00000001'dfd85c45_int);

const Dlp3CurveA Dlp3Gen2(
    Dlp3Field(
        0xba90058b'535fa443'7418d8a0'cb3be8bc'b3ab8171'd25c1af2'ada23380'84f96137_int),
    Dlp3Field(
        0x36fb2623'484b9e96'2f1e027c'7a96bc35'644fd629'b48707ed'7226bc75'4628e702_int));
GEC_DEF_GLOBAL(Dlp3G2Card, Dlp3G2SArray, //
               0x000014d8'c7ebee77_int);
GEC_DEF_GLOBAL(Dlp3G2Card_RR, Dlp3G2SArray, //
               0x00000ca3'38d33a63_int);
GEC_DEF_GLOBAL(Dlp3G2Card_OneR, Dlp3G2SArray, //
               0x000002c3'613d3042_int);

GEC_DEF_GLOBAL(
    Dlp3P2, Dlp3Array_2, //
    0xffffffff'ffffffff'ffffffff'ffffffff'ffffffff'ffffffff'fffffffe'fffffc2f_int);
GEC_DEF_GLOBAL(
    Dlp3P2_RR, Dlp3Array_2, //
    0x00000000'00000000'00000000'00000000'00000000'00000001'000007a2'000e90a1_int);
GEC_DEF_GLOBAL(
    Dlp3P2_OneR, Dlp3Array_2, //
    0x00000000'00000000'00000000'00000000'00000000'00000000'00000001'000003d1_int);

// GEC_DEF_GLOBAL(Dlp3A2, Dlp3Field2, //
//                0);
GEC_DEF_GLOBAL(Dlp3B2, Dlp3Field2, //
               0x0000000d'0000319d_int);

GEC_DEF_GLOBAL(Dlp3Card2, Dlp3SArray_2, //
               0x00000002'85b3b1fb_int);
GEC_DEF_GLOBAL(Dlp3Card2_RR, Dlp3SArray_2, //
               0x00000000'1a7f0f15_int);
GEC_DEF_GLOBAL(Dlp3Card2_OneR, Dlp3SArray_2, //
               0x00000001'dfd85c45_int);

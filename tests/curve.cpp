#include "curve.hpp"

const Field160 AR_160(0x821006a8u, 0x420792f5u, 0x3a8009a5u, 0xd1ddf060u,
                      0x5d2d2098u);
const Field160 BR_160(0x60c3a8e6u, 0x5781342fu, 0x242d1db1u, 0x2583f9d6u,
                      0xcac79e59u);

const Field160_2 AR2_160(0x821006a8llu, 0x3a8009a5420792f5llu,
                         0x5d2d2098d1ddf060llu);
const Field160_2 BR2_160(0x60c3a8e6llu, 0x242d1db15781342fllu,
                         0xcac79e592583f9d6llu);

alignas(8) const LIMB_T DlpP[LN_160] = {0x9b7ed883u, 0x1ddf5414u, 0x448756f6u,
                                        0xd5a0ed72u, 0x8049a325u};
alignas(8) const LIMB_T DlpP_RR[LN_160] = {
    0x9166fb51u, 0xfd7ddbbdu, 0x6c613523u, 0xa64010c6u, 0x4a86f4adu};
alignas(8) const LIMB_T DlpP_OneR[LN_160] = {
    0x6481277du, 0xe220abebu, 0xbb78a909u, 0x2a5f128du, 0x7fb65cdau};

const DlpField DlpA(0);
const DlpField DlpB(0x7baf70c8u, 0x7b92164du, 0xfc11e794u, 0x3fea12cau,
                    0xe3915053u);

alignas(8) const LIMB_T DlpCard[LN_160] = {
    0xfa89e3f2u, 0x9101e4f5u, 0x2244486cu, 0xead076b9u, 0x4024d192u};

alignas(32) const LIMB2_T DlpP2[LN2_160] = {
    0x1ddf54149b7ed883llu, 0xd5a0ed72448756f6llu, 0x8049a325llu};
alignas(32) const LIMB2_T DlpP2_RR[LN2_160] = {
    0xf6d8bed667130c77llu, 0x880eba1d4858bd67llu, 0x564b44fallu};
alignas(32) const LIMB2_T DlpP2_OneR[LN2_160] = {
    0x912d3bbe10d1a50fllu, 0xe9e1701be85df2ccllu, 0x00f7f560llu};

const DlpField2 DlpA2(0);
const DlpField2 DlpB2(0x07bfab07llu, 0x4f0b80df42ef9664llu,
                      0x8969ddf0868d2878llu);

alignas(32) const LIMB2_T DlpCard2[LN2_160] = {
    0x9101e4f5fa89e3f2llu, 0xead076b92244486cllu, 0x4024d192llu};

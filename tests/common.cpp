#include "common.hpp"

alignas(8) const LIMB_T MOD_160[LN_160] = {
    0x5e3b0969u, 0x5c17ef6cu, 0xf5d7cecau, 0xd8db9627u, 0xb77902abu};
alignas(8) const LIMB_T RR_160[LN_160] = {0x61260cf2u, 0x399ce6a5u, 0x46c1c15au,
                                          0x8aec7519u, 0x7cd393b3u};
alignas(8) const LIMB_T OneR_160[LN_160] = {
    0xa1c4f697u, 0xa3e81093u, 0x0a283135u, 0x272469d8u, 0x4886fd54u};

alignas(8) const LIMB2_T MOD2_160[LN2_160] = {0x5c17ef6c5e3b0969u,
                                              0xd8db9627f5d7cecau, 0xb77902abu};
alignas(8) const LIMB2_T RR2_160[LN2_160] = {0x399ce6a561260cf2u,
                                             0x8aec751946c1c15au, 0x7cd393b3u};
alignas(8) const LIMB2_T OneR2_160[LN2_160] = {
    0xa3e81093a1c4f697llu, 0x272469d80a283135llu, 0x4886fd54llu};

alignas(32) const LIMB_T MOD_256[LN_256] = {
    0xfffffc2fu, 0xfffffffeu, 0xffffffffu, 0xffffffffu,
    0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
alignas(32) const LIMB_T RR_256[LN_256] = {
    0x000e90a1u, 0x000007a2u, 0x00000001u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};
alignas(32) const LIMB_T OneR_256[LN_256] = {
    0x000003d1u, 0x00000001u, 0x00000001u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};

alignas(32) const LIMB2_T MOD2_256[LN2_256] = {
    0xfffffffefffffc2fllu, 0xffffffffffffffffllu, 0xffffffffffffffffllu,
    0xffffffffffffffffllu};
alignas(32) const LIMB2_T RR2_256[LN2_256] = {
    0x000007a2000e90a1llu, 0x0000000000000001llu, 0x0000000000000000llu,
    0x0000000000000000llu};
alignas(32) const LIMB2_T OneR2_256[LN2_256] = {
    0x00000001000003d1llu, 0x0000000000000001llu, 0x0000000000000000llu,
    0x0000000000000000llu};

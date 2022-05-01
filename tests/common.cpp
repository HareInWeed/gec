#include "common.hpp"

const LIMB_T MOD_160[LN_160] = {0x5e3b0969u, 0x5c17ef6cu, 0xf5d7cecau,
                                0xd8db9627u, 0xb77902abu};
const LIMB_T RR_160[LN_160] = {0x61260cf2u, 0x399ce6a5u, 0x46c1c15au,
                               0x8aec7519u, 0x7cd393b3u};

alignas(32) const LIMB_T MOD_256[LN_256] = {
    0xfffffc2fu, 0xfffffffeu, 0xffffffffu, 0xffffffffu,
    0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
alignas(32) const LIMB_T RR_256[LN_256] = {
    0x000e90a1u, 0x000007a2u, 0x00000001u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};

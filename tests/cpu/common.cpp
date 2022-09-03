#define ARRAY_DEF
#include <common.hpp>

GEC_DEF_GLOBAL(MOD_160, Array160, //
               0xb77902abu, 0xd8db9627u, 0xf5d7cecau, 0x5c17ef6cu, 0x5e3b0969u);
GEC_DEF_GLOBAL(RR_160, Array160, //
               0x7cd393b3u, 0x8aec7519u, 0x46c1c15au, 0x399ce6a5u, 0x61260cf2u);
GEC_DEF_GLOBAL(OneR_160, Array160, //
               0x4886fd54u, 0x272469d8u, 0x0a283135u, 0xa3e81093u, 0xa1c4f697u);

GEC_DEF_GLOBAL(MOD2_160, Array160_2, //
               0xb77902abllu, 0xd8db9627f5d7cecallu, 0x5c17ef6c5e3b0969llu);
GEC_DEF_GLOBAL(RR2_160, Array160_2, //
               0x158d01edllu, 0xcf41f1cd75ad34a8llu, 0x87ada0ed26f392f0llu);
GEC_DEF_GLOBAL(OneR2_160, Array160_2, //
               0x3e45aeb8llu, 0x73a542628a520aeellu, 0xad68a50f4a90f52allu);

GEC_DEF_GLOBAL(MOD_256, Array256, //
               0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
               0xffffffffu, 0xfffffffeu, 0xfffffc2fu);
GEC_DEF_GLOBAL(RR_256, Array256, //
               0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
               0x00000001u, 0x000007a2u, 0x000e90a1u);
GEC_DEF_GLOBAL(OneR_256, Array256, //
               0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
               0x00000001u, 0x00000001u, 0x000003d1u);

GEC_DEF_GLOBAL(MOD2_256, Array256_2, //
               0xffffffffffffffffllu, 0xffffffffffffffffllu,
               0xffffffffffffffffllu, 0xfffffffefffffc2fllu);
GEC_DEF_GLOBAL(RR2_256, Array256_2, //
               0x0000000000000000llu, 0x0000000000000000llu,
               0x0000000000000001llu, 0x000007a2000e90a1llu);
GEC_DEF_GLOBAL(OneR2_256, Array256_2, //
               0x0000000000000000llu, 0x0000000000000000llu,
               0x0000000000000001llu, 0x00000001000003d1llu);

#pragma once
#ifndef GEC_TEST_FIELD_HPP
#define GEC_TEST_FIELD_HPP

#include "common.hpp"

#include <gec/bigint.hpp>

#include <type_traits>

using Field160 = GEC_BASE_FIELD(Array160, MOD_160, MOD_P_160, RR_160, OneR_160);

using Field160_2 = GEC_BASE_FIELD(Array160_2, MOD2_160, MOD2_P_160, RR2_160,
                                  OneR2_160);

#endif // !GEC_TEST_FIELD_HPP
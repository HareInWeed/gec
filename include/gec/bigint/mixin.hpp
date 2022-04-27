#pragma once
#ifndef GEC_BIGINT_MIXIN_HPP
#define GEC_BIGINT_MIXIN_HPP

#include "mixin/add_sub.hpp"
#include "mixin/bit_ops.hpp"
#include "mixin/mod_add_sub.hpp"
#include "mixin/montgomery.hpp"
#include "mixin/vt_compare.hpp"

// IO mixins are not include by default
#ifdef GEC_DEBUG
#include "mixin/ostream.hpp"
#include "mixin/print.hpp"
#endif // GEC_DEBUG

#endif // !GEC_BIGINT_MIXIN_HPP

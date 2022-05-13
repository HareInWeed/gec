#pragma once
#ifndef GEC_TEST_FIELD_HPP
#define GEC_TEST_FIELD_HPP

#include "common.hpp"

#include <gec/bigint.hpp>

class Field160
    : public gec::bigint::ArrayBE<LIMB_T, LN_160>,
      public gec::bigint::Constants<Field160, LIMB_T, LN_160>,
      public gec::bigint::VtCompare<Field160, LIMB_T, LN_160>,
      public gec::bigint::BitOps<Field160, LIMB_T, LN_160>,
      public gec::bigint::ModAddSub<Field160, LIMB_T, LN_160, MOD_160>,
      public gec::bigint::Montgomery<Field160, LIMB_T, LN_160, MOD_160,
                                     MOD_P_160, RR_160, OneR_160>,
      public gec::bigint::Exponentiation<Field160>,
      public gec::bigint::BigintContext<Field160>,
      public gec::bigint::ArrayOstream<Field160, LIMB_T, LN_160>,
      public gec::bigint::ArrayPrint<Field160, LIMB_T, LN_160> {
    using ArrayBE::ArrayBE;
};

class Field160_2
    : public gec::bigint::ArrayBE<LIMB2_T, LN2_160>,
      public gec::bigint::Constants<Field160_2, LIMB2_T, LN2_160>,
      public gec::bigint::VtCompare<Field160_2, LIMB2_T, LN2_160>,
      public gec::bigint::BitOps<Field160_2, LIMB2_T, LN2_160>,
      public gec::bigint::ModAddSub<Field160_2, LIMB2_T, LN2_160, MOD2_160>,
      public gec::bigint::MontgomeryCarryFree<Field160_2, LIMB2_T, LN2_160,
                                              MOD2_160, MOD2_P_160, RR2_160,
                                              OneR2_160>,
      public gec::bigint::Exponentiation<Field160_2>,
      public gec::bigint::BigintContext<Field160_2>,
      public gec::bigint::ArrayOstream<Field160_2, LIMB2_T, LN2_160>,
      public gec::bigint::ArrayPrint<Field160_2, LIMB2_T, LN2_160> {
    using ArrayBE::ArrayBE;
};

#endif // !GEC_TEST_FIELD_HPP
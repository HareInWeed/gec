#pragma once
#ifndef GEC_BIGINT_MIXIN_QUADRATIC_RESIDUE_HPP
#define GEC_BIGINT_MIXIN_QUADRATIC_RESIDUE_HPP

#include <gec/utils/basic.hpp>
#include <gec/utils/crtp.hpp>

namespace gec {

namespace bigint {

template <class Core>
class GEC_EMPTY_BASES QuadraticResidue
    : protected CRTP<Core, QuadraticResidue<Core>> {
    friend CRTP<Core, QuadraticResidue<Core>>;

  public:
    template <typename CTX>
    __host__ __device__ GEC_INLINE int legendre(const Core &a, CTX ctx) {
        return legendre(a, a->mod(), ctx);
    }
    template <typename CTX>
    int legendre(const Core &a, const Core &p, CTX ctx) {
        // TODO
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_QUADRATIC_RESIDUE_HPP
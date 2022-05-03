#pragma once
#ifndef GEC_BIGINT_MIXIN_OSTREAM_HPP
#define GEC_BIGINT_MIXIN_OSTREAM_HPP

#include <gec/utils/crtp.hpp>

#include <iomanip>
#include <iostream>

namespace gec {

namespace bigint {

/** @brief mixin that enables output array with ostream
 */
template <class Core, class LIMB_T, size_t LIMB_N>
class ArrayOstreamMixin
    : public CRTP<Core, ArrayOstreamMixin<Core, LIMB_T, LIMB_N>> {
  public:
    friend std::ostream &operator<<(std::ostream &o, const Core &bigint) {
        using namespace std;
        o << "0x";
        for (size_t i = 0; i < LIMB_N; ++i) {
            o << setw(2 * sizeof(LIMB_T)) << setfill('0') << hex
              << bigint.get_arr()[LIMB_N - 1 - i] << ' ';
        }
        return o;
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_OSTREAM_HPP
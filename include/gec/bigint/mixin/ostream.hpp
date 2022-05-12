#pragma once
#ifndef GEC_BIGINT_MIXIN_OSTREAM_HPP
#define GEC_BIGINT_MIXIN_OSTREAM_HPP

#include <gec/utils/crtp.hpp>

#include <iomanip>
#include <ostream>

namespace gec {

namespace bigint {

/** @brief mixin that enables output array() with ostream
 */
template <class Core, class LIMB_T, size_t LIMB_N>
class ArrayOstream : protected CRTP<Core, ArrayOstream<Core, LIMB_T, LIMB_N>> {
    friend CRTP<Core, ArrayOstream<Core, LIMB_T, LIMB_N>>;

  public:
    friend std::ostream &operator<<(std::ostream &o, const Core &bigint) {
        using namespace std;
        for (size_t i = 0; i < LIMB_N; ++i) {
            o << (i == 0 ? "0x" : " ") << setw(2 * sizeof(LIMB_T))
              << setfill('0') << hex << bigint.array()[LIMB_N - 1 - i];
        }
        return o;
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_OSTREAM_HPP
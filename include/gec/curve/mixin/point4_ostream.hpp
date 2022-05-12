#pragma once
#ifndef GEC_CURVE_MIXIN_POINT4_OSTREAM_HPP
#define GEC_CURVE_MIXIN_POINT4_OSTREAM_HPP

#include <gec/utils/crtp.hpp>
#include <iomanip>
#include <ostream>

namespace gec {

namespace curve {

/** @brief mixin that enables output x(), y(), z() and z1() with ostream
 */
template <typename Core>
class Point4Ostream : protected CRTP<Core, Point4Ostream<Core>> {
    friend CRTP<Core, Point4Ostream<Core>>;

  public:
    friend std::ostream &operator<<(std::ostream &o, const Core &point) {
        using namespace std;
        o << '{' << point.x() << ',' << endl;
        o << ' ' << point.y() << ',' << endl;
        o << ' ' << point.z() << ',' << endl;
        o << ' ' << point.z1() << '}' << endl;
        return o;
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_POINT4_OSTREAM_HPP

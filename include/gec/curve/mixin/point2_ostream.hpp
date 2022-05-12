#pragma once
#ifndef GEC_CURVE_MIXIN_POINT2_OSTREAM_HPP
#define GEC_CURVE_MIXIN_POINT2_OSTREAM_HPP

#include <gec/utils/crtp.hpp>
#include <iomanip>
#include <ostream>

namespace gec {

namespace curve {

/** @brief mixin that enables output x() and y() with ostream
 */
template <typename Core>
class Point2Ostream : protected CRTP<Core, Point2Ostream<Core>> {
    friend CRTP<Core, Point2Ostream<Core>>;

  public:
    friend std::ostream &operator<<(std::ostream &o, const Core &point) {
        using namespace std;
        o << '{' << point.x() << ',' << endl;
        o << ' ' << point.y() << '}' << endl;
        return o;
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_POINT2_OSTREAM_HPP

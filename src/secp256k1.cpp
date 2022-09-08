#include <gec/curve/secp256k1.hpp>

namespace gec {

namespace curve {

namespace secp256k1 {

using namespace gec::bigint::literal;

// ----- Field -----
namespace _secp256k1_ {

const FBase MOD( //
    0xffffffff'ffffffff'ffffffff'ffffffff'ffffffff'ffffffff'fffffffe'fffffc2f_int);
const FBase RR( //
    0x01'000007a2'000e90a1_int);
const FBase ONE_R( //
    0x1'000003d1_int);

#ifdef __CUDACC__
__constant__ const FBase d_MOD( //
    0xffffffff'ffffffff'ffffffff'ffffffff'ffffffff'ffffffff'fffffffe'fffffc2f_int);
__constant__ const FBase d_RR( //
    0x1'000007a2'000e90a1_int);
__constant__ const FBase d_ONE_R( //
    0x1'000003d1_int);
#endif // __CUDACC__

const SBase CARD( //
    0xffffffff'ffffffff'ffffffff'fffffffe'baaedce6'af48a03b'bfd25e8c'd0364141_int);
const SBase CARD_RR( //
    0x9d671cd5'81c69bc5'e697f5e4'5bcd07c6'741496c2'0e7cf878'896cf214'67d7d140_int);
const SBase CARD_ONE_R( //
    0x1'45512319'50b75fc4'402da1732fc9bebf_int);

#ifdef __CUDACC__
__constant__ const SBase d_CARD( //
    0xffffffff'ffffffff'ffffffff'fffffffe'baaedce6'af48a03b'bfd25e8c'd0364141_int);
__constant__ const SBase d_CARD_RR( //
    0x9d671cd5'81c69bc5'e697f5e4'5bcd07c6'741496c2'0e7cf878'896cf214'67d7d140_int);
__constant__ const SBase d_CARD_ONE_R( //
    0x1'45512319'50b75fc4'402da173'2fc9bebf_int);
#endif // __CUDACC__

// A = 0
const Field B(0x7'00001ab7_int);

} // namespace _secp256k1_

const Curve<> Gen(
    Field( //
        0x9981e643'e9089f48'979f48c0'33fd129c'231e2953'29bc66db'd7362e5a'487e2097_int),
    Field( //
        0xcf3f851f'd4a582d6'70b6b59a'ac19c136'8dfc5d5d'1f1dc64d'b15ea6d2'd3dbabe2_int),
    Field( //
        0x1'000003d1_int));

#ifdef __CUDACC__
__constant__ const Curve<> d_Gen(
    Field( //
        0x9981e643'e9089f48'979f48c0'33fd129c'231e2953'29bc66db'd7362e5a'487e2097_int),
    Field( //
        0xcf3f851f'd4a582d6'70b6b59a'ac19c136'8dfc5d5d'1f1dc64d'b15ea6d2'd3dbabe2_int),
    Field( //
        0x1'000003d1_int));
#endif // __CUDACC__

} // namespace secp256k1

} // namespace curve

} // namespace gec
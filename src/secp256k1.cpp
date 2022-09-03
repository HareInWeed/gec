#include <gec/curve/secp256k1.hpp>

namespace gec {

namespace curve {

namespace secp256k1 {

// ----- Field -----
namespace _secp256k1_ {

const FBase MOD(0xffffffffffffffffllu, 0xffffffffffffffffllu,
                0xffffffffffffffffllu, 0xfffffffefffffc2fllu);
const FBase RR(0x0llu, 0x0llu, 0x1llu, 0x000007a2000e90a1llu);
const FBase ONE_R(0x1000003d1llu);

#ifdef __CUDACC__
__constant__ const FBase d_MOD(0xffffffffffffffffllu, 0xffffffffffffffffllu,
                               0xffffffffffffffffllu, 0xfffffffefffffc2fllu);
__constant__ const FBase d_RR(0x0llu, 0x0llu, 0x1llu, 0x000007a2000e90a1llu);
__constant__ const FBase d_ONE_R(0x1000003d1llu);
#endif // __CUDACC__

const SBase CARD(0xffffffffffffffffllu, 0xfffffffffffffffellu,
                 0xbaaedce6af48a03bllu, 0xbfd25e8cd0364141llu);
const SBase CARD_RR(0x9d671cd581c69bc5llu, 0xe697f5e45bcd07c6llu,
                    0x741496c20e7cf878llu, 0x896cf21467d7d140llu);
const SBase CARD_ONE_R(0x0llu, 0x1llu, 0x4551231950b75fc4llu,
                       0x402da1732fc9bebfllu);

#ifdef __CUDACC__
__constant__ const SBase d_CARD(0xffffffffffffffffllu, 0xfffffffffffffffellu,
                                0xbaaedce6af48a03bllu, 0xbfd25e8cd0364141llu);
__constant__ const SBase d_CARD_RR(0x9d671cd581c69bc5llu, 0xe697f5e45bcd07c6llu,
                                   0x741496c20e7cf878llu,
                                   0x896cf21467d7d140llu);
__constant__ const SBase d_CARD_ONE_R(0x0llu, 0x1llu, 0x4551231950b75fc4llu,
                                      0x402da1732fc9bebfllu);
#endif // __CUDACC__

// A = 0
const Field B(0x700001ab7);

} // namespace _secp256k1_

const Curve<> Gen(Field(0x9981e643e9089f48llu, 0x979f48c033fd129cllu,
                        0x231e295329bc66dbllu, 0xd7362e5a487e2097llu),
                  Field(0xcf3f851fd4a582d6llu, 0x70b6b59aac19c136llu,
                        0x8dfc5d5d1f1dc64dllu, 0xb15ea6d2d3dbabe2llu),
                  Field(0x1000003d1llu));

#ifdef __CUDACC__
__constant__ const Curve<>
    d_Gen(Field(0x9981e643e9089f48llu, 0x979f48c033fd129cllu,
                0x231e295329bc66dbllu, 0xd7362e5a487e2097llu),
          Field(0xcf3f851fd4a582d6llu, 0x70b6b59aac19c136llu,
                0x8dfc5d5d1f1dc64dllu, 0xb15ea6d2d3dbabe2llu),
          Field(0x1000003d1llu));
#endif // __CUDACC__

} // namespace secp256k1

} // namespace curve

} // namespace gec
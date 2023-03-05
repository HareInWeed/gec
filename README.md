# GEC

Elliptic curve cryptography with GPU acceleration

**Notice**: **This implementation is experimental and has not undergone any code audit. Use it at your own risk.**

## Features

- finite field
  - modular addition/subtraction/multiplication/inversion/exponentiation
  - quadratic residue
  - random element generator
- elliptic curve
  - affine/projective/Jacobian coordinates
  - point add
  - scalar multiplication
  - map to curve
  - pre-defined curves
    - secp256k1
    - sm2
- discrete logarithm for elliptic curve
  - pollard lambda
  - pollard rho
- acceleration
  - AVX2
  - multi-thread
  - GPU (with CUDA)

## Benchmarks

See [benchmarks.md](docs/benchmarks.md).

## Usage

### Basic Usage

Elliptic curves need to be defined before any ECC operations can be carried out.
Take secp256k1 as an example,
define the finite field `Field` for secp256k1 first.

```c++
#include <gec/utils/macros.hpp>
#include <gec/bigint/preset.hpp>
#include <gec/curve/preset.hpp>

using namespace gec::bigint::literal; // use the bigint literal

// use uint64 x 4 to store a single element on finite field
using Bigint256 = bigint::ArrayBE<uint64_t, 4>;

// define parameters required by montgomery multiplication:
GEC_DEF_GLOBAL(MOD, Bigint256,     // cardinality of finite field
    0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f_int);
constexpr Bigint256::LimbT MOD_P = // -MOD^-1 mod 2^64
    0xd838091dd2253531ull;
GEC_DEF_GLOBAL(RR, Bigint256,      // 2^512 mod MOD
    0x01000007a2000e90a1_int);
GEC_DEF_GLOBAL(ONE_R, Bigint256,   // 2^256 mod MOD
    0x1000003d1_int);

// define the finite field type
using Field = GEC_BASE_FIELD(Bigint256, MOD, MOD_P, RR, ONE_R);
```

Then define `Scalar` as the scalar of secp256k1.

```c++
// define parameters required by montgomery multiplication:
GEC_DEF_GLOBAL(CARD, Bigint256,       // cardinality of the elliptic curve
    0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141_int);
constexpr Bigint256::LimbT CARD_P =   // -CARD^-1 mod 2^64
    0x4b0dff665588b13full;
GEC_DEF_GLOBAL(CARD_RR, Bigint256,    // 2^512 mod CARD
    0x9d671cd581c69bc5e697f5e45bcd07c6741496c20e7cf878896cf21467d7d140_int);
GEC_DEF_GLOBAL(CARD_ONE_R, Bigint256, // 2^256 mod CARD
    0x14551231950b75fc4402da1732fc9bebf_int);

// define the scalar type
using Scalar = GEC_BASE_FIELD(Bigint256, CARD, CARD_P, CARD_RR, CARD_ONE_R);
```

Finally, define `Secp256k1` as curve secp256k1.

```c++
// parameters of the elliptic curve, in montgomery form
const Field A(0);                 // = A * 2^256 mod MOD
const Field B(0x700001ab7_int);  // = B * 2^256 mod MOD

// define the curve with Jacobian coordinate
using Secp256k1_  = GEC_CURVE(gec::curve::JacobianCurve, Field, A, B);
// use the specialized implementation for curves whose A = 0 to boost performance
using Secp256k1   = GEC_CURVE_B(gec::curve::JacobianCurve, Field, B);

// define the generator, in montgomery form
const Secp256k1 GEN(
    Field(0x9981e643e9089f48979f48c033fd129c231e295329bc66dbd7362e5a487e2097_int),
    Field(0xcf3f851fd4a582d670b6b59aac19c1368dfc5d5d1f1dc64db15ea6d2d3dbabe2_int),
    Field(0x1000003d1_int),
);
```

Now you are free to use `Secp256k1` for elliptic curve arithmetics.

```c++
Secp256k1 p1, p2, p3;
Secp256k1::mul(p1, 3, GEN);

Scalar s;
Scalar::neg(s, 0x3_int);
Secp256k1::mul(p2, s, GEN);

Secp256k1::add(p3, p1, p2);
p3.is_inf() // == true
```

With CUDA support enabled, you can use `Secp256k1` in CUDA kernels as well.

```c++
__constant__ const Secp256k1 d_GEN(
    Field(0x9981e643e9089f48979f48c033fd129c231e295329bc66dbd7362e5a487e2097_int),
    Field(0xcf3f851fd4a582d670b6b59aac19c1368dfc5d5d1f1dc64db15ea6d2d3dbabe2_int),
    Field(0x1000003d1_int),
);

__global__ void cuda_kernel() {
    Secp256k1 p1, p2, p3;
    Secp256k1::mul(p1, 3, d_GEN);

    Scalar s;
    Scalar::neg(s, 0x3_int);
    Secp256k1::mul(p2, s, d_GEN);

    Secp256k1::add(p3, p1, p2);
    p3.is_inf() // == true
}
```

### AVX2

To enable AVX2, set `GEC_ENABLE_AVX2` option to `On` in CMake.

Note that this option only adds some mixins built with AVX2.
To enable AVX2 acceleration in finite field and elliptic curve algorithms, 
make sure the finite field and elliptic curve classes
are built with AVX2 accelerated mixins, 
such as [`AVX2MontgomeryOps`](include/gec/bigint/mixin/montgomery.hpp#L297).

### Multi-thread Discrete Logarithm

To enable multi-thread discrete logarithm, set `GEC_ENABLE_PTHREADS` option to `On` in CMake.
Make sure pthreads is available before turning this flag on.

### CUDA

To enable CUDA support, set `GEC_ENABLE_CUDA` option to `On` in CMake.

GEC has been tested under CUDA 11.1. Older CUDA might work as well, but it is not guaranteed.

**Notice**: When `GEC_ENABLE_CUDA` is set to `On`, in order to support CUDA,
the macro `GEC_DEF_GLOBAL(VAR, ...)` will define an additional variable, 
`d_VAR`, in constant memory with the same value.
In case of naming collision, 
you can define the variables for CPU and variables for CUDA manually.
Other macros such as `GEC_BASE_FIELD` and `GEC_CURVE` also implicitly
refer to `d_VAR` when CUDA support is enabled. 
You may need to replace them as well.
Check out [secp256k1.cpp](src/secp256k1.cpp) for an example.

## License

MIT

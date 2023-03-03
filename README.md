# GEC

Elliptic curve cryptography with GPU acceleration

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
- discrete logarithm
  - pollard lambda
  - pollard rho

### AVX2

To enable AVX2, set `GEC_ENABLE_AVX2` option to `On` in CMake

Notice that the option only adds some mixins built with AVX2. To 

### Multi-thread Discrete Logarithm

To enable multi-thread discrete logarithm, set `GEC_ENABLE_PTHREADS` option to `On` in CMake

### CUDA

To enable CUDA support, set `GEC_ENABLE_CUDA` option to `On` in CMake

GEC has been tested under CUDA 11.1. Older CUDA might work as well, but it is not guaranteed.

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
    0xffffffff'ffffffff'ffffffff'ffffffff'ffffffff'ffffffff'fffffffe'fffffc2f_int);
constexpr Bigint256::LimbT MOD_P = // -MOD^-1 mod 2^64
    0xd838091d'd2253531ull;
GEC_DEF_GLOBAL(RR, Bigint256,      // 2^512 mod MOD
    0x01'000007a2'000e90a1_int);
GEC_DEF_GLOBAL(ONE_R, Bigint256,   // 2^256 mod MOD
    0x1'000003d1_int);

// define the finite field type
using Field = GEC_BASE_FIELD(Bigint256, MOD, MOD_P, RR, ONE_R);
```

Then define `Scalar` as the scalar of secp256k1.

```c++
// define parameters required by montgomery multiplication:
GEC_DEF_GLOBAL(CARD, Bigint256,       // cardinality of the elliptic curve
    0xffffffff'ffffffff'ffffffff'fffffffe'baaedce6'af48a03b'bfd25e8c'd0364141_int);
constexpr Bigint256::LimbT CARD_P =   // -CARD^-1 mod 2^64
    0x4b0dff6'65588b13full;
GEC_DEF_GLOBAL(CARD_RR, Bigint256,    // 2^512 mod CARD
    0x9d671cd5'81c69bc5'e697f5e4'5bcd07c6'741496c2'0e7cf878'896cf214'67d7d140_int);
GEC_DEF_GLOBAL(CARD_ONE_R, Bigint256, // 2^256 mod CARD
    0x1'45512319'50b75fc4'402da173'2fc9bebf_int);

// define the scalar type
using Scalar = GEC_BASE_FIELD(Bigint256, CARD, CARD_P, CARD_RR, CARD_ONE_R);
```

Finally, define `Secp256k1` as curve secp256k1.

```c++
// parameters of the elliptic curve
const Field A(0);
const Field B(0x7'00001ab7_int);

// define the curve with Jacobian coordinate
using Secp256k1_  = GEC_CURVE(gec::curve::JacobianCurve, Field, A, B);
// use the specialized implementation for curves whose A = 0 to boost performance
using Secp256k1   = GEC_CURVE_B(gec::curve::JacobianCurve, Field, B);

// define the generator
const Secp256k1 GEN(
    Field(0x9981e643'e9089f48'979f48c0'33fd129c'231e2953'29bc66db'd7362e5a'487e2097_int),
    Field(0xcf3f851f'd4a582d6'70b6b59a'ac19c136'8dfc5d5d'1f1dc64d'b15ea6d2'd3dbabe2_int),
    Field(0x1'000003d1_int),
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
    Field(0x9981e643'e9089f48'979f48c0'33fd129c'231e2953'29bc66db'd7362e5a'487e2097_int),
    Field(0xcf3f851f'd4a582d6'70b6b59a'ac19c136'8dfc5d5d'1f1dc64d'b15ea6d2'd3dbabe2_int),
    Field(0x1'000003d1_int),
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


**Notice**: When `GEC_ENABLE_CUDA` is set to `On`, in order to support CUDA,
the macro `GEC_DEF_GLOBAL(VAR, ...)` will define an additional variable, 
`d_VAR`, in constant memory with the same value.
In case of naming collision, 
you can define the variables for CPU and variables for CUDA manually.
Other macros such as `GEC_BASE_FIELD` and `GEC_CURVE` also implicitly
refer to `d_VAR` when CUDA support is enabled. 
You may need to replace them as well.

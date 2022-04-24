# GEC: Generic Elliptic Curve Library

## TODO

- [x] crtp
- [ ] divide `sequence.hpp` into smaller pieces
- [ ] bigint
  - [ ] bit operations
    - [ ] inplace bit operations
  - [ ] add
    - [x] `add_with_carry`
      - [x] with intrinsics
    - [x] `add_with_carry` inplace
      - [ ] with intrinsics
    - [x] `add`: $c = a + b \pmod{\mathrm{LIMB_BITS}}$
    - [x] `add` inplace: $a = a + b \pmod{\mathrm{LIMB_BITS}}$
    - [x] add group `add`: $c = a + b \pmod{M}$
    - [ ] add group `add` inplace: $a = a + b \pmod{M}$
  - [ ] sub
    - [x] `sub_with_borrow`
      - [ ] with intrinsics
    - [x] `sub_with_borrow` inplace
      - [ ] with intrinsics
    - [x] `sub`: $a = b - c \pmod{\mathrm{LIMB_BITS}}$
    - [x] `sub` inplace: $a = a - b \pmod{\mathrm{LIMB_BITS}}$
    - [x] add group `sub`: $a = b - c \pmod{M}$
    - [ ] add group `sub` inplace: $a = a - b \pmod{M}$
  - [ ] mul
  - [ ] inv
- [ ] use concepts

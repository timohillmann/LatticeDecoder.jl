# Example fixture provenance

These are the only generated data files needed by the public notebooks. Arrays
are stored as dense NumPy-compatible NPZ files so their dimensions and contents
can be inspected without project-specific serialization.

## `qldlc_n13_3.npz`

- Purpose: generated quantum LDLC example, including its 28 × 26 overcomplete
  BP matrix and 26 × 26 square decision matrix.
- Origin: the third retained 13-mode instance (`code_index = 3`) from the
  legacy `reduced_ldlc_gkp_n_13_3.jld2` collection. The suffix `_3` is an
  instance index, not a degree. Its 26 × 26 integer classical generator has
  four nonzeros in every row.
- Derivation: let `M` be `classical_generator`, `R` be `reduced_generator`,
  `L` be `logical_rows`, and `J = [0 I; -I 0]` be the 26-dimensional
  symplectic form in `qqpp` order. Then `decision_H = -R*J`,
  `G = inv(decision_H)`, `logical_check = inv(R)`, and
  `bp_H = vcat(-M*J, -2L*J)`. These identities are checked by the test suite.
- Contents: `classical_generator`, `reduced_generator`, `logical_rows`,
  `bp_H`, `decision_H`, `G`, and `logical_check`.
- SHA-256: `0320e486a1b81c77453ce0c908e289ad930034b388955d5271d421589205de42`.

The legacy archive contains matrices and distance diagnostics but no RNG seed
or complete stochastic-construction record. Consequently this exact instance
cannot honestly be regenerated from a seed. The retained arrays and checksum
make every public calculation reproducible, but reproducing the original
random search requires provenance that was not recorded. This limitation is
explicit rather than assigning an unverified seed.

## Bivariate-bicycle fixtures

The files under `bivariate_bicycle/` contain parity-check and logical-operator
arrays for the named `[[n,k,d]]` binary CSS instances. The notebooks use only
`hx` and `hz` to demonstrate loading and deterministic systematic enlargement.
They follow the bivariate-bicycle convention of Bravyi et al., *High-threshold
and low-overhead fault-tolerant quantum memory*, Nature 627, 778–782 (2024),
DOI `10.1038/s41586-024-07107-7`.

For both instances, work over `GF(2)[x,y]/(x^ell-1, y^m-1)`. Let `A` and `B`
be the binary circulant matrices of polynomials `a` and `b`, with
`x = P_ell ⊗ I_m` and `y = I_ell ⊗ P_m`, where `P_s` maps basis index `i` to
`i+1 mod s`. The stored checks are `hx = [A B]` and
`hz = [transpose(B) transpose(A)]`. These conventions and the parameters below
regenerate the stored check arrays without randomness.

- `30_4_5_p2.npz`: `[[30,4,5]]`, `ell=5`, `m=3`, `a=1+x`,
  `b=1+y+x^2*y^2`, binary field (`p=2`), SHA-256
  `72bc37894559a83efac2cf618f83dc469ba004a427b885e76e13d0fe6817e0bc`.
- `48_4_7_p2.npz`: `[[48,4,7]]`, `ell=8`, `m=3`, `a=1+x`,
  `b=1+y+x^3*y^2`, binary field (`p=2`), SHA-256
  `e5d2a91bf5287afc9b7189f9975c8ce6b5957d6ab99bf662c35b6c8e5c821614`.

The stored `lx` and `lz` provide one valid logical basis. Logical bases are not
unique; they are retained and checksum-protected rather than treated as a
canonical output of the polynomial construction.

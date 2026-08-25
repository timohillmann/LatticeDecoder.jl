# Test fixture provenance

`ldlc_n128_d5_H.npy` is a 128 × 128 degree-5 classical LDLC check matrix used
for decoder regression. `ldlc_n128_d5_decoder_goldens.npz` contains the expected
decoder outputs generated from that matrix by
`generate_ldlc_decoder_goldens.jl`. They are test inputs, not simulation data.

- `ldlc_n128_d5_H.npy` SHA-256:
  `e356c7573c0f6adf80bd27ed1a56fe8f2c70e8ba7eb2f7ed356e5ecc49dbb161`
- `ldlc_n128_d5_decoder_goldens.npz` SHA-256:
  `101e6b4c3fa55b82c544c84f80ad6d8e4d0b3593117e412d17bae1ae18cdf1bf`

The generator script documents the deterministic decoder configurations used
to refresh the golden outputs if an intentional algorithm change requires it.

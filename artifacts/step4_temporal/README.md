# Step 4: motion-corrected time operator

This directory records the Step 4 experiment. It does not contain a Tucker
rank decision or other Step 5 work.

## Operators

For a positive ENU displacement `(dx, dy)`, the horizontal operator uses

```text
S(dx,dy) f(x,y,z) = f(x-dx,y-dy,z).
```

The implementation is separable bilinear interpolation on the regular x-y
grid. It uses non-periodic zero extension computationally. A separate stencil
mask marks an output valid only when every nonzero interpolation contributor is
inside the domain and valid. Thus the zero extension is never classified as an
observation or physics-domain value.

`S*` is implemented independently by transposed interpolation-weight scatter.
For this exact regular-grid, symmetric linear-hat kernel and zero-extension
rule, `A_delta[i,j]=phi(i-j-delta)` and the evenness of `phi` proves
`S(delta)*=S(-delta)`, including the truncated finite domain. The implementation
does not assume that identity, and an independent test verifies it; changing
the interpolation or boundary rule would require re-verification.

For successive three-component wind fields,

```text
Tc(U)[t] = U[t+1] - S[t] U[t].
```

The exact Euclidean discrete adjoint accumulates `-S[t]* q[t]` into time `t`
and `q[t]` into time `t+1`.

## Real-case protocol

- Six Z9539/Z9532 volume pairs are ordered by the midpoint of their two nominal
  timestamps.
- No per-ray timestamps exist. No synthetic ray timestamps are generated, and
  within-volume asynchrony cannot be corrected exactly.
- Post-QC radar-indexed observation masks are rebuilt with the frozen gridding
  settings without running PyDDA.
- Storm motion is estimated independently of retrieved vortex centres using
  the maximum two-radar reflectivity composite from 2-6 km, a fixed 20 dBZ
  feature floor, and normalized-correlation search within 30 km.
- Temporal wind norms use the same strict common mask for corrected and
  uncorrected differences.
- The singular-value table is a Step 5 pre-diagnostic only; it does not select a
  Tucker rank.

## Reproduction

```powershell
python -m unittest tests.test_temporal_operator -v
python tools/verify_temporal_operator.py
python tools/run_step4_temporal_diagnostics.py
python -m unittest discover -s tests -v
python -m compileall -q src tests tools
```

`step4_summary.json` is the machine-readable main report. CSV files retain all
volume, motion, window, per-height, temporal-norm, and singular-value records.
`step4_masks_and_motion.nc` stores the recovered masks and reflectivity motion
inputs.

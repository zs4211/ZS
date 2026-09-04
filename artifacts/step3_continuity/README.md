# Step 3: anelastic mass-continuity operator

This directory records the reproducible Step 3 verification and the comparison
against the frozen Step 0 wind field. It is an experiment note, not the paper
draft.

## Discrete operator

The independent, matrix-free operator is

```text
D_rho(u, v, w) = D_x u + D_y v + rho0^-1 D_z(rho0 w).
```

Coordinates supplied in metres or kilometres are converted internally to
metres. Wind components are in m s^-1, so the residual is in s^-1. Interior
derivatives use a centred two-point secant on the coordinate arrays. The two
outer domain faces use first-order one-sided differences. No impermeability or
zero-wind boundary value is built into the operator; future boundary penalties
must remain separate.

`rho0` can be a positive finite one-dimensional array on the z levels or a
callable evaluated at z in metres. The real-case comparison uses the
dimensionless standard profile `exp(-z / 10000 m)`. Its normalization cancels
from the operator.

The exact Euclidean discrete adjoint is

```text
D_rho*(q) = [D_x^T q, D_y^T q, rho0 D_z^T(rho0^-1 q)].
```

It is implemented by stencil-transpose scatter operations, without assembling
a derivative matrix.

## Physics mask

The physics mask is not part of `D_rho`. It is derived separately from the base
validity mask and finite u/v/w values. A residual is physically defined only
where all samples required by each component's derivative stencil are valid.
External domain edges may be valid through their one-sided stencil; internal
mask boundaries are invalid and are emitted as NaN. Computational fill values
therefore cannot turn missing cells into physical zero wind.

## Reproduction

Run from the repository root:

```powershell
python -m unittest tests.test_mass_continuity -v
python tools/verify_mass_continuity_operator.py
python tools/run_step3_continuity_diagnostics.py
python -m unittest discover -s tests -v
python -m compileall -q src tests tools
```

Machine-readable results are in `operator_verification.json` and
`step3_summary.json`. The compressed NetCDF file contains the masks and old/new
residual fields used by the comparison.

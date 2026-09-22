# Data

The full descriptor matrix (511 984 environments × 252, float32, ≈ 500 MB) is **not** stored here;
it is regenerated from the LAMMPS dumps by `bench/bench_descriptor_ann.py` (first run with `--lammps_dumps`).

This directory holds a stratified sample for quick end-to-end tests, produced by `bench/make_sample.py`:

* `soap_sample.npy`        — (20 000 × 252) float32, L2-normalised SOAP vectors, 2 500 per state
* `soap_sample_labels.npy` — state label per row (`T300` … `T1300`, `melt`, `vac`)

The sample supports `--split loso` (labels included) but is far too small to reproduce the throughput
numbers of the paper; use the full set for that.

Potential file: `Cu01.eam.alloy` (Mishin et al., Phys. Rev. B 63, 224106) — download from the
NIST Interatomic Potentials Repository and place it in `md/`.

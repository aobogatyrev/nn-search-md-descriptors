# Nearest-neighbour search in the space of atomic-environment descriptors

Code, configuration files and results for the article

> **Bogatyrev O. O.** Algorithms and data structures for nearest neighbour search in molecular dynamics: approximate methods in the space of atomic-environment descriptors. *Visnyk of Cherkasy University. Series Physical and Mathematical Sciences* (2026, submitted).

This is the second article of a two-part cycle. The first part (exact methods in real space) is published as
[10.31651/2076-5851-2025-199-210](https://doi.org/10.31651/2076-5851-2025-199-210).

*Українське резюме — наприкінці файлу.*

---

## What is here

| Path | Contents |
|---|---|
| `md/in.cu_generate` | LAMMPS input deck that generates the configurations (6 crystalline temperatures, melt, single vacancy). |
| `bench/bench_descriptor_ann.py` | The benchmark: SOAP descriptors → exact ground truth → Ball-Tree / LSH / HNSW → metrics, tables, figures. |
| `bench/print_versions.py` | Prints the software environment (used for the reproducibility paragraph). |
| `bench/make_sample.py` | Builds a small stratified subset of the descriptors for quick tests (see *Data*). |
| `results/*.json`, `results/*.csv` | All measurements reported in the paper, with full benchmark conditions in `meta`. |
| `figures/*.png` | Figures 1–3 of the paper and the auxiliary plots. |
| `data/` | Descriptor sample (see *Data*); the full set is regenerated from the MD dumps. |

## The measurements at a glance

All numbers below are single-threaded (`--threads 1`) against a **double-precision, deterministically tie-broken** ground truth; see *Ground truth* for why this matters.

| Method, parameter | recall@10 | strict recall | excess distance | queries/s | × brute force |
|---|---|---|---|---|---|
| Exhaustive search (FAISS `IndexFlatL2`, float32) | 0.973 | 0.943 | 1.4·10⁻³ | 1 795 | 1 |
| Ball-Tree (scikit-learn, float64) | 1.000 | 1.000 | 0 | 51 | 0.03 |
| HNSW, efSearch = 10 | 0.982 | 0.958 | 2.1·10⁻³ | 14 601 | 8.1 |
| HNSW, efSearch = 40 | 0.9999 | 0.9995 | 1.7·10⁻⁵ | 6 835 | 3.8 |
| HNSW, efSearch = 160 | 1.000 | 1.000 | < 10⁻⁶ | 2 705 | 1.5 |
| LSH (FAISS `IndexLSH`), 2048 bits | 0.099 | 0.093 | 0.74 | 224 | 0.12 |

Dataset: 511 984 SOAP environments (D = 252) of fcc copper, base 501 984 / queries 10 000, means over 5 random splits (`results/results_repeats_t1.json`). On all 10 cores (`results_t10.json`) the HNSW/brute-force ratios rise to 27.8× (efSearch = 10) and 8.3× (efSearch = 80); recall does not depend on threads.

## Ground truth — read this before re-using the benchmark

The standard "exact" search of FAISS (`IndexFlatL2`, float32) is **not exact on this data**. It computes ‖q−x‖² as ‖q‖²+‖x‖²−2q·x, which suffers catastrophic cancellation for near-identical vectors — and molecular-dynamics environments are highly redundant. Against a float64 ground truth with direct ‖q−x‖² and deterministic (distance, index) tie-breaking, float32 exhaustive search identifies the true nearest neighbour in only 92.7 % of queries at D = 252, and in 65.9 % at D = 50; the error vanishes for genuinely novel queries (melt against a crystalline base). See `results/results_dscan_t1.json`, `results_nscan_t1.json`, `results_loso_*_t1.json` and Section 6.2 of the paper.

Consequently the benchmark

* builds the ground truth in float64 with a direct difference and deterministic tie-breaking (`exact_knn_f64`), never timed;
* times float32 exhaustive search as an ordinary method and reports *its* recall;
* reports three metrics per method: tie-aware `recall_at_k` (ann-benchmarks convention, 1 % tolerance), strict index `recall_strict`, and the excess distance `rel_dist_err`;
* pins the thread count for BLAS, FAISS and hnswlib and warms up before timing;
* reports memory as a structural estimate from the storage layout, not as measured RSS.

An exact float64 method (Ball-Tree) reads exactly 1.0 / 1.0 / 0 — this is the built-in check that the ground truth is correct.

## Reproducing

### 1. Environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python bench/print_versions.py        # should list the versions in requirements.txt
```

LAMMPS is a separate binary (`brew install lammps` on macOS; version used: 22 Jul 2025, update 4). The EAM potential `Cu01.eam.alloy` (Mishin et al., 2001) is downloaded from the NIST Interatomic Potentials Repository and placed next to the deck.

### 2. Molecular dynamics (≈ 1 h on one core)

```bash
cd md && lmp_serial -in in.cu_generate
```

Produces `dump_T{300,500,700,900,1100,1300}.lammpstrj`, `dump_melt.lammpstrj`, `dump_vac.lammpstrj` — 8 states × 16 frames (frame 0 + every 200 steps of a 3000-step production run after 2000/4000 steps of NVT equilibration; timestep 2 fs; Nosé–Hoover with 0.1 ps damping; velocity seeds 12345 / 23456 / 45678). Everything else in the paper is derived from these dumps.

### 3. Descriptors and benchmark

The first run reads the dumps, computes SOAP (DScribe; n_max = 8, l_max = 6, r_cut = 5.5 Å, σ = 0.5 Å; L2-normalised) and caches the descriptors and state labels; later runs reuse the cache.

```bash
cd bench
# base run + cache + state labels (single thread)
python bench_descriptor_ann.py --mode single  --threads 1 --lammps_dumps '../md/dump_*.lammpstrj'
# error bars, scaling with N, scaling with D
python bench_descriptor_ann.py --mode repeats --threads 1 --n_runs 5
python bench_descriptor_ann.py --mode nscan   --threads 1
python bench_descriptor_ann.py --mode dscan   --threads 1 --lammps_dumps '../md/dump_*.lammpstrj'
# leave-one-state-out (novelty test)
python bench_descriptor_ann.py --mode single --split loso --holdout vac  --threads 1
python bench_descriptor_ann.py --mode single --split loso --holdout melt --threads 1
# best practical configuration — all cores
python bench_descriptor_ann.py --mode single  --threads 0
```

Output files carry the suffix `_t<threads>` (and `_loso_<state>`), so runs never overwrite each other. Every JSON has a `meta.benchmark_conditions` block stating dtypes, threads, timing method, memory model and split.

Hardware used for the paper: Apple M1 Max (10 cores), 32 GB, macOS 26.5.1. Absolute throughputs are platform-dependent; the ratios and scaling laws are what transfer.

## Data

The full descriptor matrix (511 984 × 252 float32 ≈ 500 MB) is not stored in the repository — it is regenerated from the dumps in a few minutes. `data/` contains a stratified 20 000-environment sample (2 500 per state) with its state labels, sufficient to run the benchmark end-to-end on a laptop:

```bash
python bench/make_sample.py --cache soap_descriptors.npy --n_per_state 2500   # creates data/soap_sample.npy + labels
cd bench && python bench_descriptor_ann.py --mode single --cache ../data/soap_sample.npy --threads 1
```

## Citation

See `CITATION.cff`. Please cite the article; the code is released under the MIT licence.

---

## Українською

Код, конфігурації та результати до статті *«Алгоритми та структури даних для пошуку найближчих сусідів у молекулярній динаміці: наближені методи у просторі дескрипторів атомного оточення»* (Вісник Черкаського університету, серія «Фізико-математичні науки», 2026, подано). Перша стаття циклу (точні методи в реальному просторі): [10.31651/2076-5851-2025-199-210](https://doi.org/10.31651/2076-5851-2025-199-210).

* `md/in.cu_generate` — LAMMPS-дек генерування конфігурацій (6 температур, розплав, вакансія);
* `bench/bench_descriptor_ann.py` — бенчмарк: SOAP → float64-еталон → Ball-Tree / LSH / HNSW → метрики, таблиці, рисунки;
* `results/` — усі вимірювання статті з повним описом умов у `meta`; `figures/` — рисунки;
* `data/` — стратифікована вибірка дескрипторів для швидкої перевірки; повний набір відтворюється з дампів.

Головний методичний момент: стандартний float32-перебір FAISS **не є точним** на надлишкових даних молекулярної динаміки (найближчий сусід правильний лише у 92,7 % запитів за D = 252) — еталон побудовано у подвійній точності з детермінованим розв'язанням рівних відстаней; деталі в розділі 6.2 статті та в коментарях до `exact_knn_f64`.

Послідовність відтворення — у розділі *Reproducing* вище (команди однакові).

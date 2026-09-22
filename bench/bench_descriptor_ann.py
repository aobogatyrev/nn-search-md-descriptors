#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_descriptor_ann.py  —  ВЕРСІЯ 2 (доопрацювання за зауваженнями редакції)
=============================================================================
Бенчмарк наближеного пошуку найближчих сусідів у просторі SOAP-дескрипторів
атомного оточення ГЦК-міді (стаття 2 циклу).

ЩО ЗМІНЕНО у v2 (і чому):
  1. ЕТАЛОН тепер обчислюється у float64 з ПРЯМИМ ||q-x||² та ДЕТЕРМІНОВАНИМ
     розв'язанням рівних відстаней (за парою відстань, індекс). Раніше еталон
     давав FAISS IndexFlatL2 у float32 через ||q||²+||x||²-2q·x — для майже
     тотожних оточень це катастрофічне скорочення, і сам «еталон» був неточним;
     звідси хибна «стеля» recall=0,979 точного Ball-Tree. Тепер точний метод
     дає рівно 1,0 за побудовою.
  2. МЕТРИКИ: recall_at_k (tie-aware, ann-benchmarks), recall_strict (строга
     індексна), rel_dist_err (надлишкова відстань). Повнота хронометрованого
     float32-перебору проти float64-еталона теж звітується — чесно.
  3. ПОТОКИ задано явно (--threads): BLAS/FAISS/hnswlib закріплено; типово 1 —
     чесне однопотокове порівняння. Warm-up перед кожним таймінгом.
  4. SPLIT: --split loso --holdout vac|melt — leave-one-state-out: база лише
     кристалічні стани, запити з утримуваного стану (тест «новизни»).
  5. META results.json містить блок benchmark_conditions (dtype, потоки, спосіб
     таймінгу, що пам'ять — структурна оцінка, а не вимір).
  6. PARAMS узгоджено з фактичним LAMMPS-протоколом (melt_T=2000, 16 знімків,
     без поверхні).

ЗАПУСК (усе на наявному кеші soap_descriptors.npy; МД НЕ повторюється):
  # 0) мітки станів для кешу (один раз; SOAP не рахується, лише читає дампи):
  python bench_descriptor_ann.py --mode single --cache soap_descriptors.npy \
         --lammps_dumps 'dump_*.lammpstrj' --threads 1
  # 1) базовий прогін, 1 потік (головні числа статті):
  python bench_descriptor_ann.py --mode single  --threads 1
  # 2) повтори ×5 (похибки):
  python bench_descriptor_ann.py --mode repeats --threads 1 --n_runs 5
  # 3) масштабованість за N та за D:
  python bench_descriptor_ann.py --mode nscan   --threads 1
  python bench_descriptor_ann.py --mode dscan   --threads 1 --lammps_dumps 'dump_*.lammpstrj'
  # 4) leave-one-state-out (тест новизни):
  python bench_descriptor_ann.py --mode single --split loso --holdout vac  --threads 1
  python bench_descriptor_ann.py --mode single --split loso --holdout melt --threads 1
  # 5) «найкраща практична конфігурація» — усі ядра (для порівняння з п.1):
  python bench_descriptor_ann.py --mode single  --threads 0

Вихідні файли отримують суфікс _t<потоки> (та _loso_<стан>), тож варіанти не
перезаписують один одного. Надіслати назад: усі results*.json / *.csv / *.png.

Параметри узгоджено в §4.3 та §5 статті.
"""

import os
import sys
import json
import time
import argparse

# --------------------------------------------------------------------------
#  КОНТРОЛЬ ПОТОКІВ — має бути ДО імпорту numpy/BLAS.
#  --threads N : потоки для BLAS/FAISS/hnswlib. 1 = чесне однопотокове
#  порівняння (типово); 0 = усі ядра ("найкраща практична конфігурація").
# --------------------------------------------------------------------------
def _early_threads():
    n = 1
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            try: n = int(sys.argv[i + 1])
            except ValueError: pass
        elif a.startswith("--threads="):
            try: n = int(a.split("=", 1)[1])
            except ValueError: pass
    if n <= 0:
        n = os.cpu_count() or 1
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    return n
THREADS = _early_threads()

import numpy as np

# --------------------------------------------------------------------------
#  ПАРАМЕТРИ (узгоджені в §4.3 та §5)
# --------------------------------------------------------------------------
PARAMS = dict(
    # --- фізична система / генерування (ASE) ---
    element="Cu",
    a0=3.615,                  # стала ґратки ГЦК-міді, Å
    supercell=(10, 10, 10),    # ~4000 атомів
    temperatures=[300, 500, 700, 900, 1100, 1300],  # K, кристал
    melt_T=2000,               # K, розплав (LAMMPS-дек; вище Tпл ≈ 1357 K)
    n_snapshots=16,            # знімків на стан (LAMMPS: кадр 0 + 15 через 200 кроків)
    md_timestep_fs=2.0,        # крок МД, фс
    md_friction=0.02,          # тертя Ланжевена, 1/фс
    equilib_steps=2000,        # кроків врівноваження перед відбором
    sample_every=200,          # кроків між знімками (декореляція)
    add_vacancy=True,          # конфігурації з вакансією
    add_surface=False,         # поверхню НЕ генерували (лише вакансія + розплав)

    # --- SOAP (DScribe), §4.3 ---
    r_cut=5.5, n_max=8, l_max=6, sigma=0.5,

    # --- бенчмарк пошуку, §5 ---
    k=10,                      # основне k (recall@10); також рахуємо recall@1
    query_size=10000,          # розмір запитової вибірки (решта — база)
    metric="euclidean",        # основна метрика (§2.2)
    # HNSW (hnswlib)
    hnsw_M=16, hnsw_efC=200,
    hnsw_efSearch=[10, 20, 40, 80, 160, 320, 640],
    # LSH (FAISS): кількість бітів хеш-коду (кратно D зручно, але не обов'язково)
    lsh_nbits=[64, 128, 256, 512, 1024, 2048],

    # --- інше ---
    potential="Cu_mishin.eam.alloy",  # setfl-файл потенціалу Мішина (NIST)
    seed=42,
    n_repeats=3,               # повтори таймінгу (медіана)
    n_jobs=1,                  # потоки для SOAP (DScribe); -1 = усі ядра
    out_prefix="results",
    threads=THREADS,           # потоки (див. _early_threads)
    split="random",            # random | loso (leave-one-state-out)
    holdout=None,              # для loso: 'vac' | 'melt'
    gt_chunk=512,              # розмір блоку запитів для float64-еталона
)


# ==========================================================================
#  1–2. ГЕНЕРУВАННЯ КОНФІГУРАЦІЙ (ASE) + SOAP (DScribe)
# ==========================================================================
def generate_descriptors(P):
    """Повертає X (N, D) float32 — нормовані SOAP-вектори всіх оточень."""
    from ase.build import bulk, fcc111
    from ase.calculators.eam import EAM
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units
    from dscribe.descriptors import SOAP

    rng = np.random.default_rng(P["seed"])
    calc = EAM(potential=P["potential"])

    def run_md(atoms, T, n_snap, label=""):
        """Врівноваження + відбір n_snap знімків за температури T (K)."""
        t0 = time.perf_counter()
        print(f"  [МД] {label}: врівноваження "
              f"({P['equilib_steps']} кроків)…", flush=True)
        atoms = atoms.copy()
        atoms.calc = calc
        MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=rng)
        dyn = Langevin(atoms, P["md_timestep_fs"] * units.fs,
                       temperature_K=T, friction=P["md_friction"])
        dyn.run(P["equilib_steps"])
        snaps = []
        for i in range(n_snap):
            dyn.run(P["sample_every"])
            snaps.append(atoms.copy())
            if (i + 1) % 5 == 0 or (i + 1) == n_snap:
                print(f"      {label}: знімок {i + 1}/{n_snap} "
                      f"({time.perf_counter() - t0:.0f} с)", flush=True)
        print(f"  [МД] {label}: готово, +{len(snaps)} знімків "
              f"({time.perf_counter() - t0:.1f} с)", flush=True)
        return snaps

    configs = []
    n_states = (len(P["temperatures"]) + 1
                + int(P["add_vacancy"]) + int(P["add_surface"]))
    print(f"[gen] станів: {n_states}; знімків на стан: {P['n_snapshots']}; "
          f"суперкомірка {P['supercell']}", flush=True)
    si = 0

    # --- кристал за діапазону температур ---
    base = bulk(P["element"], "fcc", a=P["a0"], cubic=True).repeat(P["supercell"])
    for T in P["temperatures"]:
        si += 1
        configs += run_md(base, T, P["n_snapshots"],
                          label=f"[{si}/{n_states}] кристал {T} K")

    # --- розплав ---
    si += 1
    configs += run_md(base, P["melt_T"], P["n_snapshots"],
                      label=f"[{si}/{n_states}] розплав {P['melt_T']} K")

    # --- вакансія ---
    if P["add_vacancy"]:
        si += 1
        vac = base.copy()
        del vac[rng.integers(len(vac))]
        configs += run_md(vac, P["temperatures"][0], P["n_snapshots"],
                          label=f"[{si}/{n_states}] вакансія {P['temperatures'][0]} K")

    # --- поверхня (111) ---
    if P["add_surface"]:
        si += 1
        slab = fcc111(P["element"], size=(8, 8, 8), a=P["a0"], vacuum=10.0,
                      periodic=True)
        configs += run_md(slab, P["temperatures"][0], P["n_snapshots"],
                          label=f"[{si}/{n_states}] поверхня(111) {P['temperatures'][0]} K")

    total_atoms = sum(len(c) for c in configs)
    print(f"[gen] зібрано {len(configs)} конфігурацій, "
          f"~{total_atoms} оточень", flush=True)

    # --- SOAP по всіх оточеннях ---
    print(f"[soap] обчислення SOAP (n_jobs={P.get('n_jobs', 1)})… "
          f"може зайняти кілька хвилин", flush=True)
    t_soap = time.perf_counter()
    soap = SOAP(species=[P["element"]], periodic=True,
                r_cut=P["r_cut"], n_max=P["n_max"], l_max=P["l_max"],
                sigma=P["sigma"])
    feats = soap.create(configs, n_jobs=P.get("n_jobs", 1))  # список (n_atoms_i, D)
    if isinstance(feats, np.ndarray):
        X = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
    else:
        X = np.vstack(feats).astype(np.float32)
    print(f"[soap] готово ({time.perf_counter() - t_soap:.1f} с), "
          f"X={X.shape}", flush=True)

    # L2-нормування (узгоджено з косинусною мірою / SOAP-ядром, §2.2)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X /= norms
    return X


def _state_label(path):
    """Мітка стану з імені дампа: dump_T300.lammpstrj -> 'T300', dump_melt -> 'melt'."""
    base = os.path.basename(path)
    base = base.split(".")[0]
    return base[5:] if base.startswith("dump_") else base


def labels_from_dumps(dump_glob, element="Cu"):
    """Мітки стану для КОЖНОГО оточення у тому ж порядку, що й дескриптори
    (посортовані файли -> кадри -> атоми). SOAP НЕ рахується — швидко;
    дозволяє відновити мітки для вже наявного кешу дескрипторів."""
    import glob
    from ase.io import read
    files = sorted(glob.glob(dump_glob))
    if not files:
        raise FileNotFoundError(f"не знайдено дампів за шаблоном: {dump_glob}")
    labels = []
    for fp in files:
        frames = read(fp, index=":", format="lammps-dump-text")
        if not isinstance(frames, list):
            frames = [frames]
        lab = _state_label(fp)
        for at in frames:
            labels += [lab] * len(at)
    return np.array(labels)


def generate_descriptors_from_dumps(P, dump_glob):
    """Читає знімки з LAMMPS-дампів і обчислює SOAP (МД виконано в LAMMPS).
    Повертає (X, labels)."""
    import glob
    from ase.io import read
    from dscribe.descriptors import SOAP

    files = sorted(glob.glob(dump_glob))
    if not files:
        raise FileNotFoundError(f"не знайдено дампів за шаблоном: {dump_glob}")
    configs, labels = [], []
    for fp in files:
        frames = read(fp, index=":", format="lammps-dump-text")
        if not isinstance(frames, list):
            frames = [frames]
        lab = _state_label(fp)
        for at in frames:
            at.set_chemical_symbols([P["element"]] * len(at))  # type 1 -> Cu
            at.set_pbc(True)
            labels += [lab] * len(at)
        configs += frames
        print(f"[gen] {fp}: +{len(frames)} знімків (стан {lab})", flush=True)

    total_atoms = sum(len(c) for c in configs)
    print(f"[gen] зібрано {len(configs)} конфігурацій, ~{total_atoms} оточень",
          flush=True)
    print(f"[soap] обчислення SOAP (n_jobs={P.get('n_jobs', 1)})…", flush=True)
    t_soap = time.perf_counter()
    soap = SOAP(species=[P["element"]], periodic=True,
                r_cut=P["r_cut"], n_max=P["n_max"], l_max=P["l_max"],
                sigma=P["sigma"])
    feats = soap.create(configs, n_jobs=P.get("n_jobs", 1))
    if isinstance(feats, np.ndarray):
        X = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
    else:
        X = np.vstack(feats).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X /= norms
    print(f"[soap] готово ({time.perf_counter() - t_soap:.1f} с), X={X.shape}",
          flush=True)
    return X, np.array(labels)


# ==========================================================================
#  4. ЕТАЛОН (ground truth) — точний k-NN повним перебором
# ==========================================================================
def exact_knn_f64(db, queries, k, chunk=512, margin=64):
    """ЕТАЛОН: точний k-NN у float64 з ДЕТЕРМІНОВАНИМ розв'язанням рівних
    відстаней — упорядкування за парою (відстань, індекс). Не хронометрується.

    Чому не FAISS IndexFlatL2: він працює у float32 і рахує ||q-x||² як
    ||q||²+||x||²-2q·x, що для майже тотожних векторів (дублікати оточень)
    дає катастрофічне скорочення; такий «еталон» сам містить похибку, і точний
    float64-метод (Ball-Tree) «не збігається» з ним на ~2% (артефакт, який у
    попередній версії помилково трактовано як «стелю» точних методів).

    Алгоритм: (1) кандидати через float64-матмул + argpartition з запасом
    margin; (2) для кандидатів — ПРЯМИЙ ||q-x||² без скорочення; (3) стабільне
    сортування за (d², індекс); (4) контроль, що межа k-го сусіда не впирається
    у запас (інакше запас розширюється)."""
    dbd = np.ascontiguousarray(db, dtype=np.float64)
    qd = np.ascontiguousarray(queries, dtype=np.float64)
    N = len(dbd)
    db_sq = np.einsum("ij,ij->i", dbd, dbd)
    I = np.empty((len(qd), k), dtype=np.int64)
    Dk = np.empty((len(qd), k), dtype=np.float64)
    for s in range(0, len(qd), chunk):
        q = qd[s:s + chunk]
        d2 = (np.einsum("ij,ij->i", q, q)[:, None] + db_sq[None, :]
              - 2.0 * (q @ dbd.T))
        np.maximum(d2, 0.0, out=d2)
        kk = min(k + margin, N)
        while True:
            cand = np.argpartition(d2, kk - 1, axis=1)[:, :kk]      # (m, kk)
            # прямі точні відстані до кандидатів (без скорочення)
            diff = q[:, None, :] - dbd[cand]                        # (m, kk, D)
            d2c = np.einsum("mkd,mkd->mk", diff, diff)
            # стабільний порядок: за d², а при рівності — за меншим індексом
            order = np.lexsort((cand, d2c), axis=1)                 # (m, kk)
            top = np.take_along_axis(cand, order[:, :k], axis=1)
            topd = np.take_along_axis(d2c, order[:, :k], axis=1)
            # контроль межі: k-й ≤ будь-який поза top? (запас не вичерпано)
            if kk >= N:
                break
            kth = topd[:, -1:]
            rest = np.take_along_axis(d2c, order[:, k:], axis=1)
            if np.all(rest.min(axis=1, keepdims=True) > kth):
                break
            kk = min(kk * 4, N)                                    # розширити запас
        I[s:s + chunk] = top
        Dk[s:s + chunk] = np.sqrt(topd)
    return I, Dk


def exact_search_timed(db, queries, k, P):
    """ХРОНОМЕТРОВАНИЙ практичний точний пошук (повний перебір, float32 FAISS
    IndexFlatL2; запасний варіант — sklearn brute). Повертає індекси та qps.
    Його власну повноту проти float64-еталона теж вимірюємо — чесно показує,
    що float32-перебір не є ідеально точним для дублікатів."""
    n_rep = P["n_repeats"]
    try:
        import faiss
        faiss.omp_set_num_threads(P["threads"])
        index = faiss.IndexFlatL2(db.shape[1])
        index.add(db)
        index.search(queries[:min(256, len(queries))], k)           # warm-up
        qps = []
        for _ in range(n_rep):
            t0 = time.perf_counter()
            _, I = index.search(queries, k)
            qps.append(queries.shape[0] / (time.perf_counter() - t0))
        return I, float(np.median(qps)), "faiss.IndexFlatL2 (float32)"
    except Exception:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
        nn.fit(db)
        nn.kneighbors(queries[:min(256, len(queries))], return_distance=False)
        qps = []
        for _ in range(n_rep):
            t0 = time.perf_counter()
            I = nn.kneighbors(queries, return_distance=False)
            qps.append(queries.shape[0] / (time.perf_counter() - t0))
        return I, float(np.median(qps)), "sklearn brute (float32)"


def _dists_f64(queries, db, I):
    """Евклідові відстані (float64) від запитів до повернутих сусідів I (n_q, k)."""
    q = queries.astype(np.float64)
    nbr = db[np.asarray(I)].astype(np.float64)          # (n_q, k, D)
    return np.sqrt(((q[:, None, :] - nbr) ** 2).sum(axis=2))


def recall_strict(I, true_I, kk):
    """Строга індексна повнота: |I ∩ I_true| / k проти ДЕТЕРМІНОВАНОГО еталона.
    Для точного методу має дорівнювати 1.0 (з точністю до справжніх, побітово
    рівних відстаней, розв'язаних інакше)."""
    I = np.asarray(I)[:, :kk]; T = np.asarray(true_I)[:, :kk]
    hits = 0
    for a, b in zip(I, T):
        hits += len(np.intersect1d(a, b, assume_unique=False))
    return hits / (T.shape[0] * kk)


def rel_dist_err(I, queries, db, true_D, kk, eps=1e-12):
    """Надлишкова відстань (excess distance): середнє по запитах відношення
    (середня істинна відстань до повернутих k) / (середня відстань до справжніх k) − 1.
    Для точного методу = 0; показує, чи повертає наближений метод інші, але
    практично рівновіддалені дублікати (тоді ≈0 попри recall<1)."""
    approx_D = _dists_f64(queries, db, np.asarray(I)[:, :kk])
    ma = approx_D.mean(axis=1); mt = np.asarray(true_D)[:, :kk].mean(axis=1)
    ratio = np.where(mt > eps, ma / np.maximum(mt, eps), 1.0)
    return float(np.mean(ratio) - 1.0)


def all_metrics(I, queries, db, true_I, true_D, k):
    """Повний набір метрик одного методу."""
    return dict(
        recall_at_k=recall_from_indices(I, queries, db, true_D, k),   # tie-aware (rtol=1%)
        recall_strict=recall_strict(I, true_I, k),                    # строга індексна
        recall_at_1=recall_from_indices(I, queries, db, true_D, 1),
        rel_dist_err=rel_dist_err(I, queries, db, true_D, k),         # excess distance
    )


def recall_from_indices(I, queries, db, true_D, kk, rtol=1e-2, atol=1e-6):
    """Повнота, стійка до рівновіддалених сусідів і НЕЗАЛЕЖНА від внутрішньої
    метрики методу: відстані повернутих сусідів перераховуються у вихідному
    просторі (float64) і порівнюються з k-ю ІСТИННОЮ відстанню. Невеликий
    відносний допуск rtol (за домовленістю ann-benchmarks [15]) зараховує сусіда
    в межах 1% від k-ї істинної відстані — це усуває float-артефакти впорядкування
    майже-рівновіддалених майже-дублікатів, тож точний метод дає рівно 1.0, а
    для LSH враховуються справжні евклідові відстані, а не геммінгові коди FAISS."""
    approx_D = _dists_f64(queries, db, np.asarray(I)[:, :kk])
    thr = np.asarray(true_D)[:, kk - 1] * (1.0 + rtol) + atol
    hits = int((approx_D <= thr[:, None]).sum())
    return hits / (true_D.shape[0] * kk)


def _est_mem_mb(method, N, D, P, nbits=None):
    """Детермінована оцінка пам'яті індексу за моделлю зберігання (МБ).
    Надійніша за різницю RSS процесу, яка зашумлена звільненням інших структур."""
    if method == "balltree":
        return N * D * 8 / 1e6                       # дані (float64) — домінують
    if method == "hnsw":
        M = P["hnsw_M"]
        return N * (D * 4 + 2 * M * 4) / 1e6         # вектори(float32) + зв'язки графа
    if method == "lsh":
        return (N * nbits / 8 + D * nbits * 4) / 1e6  # бінарні коди + матриця проєкцій
    return None


# ==========================================================================
#  5. НАБЛИЖЕНІ МЕТОДИ
# ==========================================================================
def bench_balltree(db, queries, true_I, true_D, P):
    """scikit-learn Ball-Tree — точний (float64, однопотоковий); проти
    детермінованого float64-еталона має давати recall = 1."""
    from sklearn.neighbors import NearestNeighbors
    k = P["k"]
    t0 = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", metric="euclidean",
                          leaf_size=P.get("balltree_leaf_size", 40))
    nn.fit(db)
    build_t = time.perf_counter() - t0
    mem = _est_mem_mb("balltree", len(db), db.shape[1], P)
    nn.kneighbors(queries[:min(256, len(queries))], return_distance=False)  # warm-up
    qps = []
    for _ in range(P["n_repeats"]):
        t0 = time.perf_counter()
        I = nn.kneighbors(queries, return_distance=False)
        qps.append(queries.shape[0] / (time.perf_counter() - t0))
    return [dict(method="BallTree(sklearn)", param="exact", dtype="float64",
                 **all_metrics(I, queries, db, true_I, true_D, k),
                 qps=float(np.median(qps)), build_s=build_t, mem_mb=mem,
                 threads=1)]


def bench_hnsw(db, queries, true_I, true_D, P):
    """hnswlib — крива recall–throughput за скану efSearch (float32; потоки задано явно)."""
    import hnswlib
    k = P["k"]; N, D = db.shape
    p = hnswlib.Index(space="l2", dim=D)
    p.set_num_threads(P["threads"])
    t0 = time.perf_counter()
    p.init_index(max_elements=N, ef_construction=P["hnsw_efC"], M=P["hnsw_M"])
    p.add_items(db, np.arange(N))
    build_t = time.perf_counter() - t0
    mem = _est_mem_mb("hnsw", N, D, P)
    rows = []
    for ef in P["hnsw_efSearch"]:
        p.set_ef(max(ef, k))
        p.knn_query(queries[:min(256, len(queries))], k=k)                 # warm-up
        qps = []
        for _ in range(P["n_repeats"]):
            t0 = time.perf_counter()
            I, _ = p.knn_query(queries, k=k)
            qps.append(queries.shape[0] / (time.perf_counter() - t0))
        rows.append(dict(method="HNSW(hnswlib)", param=f"efSearch={ef}", dtype="float32",
                         **all_metrics(I, queries, db, true_I, true_D, k),
                         qps=float(np.median(qps)), build_s=build_t, mem_mb=mem,
                         threads=P["threads"]))
    return rows


def bench_lsh(db, queries, true_I, true_D, P):
    """FAISS IndexLSH (бінарний LSH на знаках випадкових проєкцій; float32) —
    крива recall–throughput за скану nbits. Потоки задано явно."""
    import faiss
    faiss.omp_set_num_threads(P["threads"])
    k = P["k"]; D = db.shape[1]
    rows = []
    for nbits in P["lsh_nbits"]:
        index = faiss.IndexLSH(D, int(nbits))
        t0 = time.perf_counter()
        index.train(db)
        index.add(db)
        build_t = time.perf_counter() - t0
        mem = _est_mem_mb("lsh", db.shape[0], D, P, nbits=nbits)
        index.search(queries[:min(256, len(queries))], k)                  # warm-up
        qps = []
        for _ in range(P["n_repeats"]):
            t0 = time.perf_counter()
            _, I = index.search(queries, k)        # I — індекси; відстані FAISS тут геммінгові
            qps.append(queries.shape[0] / (time.perf_counter() - t0))
        rows.append(dict(method="LSH(FAISS)", param=f"nbits={nbits}", dtype="float32",
                         **all_metrics(I, queries, db, true_I, true_D, k),
                         qps=float(np.median(qps)), build_s=build_t, mem_mb=mem,
                         threads=P["threads"]))
    return rows


# ==========================================================================
#  ВІЗУАЛІЗАЦІЯ + ЗБЕРЕЖЕННЯ
# ==========================================================================
def save_outputs(rows, meta, P):
    import csv
    with open(f"{P['out_prefix']}.json", "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, results=rows), f, ensure_ascii=False, indent=2)

    cols = ["method", "param", "dtype", "threads", "recall_at_k", "recall_strict",
            "recall_at_1", "rel_dist_err", "qps", "build_s", "mem_mb"]
    with open(f"{P['out_prefix']}_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for method in sorted(set(r["method"] for r in rows)):
            pts = sorted([r for r in rows if r["method"] == method],
                         key=lambda r: r["recall_at_k"])
            xs = [r["recall_at_k"] for r in pts]
            ys = [r["qps"] for r in pts]
            style = "o-" if len(pts) > 1 else "s"
            ax.plot(xs, ys, style, label=method, markersize=7)
        ax.set_yscale("log")
        ax.set_xlabel(f"Повнота recall@{P['k']}")
        ax.set_ylabel("Пропускна здатність, запитів/с (log)")
        ax.set_title("Крива компромісу «повнота–швидкодія»")
        ax.grid(True, which="both", alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(f"{P['out_prefix']}_figure.png", dpi=150)
        print(f"[ok] рисунок: {P['out_prefix']}_figure.png")
    except Exception as e:
        print(f"[warn] рисунок не збудовано: {e}")


# ==========================================================================
#  СПІЛЬНЕ ЯДРО + РЕЖИМИ РОЗГОРТОК (repeats / nscan / dscan)
# ==========================================================================
def make_split(X, P, seed, labels=None):
    """Поділ на базу пошуку та запитову вибірку.
    split='random' — випадкове розбиття (алгоритмічний бенчмарк; «легкий» тест,
                     бо в базі лишаються майже тотожні оточення того ж знімка);
    split='loso'   — leave-one-state-out: база = ЛИШЕ кристалічні стани (T*),
                     запити = вибірка з утримуваного стану (holdout: 'vac'|'melt').
                     Імітує заявлене застосування — пошук для НОВИХ оточень."""
    rng = np.random.default_rng(seed)
    if P.get("split") == "loso":
        if labels is None:
            raise ValueError("split=loso потребує міток станів (labels)")
        hold = P["holdout"]
        labels = np.asarray(labels)
        crystal = np.array([str(l).startswith("T") for l in labels])
        qpool = np.flatnonzero(labels == hold)
        if len(qpool) == 0:
            raise ValueError(f"немає оточень зі станом holdout={hold!r}; "
                             f"наявні: {sorted(set(labels.tolist()))}")
        nq = min(P["query_size"], len(qpool))
        qidx = rng.choice(qpool, size=nq, replace=False)
        return (np.ascontiguousarray(X[crystal]), np.ascontiguousarray(X[qidx]))
    n = len(X)
    perm = rng.permutation(n)
    nq = min(P["query_size"], n // 5)
    db = np.ascontiguousarray(X[perm[nq:]])
    queries = np.ascontiguousarray(X[perm[:nq]])
    return db, queries


def run_benchmark(db, queries, P, verbose=False):
    """Один прогін: float64-еталон + хронометрований перебір + три методи."""
    k = P["k"]
    if verbose: print("[info] float64-еталон (детермінований tie-break)…", flush=True)
    t0 = time.perf_counter()
    true_I, true_D = exact_knn_f64(db, queries, k, chunk=P["gt_chunk"])
    if verbose: print(f"[info] еталон готовий ({time.perf_counter() - t0:.0f} с)", flush=True)
    if verbose: print("[info] хронометрований перебір…", flush=True)
    I_br, exact_qps, brute_impl = exact_search_timed(db, queries, k, P)
    rows = []
    for fn, name in [(bench_balltree, "BallTree"),
                     (bench_lsh, "LSH"), (bench_hnsw, "HNSW")]:
        try:
            if verbose: print(f"[info] {name}…", flush=True)
            rows += fn(db, queries, true_I, true_D, P)
        except Exception as e:
            print(f"[warn] {name} пропущено: {e}")
    # перебір: хронометрований float32-пошук, повнота — проти float64-еталона
    rows.append(dict(method="Exact(brute)", param=brute_impl, dtype="float32",
                     **all_metrics(I_br, queries, db, true_I, true_D, k),
                     qps=float(exact_qps), build_s=0.0, mem_mb=None,
                     threads=P["threads"]))
    return rows


def mode_repeats(X, P, n_runs=5, labels=None):
    """R прогонів з різними seed розбиття -> середнє ± стд за (метод, параметр)."""
    acc = {}
    for s in range(n_runs):
        print(f"[repeats] прогін {s + 1}/{n_runs}", flush=True)
        db, queries = make_split(X, P, seed=1000 + s, labels=labels)
        for r in run_benchmark(db, queries, P):
            key = (r["method"], r["param"])
            d = acc.setdefault(key, {f: [] for f in
                              ["recall_at_k", "recall_strict", "recall_at_1",
                               "rel_dist_err", "qps", "build_s", "mem_mb"]})
            for f in ["recall_at_k", "recall_strict", "recall_at_1", "rel_dist_err",
                      "qps", "build_s"]:
                d[f].append(r[f])
            if r.get("mem_mb") is not None:
                d["mem_mb"].append(r["mem_mb"])
    out = []
    for (method, param), v in acc.items():
        rec = dict(method=method, param=param, n_runs=len(v["qps"]))
        for f in ["recall_at_k", "recall_strict", "recall_at_1", "rel_dist_err",
                  "qps", "build_s"]:
            a = np.array(v[f], float)
            rec[f + "_mean"] = float(a.mean())
            rec[f + "_std"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        rec["mem_mb_mean"] = float(np.mean(v["mem_mb"])) if v["mem_mb"] else None
        out.append(rec)
    return out


def mode_nscan(X, P, sizes):
    """Прогони на підвибірках різного обсягу N (масштабованість)."""
    rng = np.random.default_rng(P["seed"])
    out = []
    for n in sizes:
        if n > len(X):
            print(f"[nscan] N={n} > доступного ({len(X)}) — пропущено")
            continue
        idx = rng.choice(len(X), size=n, replace=False)
        db, queries = make_split(X[idx], P, seed=P["seed"])
        for r in run_benchmark(db, queries, P):
            r["N"] = int(n); r["n_db"] = int(len(db)); r["n_queries"] = int(len(queries))
            out.append(r)
        print(f"[nscan] N={n} готово", flush=True)
    return out


def mode_dscan(P, dump_glob, configs):
    """Прогони за різної розмірності D (через n_max, l_max) — ядро статті."""
    import os
    out = []
    for (nm, lm) in configs:
        Pc = dict(P); Pc["n_max"] = nm; Pc["l_max"] = lm
        cache = f"soap_n{nm}_l{lm}.npy"
        if os.path.exists(cache):
            print(f"[dscan] кеш {cache}")
            X = np.load(cache)
        else:
            if not dump_glob:
                print(f"[dscan] немає кешу {cache} і не задано --lammps_dumps — пропуск")
                continue
            X, _ = generate_descriptors_from_dumps(Pc, dump_glob)
            np.save(cache, X)
        db, queries = make_split(X, Pc, seed=P["seed"])
        D = int(X.shape[1])
        for r in run_benchmark(db, queries, Pc):
            r["D"] = D; r["n_max"] = nm; r["l_max"] = lm
            out.append(r)
        print(f"[dscan] n_max={nm} l_max={lm} D={D} готово", flush=True)
    return out


def _save_json_csv(payload, prefix, csv_cols):
    import csv
    with open(f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    rows = payload["results"]
    with open(f"{prefix}_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[ok] збережено {prefix}.json та {prefix}_table.csv")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_repeats(rows, P):
    try:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(7, 5))
        for method in sorted(set(r["method"] for r in rows)):
            pts = sorted([r for r in rows if r["method"] == method],
                         key=lambda r: r["recall_at_k_mean"])
            x = [r["recall_at_k_mean"] for r in pts]
            y = [r["qps_mean"] for r in pts]
            xe = [r["recall_at_k_std"] for r in pts]
            ye = [r["qps_std"] for r in pts]
            style = "o-" if len(pts) > 1 else "s"
            ax.errorbar(x, y, xerr=xe, yerr=ye, fmt=style, capsize=3,
                        label=method, markersize=6)
        ax.set_yscale("log")
        ax.set_xlabel(f"Повнота recall@{P['k']}")
        ax.set_ylabel("Пропускна здатність, запитів/с (log)")
        ax.set_title(f"Крива компромісу «повнота–швидкодія» "
                     f"(середнє ± стд, {rows[0]['n_runs']} прогонів)")
        ax.grid(True, which="both", alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(f"{P.get('out_prefix','results_repeats')}_figure.png", dpi=150)
        print(f"[ok] рисунок: {P.get('out_prefix','results_repeats')}_figure.png")
    except Exception as e:
        print(f"[warn] рисунок repeats: {e}")


def plot_scan(rows, xkey, xlabel, fname, P, ef_for_recall="efSearch=10",
              hnsw_ef_for_qps="efSearch=80"):
    """Універсальний рисунок для nscan/dscan: qps vs x (log-log) + recall vs x."""
    try:
        plt = _plt()
        xs_all = sorted(set(r[xkey] for r in rows))
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
        # ліва панель: пропускна здатність
        series = {
            "BallTree(sklearn)": lambda r: r["method"] == "BallTree(sklearn)",
            "HNSW " + hnsw_ef_for_qps: lambda r: r["method"] == "HNSW(hnswlib)"
                                        and r["param"] == hnsw_ef_for_qps,
            "Exact(brute)": lambda r: r["method"] == "Exact(brute)",
        }
        for lbl, sel in series.items():
            pts = sorted([r for r in rows if sel(r)], key=lambda r: r[xkey])
            if pts:
                a1.plot([p[xkey] for p in pts], [p["qps"] for p in pts],
                        "o-", label=lbl)
        a1.set_xscale("log"); a1.set_yscale("log")
        a1.set_xlabel(xlabel); a1.set_ylabel("запитів/с (log)")
        a1.set_title("Пропускна здатність"); a1.grid(True, which="both", alpha=0.3)
        a1.legend()
        # права панель: повнота HNSW@ef та BallTree
        for lbl, sel in {
            "HNSW " + ef_for_recall: lambda r: r["method"] == "HNSW(hnswlib)"
                                      and r["param"] == ef_for_recall,
            "BallTree(sklearn)": lambda r: r["method"] == "BallTree(sklearn)",
        }.items():
            pts = sorted([r for r in rows if sel(r)], key=lambda r: r[xkey])
            if pts:
                a2.plot([p[xkey] for p in pts], [p["recall_at_k"] for p in pts],
                        "o-", label=lbl)
        a2.set_xscale("log")
        a2.set_xlabel(xlabel); a2.set_ylabel(f"recall@{P['k']}")
        a2.set_title("Повнота"); a2.grid(True, which="both", alpha=0.3); a2.legend()
        fig.tight_layout(); fig.savefig(fname, dpi=150)
        print(f"[ok] рисунок: {fname}")
    except Exception as e:
        print(f"[warn] рисунок scan: {e}")


def _disclosure(P):
    """Умови бенчмарку для розділу «Відтворюваність» (потрапляють у meta)."""
    return dict(benchmark_conditions=dict(
        ground_truth="exact k-NN, float64, direct ||q-x||^2, deterministic tie-break "
                     "by (distance, index); NOT timed",
        timed_exact_search="FAISS IndexFlatL2 float32 (fallback: sklearn brute); "
                           "its own recall vs float64 ground truth is reported",
        dtype=dict(balltree="float64 (scikit-learn)", hnsw="float32 (hnswlib)",
                   lsh="float32 (FAISS IndexLSH, binary sign-of-projection)",
                   brute="float32"),
        threads=P["threads"],
        threads_note="BLAS/FAISS/hnswlib pinned via env + explicit API; "
                     "scikit-learn Ball-Tree is single-threaded by design",
        batch="all queries in one batch call per method",
        timing="time.perf_counter over the whole query batch; median of "
               f"{P['n_repeats']} repeats; warm-up batch of 256 queries before timing; "
               "excludes index build and any format conversion",
        memory="STRUCTURAL ESTIMATE from data layout (not measured RSS): "
               "BallTree=N*D*8; HNSW=N*(D*4+2*M*4); LSH=N*nbits/8+D*nbits*4",
        recall=dict(recall_at_k="tie-aware: neighbour counts if true dist <= "
                                "d_k*(1+1e-2)+1e-6 (ann-benchmarks convention)",
                    recall_strict="|I ∩ I_true|/k vs deterministic ground truth",
                    rel_dist_err="excess distance: mean(returned dists)/mean(true dists) - 1"),
        split=P["split"], holdout=P["holdout"],
        balltree_leaf_size=P.get("balltree_leaf_size", 40),
    ))


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--potential", default=PARAMS["potential"],
                    help="шлях до setfl-файлу потенціалу Мішина")
    ap.add_argument("--cache", default="soap_descriptors.npy",
                    help="кеш SOAP-векторів (.npy)")
    ap.add_argument("--n_snapshots", type=int, default=PARAMS["n_snapshots"],
                    help="знімків на стан (менше = швидше; напр. 15)")
    ap.add_argument("--n_jobs", type=int, default=PARAMS["n_jobs"],
                    help="потоки для SOAP; -1 = усі ядра")
    ap.add_argument("--lammps_dumps", default=None,
                    help="шаблон LAMMPS-дампів, напр. 'dump_*.lammpstrj' "
                         "(використати замість повільної ASE-МД)")
    ap.add_argument("--mode", default="single",
                    choices=["single", "repeats", "nscan", "dscan"],
                    help="single — базовий прогін; repeats — повтори з дисперсією; "
                         "nscan — масштабованість за N; dscan — залежність від D")
    ap.add_argument("--n_runs", type=int, default=5,
                    help="кількість повторів для mode=repeats")
    ap.add_argument("--threads", type=int, default=1,
                    help="потоки для BLAS/FAISS/hnswlib: 1 = однопотокове порівняння "
                         "(типово); 0 = усі ядра. (Зчитується ще до імпорту numpy.)")
    ap.add_argument("--split", default="random", choices=["random", "loso"],
                    help="random — випадкове розбиття; loso — leave-one-state-out "
                         "(база = кристалічні стани, запити = --holdout)")
    ap.add_argument("--holdout", default=None, choices=[None, "vac", "melt"],
                    help="для --split loso: утримуваний стан як джерело запитів")
    ap.add_argument("--out_prefix", default=None,
                    help="префікс вихідних файлів (типово — автоматично за режимом/split)")
    args = ap.parse_args()
    P = dict(PARAMS)
    P["potential"] = args.potential
    P["n_snapshots"] = args.n_snapshots
    P["n_jobs"] = args.n_jobs
    P["threads"] = THREADS
    P["split"] = args.split
    P["holdout"] = args.holdout
    if args.split == "loso" and not args.holdout:
        ap.error("--split loso потребує --holdout vac|melt")
    sfx = (f"_loso_{args.holdout}" if args.split == "loso" else "") + f"_t{THREADS}"
    np.random.seed(P["seed"])
    print(f"[info] потоки={THREADS} split={args.split}"
          + (f" holdout={args.holdout}" if args.holdout else ""), flush=True)

    # --- режим dscan: власні дескриптори для кожної (n_max, l_max) ---
    if args.mode == "dscan":
        configs = [(4, 4), (6, 5), (8, 6), (10, 8), (12, 9)]   # D ≈ 50..780
        rows = mode_dscan(P, args.lammps_dumps, configs)
        payload = dict(meta=dict(mode="dscan", configs=configs,
                                 k=P["k"], query_size=P["query_size"]),
                       results=rows)
        pre = args.out_prefix or f"results_dscan{sfx}"
        payload["meta"].update(_disclosure(P))
        _save_json_csv(payload, pre,
                       ["method", "param", "dtype", "threads", "D", "n_max", "l_max",
                        "recall_at_k", "recall_strict", "recall_at_1", "rel_dist_err",
                        "qps", "build_s", "mem_mb"])
        plot_scan(rows, "D", "Розмірність дескриптора D", f"{pre}_figure.png", P)
        return

    # 1–2. дескриптори (з кешем, бо генерування — найдовший крок)
    labels = None
    labels_cache = args.cache.replace(".npy", "") + "_labels.npy"
    if os.path.exists(args.cache):
        print(f"[info] завантажую кеш {args.cache}")
        X = np.load(args.cache)
        if os.path.exists(labels_cache):
            labels = np.load(labels_cache, allow_pickle=False)
        elif args.lammps_dumps:
            # мітки для наявного кешу — з дампів (без перерахунку SOAP)
            print("[info] відновлюю мітки станів із дампів…", flush=True)
            labels = labels_from_dumps(args.lammps_dumps, P["element"])
            np.save(labels_cache, labels)
    elif args.lammps_dumps:
        print("[info] читаю LAMMPS-дампи та рахую SOAP (DScribe)…")
        X, labels = generate_descriptors_from_dumps(P, args.lammps_dumps)
        np.save(args.cache, X); np.save(labels_cache, labels)
    else:
        print("[info] генерую конфігурації (ASE+EAM) та SOAP (DScribe)…")
        X = generate_descriptors(P)
        np.save(args.cache, X)
    N, D = X.shape
    print(f"[info] N={N} оточень, D={D}")
    if labels is not None:
        if len(labels) != N:
            raise RuntimeError(f"мітки ({len(labels)}) не збігаються з дескрипторами ({N}); "
                               f"видаліть {labels_cache} або перевірте --lammps_dumps")
        u, c = np.unique(labels, return_counts=True)
        print("[info] стани: " + ", ".join(f"{a}={b}" for a, b in zip(u, c)))
    if args.split == "loso" and labels is None:
        ap.error("--split loso потребує міток: задайте --lammps_dumps (мітки відновляться "
                 "з дампів без перерахунку SOAP)")

    # --- режим repeats: повтори з дисперсією ---
    if args.mode == "repeats":
        rows = mode_repeats(X, P, n_runs=args.n_runs, labels=labels)
        payload = dict(meta=dict(mode="repeats", n_runs=args.n_runs,
                                 N=int(N), D=int(D), k=P["k"], **_disclosure(P)),
                       results=rows)
        pre = args.out_prefix or f"results_repeats{sfx}"
        P["out_prefix"] = pre
        _save_json_csv(payload, pre,
                       ["method", "param", "n_runs",
                        "recall_at_k_mean", "recall_at_k_std",
                        "recall_strict_mean", "recall_strict_std",
                        "recall_at_1_mean", "recall_at_1_std",
                        "rel_dist_err_mean", "rel_dist_err_std",
                        "qps_mean", "qps_std", "build_s_mean", "mem_mb_mean"])
        plot_repeats(rows, P)
        return

    # --- режим nscan: масштабованість за обсягом N ---
    if args.mode == "nscan":
        sizes = [10000, 30000, 100000, 300000, min(N, 500000)]
        rows = mode_nscan(X, P, sizes)
        payload = dict(meta=dict(mode="nscan", sizes=sizes, D=int(D), k=P["k"]),
                       results=rows)
        pre = args.out_prefix or f"results_nscan{sfx}"
        payload["meta"].update(_disclosure(P))
        _save_json_csv(payload, pre,
                       ["method", "param", "dtype", "threads", "N", "n_db", "n_queries",
                        "recall_at_k", "recall_strict", "recall_at_1", "rel_dist_err",
                        "qps", "build_s", "mem_mb"])
        plot_scan(rows, "N", "Обсяг бази N", f"{pre}_figure.png", P)
        return

    # 3. поділ база / запити (random або leave-one-state-out)
    db, queries = make_split(X, P, seed=P["seed"], labels=labels)
    print(f"[info] база={len(db)}, запити={len(queries)}")

    # 4–5. еталон + методи
    rows = run_benchmark(db, queries, P, verbose=True)

    meta = dict(N=int(N), D=int(D), n_db=int(len(db)), n_queries=int(len(queries)),
                source=("lammps_dumps" if args.lammps_dumps else
                        ("cache" if os.path.exists(args.cache) else "ase_md")),
                params={k: P[k] for k in
                        ["k", "r_cut", "n_max", "l_max", "sigma", "supercell",
                         "temperatures", "melt_T", "n_snapshots",
                         "hnsw_M", "hnsw_efC", "hnsw_efSearch", "lsh_nbits"]},
                **_disclosure(P))
    P["out_prefix"] = args.out_prefix or f"results{sfx}"
    save_outputs(rows, meta, P)
    print(f"[ok] збережено {P['out_prefix']}.json та {P['out_prefix']}_table.csv")
    print("\n=== ПІДСУМОК ===")
    for r in rows:
        print(f"{r['method']:18} {str(r['param']):26} "
              f"recall@{P['k']}={r['recall_at_k']:.4f} strict={r['recall_strict']:.4f} "
              f"excess={r['rel_dist_err']:+.2e}  qps={r['qps']:.0f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Стратифікована вибірка SOAP-дескрипторів для репозиторію (швидкі тести).

Бере повний кеш (soap_descriptors.npy + soap_descriptors_labels.npy) і зберігає
data/soap_sample.npy + data/soap_sample_labels.npy — по n_per_state оточень із
кожного стану, з фіксованим seed. Повний набір у репозиторій не кладемо (≈500 МБ).

    python make_sample.py --cache soap_descriptors.npy --n_per_state 2500
"""
import argparse
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="soap_descriptors.npy")
    ap.add_argument("--n_per_state", type=int, default=2500)
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    X = np.load(a.cache)
    lab_path = a.cache.replace(".npy", "") + "_labels.npy"
    if not os.path.exists(lab_path):
        raise SystemExit(f"немає міток {lab_path}: запустіть бенчмарк один раз із --lammps_dumps")
    labels = np.load(lab_path)
    assert len(labels) == len(X), "мітки не збігаються з дескрипторами"

    rng = np.random.default_rng(a.seed)
    idx = []
    for s in sorted(set(labels.tolist())):
        pool = np.flatnonzero(labels == s)
        take = min(a.n_per_state, len(pool))
        idx.append(rng.choice(pool, size=take, replace=False))
        print(f"  {s:6}: {take} з {len(pool)}")
    idx = np.sort(np.concatenate(idx))

    os.makedirs(a.out_dir, exist_ok=True)
    np.save(os.path.join(a.out_dir, "soap_sample.npy"), X[idx].astype(np.float32))
    np.save(os.path.join(a.out_dir, "soap_sample_labels.npy"), labels[idx])
    print(f"збережено {len(idx)} × {X.shape[1]} у {os.path.abspath(a.out_dir)}")


if __name__ == "__main__":
    main()

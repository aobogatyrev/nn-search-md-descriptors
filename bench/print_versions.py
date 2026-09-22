#!/usr/bin/env python3
"""Зʼясовує версії середовища для рядка про обладнання у §5.5 / §6.
Запуск:  python print_versions.py
(за потреби вкажіть бінарник LAMMPS:  python print_versions.py --lammps lmp_serial)
"""
import argparse
import platform
import shutil
import subprocess
import sys


def pkg_version(import_name, dist_names=None):
    """Версія пакета: спершу importlib.metadata, потім module.__version__."""
    dist_names = dist_names or [import_name]
    # 1) метадані дистрибутиву (найнадійніше)
    try:
        from importlib.metadata import version, PackageNotFoundError
        for d in dist_names:
            try:
                return version(d)
            except PackageNotFoundError:
                continue
    except Exception:
        pass
    # 2) атрибут __version__ модуля
    try:
        mod = __import__(import_name)
        return getattr(mod, "__version__", "невідомо (немає __version__)")
    except Exception as e:
        return f"не встановлено ({e.__class__.__name__})"


def lammps_version(binary):
    """Версія LAMMPS: через python-модуль lammps, інакше через бінарник."""
    # 1) python-модуль lammps
    try:
        import lammps
        try:
            v = lammps.lammps().version()       # ціле число дати, напр. 20230802
            return f"{v} (python-модуль lammps)"
        except Exception:
            v = getattr(lammps, "__version__", None)
            if v:
                return f"{v} (python-модуль lammps)"
    except Exception:
        pass
    # 2) бінарник: банер містить рядок 'LAMMPS (DD Mon YYYY)'
    for binname in [binary, "lmp_serial", "lmp", "lammps_serial", "lammps"]:
        if binname and shutil.which(binname):
            try:
                out = subprocess.run([binname, "-h"], capture_output=True,
                                     text=True, timeout=20)
                text = (out.stdout or "") + (out.stderr or "")
                for line in text.splitlines():
                    if "LAMMPS (" in line:
                        return line.strip().split("LAMMPS (", 1)[1].rstrip(")").strip() \
                               + f"  [бінарник {binname}]"
                # запасний варіант: перший рядок банера
                if text.strip():
                    return f"невідомо (банер не містить дати) [бінарник {binname}]"
            except Exception as e:
                return f"помилка запуску {binname}: {e.__class__.__name__}"
    return "не знайдено (вкажіть --lammps <шлях до бінарника>)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lammps", default="lmp_serial",
                    help="імʼя/шлях бінарника LAMMPS (типово lmp_serial)")
    args = ap.parse_args()

    mac = platform.mac_ver()[0]
    os_str = f"macOS {mac}" if mac else platform.platform()

    rows = [
        ("Python", platform.python_version()),
        ("NumPy", pkg_version("numpy")),
        ("SciPy", pkg_version("scipy")),
        ("scikit-learn", pkg_version("sklearn", ["scikit-learn"])),
        ("ASE", pkg_version("ase")),
        ("DScribe", pkg_version("dscribe")),
        ("hnswlib", pkg_version("hnswlib")),
        ("FAISS", pkg_version("faiss", ["faiss-cpu", "faiss-gpu", "faiss"])),
        ("LAMMPS", lammps_version(args.lammps)),
    ]

    print("\n=== Середовище ===")
    print(f"CPU: Apple M1 Max    RAM: 32 ГБ    ОС: {os_str}")
    w = max(len(n) for n, _ in rows)
    for name, ver in rows:
        print(f"  {name:<{w}} : {ver}")

    # Рядок, готовий для статті (україномовний)
    d = dict(rows)
    print("\n=== Рядок для §5.5 / §6 (скопіювати) ===")
    print(
        f"Обчислення виконано на одному вузлі (Apple M1 Max, 32 ГБ оперативної "
        f"памʼяті, {os_str}). Версії програмного забезпечення: Python {d['Python']}, "
        f"LAMMPS {d['LAMMPS'].split('  [')[0].split(' [')[0]}, DScribe {d['DScribe']}, "
        f"ASE {d['ASE']}, hnswlib {d['hnswlib']}, FAISS {d['FAISS']}, "
        f"scikit-learn {d['scikit-learn']}, NumPy {d['NumPy']}."
    )


if __name__ == "__main__":
    main()

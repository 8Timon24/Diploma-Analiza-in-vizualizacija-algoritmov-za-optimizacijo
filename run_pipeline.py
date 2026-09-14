"""
Orkestracijska skripta: zaporedno požene celoten cevovod priprave podatkov
in analize, od surovih optimizacijskih zagonov do končne Spearmanove analize.

Vsak korak se izvede kot ločen podproces (enako, kot bi ga pognal ročno v
terminalu), zaporedje pa se ustavi takoj, če kateri koli korak vrne napako
(neničelno izhodno kodo) - tako se ne nadaljuje na pokvarjenih podatkih.

Uporaba:
    python run_pipeline.py                  # poženi vse korake
    python run_pipeline.py --from gručenje   # nadaljuj od danega koraka
    python run_pipeline.py --only benchmark  # poženi samo en korak
    python run_pipeline.py --dry-run         # samo izpiši ukaze, ne poganjaj

TODO pred prvim zagonom:
  - preveri, da so poti/imena skript in argumenti spodaj pravi za tvoje okolje
    (posebej -c kmeans za korak gručenja, in imena pairwise skript)
  - preveri, da so vse skripte v istem direktoriju kot ta orkestrator, ali
    popravi poti spodaj
"""
import subprocess
import sys
import time
import argparse

# ---------------------------------------------------------------------------
# Definicija korakov: (ime, ukaz kot seznam argumentov, opis)
# Vrstni red je pomemben - vsak korak je odvisen od izhoda prejšnjega.
# ---------------------------------------------------------------------------
STEPS = [
    ("benchmark",
     [sys.executable, "ClustOpts_algorithm_benchmarks.py", "-p", "a", "-e"],
     "Zagon vseh optimizacijskih algoritmov (mealpy/cocoex) - populacijska "
     "+ g_best trajektorija + raznolikost, epoch=10*dimenzija."),

    ("harvest_results",
     [sys.executable, "harvest_results.py"],
     "Gradnja results.csv iz g_best trajektorij."),

    ("preprocess",
     [sys.executable, "preprocess_data.py"],
     "Pretvorba outputs/ v data/processed/dim_{d}/F{f}_I{i}.csv (vhod za gručenje)."),

    ("gručenje",
     [sys.executable, "1_clustering_calculate_selected_algorithms_refactored.py", "-c", "kmeans"],
     "KMeans gručenje trajektorij -> cluster_centers, cluster_distributions, clustering_results."),

    ("agregatna_kosinusna",
     [sys.executable, "2_clustering_calculate_algorithm_similarity_refactored.py", "-c", "kmeans"],
     "Agregatna kosinusna podobnost (za clustermap slike)."),

    ("entropija_calc",
     [sys.executable, "entropy_calculation.py"],
     "Izračun entropije zasedenosti gruč (granularna + agregirana)."),

    ("entropija_pairwise",
     [sys.executable, "entropy_pairwise.py"],
     "Pairwise mera razlike v entropiji."),

    ("return_rate_calc",
     [sys.executable, "return_rate_calculation.py"],
     "Izračun revisiting history + matrike deleža vračanja."),

    ("return_rate_pairwise",
     [sys.executable, "return_rate_pairwise.py"],
     "Pairwise mera deleža vračanja (granularna)."),

    ("cosine_pairwise",
     [sys.executable, "cosine_pairwise.py"],
     "Pairwise globalna kosinusna razdalja."),

    ("cosine_columns_pairwise",
     [sys.executable, "cosine_rr.py"],
     "Pairwise stolpčna kosinusna razdalja (po gručah)."),

    ("exploration_pairwise",
     [sys.executable, "exploration_pairwise.py"],
     "Pairwise mera razlike v raziskovanju/izkoriščanju."),

    ("solutions_pairwise",
     [sys.executable, "solutions_pairwise.py"],
     "Pairwise mera razlike v lokaciji in fitnessu."),

    ("merge",
     [sys.executable, "merge_metrics.py"],
     "Združitev vseh metrik v merged_dim_{d}.csv."),

    ("spearman",
     [sys.executable, "spearman.py"],
     "Spearmanova korelacijska analiza med metrikami."),
]

STEP_NAMES = [name for name, _, _ in STEPS]


def run_step(name, cmd, description, dry_run=False):
    print(f"\n{'='*70}")
    print(f"KORAK: {name}")
    print(f"  {description}")
    print(f"  ukaz: {' '.join(cmd)}")
    print(f"{'='*70}")

    if dry_run:
        print("  [dry-run] preskočeno")
        return True

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n!!! KORAK '{name}' JE SPODLETEL (izhodna koda {result.returncode}), "
              f"po {elapsed:.1f}s. Ustavljam cevovod.")
        return False

    print(f"\n  korak '{name}' uspešno zaključen v {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Zaporedni cevovod priprave podatkov in analize")
    parser.add_argument("--from", dest="from_step", choices=STEP_NAMES,
                        help="Nadaljuj od tega koraka naprej (vključno)")
    parser.add_argument("--only", dest="only_step", choices=STEP_NAMES,
                        help="Poženi samo ta en korak")
    parser.add_argument("--dry-run", action="store_true",
                        help="Samo izpiši ukaze, brez dejanskega poganjanja")
    args = parser.parse_args()

    if args.only_step:
        steps_to_run = [s for s in STEPS if s[0] == args.only_step]
    elif args.from_step:
        start_idx = STEP_NAMES.index(args.from_step)
        steps_to_run = STEPS[start_idx:]
    else:
        steps_to_run = STEPS

    print(f"Cevovod: {len(steps_to_run)} korakov -> {[s[0] for s in steps_to_run]}")

    pipeline_start = time.time()
    for name, cmd, description in steps_to_run:
        ok = run_step(name, cmd, description, dry_run=args.dry_run)
        if not ok:
            sys.exit(1)

    total = time.time() - pipeline_start
    print(f"\n{'='*70}")
    print(f"CELOTEN CEVOVOD USPEŠNO ZAKLJUČEN v {total/60:.1f} min")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
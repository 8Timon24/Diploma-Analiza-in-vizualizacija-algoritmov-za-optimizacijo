import pandas as pd
import os

OUTPUTS_DIR = "outputs"
DIMENSIONS = [2, 5, 10]
SEEDS = [1, 2, 3, 4, 5]


def best_row_from_gbest(path):
    df = pd.read_csv(path)
    coord_cols = [c for c in df.columns if c.startswith('x')]
    best = df.loc[df['fitness'].idxmin()]
    return coord_cols, best


def main():
    for d in DIMENSIONS:
        base = f"{OUTPUTS_DIR}/dim_{d}"
        rows = []
        coord_cols_ref = None

        for algorithm in sorted(os.listdir(base)):
            alg_dir = f"{base}/{algorithm}"
            if not os.path.isdir(alg_dir):
                continue
            for problem_folder in sorted(os.listdir(alg_dir)):
                pf = problem_folder
                parts = pf.replace('F', '').replace('I', '').split('_')
                problem_id, instance_id = int(parts[0]), int(parts[1])

                for seed in SEEDS:
                    path = f"{alg_dir}/{pf}/gbest_trajectory_{seed}.csv"
                    coord_cols, best = best_row_from_gbest(path)
                    coord_cols_ref = coord_cols
                    row = {
                        "algorithm": algorithm,
                        "problem_id": problem_id,
                        "instance_id": instance_id,
                        "seed": seed,
                        "fitness": best["fitness"],
                    }
                    for c in coord_cols:
                        row[c] = best[c]
                    rows.append(row)

        result = pd.DataFrame(rows)
        ordered = (["algorithm", "problem_id", "instance_id", "seed"]
                   + coord_cols_ref + ["fitness"])
        result = result[ordered]
        out_path = f"{base}/results.csv"
        result.to_csv(out_path, index=False)
        print(f"wrote {out_path}  ({len(result)} rows)")


if __name__ == "__main__":
    main()

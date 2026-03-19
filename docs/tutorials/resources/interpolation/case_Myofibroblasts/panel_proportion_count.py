import argparse
import json
from pathlib import Path

import anndata
import pandas as pd


def parse_args():
    case_dir = Path(__file__).resolve().parent
    section_dir = case_dir.parent

    parser = argparse.ArgumentParser(description="Predicted trajectory composition summary.")
    parser.add_argument(
        "--pred_dir",
        default=str(section_dir / "outputs" / "02_myofibroblasts_end_to_end" / "00_model_inference_ckpt6"),
        help="Directory containing pred_t*_to_t*.h5ad files.",
    )
    parser.add_argument(
        "--ct_to_trajectory_json",
        default=str(section_dir / "ct_to_trajectory.json"),
        help="Cell-type to trajectory mapping JSON.",
    )
    parser.add_argument("--cell_type", default="Myofibroblasts", help="Query cell type.")
    parser.add_argument("--target_day", type=float, default=18.25, help="Target day (E-day numeric).")
    parser.add_argument("--start_day", type=float, default=8.5, help="Timeline start day used for index mapping.")
    parser.add_argument("--step", type=float, default=0.25, help="Timeline step size for index mapping.")
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional output CSV with per-cell-type counts/ratios.",
    )
    return parser.parse_args()


def day_to_point(target_day: float, start_day: float, step: float) -> int:
    return int(round((target_day - start_day) / step))


def main():
    args = parse_args()

    pred_dir = Path(args.pred_dir)
    ct_json = Path(args.ct_to_trajectory_json)

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not ct_json.exists():
        raise FileNotFoundError(f"ct_to_trajectory JSON not found: {ct_json}")

    point = day_to_point(args.target_day, args.start_day, args.step)
    pred_file = pred_dir / f"pred_t{point - 1}_to_t{point}.h5ad"
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    adata_pred = anndata.read_h5ad(pred_file)

    with open(ct_json, "r") as f:
        ct_to_traj = json.load(f)

    ct = args.cell_type
    traj = ct_to_traj.get(ct)
    if not traj:
        raise ValueError(f"Cell type '{ct}' not found in trajectory mapping JSON.")

    traj_cts = [k for k, v in ct_to_traj.items() if v == traj]
    pred_ct = adata_pred.obs["predicted_cell_type"].astype(str)

    counts = pred_ct[pred_ct.isin(traj_cts)].value_counts().sort_values(ascending=False)
    traj_total = int(counts.sum())

    print(f"Trajectory: {traj}")
    print(f"Prediction file: {pred_file}")
    print(f"Total cells in trajectory: {traj_total}\n")

    out_rows = []
    for traj_ct in traj_cts:
        ct_count = int(counts.get(traj_ct, 0))
        ratio = ct_count / traj_total if traj_total > 0 else 0.0
        print(f"{traj_ct}: {ct_count} ({ratio:.4f})")
        out_rows.append(
            {
                "trajectory": traj,
                "cell_type": traj_ct,
                "count": ct_count,
                "ratio": ratio,
                "target_day": float(args.target_day),
                "prediction_file": str(pred_file),
            }
        )

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out_rows).to_csv(output_csv, index=False)
        print(f"\n[OK] Saved: {output_csv}")


if __name__ == "__main__":
    main()

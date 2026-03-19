import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy import sparse


def parse_args():
    case_dir = Path(__file__).resolve().parent
    section_dir = case_dir.parent

    parser = argparse.ArgumentParser(description="Marker-gene panel reconstruction for imputation case.")
    parser.add_argument(
        "--input_data",
        default="/workspace/fuchenghao/dynflow_codebase/dynflow_dataset/aggregated_full_hvg_4000.h5ad",
        help="Path to full reference AnnData.",
    )
    parser.add_argument(
        "--pred_dir",
        default=str(section_dir / "outputs" / "02_myofibroblasts_end_to_end" / "00_model_inference_ckpt6"),
        help="Directory containing pred_t*_to_t*.h5ad files.",
    )
    parser.add_argument(
        "--deg_dir",
        default=str(section_dir / "deg_results_full"),
        help="Directory containing *_deg.csv files.",
    )
    parser.add_argument("--cell_type", default="Myofibroblasts", help="Cell type to analyze.")
    parser.add_argument("--target_day", type=float, default=18.25, help="Target day (E-day numeric).")
    parser.add_argument("--start_day", type=float, default=8.5, help="Timeline start day used for index mapping.")
    parser.add_argument("--step", type=float, default=0.25, help="Timeline step size for index mapping.")
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Output CSV path. Defaults to case_Myofibroblasts/<cell_type>_marker_genes_t<point>.csv",
    )
    return parser.parse_args()


def sanitize_cell_type(cell_type: str) -> str:
    return (
        cell_type.replace("/", "_")
        .replace(" ", "_")
        .replace("(", "|")
        .replace(")", "|")
    )


def day_to_point(target_day: float, start_day: float, step: float) -> int:
    return int(round((target_day - start_day) / step))


def day_str(day_value: float) -> str:
    return f"E{float(day_value)}"


def to_dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def mean_vector(adata: anndata.AnnData, mask: np.ndarray) -> np.ndarray:
    if int(mask.sum()) == 0:
        return np.zeros(adata.n_vars, dtype=float)
    return np.asarray(adata[mask].X.mean(axis=0)).ravel()


def normalize_to_x(adata: anndata.AnnData) -> anndata.AnnData:
    adata_m = np.concatenate([to_dense(adata.layers["Ms"]), to_dense(adata.layers["Mu"])], axis=1)
    data = (adata_m - adata_m.min(axis=0)) / (adata_m.max(axis=0) - adata_m.min(axis=0) + 1e-7)
    n_vars = adata.n_vars
    adata.X = data[:, :n_vars] + data[:, n_vars:]
    return adata


def main():
    args = parse_args()

    input_data = Path(args.input_data)
    pred_dir = Path(args.pred_dir)
    deg_dir = Path(args.deg_dir)

    if not input_data.exists():
        raise FileNotFoundError(f"Input AnnData not found: {input_data}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not deg_dir.exists():
        raise FileNotFoundError(f"DEG directory not found: {deg_dir}")

    ct = args.cell_type
    ct_file = sanitize_cell_type(ct)
    point = day_to_point(args.target_day, args.start_day, args.step)

    pred_file = pred_dir / f"pred_t{point - 1}_to_t{point}.h5ad"
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    timepoints = [args.target_day - 0.25, args.target_day, args.target_day + 0.25, args.target_day + 0.5]
    required_days = {day_str(t) for t in timepoints}

    adata_backed = anndata.read_h5ad(input_data, backed="r")
    obs = adata_backed.obs.copy()

    gene_id_to_name = {}
    if "gene_name" in adata_backed.var.columns:
        gene_id_to_name = dict(zip(adata_backed.var.index.astype(str), adata_backed.var["gene_name"].astype(str)))

    mask = (
        (obs["cell_type"].astype(str) == ct)
        & (obs["day"].astype(str).isin(required_days))
    ).values
    if int(mask.sum()) == 0:
        raise ValueError(f"No cells found for {ct} in days {sorted(required_days)}")

    adata_full = adata_backed[mask].to_memory()
    adata_full = normalize_to_x(adata_full)

    if "gene_name" in adata_full.var.columns:
        adata_full.var.index = adata_full.var["gene_name"].astype(str)

    deg_files = sorted(deg_dir.glob(f"{ct_file}_E*_deg.csv"))
    if not deg_files:
        raise FileNotFoundError(
            f"No DEG files found for {ct} under {deg_dir}. "
            f"Expected pattern: {ct_file}_E*_deg.csv"
        )

    deg_genes = set()
    for f in deg_files:
        df = pd.read_csv(f)
        gene_col = "names" if "names" in df.columns else df.columns[0]
        genes = df[gene_col].dropna().astype(str)
        if gene_id_to_name:
            genes = genes.map(gene_id_to_name).fillna(genes)
        deg_genes.update(genes.values)

    adata_pred = anndata.read_h5ad(pred_file)
    pred_ct_mask = adata_pred.obs["predicted_cell_type"].astype(str) == ct
    pred_ct_cells = adata_pred[pred_ct_mask]
    if pred_ct_cells.n_obs == 0:
        raise ValueError(f"No predicted cells for cell_type={ct} in {pred_file}")

    expr_by_time = []
    for t in timepoints:
        mask_t = (
            (adata_full.obs["cell_type"].astype(str) == ct)
            & (adata_full.obs["day"].astype(str) == day_str(t))
        ).values
        expr_by_time.append(mean_vector(adata_full, mask_t))
        print(f"[INFO] {ct} @ {day_str(t)} cells: {int(mask_t.sum())}")

    surrounding_expr = np.mean([expr_by_time[0], expr_by_time[2], expr_by_time[3]], axis=0)
    center_expr = expr_by_time[1]
    pred_expr = np.asarray(pred_ct_cells.X.mean(axis=0)).ravel()

    high_in_surrounding = surrounding_expr > center_expr * 1.5
    high_in_pred = pred_expr > center_expr * 1.5
    consistent_pred_surround = np.abs(pred_expr - surrounding_expr) < 0.1 * np.maximum(surrounding_expr, 1e-8)

    marker_mask = high_in_surrounding & high_in_pred & consistent_pred_surround
    marker_candidates = set(adata_full.var_names[np.where(marker_mask)[0]].astype(str))
    marker_genes = sorted(marker_candidates & deg_genes)

    print(f"[INFO] Found {len(marker_genes)} marker genes in DEG union")

    if args.output_csv is None:
        output_csv = Path(__file__).resolve().parent / f"{ct_file}_marker_genes_t{point}.csv"
    else:
        output_csv = Path(args.output_csv)

    marker_indices = [list(adata_full.var_names).index(g) for g in marker_genes]
    expr_df = pd.DataFrame(
        {
            "gene": marker_genes,
            "t_minus_0.25": expr_by_time[0][marker_indices],
            "t_center": expr_by_time[1][marker_indices],
            "t_plus_0.25": expr_by_time[2][marker_indices],
            "t_plus_0.5": expr_by_time[3][marker_indices],
            "t_pred": pred_expr[marker_indices],
        }
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    expr_df.to_csv(output_csv, index=False)
    print(f"[OK] Saved marker-gene table: {output_csv}")


if __name__ == "__main__":
    main()

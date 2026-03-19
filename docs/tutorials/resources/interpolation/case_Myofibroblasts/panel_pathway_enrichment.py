import argparse
import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy import sparse, stats


def parse_args():
    case_dir = Path(__file__).resolve().parent
    section_dir = case_dir.parent

    parser = argparse.ArgumentParser(description="Pathway enrichment reconstruction for imputation case.")
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
    parser.add_argument(
        "--msigdb_path",
        default=str(section_dir / "msigdb.v2025.1.Mm.json"),
        help="Path to MSigDB JSON.",
    )
    parser.add_argument("--cell_type", default="Myofibroblasts", help="Cell type to analyze.")
    parser.add_argument("--target_day", type=float, default=18.25, help="Target day (E-day numeric).")
    parser.add_argument("--start_day", type=float, default=8.5, help="Timeline start day used for index mapping.")
    parser.add_argument("--step", type=float, default=0.25, help="Timeline step size for index mapping.")
    parser.add_argument(
        "--real_days",
        default="18.0,18.25,18.5,18.75",
        help="Comma-separated real days to compare against prediction.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(case_dir),
        help="Directory for output pathway CSV files.",
    )
    parser.add_argument(
        "--pval_thresh",
        type=float,
        default=0.05,
        help="Adjusted p-value threshold for top-pathway set operations.",
    )
    parser.add_argument("--top_n", type=int, default=50, help="Top-N pathways for overlap/Jaccard.")
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


def day_label(day_value: float) -> str:
    return f"E{float(day_value)}"


def _to_dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]

    adj_ranked = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * n / rank)
        adj_ranked[i] = val
        prev = val

    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.clip(adj_ranked, 0.0, 1.0)
    return adjusted


def _wilcoxon_deg(x_target: np.ndarray, x_other: np.ndarray, gene_names: np.ndarray) -> pd.DataFrame:
    mean_target = x_target.mean(axis=0)
    mean_other = x_other.mean(axis=0)
    logfc = np.log2((mean_target + 1e-9) / (mean_other + 1e-9))

    pvals = np.ones(x_target.shape[1], dtype=float)
    scores = np.zeros(x_target.shape[1], dtype=float)

    for j in range(x_target.shape[1]):
        a = x_target[:, j]
        b = x_other[:, j]

        if np.allclose(a, a[0]) and np.allclose(b, b[0]) and np.isclose(a[0], b[0]):
            pvals[j] = 1.0
            scores[j] = 0.0
            continue

        res = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
        pvals[j] = res.pvalue
        scores[j] = res.statistic

    pvals_adj = _bh_fdr(pvals)
    df = pd.DataFrame(
        {
            "names": gene_names,
            "scores": scores,
            "logfoldchanges": logfc,
            "pvals": pvals,
            "pvals_adj": pvals_adj,
        }
    )
    return df.sort_values(["pvals_adj", "pvals", "logfoldchanges"], ascending=[True, True, False]).reset_index(
        drop=True
    )


def normalize_to_x(adata: anndata.AnnData) -> anndata.AnnData:
    adata_m = np.concatenate([_to_dense(adata.layers["Ms"]), _to_dense(adata.layers["Mu"])], axis=1)
    data = (adata_m - adata_m.min(axis=0)) / (adata_m.max(axis=0) - adata_m.min(axis=0) + 1e-7)
    n_vars = adata.n_vars
    adata.X = data[:, :n_vars] + data[:, n_vars:]
    return adata


def compute_pathway_enrichment(pathways, msigdb, all_genes, up_genes):
    M = len(all_genes)
    N = len(up_genes)
    enrich_data = []

    for pw, pw_name in pathways.items():
        pw_genes = set(msigdb[pw]["geneSymbols"]) & all_genes
        n = len(pw_genes)
        k = len(pw_genes & up_genes)
        if n == 0 or k == 0:
            continue

        table = [[k, N - k], [n - k, M - N - n + k]]
        if min(table[0] + table[1]) < 0:
            continue

        _, p_val = stats.fisher_exact(table, alternative="greater")
        enrich_data.append(
            {
                "pathway": pw_name,
                "overlap": k,
                "gene_set_size": N,
                "pathway_size": n,
                "p_value": p_val,
            }
        )

    if not enrich_data:
        return pd.DataFrame(columns=["pathway", "overlap", "gene_set_size", "pathway_size", "p_value"])

    return pd.DataFrame(enrich_data).sort_values("p_value")


def get_top_pathways(df: pd.DataFrame, n: int, pval_thresh: float):
    if df.empty:
        return set()
    return set(df[df["p_value"] < pval_thresh].head(n)["pathway"])


def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def get_top_pathways_dict(df: pd.DataFrame, n: int, pval_thresh: float):
    if df.empty:
        return {}
    df_filtered = df[df["p_value"] < pval_thresh].head(n)
    return dict(zip(df_filtered["pathway"], df_filtered["p_value"]))


def get_all_pathways_dict(df: pd.DataFrame):
    if df.empty:
        return {}
    return dict(zip(df["pathway"], df["p_value"]))


def main():
    args = parse_args()

    input_data = Path(args.input_data)
    pred_dir = Path(args.pred_dir)
    deg_dir = Path(args.deg_dir)
    msigdb_path = Path(args.msigdb_path)
    output_dir = Path(args.output_dir)

    for p in [input_data, pred_dir, deg_dir, msigdb_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required path not found: {p}")

    output_dir.mkdir(parents=True, exist_ok=True)

    ct = args.cell_type
    ct_file = sanitize_cell_type(ct)

    point = day_to_point(args.target_day, args.start_day, args.step)
    pred_file = pred_dir / f"pred_t{point - 1}_to_t{point}.h5ad"
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    adata_backed = anndata.read_h5ad(input_data, backed="r")
    obs = adata_backed.obs.copy()
    ct_mask = (obs["cell_type"].astype(str) == ct).values
    if int(ct_mask.sum()) == 0:
        raise ValueError(f"No cells found for cell type: {ct}")

    adata_full_ct = adata_backed[ct_mask].to_memory()
    adata_full_ct = normalize_to_x(adata_full_ct)

    if "gene_name" not in adata_full_ct.var.columns:
        raise ValueError("Expected var['gene_name'] to exist in input AnnData.")

    gene_id_to_name = dict(zip(adata_full_ct.var.index.astype(str), adata_full_ct.var["gene_name"].astype(str)))

    with open(msigdb_path, "r") as f:
        msigdb = json.load(f)

    pathways = {pw: pw.replace("GOBP_", "").replace("_", " ") for pw in msigdb if pw.startswith("GOBP_")}

    adata_pred = anndata.read_h5ad(pred_file)
    adata_pred = adata_pred[adata_pred.obs["predicted_cell_type"].astype(str) == ct].copy()
    if adata_pred.n_obs == 0:
        raise ValueError(f"No predicted cells for cell_type={ct} in {pred_file}")

    adata_pred.var = adata_full_ct.var.copy()

    target_day_label = day_label(args.target_day)
    adata_ct_other = adata_full_ct[adata_full_ct.obs["day"].astype(str) != target_day_label].copy()

    adata_combined = anndata.concat([adata_pred, adata_ct_other])
    adata_combined.obs["is_pred"] = pd.Categorical(["pred"] * adata_pred.n_obs + ["other"] * adata_ct_other.n_obs)
    adata_combined.var.index = adata_full_ct.var["gene_name"].astype(str)

    x_combined = _to_dense(adata_combined.X)
    is_pred = adata_combined.obs["is_pred"].astype(str).values == "pred"
    pred_deg = _wilcoxon_deg(
        x_combined[is_pred],
        x_combined[~is_pred],
        np.asarray(adata_combined.var_names.astype(str)),
    )
    pred_up = set(pred_deg[(pred_deg["pvals_adj"] < 0.01) & (pred_deg["logfoldchanges"] > 1)]["names"].astype(str))

    all_genes = set(adata_combined.var_names.astype(str))
    pred_enrich = compute_pathway_enrichment(pathways, msigdb, all_genes, pred_up)

    pred_out = output_dir / f"{ct_file}_pred_pathway_enrichment_E{float(args.target_day)}.csv"
    pred_enrich.to_csv(pred_out, index=False)
    print(f"[OK] {pred_out}")

    real_days = [float(x.strip()) for x in args.real_days.split(",") if x.strip()]
    real_enrich = {}

    for day in real_days:
        gt_deg_file = deg_dir / f"{ct_file}_E{float(day)}_deg.csv"
        if not gt_deg_file.exists():
            raise FileNotFoundError(f"Missing DEG file: {gt_deg_file}")

        gt_deg = pd.read_csv(gt_deg_file)
        if "names" in gt_deg.columns:
            names = gt_deg["names"].astype(str)
        else:
            names = gt_deg.iloc[:, 0].astype(str)

        gt_deg["gene_name"] = names.map(gene_id_to_name).fillna(names)
        gt_up = set(
            gt_deg[(gt_deg["pvals_adj"] < 0.01) & (gt_deg["logfoldchanges"] > 1)]["gene_name"]
            .dropna()
            .astype(str)
        )

        day_enrich = compute_pathway_enrichment(pathways, msigdb, all_genes, gt_up)
        real_enrich[day] = day_enrich

        out_file = output_dir / f"{ct_file}_real_pathway_enrichment_E{float(day)}.csv"
        day_enrich.to_csv(out_file, index=False)
        print(f"[OK] {out_file}")

    if 18.0 in real_enrich and 18.5 in real_enrich and 18.75 in real_enrich and 18.25 in real_enrich:
        pred_top = get_top_pathways(pred_enrich, args.top_n, args.pval_thresh)
        real_180_top = get_top_pathways(real_enrich[18.0], args.top_n, args.pval_thresh)
        real_1825_top = get_top_pathways(real_enrich[18.25], args.top_n, args.pval_thresh)
        real_185_top = get_top_pathways(real_enrich[18.5], args.top_n, args.pval_thresh)
        real_1875_top = get_top_pathways(real_enrich[18.75], args.top_n, args.pval_thresh)

        all_tops = [pred_top, real_180_top, real_185_top, real_1875_top]
        labels = ["Pred E18.25", "Real E18.0", "Real E18.5", "Real E18.75"]
        matrix = [[jaccard_similarity(all_tops[i], all_tops[j]) for j in range(4)] for i in range(4)]
        df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)

        print("\nJaccard Similarity Matrix:")
        print(df_matrix.round(4))

        all_real_tops = [real_1825_top, real_180_top, real_185_top, real_1875_top]
        real_labels = ["Real E18.25", "Real E18.0", "Real E18.5", "Real E18.75"]
        real_matrix = [[jaccard_similarity(all_real_tops[i], all_real_tops[j]) for j in range(4)] for i in range(4)]
        df_real_matrix = pd.DataFrame(real_matrix, index=real_labels, columns=real_labels)

        print("\nReal-only Jaccard Similarity Matrix:")
        print(df_real_matrix.round(4))

        pred_dict = get_top_pathways_dict(pred_enrich, args.top_n, args.pval_thresh)
        real_180_dict = get_top_pathways_dict(real_enrich[18.0], args.top_n, args.pval_thresh)
        real_185_dict = get_top_pathways_dict(real_enrich[18.5], args.top_n, args.pval_thresh)

        pred_all = get_all_pathways_dict(pred_enrich)
        real_180_all = get_all_pathways_dict(real_enrich[18.0])
        real_185_all = get_all_pathways_dict(real_enrich[18.5])
        real_1825_all = get_all_pathways_dict(real_enrich[18.25])

        shared = set(pred_dict.keys()) & set(real_180_dict.keys()) & set(real_185_dict.keys())
        shared_data = []

        for pw in sorted(shared):
            pred_pval = pred_all.get(pw, np.nan)
            r180_pval = real_180_all.get(pw, np.nan)
            r185_pval = real_185_all.get(pw, np.nan)
            r1825_pval = real_1825_all.get(pw, np.nan)

            shared_data.append(
                {
                    "pathway": pw,
                    "pred_E18.25_pval": pred_pval,
                    "real_E18.0_pval": r180_pval,
                    "real_E18.5_pval": r185_pval,
                    "real_E18.25_pval": r1825_pval,
                    "pred_E18.25_neg_log_pval": -np.log10(pred_pval) if not np.isnan(pred_pval) else np.nan,
                    "real_E18.0_neg_log_pval": -np.log10(r180_pval) if not np.isnan(r180_pval) else np.nan,
                    "real_E18.5_neg_log_pval": -np.log10(r185_pval) if not np.isnan(r185_pval) else np.nan,
                    "real_E18.25_neg_log_pval": -np.log10(r1825_pval) if not np.isnan(r1825_pval) else np.nan,
                }
            )

        # Keep panel-focused pathways present even if set-intersection is empty,
        # so downstream plotting notebooks can still build a stable table.
        panel_focus = [
            "SMALL MOLECULE METABOLIC PROCESS",
            "SKELETAL MUSCLE CONTRACTION",
            "REGULATION OF RESPONSE TO STRESS",
            "REGULATION OF DEFENSE RESPONSE",
            "ORGANIC ACID METABOLIC PROCESS",
            "MULTICELLULAR ORGANISMAL MOVEMENT",
            "LIPID METABOLIC PROCESS",
        ]
        already_present = {row["pathway"] for row in shared_data}
        for pw in panel_focus:
            if pw in already_present:
                continue
            pred_pval = pred_all.get(pw, np.nan)
            r180_pval = real_180_all.get(pw, np.nan)
            r185_pval = real_185_all.get(pw, np.nan)
            r1825_pval = real_1825_all.get(pw, np.nan)
            shared_data.append(
                {
                    "pathway": pw,
                    "pred_E18.25_pval": pred_pval,
                    "real_E18.0_pval": r180_pval,
                    "real_E18.5_pval": r185_pval,
                    "real_E18.25_pval": r1825_pval,
                    "pred_E18.25_neg_log_pval": -np.log10(pred_pval) if not np.isnan(pred_pval) else np.nan,
                    "real_E18.0_neg_log_pval": -np.log10(r180_pval) if not np.isnan(r180_pval) else np.nan,
                    "real_E18.5_neg_log_pval": -np.log10(r185_pval) if not np.isnan(r185_pval) else np.nan,
                    "real_E18.25_neg_log_pval": -np.log10(r1825_pval) if not np.isnan(r1825_pval) else np.nan,
                }
            )

        shared_cols = [
            "pathway",
            "pred_E18.25_pval",
            "real_E18.0_pval",
            "real_E18.5_pval",
            "real_E18.25_pval",
            "pred_E18.25_neg_log_pval",
            "real_E18.0_neg_log_pval",
            "real_E18.5_neg_log_pval",
            "real_E18.25_neg_log_pval",
        ]
        df_shared = pd.DataFrame(shared_data, columns=shared_cols)
        shared_out = output_dir / f"{ct_file}_shared_pathways.csv"
        df_shared.to_csv(shared_out, index=False)
        print(f"[OK] {shared_out}")
        print(f"[INFO] Shared pathways count: {len(shared)}")


if __name__ == "__main__":
    main()

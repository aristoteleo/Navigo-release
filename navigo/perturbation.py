import json
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .model import MLPTimeGRN, Navigo


def _minmax_normalize(x):
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    return (x - x_min) / (x_max - x_min + 1e-7)


def _resolve_device(device):
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_gene_names(adata):
    if "gene_name" in adata.var.columns:
        return list(adata.var["gene_name"].astype(str).values)
    return list(adata.var_names.astype(str).values)


def _first_index_map(gene_names):
    mapping = {}
    for i, g in enumerate(gene_names):
        if g not in mapping:
            mapping[g] = i
    return mapping


def run_perturbation_inference(
    data_path,
    checkpoint_path,
    target_list_path,
    output_dir,
    fibro_cell_type="Fibroblasts",
    input_dim=7804,
    hidden_1=5012,
    hidden_2=5012,
    flow_num_steps=100,
    simulation_steps=10,
    knockout_scale=3.0,
    device=None,
    strict_target_presence=False,
):
    """Run Navigo perturbation inference and write one CSV per target set.

    The implementation follows the original inhibition script behavior while
    exposing a reusable package API.
    """
    data_path = Path(data_path)
    checkpoint_path = Path(checkpoint_path)
    target_list_path = Path(target_list_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "data_path": data_path,
        "checkpoint_path": checkpoint_path,
        "target_list_path": target_list_path,
    }
    missing = [f"{k}: {v}" for k, v in required.items() if not v.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    run_device = _resolve_device(device)
    started = time.time()

    adata = anndata.read_h5ad(data_path)
    if "time" not in adata.obs.columns or "cell_type" not in adata.obs.columns:
        raise ValueError("AnnData must include obs['time'] and obs['cell_type']")
    if "Ms" not in adata.layers.keys() or "Mu" not in adata.layers.keys():
        raise ValueError("AnnData must include layers['Ms'] and layers['Mu']")

    gene_names = _load_gene_names(adata)
    gene_to_idx = _first_index_map(gene_names)

    adata_fib = adata[adata.obs["cell_type"].astype(str) == fibro_cell_type].copy()
    if adata_fib.n_obs == 0:
        raise ValueError(f"No cells found for cell_type='{fibro_cell_type}'")

    ms_fib = np.asarray(adata_fib.layers["Ms"])
    mu_fib = np.asarray(adata_fib.layers["Mu"])
    data_fib_np = _minmax_normalize(np.concatenate([ms_fib, mu_fib], axis=1))

    data_fib_original = torch.tensor(data_fib_np, dtype=torch.float32).to(run_device)
    time_label_fib = torch.tensor(
        np.asarray(adata_fib.obs["time"], dtype=np.float32), dtype=torch.float32
    ).to(run_device)

    state = torch.load(checkpoint_path, map_location=run_device)
    model = MLPTimeGRN(input_dim=input_dim, hidden_1=hidden_1, hidden_2=hidden_2).to(run_device)
    model.load_state_dict(state, strict=True)
    flow = Navigo(model=model, num_steps=flow_num_steps, device=run_device)

    lines = [line.strip() for line in target_list_path.read_text().splitlines() if line.strip()]

    # baseline is shared across perturbations
    pred_forward_base = flow.sample_ode_time_interval(
        z_full=data_fib_original,
        t_start=time_label_fib,
        t_end=time_label_fib + 1,
        N=simulation_steps,
    )

    missing_targets = []
    written_files = []

    for line in tqdm(lines, desc="Perturbation inference"):
        targets = [t.strip() for t in line.split(",") if t.strip()]
        data_fib = data_fib_original.clone()

        index = []
        value_s = []
        value_u = []

        for target in targets:
            if target not in gene_to_idx:
                missing_targets.append(target)
                if strict_target_presence:
                    raise KeyError(f"Target gene '{target}' not found in dataset genes")
                continue

            target_idx = gene_to_idx[target]
            index.append(target_idx)

            s_pos = data_fib[:, target_idx][data_fib[:, target_idx] > 0]
            u_pos = data_fib[:, target_idx + (input_dim // 2)][
                data_fib[:, target_idx + (input_dim // 2)] > 0
            ]

            s_val = knockout_scale * torch.mean(s_pos) if s_pos.numel() > 0 else torch.tensor(0.0, device=run_device)
            u_val = knockout_scale * torch.mean(u_pos) if u_pos.numel() > 0 else torch.tensor(0.0, device=run_device)

            value_s.append(s_val)
            value_u.append(u_val)

            data_fib[:, target_idx] = s_val
            data_fib[:, target_idx + (input_dim // 2)] = u_val

        if len(index) == 0:
            continue

        pred_forward = pred_forward_base.copy()
        pred_forward_ko = flow.sample_ode_time_interval_knockout(
            z_full=data_fib,
            t_start=time_label_fib,
            t_end=time_label_fib + 1,
            N=simulation_steps,
            index=index,
            value_s=value_s,
            value_u=value_u,
        )

        index_arr = np.array(index, dtype=int)
        half_dim = input_dim // 2
        pred_forward[:, index_arr] = 0
        pred_forward_ko[:, index_arr] = 0
        pred_forward[:, index_arr + half_dim] = 0
        pred_forward_ko[:, index_arr + half_dim] = 0

        pred_forward_mean = pred_forward.mean(axis=0)
        pred_forward_ko_mean = pred_forward_ko.mean(axis=0)

        s_changes = pred_forward_ko_mean[:half_dim] - pred_forward_mean[:half_dim]
        u_changes = pred_forward_ko_mean[half_dim:] - pred_forward_mean[half_dim:]
        total_changes = s_changes + u_changes
        relative_changes = np.where(
            pred_forward_mean[:half_dim] + pred_forward_mean[half_dim:] != 0,
            total_changes / (pred_forward_mean[:half_dim] + 1e-7),
            np.zeros_like(total_changes),
        )

        df_changes = pd.DataFrame(
            {
                "gene_name": gene_names,
                "spliced_change": s_changes,
                "unspliced_change": u_changes,
                "total_change": total_changes,
                "relative_change": relative_changes,
            }
        )

        target_name = "_".join(targets)
        out_file = output_dir / f"{target_name}.csv"
        df_changes.to_csv(out_file, index=False)
        written_files.append(str(out_file))

    summary = {
        "data_path": str(data_path),
        "checkpoint_path": str(checkpoint_path),
        "target_list_path": str(target_list_path),
        "output_dir": str(output_dir),
        "device": run_device,
        "num_cells_fibro": int(adata_fib.n_obs),
        "num_targets": int(len(lines)),
        "num_outputs": int(len(written_files)),
        "missing_targets_unique": sorted(set(missing_targets)),
        "flow_num_steps": int(flow_num_steps),
        "simulation_steps": int(simulation_steps),
        "knockout_scale": float(knockout_scale),
        "elapsed_sec": float(time.time() - started),
    }

    (output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

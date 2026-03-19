import argparse
import os
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navigo.model import MLPTimeGRN, Navigo

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['WANDB_MODE'] = 'dryrun'

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='../data_processing/data_store/Limb#mesenchyme#trajectory.h5ad')
parser.add_argument('--perturb_genes', nargs='+', default=['ENSMUSG00000048402'])
parser.add_argument('--perturb_gene_names', nargs='+', default=['Gli2'])
parser.add_argument('--cell_type', default='Gli2')
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--regulation_effect', default='2')
parser.add_argument('--output_root', default='../results_full')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

adata = anndata.read_h5ad(args.data_path.replace('@', ' '))
adata = adata[adata.obs['day'].isin(['E12.5', 'E12.75', 'E13.0', 'E13.25', 'E13.5'])].copy()

data = torch.tensor(np.concatenate([adata.layers['Ms'], adata.layers['Mu']], axis=1), dtype=torch.float32).to(device)
data = (data - data.min(axis=0).values) / (data.max(axis=0).values - data.min(axis=0).values + 1e-7)
time_label = torch.tensor(np.asarray(adata.obs['time'], dtype=np.float32), dtype=torch.float32).to(device)

model = MLPTimeGRN(input_dim=7804, hidden_1=5012, hidden_2=5012).to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
flow = Navigo(model=model, num_steps=100, device=device)

pred_forward = flow.sample_ode_time_interval(z_full=data, t_start=time_label, t_end=time_label + 1, N=10)

indices = []
values_s = []
values_u = []
for gene in args.perturb_genes:
    idx = list(adata.var.index).index(gene)
    indices.append(idx)
    zero = torch.tensor(0.0, device=device)
    values_s.append(zero)
    values_u.append(zero)
    data[:, idx] = zero
    data[:, idx + 3902] = zero

pred_forward_ko = flow.sample_ode_time_interval_knockout(
    z_full=data,
    t_start=time_label,
    t_end=time_label + 1,
    N=10,
    index=indices,
    value_s=values_s,
    value_u=values_u,
)

pred_ko_mean = pred_forward_ko.mean(axis=0)
pred_mean = pred_forward.mean(axis=0)

pred_ko_mean = pred_ko_mean[0:3902] + pred_ko_mean[3902:]
pred_mean = pred_mean[0:3902] + pred_mean[3902:]
for idx in indices:
    pred_ko_mean[idx] = pred_mean[idx] = 0
res_background = pred_ko_mean - pred_mean

gene_names = '_'.join(args.perturb_gene_names)
output_dir = Path(args.output_root) / f'forward_{gene_names}_{args.cell_type}_{args.regulation_effect}'
output_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame(res_background, index=adata.var.index).to_csv(output_dir / 'prediction.csv')
print('Saved:', output_dir / 'prediction.csv')

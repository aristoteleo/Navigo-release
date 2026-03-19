import json
import pandas as pd
import anndata
from scipy.stats import hypergeom
import numpy as np
from statsmodels.stats.multitest import multipletests
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
MSIGDB_PATH = REPO_ROOT / "data" / "knockout" / "msigdb_mouse_v2025_1.json"

with open(MSIGDB_PATH) as f:
    data = json.load(f)


adata = anndata.read("/home/yiminfan/projects/ctb-liyue/yiminfan/multigps_pathology/dynflow/aggregated_pert_3902.h5ad")
df = adata.var
del df['Unnamed: 0'],df['gene_id'],df['gene_type']

pathways = [
    "HALLMARK_HEDGEHOG_SIGNALING",
    "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
    "HALLMARK_TGF_BETA_SIGNALING",
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
    "HALLMARK_NOTCH_SIGNALING",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_APOPTOSIS",
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "REACTOME_HEDGEHOG_LIGAND_BIOGENESIS",
    "REACTOME_SIGNALING_BY_HEDGEHOG",
    "REACTOME_SIGNALING_BY_WNT",
]

pathway_genes_all = []
pathway_all = []

for pathway in pathways:
    pathway_genes = set(data[pathway]['geneSymbols']).intersection(set(adata.var['gene_short_name']))
    pathway_all.extend([pathway] * len(pathway_genes))
    pathway_genes_all.extend(list(pathway_genes))
    
pathway_genes = pd.DataFrame([pathway_all,pathway_genes_all]).T
pathway_genes.columns = ['pathway','gene']
pathway_genes.to_csv("pathway_info.csv")
            

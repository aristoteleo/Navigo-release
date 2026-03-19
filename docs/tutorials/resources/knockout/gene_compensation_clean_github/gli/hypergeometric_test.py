import json
import pandas as pd
import anndata
from scipy.stats import hypergeom
import numpy as np
from statsmodels.stats.multitest import multipletests
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
MSIGDB_PATH = REPO_ROOT / "data" / "knockout" / "msigdb_mouse_v2025_1.json"

with open(MSIGDB_PATH) as f:
    data = json.load(f)

gsea_pathways = [
    "HALLMARK_TGF_BETA_SIGNALING",
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
    "HALLMARK_NOTCH_SIGNALING",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_APOPTOSIS",
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "REACTOME_SIGNALING_BY_HEDGEHOG",
    "REACTOME_SIGNALING_BY_WNT",
    "HALLMARK_G2M_CHECKPOINT",
]

pathway_mapping = {
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": "EMT",
    "REACTOME_SIGNALING_BY_WNT": "WNT",
    "REACTOME_SIGNALING_BY_HEDGEHOG": "Hedgehog",
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING": "PI3K/Akt/mTOR",
    "HALLMARK_G2M_CHECKPOINT": "G2M Checkpoint",
    "HALLMARK_TGF_BETA_SIGNALING": "TGF Beta",
    "HALLMARK_NOTCH_SIGNALING": "Notch",
    "HALLMARK_APOPTOSIS": "Apoptosis",
    "HALLMARK_INFLAMMATORY_RESPONSE": "Inflammatory"
}

adata = anndata.read("/home/yiminfan/projects/ctb-liyue/yiminfan/multigps_pathology/dynflow/aggregated_pert_3902.h5ad")
cell_types = [d for d in os.listdir("output") if d.endswith("#trajectory")]
total_genes = len(adata.var['gene_short_name'])

os.makedirs("output/hypergeometric_tests", exist_ok=True)

def hypergeometric_test(selected_genes, total_genes, data, gsea_pathways, gene_names):
    results = []
    for pathway in gsea_pathways:
        if "REACTOME" in pathway or "HALLMARK" in pathway:
            pathway_genes = set(data[pathway]['geneSymbols']).intersection(set(gene_names))
            overlap_genes = selected_genes.intersection(pathway_genes)
            
            M = total_genes
            n = len(pathway_genes)
            N = len(selected_genes)
            k = len(overlap_genes)
            
            p_value = hypergeom.sf(k-1, M, n, N)
            expected = (N * n) / M
            fold_enrichment = (k / expected) if expected > 0 else np.inf
            
            results.append({
                'pathway': pathway,
                'pathway_size': n,
                'selected_size': N,
                'overlap': k,
                'expected': expected,
                'fold_enrichment': fold_enrichment,
                'p_value': p_value,
                'overlapping_genes': ','.join(sorted(overlap_genes))
            })
    
    results_df = pd.DataFrame(results)
    _, results_df['p_adj'], _, _ = multipletests(results_df['p_value'], method='fdr_bh')
    results_df = results_df.sort_values('p_value')
    return results_df

for cell_type in cell_types:
    df_pred = pd.read_csv(f"/home/yiminfan/projects/ctb-liyue/yiminfan/multigps_pathology/dynflow/dynflow_project/perturbation_ko_clean/results_full/forward_Gli2_{cell_type}_2/prediction.csv")
    df_pred.index = list(adata.var['gene_short_name'])
    df_pred = df_pred.sort_values(by='0', ascending=False)
    selected_genes_pred = set(df_pred.index[:100])
    
    df_real = pd.read_csv(f"/home/yiminfan/projects/ctb-liyue/yiminfan/multigps_pathology/dynflow/dynflow_project/perturbation_ko_clean/data_processing/de_tests/Gli2 KO_{cell_type}.csv", index_col=0)
    df_real = df_real[df_real['group'] == 'perturb']
    df_real['gene_short_name'] = df_real['names'].map(adata.var['gene_short_name'])
    df_real.index = list(df_real['gene_short_name'])
    df_real = df_real[df_real['pvals_adj'] < 0.05]
    df_real = df_real[df_real['logfoldchanges'] > 0]
    selected_genes_real = set(df_real.index[:100])
    
    results_pred = hypergeometric_test(selected_genes_pred, total_genes, data, gsea_pathways, adata.var['gene_short_name'])
    results_real = hypergeometric_test(selected_genes_real, total_genes, data, gsea_pathways, adata.var['gene_short_name'])
    
    results_pred['neg_log10_p'] = -np.log10(results_pred['p_value'])
    results_real['neg_log10_p'] = -np.log10(results_real['p_value'])
    
    combined = pd.merge(results_pred[['pathway', 'neg_log10_p']],results_real[['pathway', 'neg_log10_p']],on='pathway')
    combined.columns = ['pathway', 'prediction', 'groundtruth']
    combined['pathway_short'] = combined['pathway'].map(pathway_mapping)
    combined = combined.set_index('pathway_short')
    del combined['pathway']
    
    combined.to_csv(f'output/hypergeometric_tests/{cell_type}.csv')
    
    print(f"{cell_type}:")
    print(combined)

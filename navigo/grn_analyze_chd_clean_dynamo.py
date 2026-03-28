import pandas as pd
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[6]
JACOBIAN_PATH = REPO_ROOT / "data" / "grn" / "jacobian_dynamo.csv"

df_class = pd.read_csv('../gene_classification_summary.csv')
degs = pd.read_csv("Atrial#cardiomyocytes_Ventricular#cardiomyocytes_deg.csv", index_col=0)
tf_list = pd.read_csv('../mouse_tf_list_in_data.csv')
tf_genes = set(tf_list.iloc[:, 0].values)

filtered_degs = degs[degs['pvals_adj'] < 0.05]
atrial_markers = set(filtered_degs.nlargest(50, 'logfoldchanges')['names'])
ventricular_markers = set(filtered_degs.nsmallest(50, 'logfoldchanges')['names'])

jac = pd.read_csv(JACOBIAN_PATH, index_col=0)

results = []

for _, row in df_class.iterrows():
    gene = row['Gene']
    gene_type = row['classification']
    
    if gene not in jac.columns:
        print(f"Gene {gene} not in jacobian")
        continue
    
    gene_jac = jac[gene].abs()
    
    atrial_changes = gene_jac[gene_jac.index.isin(atrial_markers)].abs().mean()
    ventricular_changes = gene_jac[gene_jac.index.isin(ventricular_markers)].abs().mean()
    marker_ratio = ventricular_changes / atrial_changes if atrial_changes > 0 else 0
    
    results.append({
        'gene': gene,
        'type': gene_type,
        'is_tf': gene in tf_genes,
        'atrial_marker_change': atrial_changes,
        'ventricular_marker_change': ventricular_changes,
        'marker_v_to_a_ratio': marker_ratio,
    })
    
   
df_results = pd.DataFrame(results)
type_order = {'VSD_only': 0, 'ASD_only': 1, 'Both': 2, 'Other': 3}
df_results['type_order'] = df_results['type'].map(type_order)
df_results = df_results.sort_values('type_order').drop('type_order', axis=1)

for gene_type in ['VSD_only', 'ASD_only', 'Both', 'Other']:
    subset = df_results[df_results['type'] == gene_type]
    print(f"\n=== {gene_type} Genes (n={len(subset)}) ===")
    print(f"Atrial: {subset['atrial_marker_change'].mean():.6f}")
    print(f"Ventricular: {subset['ventricular_marker_change'].mean():.6f}")
    print(f"V/A ratio: {subset['marker_v_to_a_ratio'].mean():.6f}")

df_results.to_csv('gene_marker_change_analysis_dynamo.csv', index=False)

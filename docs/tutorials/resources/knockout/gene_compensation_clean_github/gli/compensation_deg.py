import pandas as pd
import numpy as np
import os
from scipy.stats import hypergeom

cell_types = [
    "Limb#mesenchyme#trajectory",
    "Chondrocyte#trajectory", 
    "Intermediate#mesoderm#trajectory",
    "Motor#neuron#trajectory",
    "Spinal#cord#dorsal#trajectory",
    "Osteoblast#trajectory"
]

categories = ['gli1', 'gli3', 'gli1_gli3', 'gli1_gli3_combined']

for cat in categories:
    results = []
    
    for cell_type in cell_types:
        expression_change_path = f"./output/{cell_type}/expression_change_results.csv"
        deg_path = f"/home/yiminfan/projects/ctb-liyue/yiminfan/multigps_pathology/dynflow/dynflow_project/perturbation_ko_clean/data_processing/de_tests/Gli2 KO_{cell_type}.csv"
        
        expression_change = pd.read_csv(expression_change_path, index_col=0)
        deg = pd.read_csv(deg_path)
        deg = deg[(deg['group'] == 'perturb') & (deg['pvals_adj'] < 0.05)]
        
        deg_genes = set(deg['gene_short_name'])
        total_genes = len(expression_change)
        total_deg_genes = len(deg_genes.intersection(set(expression_change.index)))
        
        same_genes = []
        opposite_genes = []
        
        for gene in expression_change.index:
            gli2 = expression_change.loc[gene, 'Gli2']
            
            if cat == 'gli1':
                target = expression_change.loc[gene, 'Gli1']
            elif cat == 'gli3':
                target = expression_change.loc[gene, 'Gli3']
            elif cat == 'gli1_gli3':
                gli1 = expression_change.loc[gene, 'Gli1']
                gli3 = expression_change.loc[gene, 'Gli3']
                if gli2 * gli1 > 0 and gli2 * gli3 > 0:
                    same_genes.append(gene)
                else:
                    opposite_genes.append(gene)
                continue
            elif cat == 'gli1_gli3_combined':
                target = expression_change.loc[gene, 'Gli1_Gli3']
            
            if cat != 'gli1_gli3':
                if gli2 * target > 0:
                    same_genes.append(gene)
                else:
                    opposite_genes.append(gene)
        
        same_count = len(same_genes)
        same_deg_count = len(set(same_genes).intersection(deg_genes))
        same_p_value = hypergeom.sf(same_deg_count - 1, total_genes, same_count, total_deg_genes)
        
        opposite_count = len(opposite_genes)
        opposite_deg_count = len(set(opposite_genes).intersection(deg_genes))
        opposite_p_value = hypergeom.sf(opposite_deg_count - 1, total_genes, opposite_count, total_deg_genes)
        
        results.append({
            'cell_type': cell_type.replace('#', '_'),
            'total_genes': total_genes,
            'deg_genes': total_deg_genes,
            'same_genes': same_count,
            'same_deg_genes': same_deg_count,
            'same_p_value': same_p_value,
            'opposite_genes': opposite_count,
            'opposite_deg_genes': opposite_deg_count,
            'opposite_p_value': opposite_p_value
        })
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(f'./output/deg_enrichment_{cat}.csv', index=False)

print("DEG enrichment analysis completed for all categories.")
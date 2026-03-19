import pandas as pd
import numpy as np
import os

pathway = pd.read_csv("./pathway_info.csv")
cell_types = [d for d in os.listdir("output") if d.endswith("#trajectory")]

gli_effects = {}

for cell_type in cell_types:
    expression_baseline = pd.read_csv(f"output/{cell_type}/expression_baseline.csv", index_col=0)
    expression_change = pd.read_csv(f"output/{cell_type}/expression_change_results.csv", index_col=0)
    
    expression_baseline = expression_baseline[expression_baseline.index.isin(pathway['gene'])]
    expression_change = expression_change[expression_change.index.isin(expression_baseline.index)]
    
    baseline_values = expression_baseline.loc[expression_change.index, '0']
    valid_genes = baseline_values > 1e-6
    expression_change = expression_change[valid_genes]
    baseline_values = baseline_values[valid_genes]
    
    percent_changes = []
    for col in expression_change.columns:
        pct = (expression_change[col].abs() / baseline_values) * 100
        percent_changes.append(pct)
    
    result = pd.concat(percent_changes, axis=1)
    result.columns = expression_change.columns
    
    col_means = result.mean().sort_values(ascending=False)
    result = result[col_means.index]
    
    result.to_csv(f"output/{cell_type}/change_by_ko_{cell_type}.csv")
    
    deg = pd.read_csv(f"/home/yiminfan/projects/ctb-liyue/yiminfan/multigps_pathology/dynflow/dynflow_project/perturbation_ko_clean/data_processing/de_tests/Gli2 KO_{cell_type}.csv")
    deg = deg[(deg['group'] == 'perturb') & (deg['pvals_adj'] < 0.05)]
    deg_count = len(deg)
    
    gli_genes = ['Gli1', 'Gli2', 'Gli3']
    gli_stats = {}
    gli_values = []
    for gli in gli_genes:
        if gli in result.columns:
            val = result[gli].mean()
            gli_stats[gli] = val
            gli_values.append(val)
            gli_rank = list(col_means.index).index(gli) + 1 if gli in col_means.index else len(col_means) + 1
            gli_stats[f'{gli}_rank'] = gli_rank
            gli_stats[f'{gli}_baseline'] = baseline_values[gli] if gli in baseline_values.index else 0
        else:
            gli_stats[gli] = 0
            gli_values.append(0)
            gli_stats[f'{gli}_rank'] = len(col_means) + 1
            gli_stats[f'{gli}_baseline'] = 0
    
    min_val = min(gli_values)
    max_val = max(gli_values)
    for gli in gli_genes:
        if max_val > min_val:
            gli_stats[f'{gli}_normalized'] = (gli_stats[gli]) / (max_val)
        else:
            gli_stats[f'{gli}_normalized'] = 0
    
    gli_stats['deg_count'] = deg_count
    gli_effects[cell_type] = gli_stats
    
    print(f"{cell_type}:")
    print(f"  DEG count: {deg_count}")
    for gli in gli_genes:
        effect = gli_stats[gli]
        norm = gli_stats[f'{gli}_normalized']
        rank = gli_stats[f'{gli}_rank']
        baseline = gli_stats[f'{gli}_baseline']
        print(f"  {gli}: {effect:.2f} (norm: {norm:.2f}, rank: {rank}, baseline: {baseline:.2f})")
    print()

gli_df = pd.DataFrame(gli_effects).T
gli_df.to_csv("output/gli_effects_by_celltype.csv")
print("GLI effects summary:")
print(gli_df)
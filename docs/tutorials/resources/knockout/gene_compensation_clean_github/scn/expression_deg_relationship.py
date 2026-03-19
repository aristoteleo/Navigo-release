import pandas as pd
import numpy as np
import os

pathway = pd.read_csv("./pathway_info.csv")
cell_types = [d for d in os.listdir("output") if d.endswith("#trajectory")]

scn_effects = {}

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
    
    result.to_csv(f"output/change_by_ko_{cell_type}.csv")
    
    deg = pd.read_csv(f"/home/yiminfan/projects/ctb-liyue/yiminfan/multigps_pathology/dynflow/dynflow_project/perturbation_ko_clean/data_processing/de_tests/Scn10a_11a DKO_{cell_type}.csv")
    deg = deg[(deg['group'] == 'perturb') & (deg['pvals_adj'] < 0.05)]
    deg_count = len(deg)
    
    scn_genes = ['Scn10a','Scn11a', 'Scn9a', 'Scn7a']
    scn_stats = {}
    scn_values = []
    for scn in scn_genes:
        if scn in result.columns:
            val = result[scn].mean()
            scn_stats[scn] = val
            scn_values.append(val)
            scn_rank = list(col_means.index).index(scn) + 1 if scn in col_means.index else len(col_means) + 1
            scn_stats[f'{scn}_rank'] = scn_rank
            scn_stats[f'{scn}_baseline'] = baseline_values[scn] if scn in baseline_values.index else 0
        else:
            scn_stats[scn] = 0
            scn_values.append(0)
            scn_stats[f'{scn}_rank'] = len(col_means) + 1
            scn_stats[f'{scn}_baseline'] = 0
    
    min_val = min(scn_values)
    max_val = max(scn_values)
    for scn in scn_genes:
        if max_val > min_val:
            scn_stats[f'{scn}_normalized'] = (scn_stats[scn]) / (max_val)
        else:
            scn_stats[f'{scn}_normalized'] = 0
    
    scn_stats['deg_count'] = deg_count
    scn_effects[cell_type] = scn_stats
    
    print(f"{cell_type}:")
    print(f"  DEG count: {deg_count}")
    for scn in scn_genes:
        effect = scn_stats[scn]
        norm = scn_stats[f'{scn}_normalized']
        rank = scn_stats[f'{scn}_rank']
        baseline = scn_stats[f'{scn}_baseline']
        print(f"  {scn}: {effect:.2f} (norm: {norm:.2f}, rank: {rank}, baseline: {baseline:.2f})")
    print()

scn_df = pd.DataFrame(scn_effects).T
scn_df.to_csv("output/scn_effects_by_celltype.csv")
print("SCN effects summary:")
print(scn_df)
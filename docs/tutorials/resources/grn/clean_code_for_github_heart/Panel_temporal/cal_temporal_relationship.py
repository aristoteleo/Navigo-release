import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CELL_TYPE = "First_heart_field"

df_cluster = pd.read_csv("../Panel_UMAP/cluster_results/gene_clusters.csv")

def find_ko_file(gene, time_range, cell_type):
    pattern = f"{gene}_neg0.0x_knockout_{cell_type}_{time_range}.csv"
    search_path = Path("/home/yiminfan/projects/ctb-liyue/yiminfan/multigps_pathology/dynflow/dynflow_project/clean_codebase_for_GRN_evaluation/heart_temporal_inference/clean/results_fhf_temporal/")
    matches = list(search_path.glob(f"**/{pattern}"))
    return matches[0] if matches else None

times = np.arange(8.5, 14.0, 0.25)
time_ranges = [f"E{t}-E{t+0.25}" for t in times]

gene_to_cluster = dict(zip(df_cluster['gene'], df_cluster['cluster']))

results = []
for time_range in time_ranges:
    time_start = float(time_range.split('-')[0][1:])
    
    for reg_gene in df_cluster['gene']:
        ko_file = find_ko_file(reg_gene, time_range, CELL_TYPE)
        if ko_file is None:
            continue
        
        try:
            df_ko = pd.read_csv(ko_file)
            for tgt_gene in df_cluster['gene']:
                row = df_ko[df_ko['gene'] == tgt_gene]
                if not row.empty:
                    results.append({
                        'time': time_start,
                        'reg_cluster': gene_to_cluster[reg_gene],
                        'tgt_cluster': gene_to_cluster[tgt_gene],
                        'effect': np.abs(row.iloc[0]['total_change'])
                    })
        except Exception as e:
            print(f"Error: {reg_gene} at {time_range}: {e}")

df_results = pd.DataFrame(results)
df_results['time_bin'] = np.floor(df_results['time'])

df_interaction = df_results.groupby(['time_bin', 'reg_cluster', 'tgt_cluster'])['effect'].mean().reset_index()
df_interaction.columns = ['time', 'reg_cluster', 'tgt_cluster', 'interaction_strength']

output_file = f'cluster_interaction_{CELL_TYPE}_binned.csv'
df_interaction.to_csv(output_file, index=False)
print(f"Saved cluster interaction to: {output_file}")

cluster_pairs = df_interaction.groupby(['reg_cluster', 'tgt_cluster']).size().reset_index()[['reg_cluster', 'tgt_cluster']]

for _, row in cluster_pairs.iterrows():
    reg_c = row['reg_cluster']
    tgt_c = row['tgt_cluster']
    
    data = df_interaction[(df_interaction['reg_cluster'] == reg_c) & 
                          (df_interaction['tgt_cluster'] == tgt_c)]
    
    plt.figure(figsize=(8, 5))
    plt.plot(data['time'], data['interaction_strength'], marker='o', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Interaction Strength')
    plt.title(f'Cluster {reg_c} -> Cluster {tgt_c}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'cluster_{reg_c}_to_{tgt_c}_temporal.png', dpi=300)
    plt.close()

print("Plots saved")
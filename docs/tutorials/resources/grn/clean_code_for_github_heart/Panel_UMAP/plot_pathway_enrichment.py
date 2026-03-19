import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap

def plot_pathway_enrichment(enrichment_file, cluster_id, output_file=None, 
                           top_n=20):
    
    plt.rcParams['font.family'] = 'Liberation Sans'
    
    df = pd.read_csv(enrichment_file).head(top_n)
    df['-log10(p)'] = -np.log10(df['p_value'])
    df['pathway_short'] = df['pathway'].str.replace('GOBP_', '').str.replace('_', ' ').str.title()
    
    df['pathway_wrapped'] = df['pathway_short'].apply(lambda x: '\n'.join(textwrap.wrap(x, width=40)))
    
    color = plt.cm.tab20(0)
    
    fig = plt.figure(figsize=(7, 10))
    ax = fig.add_axes([0.6, 0.05, 0.35, 0.9])
    
    ax.barh(range(len(df)), df['-log10(p)'].values[::-1], 
            color=color, alpha=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['pathway_wrapped'].values[::-1], fontsize=14)
    ax.set_xlabel('-log10(p-value)', fontsize=14)
    ax.set_title(f'Cluster {cluster_id}', fontsize=18, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    if output_file is None:
        output_file = f'figures/pathway_enrichment_cluster_{cluster_id}.pdf'
    
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Pathway enrichment plot for cluster {cluster_id} saved to '{output_file}'")

for i in range(4):
    plot_pathway_enrichment(
        f'cluster_results/cluster_{i}_pathway_enrichment.csv',
        cluster_id=i,
        top_n=20
    )

for i in range(4):
    plot_pathway_enrichment(
        f'dynamo_cluster_results/dynamo_cluster_{i}_pathway_enrichment.csv',
        cluster_id=i,
        output_file=f'figures/dynamo_pathway_enrichment_cluster_{i}.pdf',
        top_n=20
    )
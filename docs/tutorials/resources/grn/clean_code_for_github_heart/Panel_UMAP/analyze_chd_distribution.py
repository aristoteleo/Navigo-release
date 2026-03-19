import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def analyze_chd_cluster_distribution():
    df_class = pd.read_csv("../gene_classification_summary.csv", index_col=0)
    df_cluster = pd.read_csv("cluster_results/gene_clusters.csv")
    
    df_merged = df_cluster.merge(df_class[['CHD classification']], left_on='gene', right_index=True, how='left')
    
    chd_categories = ['Malformation of outflow tracts', 'Functional single ventricle', 
                      'Heterotaxy', 'Obstructive lesions', 'ASD', 'VSD']
    
    results = []
    for cat in chd_categories:
        cat_genes = df_merged[df_merged['CHD classification'].str.contains(cat, case=False, na=False)]
        total_cat = len(cat_genes)
        
        for cluster in sorted(df_merged['cluster'].unique()):
            count = (cat_genes['cluster'] == cluster).sum()
            results.append({'category': cat, 'cluster': cluster, 'count': count, 
                           'total': total_cat, 'percentage': 100 * count / total_cat if total_cat > 0 else 0})
    
    df_results = pd.DataFrame(results)
    
    pivot = df_results.pivot(index='category', columns='cluster', values='percentage')
    pivot.to_csv('cluster_results/chd_cluster_distribution.csv')

    # Match manuscript-style panel layout:
    # rows = clusters, cols = CHD categories (transposed from default view).
    display_order = [
        'ASD',
        'VSD',
        'Functional single ventricle',
        'Heterotaxy',
        'Malformation of outflow tracts',
        'Obstructive lesions',
    ]
    label_map = {
        'ASD': 'Atrial septal defect',
        'VSD': 'Ventricular septal defect',
        'Functional single ventricle': 'Functional single ventricle',
        'Heterotaxy': 'Heterotaxy',
        'Malformation of outflow tracts': 'Malformation of outflow tracts',
        'Obstructive lesions': 'Obstructive lesions',
    }

    plot_df = pivot.reindex(index=display_order).T
    plot_df.columns = [label_map[c] for c in plot_df.columns]
    plot_df.index = [f'Cluster {i}' for i in plot_df.index]

    # Paper-like blue heatmap: white -> tab20 blue
    tab20_blue = plt.cm.tab20.colors[0]
    blue_cmap = LinearSegmentedColormap.from_list(
        'tab20_blue_gradient',
        ['#ffffff', tab20_blue],
    )

    plt.figure(figsize=(7.2, 3.2))
    ax = sns.heatmap(
        plot_df,
        annot=True,
        fmt='.2f',
        cmap=blue_cmap,
        vmin=0,
        vmax=max(0.65, float(plot_df.values.max() / 100.0)) * 100.0,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Proportion', 'shrink': 0.92},
        annot_kws={'fontsize': 11},
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 20, 40, 60])
    cbar.set_ticklabels(['0', '0.2', '0.4', '0.6'])
    cbar.ax.set_title('Proportion', fontsize=12, pad=6)
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    # Save canonical and versioned outputs; use v2/final for manuscript-facing use.
    plt.savefig('cluster_results/chd_cluster_heatmap.png', dpi=150)
    plt.savefig('cluster_results/chd_cluster_heatmap_v2.png', dpi=150)
    plt.savefig('cluster_results/chd_cluster_heatmap_final.png', dpi=150)
    plt.close()
    
    print("\nCHD Distribution by Cluster (row sum = 100%):")
    print(pivot)
    print(f"\nResults saved to 'cluster_results/chd_cluster_distribution.csv'")
    print("Heatmap saved to:")
    print(" - cluster_results/chd_cluster_heatmap.png")
    print(" - cluster_results/chd_cluster_heatmap_v2.png")
    print(" - cluster_results/chd_cluster_heatmap_final.png")

analyze_chd_cluster_distribution()

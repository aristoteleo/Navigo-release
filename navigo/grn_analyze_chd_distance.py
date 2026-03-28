import pandas as pd
import numpy as np

def find_top_genes_matrix(distances_file='cluster_results/gene_cluster_distances.csv', top_n=5):
    df = pd.read_csv(distances_file)
    
    dist_cols = [col for col in df.columns if col.startswith('dist_to_cluster_')]
    df['mean_distance'] = df[dist_cols].mean(axis=1)
    
    result_data = []
    
    for i, col in enumerate(dist_cols):
        df_cluster = df.sort_values(col).head(top_n)
        for rank, (idx, row) in enumerate(df_cluster.iterrows(), 1):
            result_data.append({
                'Category': f'Cluster {i}',
                'Rank': rank,
                'Gene': row['gene'],
                'Distance': round(row[col], 4)
            })
    
    df_overall = df.sort_values('mean_distance').head(top_n)
    for rank, (idx, row) in enumerate(df_overall.iterrows(), 1):
        result_data.append({
            'Category': 'Overall',
            'Rank': rank,
            'Gene': row['gene'],
            'Distance': round(row['mean_distance'], 4)
        })
    
    df_result = pd.DataFrame(result_data)
    
    pivot_gene = df_result.pivot(index='Category', columns='Rank', values='Gene')
    pivot_dist = df_result.pivot(index='Category', columns='Rank', values='Distance')
    
    pivot_gene.columns = [f'Gene_Rank_{i}' for i in pivot_gene.columns]
    pivot_dist.columns = [f'Distance_Rank_{i}' for i in pivot_dist.columns]
    
    pivot_combined = pd.concat([pivot_gene, pivot_dist], axis=1)
    pivot_combined = pivot_combined.reindex(sorted(pivot_combined.columns), axis=1)
    
    pivot_combined.to_csv('cluster_results/top_genes_matrix.csv')
    
    print("\nTop genes matrix with distances:")
    print(pivot_combined)
    print("\nResults saved to cluster_results/top_genes_matrix.csv")

find_top_genes_matrix()
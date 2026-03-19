import pandas as pd
import numpy as np

def create_top_genes_distance_matrix(matrix_file='cluster_results/top_genes_matrix.csv', 
                                     distances_file='cluster_results/gene_cluster_distances.csv',
                                     output_file='cluster_results/top_genes_distance_matrix.csv'):
    df_matrix = pd.read_csv(matrix_file, index_col=0)
    df_dist = pd.read_csv(distances_file)
    
    gene_cols = [col for col in df_matrix.columns if col.startswith('Gene_Rank_')]
    
    all_genes = []
    for col in gene_cols:
        all_genes.extend(df_matrix[col].dropna().tolist())
    unique_genes = list(dict.fromkeys(all_genes))
    
    dist_cols = [col for col in df_dist.columns if col.startswith('dist_to_cluster_')]
    n_clusters = len(dist_cols)
    
    heatmap_data = []
    for gene in unique_genes:
        gene_row = df_dist[df_dist['gene'] == gene]
        if len(gene_row) > 0:
            distances = [gene_row[col].values[0] for col in dist_cols]
            heatmap_data.append(distances)
    
    heatmap_df = pd.DataFrame(heatmap_data, index=unique_genes, 
                             columns=[f'Cluster_{i}' for i in range(n_clusters)])
    
    heatmap_df['min_cluster'] = heatmap_df.idxmin(axis=1)
    heatmap_df['min_distance'] = heatmap_df.iloc[:, :n_clusters].min(axis=1)
    
    heatmap_df = heatmap_df.sort_values(['min_cluster', 'min_distance'])
    
    heatmap_df = heatmap_df.drop(['min_cluster', 'min_distance'], axis=1)
    
    heatmap_df = heatmap_df.T
    
    heatmap_df.to_csv(output_file)
    
    print(f"Distance matrix saved to '{output_file}'")
    print(f"Shape: {heatmap_df.shape}")
    print("\nPreview:")
    print(heatmap_df.iloc[:, :10])

create_top_genes_distance_matrix()
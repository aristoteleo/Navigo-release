import pandas as pd
import numpy as np
from glob import glob
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
from collections import Counter
from adjustText import adjust_text

def plot_interaction_network(ko_path, n_clusters=4, n_top_interactions=100, top_percent=50, output_file='interaction_network.pdf'):
    ko_files = glob(f"{ko_path}/*_neg0.0x_knockout_First_heart_field.csv")
    
    response_vectors, gene_names = [], []
    for ko_file in ko_files:
        ko_gene = os.path.basename(ko_file).replace('_neg0.0x_knockout_First_heart_field.csv', '')
        df_ko = pd.read_csv(ko_file).sort_values('gene').set_index('gene')
        response_vectors.append(df_ko['total_change'].values)
        gene_names.append(ko_gene)
    
    response_matrix = np.array(response_vectors)
    response_matrix = response_matrix / np.linalg.norm(response_matrix, axis=1, keepdims=True)
    
    pca = PCA(n_components=50)
    response_pca = pca.fit_transform(response_matrix)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(response_pca)
    
    reducer = umap.UMAP(n_components=2, random_state=42)
    response_umap = reducer.fit_transform(response_matrix)
    
    adata = anndata.AnnData(X=response_matrix)
    adata.obs_names = gene_names
    adata.obs['Cluster'] = [f'Cluster {i}' for i in clusters]
    adata.obsm['X_umap'] = response_umap
    
    gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}
    gene_to_cluster = {gene: clusters[i] for i, gene in enumerate(gene_names)}
    
    interaction_strengths = {}
    cluster_interactions = np.zeros((n_clusters, n_clusters))
    
    for ko_file in ko_files:
        ko_gene = os.path.basename(ko_file).replace('_neg0.0x_knockout_First_heart_field.csv', '')
        df_ko = pd.read_csv(ko_file).sort_values('gene').set_index('gene')
        
        for target_gene in gene_names:
            if target_gene in df_ko.index and target_gene != ko_gene:
                strength = abs(df_ko.loc[target_gene, 'total_change'])
                pair = tuple(sorted([ko_gene, target_gene]))
                interaction_strengths[pair] = interaction_strengths.get(pair, 0) + strength
                
                c1, c2 = gene_to_cluster[ko_gene], gene_to_cluster[target_gene]
                cluster_interactions[c1, c2] += strength
                if c1 != c2:
                    cluster_interactions[c2, c1] += strength
    
    print("Cluster Interaction Matrix:")
    print(cluster_interactions)
    print("\nStrongest cluster interactions:")
    for i in range(n_clusters):
        for j in range(i, n_clusters):
            print(f"Cluster {i} <-> Cluster {j}: {cluster_interactions[i, j]:.2f}")
    
    sorted_interactions = sorted(interaction_strengths.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {min(20, len(sorted_interactions))} strongest gene pairs:")
    gene_counter = Counter()
    for i, ((gene_a, gene_b), strength) in enumerate(sorted_interactions[:20], 1):
        c1, c2 = gene_to_cluster[gene_a], gene_to_cluster[gene_b]
        print(f"{i}. {gene_a} (C{c1}) <-> {gene_b} (C{c2}): {strength:.4f}")
        gene_counter[gene_a] += 1
        gene_counter[gene_b] += 1
    
    top5_genes = [gene for gene, _ in gene_counter.most_common(10)]
    print(f"\nTop 5 most frequent genes in top 20 pairs:")
    for gene, count in gene_counter.most_common(5):
        print(f"{gene}: {count} times")
    
    top_interactions = sorted_interactions[:n_top_interactions]
    
    threshold_idx = int(len(top_interactions) * top_percent / 100)
    strength_threshold = sorted([s for _, s in top_interactions], reverse=True)[threshold_idx - 1]
    top_interactions = [(p, s) for p, s in top_interactions if s >= strength_threshold]
    
    print(f"\nFiltering top {top_percent}% of edges: {len(top_interactions)} edges remaining")
    
    strengths = np.array([s for _, s in top_interactions])
    strengths_log = np.log1p(strengths)
    max_log, min_log = strengths_log.max(), strengths_log.min()
    
    plt.rcParams['font.family'] = 'Liberation Sans'
    
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    
    for i, ((gene_a, gene_b), strength) in enumerate(top_interactions):
        idx_a, idx_b = gene_to_idx[gene_a], gene_to_idx[gene_b]
        pos_a, pos_b = response_umap[idx_a], response_umap[idx_b]
        normalized = (strengths_log[i] - min_log) / (max_log - min_log)
        width = 0 + normalized * 5
        ax.plot([pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]], 
                color='gray', linewidth=width, alpha=0.3, zorder=1)
    
    sc.pl.umap(adata, color='Cluster', palette='tab20', frameon=False, 
               size=200, legend_loc=None, ax=ax, title='', show=False)
    
    texts = []
    for gene in top5_genes:
        idx = gene_to_idx[gene]
        pos = response_umap[idx]
        txt = ax.text(pos[0], pos[1], gene, fontsize=16, style='italic', color='black',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.7))
        texts.append(txt)
    
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
    
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

plot_interaction_network("../results", n_clusters=4, n_top_interactions=100, top_percent=50, output_file='interaction_network.pdf')
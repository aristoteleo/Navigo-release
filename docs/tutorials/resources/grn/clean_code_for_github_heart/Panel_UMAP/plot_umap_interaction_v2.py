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

def plot_circular_network(ko_path, n_clusters=4, n_top_interactions=100, min_percentile=50, output_file='circular_network.pdf'):
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
    
    cluster_colors = dict(zip([f'Cluster {i}' for i in range(n_clusters)], 
                             sc.pl.palettes.default_20[:n_clusters]))
    
    gene_to_cluster = {gene: clusters[i] for i, gene in enumerate(gene_names)}
    cluster_genes = {i: [] for i in range(n_clusters)}
    for gene, cluster in gene_to_cluster.items():
        cluster_genes[cluster].append(gene)
    
    gene_positions = {}
    angle = 0
    
    for cluster_id in range(n_clusters):
        genes = sorted(cluster_genes[cluster_id])
        for gene in genes:
            gene_positions[gene] = (np.cos(angle), np.sin(angle))
            angle += 2 * np.pi / len(gene_names)
    
    interaction_strengths = {}
    for ko_file in ko_files:
        ko_gene = os.path.basename(ko_file).replace('_neg0.0x_knockout_First_heart_field.csv', '')
        df_ko = pd.read_csv(ko_file).sort_values('gene').set_index('gene')
        
        for target_gene in gene_names:
            if target_gene in df_ko.index and target_gene != ko_gene:
                strength = abs(df_ko.loc[target_gene, 'total_change'])
                pair = tuple(sorted([ko_gene, target_gene]))
                interaction_strengths[pair] = interaction_strengths.get(pair, 0) + strength
    
    sorted_interactions = sorted(interaction_strengths.items(), key=lambda x: x[1], reverse=True)
    top_interactions = sorted_interactions[:n_top_interactions]
    
    strengths = np.array([s for _, s in top_interactions])
    threshold = np.percentile(strengths, min_percentile)
    filtered_interactions = [(pair, s) for pair, s in top_interactions if s >= threshold]
    
    gene_counter = Counter()
    for (gene_a, gene_b), _ in sorted_interactions[:20]:
        gene_counter[gene_a] += 1
        gene_counter[gene_b] += 1
    top_genes = [gene for gene, _ in gene_counter.most_common(10)]
    
    strengths_filtered = np.array([s for _, s in filtered_interactions])
    strengths_log = np.log1p(strengths_filtered)
    max_log, min_log = strengths_log.max(), strengths_log.min()
    
    plt.rcParams['font.family'] = 'Liberation Sans'
    
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    
    for i, ((gene_a, gene_b), strength) in enumerate(filtered_interactions):
        pos_a, pos_b = gene_positions[gene_a], gene_positions[gene_b]
        normalized = (strengths_log[i] - min_log) / (max_log - min_log)
        width = 0.1 + normalized * 3
        ax.plot([pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]], 
                color='gray', linewidth=width, alpha=0.3, zorder=1)
    
    for cluster_id in range(n_clusters):
        genes = cluster_genes[cluster_id]
        x = [gene_positions[g][0] for g in genes]
        y = [gene_positions[g][1] for g in genes]
        color = cluster_colors[f'Cluster {cluster_id}']
        ax.scatter(x, y, c=[color], s=100, edgecolors='black', linewidths=1.5, zorder=2)
    
    texts = []
    for gene in top_genes:
        pos = gene_positions[gene]
        txt = ax.text(pos[0], pos[1], gene, fontsize=12, style='italic', 
                     ha='center', va='center', weight='bold')
        texts.append(txt)
    
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

plot_circular_network("../results", n_clusters=4, n_top_interactions=100, min_percentile=70, output_file='circular_network.pdf')
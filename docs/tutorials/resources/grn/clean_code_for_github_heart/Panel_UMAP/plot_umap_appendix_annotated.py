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
from adjustText import adjust_text

def plot_umap(ko_path, n_clusters=4, output_file='umap_clusters.png', with_legend=True):
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
    
    plt.rcParams['font.family'] = 'Liberation Sans'
    sc.set_figure_params(transparent=False, facecolor='white')
    
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    
    legend_loc = 'right margin' if with_legend else 'none'
    sc.pl.umap(adata, color='Cluster', palette='tab20', frameon=False, 
               size=200, legend_loc=legend_loc, ax=ax, title='',
               add_outline=False, show=False)
    
    texts = [ax.text(response_umap[i, 0], response_umap[i, 1], gene, 
                     fontsize=8, ha='center', va='center', style='italic') 
             for i, gene in enumerate(gene_names)]
    
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    ax.set_aspect('equal')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

plot_umap("../results", n_clusters=4, output_file='umap_clusters_annotated.pdf', with_legend=True)
plot_umap("../results", n_clusters=4, output_file='umap_clusters_no_legend_annotated.pdf', with_legend=False)
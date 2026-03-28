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
    sc.set_figure_params(transparent=True, facecolor='none')
    
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_alpha(0)
    
    if with_legend:
        sc.pl.umap(adata, color='Cluster', palette='tab20', frameon=False, 
                   size=200, legend_loc='right margin', ax=ax, title='',
                   save=output_file.replace('.png', '.png'), show=False)
    else:
        sc.pl.umap(adata, color='Cluster', palette='tab20', frameon=False, 
                   size=200, legend_loc='none', ax=ax, title='',
                   save=output_file.replace('.png', '.png'), show=False)
    
    ax.set_aspect('equal')

plot_umap("../results", n_clusters=4, output_file='umap_clusters.png', with_legend=True)
plot_umap("../results", n_clusters=4, output_file='umap_clusters_no_legend.png', with_legend=False)
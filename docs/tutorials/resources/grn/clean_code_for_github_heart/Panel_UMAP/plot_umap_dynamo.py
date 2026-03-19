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

def plot_dynamo_umap_with_both_clusters(dynflow_ko_path, dynamo_jacobian_path, 
                                        n_clusters=4, output_prefix='dynamo_umap'):
    ko_files = glob(f"{dynflow_ko_path}/*_neg0.0x_knockout_First_heart_field.csv")
    
    response_vectors, gene_names = [], []
    for ko_file in ko_files:
        ko_gene = os.path.basename(ko_file).replace('_neg0.0x_knockout_First_heart_field.csv', '')
        df_ko = pd.read_csv(ko_file).sort_values('gene').set_index('gene')
        response_vectors.append(df_ko['total_change'].values)
        gene_names.append(ko_gene)
    
    response_matrix = np.array(response_vectors)
    response_matrix = response_matrix / np.linalg.norm(response_matrix, axis=1, keepdims=True)
    
    pca_dynflow = PCA(n_components=50)
    response_pca = pca_dynflow.fit_transform(response_matrix)
    
    kmeans_dynflow = KMeans(n_clusters=n_clusters, random_state=42)
    dynflow_clusters = kmeans_dynflow.fit_predict(response_pca)
    
    df_jacobian = pd.read_csv(dynamo_jacobian_path, index_col=0)
    df_jacobian = df_jacobian[gene_names]
    
    dynamo_matrix = df_jacobian.values.T
    dynamo_matrix = dynamo_matrix / np.linalg.norm(dynamo_matrix, axis=1, keepdims=True)
    
    pca_dynamo = PCA(n_components=50)
    dynamo_pca = pca_dynamo.fit_transform(dynamo_matrix)
    
    kmeans_dynamo = KMeans(n_clusters=n_clusters, random_state=42)
    dynamo_clusters = kmeans_dynamo.fit_predict(dynamo_pca)
    
    reducer = umap.UMAP(n_components=2, random_state=42)
    dynamo_umap = reducer.fit_transform(dynamo_matrix)
    
    plt.rcParams['font.family'] = 'Liberation Sans'
    sc.set_figure_params(transparent=False, facecolor='white')
    
    adata_dynflow = anndata.AnnData(X=dynamo_matrix)
    adata_dynflow.obs_names = gene_names
    adata_dynflow.obs['Cluster'] = [f'Cluster {i}' for i in dynflow_clusters]
    adata_dynflow.obsm['X_umap'] = dynamo_umap
    
    fig, ax = plt.subplots(figsize=(6, 6))
    sc.pl.umap(adata_dynflow, color='Cluster', palette='tab20', frameon=False, 
               size=200, legend_loc='right margin', ax=ax, title='',
               save=f'{output_prefix}_dynflow_clusters_with_legend.png', show=False)
    ax.set_aspect('equal')
    
    fig, ax = plt.subplots(figsize=(6, 6))
    sc.pl.umap(adata_dynflow, color='Cluster', palette='tab20', frameon=False, 
               size=200, legend_loc='none', ax=ax, title='',
               save=f'{output_prefix}_dynflow_clusters_no_legend.png', show=False)
    ax.set_aspect('equal')
    
    adata_dynamo = anndata.AnnData(X=dynamo_matrix)
    adata_dynamo.obs_names = gene_names
    adata_dynamo.obs['Cluster'] = [f'Cluster {i}' for i in dynamo_clusters]
    adata_dynamo.obsm['X_umap'] = dynamo_umap
    
    fig, ax = plt.subplots(figsize=(6, 6))
    sc.pl.umap(adata_dynamo, color='Cluster', palette='tab20', frameon=False, 
               size=200, legend_loc='right margin', ax=ax, title='',
               save=f'{output_prefix}_dynamo_clusters_with_legend.png', show=False)
    ax.set_aspect('equal')
    
    fig, ax = plt.subplots(figsize=(6, 6))
    sc.pl.umap(adata_dynamo, color='Cluster', palette='tab20', frameon=False, 
               size=200, legend_loc='none', ax=ax, title='',
               save=f'{output_prefix}_dynamo_clusters_no_legend.png', show=False)
    ax.set_aspect('equal')

plot_dynamo_umap_with_both_clusters(
    "../results",
    "/home/yiminfan/projects/ctb-liyue/yiminfan/multigps_pathology/dynflow/dynflow_project/clean_codebase_for_GRN_evaluation/heart_temporal_inference/clean_analysis_key_tfs/clean_chd/analysis_chd_disease_genes_pattern/jacobian_dynamo2.csv",
    n_clusters=4,
    output_prefix='figures/umapfigures'
)
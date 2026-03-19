
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def find_optimal_clusters_dynamo(jacobian_path, ko_path, max_k=10):
    ko_files = glob(f"{ko_path}/*_neg0.0x_knockout_First_heart_field.csv")
    
    gene_names = []
    for ko_file in ko_files:
        ko_gene = ko_file.split('/')[-1].replace('_neg0.0x_knockout_First_heart_field.csv', '')
        gene_names.append(ko_gene)
    
    df_jacobian = pd.read_csv(jacobian_path, index_col=0)
    df_jacobian = df_jacobian[gene_names]
    
    response_matrix = df_jacobian.values
    response_matrix = response_matrix / np.linalg.norm(response_matrix, axis=1, keepdims=True)
    
    pca = PCA(n_components=50)
    response_pca = pca.fit_transform(response_matrix.T)
    
    inertias = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(response_pca)
        inertias.append(km.inertia_)
    
    plt.rcParams['font.family'] = 'Liberation Sans'
    plt.figure(figsize=(5, 5))
    plt.plot(range(2, max_k + 1), inertias, 'o-', linewidth=2, markersize=10)
    plt.xlabel('k', fontsize=20)
    plt.ylabel('Inertia', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("elbow_method_dynamo.pdf")
    plt.close()

find_optimal_clusters_dynamo(
    "/home/yiminfan/projects/ctb-liyue/yiminfan/multigps_pathology/dynflow/dynflow_project/clean_codebase_for_GRN_evaluation/heart_temporal_inference/clean_analysis_key_tfs/clean_chd/analysis_chd_disease_genes_pattern/jacobian_dynamo2.csv",
    "../results"
)
import pandas as pd
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[6]
MSIGDB_PATH = REPO_ROOT / "data" / "grn" / "msigdb_mouse_v2025_1.json"

def analyze_cluster_characteristics(ko_path, msigdb_path, n_clusters=3, n_top_genes=200, output_dir='cluster_results'):
    os.makedirs(output_dir, exist_ok=True)
    
    ko_files = glob(f"{ko_path}/*_neg0.0x_knockout_First_heart_field.csv")
    
    response_vectors, gene_names, all_genes = [], [], None
    for ko_file in ko_files:
        ko_gene = os.path.basename(ko_file).replace('_neg0.0x_knockout_First_heart_field.csv', '')
        df_ko = pd.read_csv(ko_file).sort_values('gene').set_index('gene')
        if all_genes is None:
            all_genes = df_ko.index.tolist()
        response_vectors.append(df_ko['total_change'].values)
        gene_names.append(ko_gene)
    
    response_matrix = np.array(response_vectors)
    response_matrix = response_matrix / np.linalg.norm(response_matrix, axis=1, keepdims=True)
    
    pca = PCA(n_components=50)
    response_pca = pca.fit_transform(response_matrix)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(response_pca)
    
    distances = np.zeros((len(gene_names), n_clusters))
    for i in range(n_clusters):
        distances[:, i] = np.linalg.norm(response_pca - kmeans.cluster_centers_[i], axis=1)
    
    df_distances = pd.DataFrame(distances, columns=[f'dist_to_cluster_{i}' for i in range(n_clusters)])
    df_distances.insert(0, 'gene', gene_names)
    df_distances.insert(1, 'assigned_cluster', clusters)
    df_distances.to_csv(f'{output_dir}/gene_cluster_distances.csv', index=False)
    print(f"\nCluster distances saved to '{output_dir}/gene_cluster_distances.csv'")
    
    df_clusters = pd.DataFrame({'gene': gene_names, 'cluster': clusters})
    df_clusters.to_csv(f'{output_dir}/gene_clusters.csv', index=False)
    print(f"Cluster assignments saved to '{output_dir}/gene_clusters.csv'")
    
    with open(msigdb_path, 'r') as f:
        msigdb = json.load(f)
    
    pathways = {pw: pw for pw in msigdb if "GOBP" in pw}
    
    cluster_results = []
    
    for c in range(n_clusters):
        mask = clusters == c
        cluster_genes = np.array(gene_names)[mask]
        mean_response = response_matrix[mask].mean(axis=0)
        
        top_down_idx = np.argsort(mean_response)[:n_top_genes]
        top_down_genes = set([all_genes[i] for i in top_down_idx])
        
        enrichment = []
        for pw, pw_name in pathways.items():
            pw_genes = set(msigdb[pw]['geneSymbols'])
            overlap = pw_genes & top_down_genes
            n = len(pw_genes & set(all_genes))
            k = len(overlap)
            
            if n < 5:
                continue
            
            _, p_val = stats.fisher_exact([[k, n_top_genes-k], [n-k, len(all_genes)-n_top_genes-n+k]], 
                                         alternative='greater')
            
            enrichment.append({'pathway': pw_name, 'overlap': k, 'pathway_size': n, 
                             'p_value': p_val, 'enrichment': k/n if n > 0 else 0})
        
        df_enrich = pd.DataFrame(enrichment)
        df_enrich = df_enrich[df_enrich['p_value'] < 0.05].sort_values('p_value')
        
        print(f"\nCluster {c}: Top enriched pathways in downregulated genes:")
        for _, row in df_enrich.head(15).iterrows():
            print(f"  {row['pathway']}: {row['overlap']}/{row['pathway_size']} (p={row['p_value']:.3e})")
        
        df_enrich.head(50).to_csv(f'{output_dir}/cluster_{c}_pathway_enrichment.csv', index=False)
        df_enrich.to_csv(f'{output_dir}/cluster_{c}_pathway_enrichment_full.csv', index=False)

        cluster_results.append({
            'cluster': c,
            'ko_genes': cluster_genes,
            'mean_response': mean_response,
            'enrichment': df_enrich
        })
    
    print(f"\nAll results saved to '{output_dir}/'")
    return cluster_results

results = analyze_cluster_characteristics("../results", str(MSIGDB_PATH), n_clusters=4)

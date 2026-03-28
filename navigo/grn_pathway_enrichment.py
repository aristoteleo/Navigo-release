import pandas as pd
from scipy import stats
from glob import glob
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[6]
MSIGDB_PATH = REPO_ROOT / "data" / "grn" / "msigdb_mouse_v2025_1.json"

def analyze_pathway_enrichment(classification_term, msigdb_path, ko_path):
    df = pd.read_csv("../gene_classification_summary.csv", index_col=0)
    df_with = df[df['CHD classification'].str.contains(classification_term, case=False, na=False)]
    df_without = df[~df['CHD classification'].str.contains(classification_term, case=False, na=False)]
    
    with_genes, without_genes = set(df_with.index), set(df_without.index)
    
    with open(msigdb_path, 'r') as f:
        msigdb = json.load(f)
    
    ko_files = glob(f"{ko_path}/*_neg0.0x_knockout_First_heart_field.csv")
    pathways = {pw: pw.replace('GOBP_', '').replace('_', ' ') for pw in msigdb if pw.startswith('GOBP_')}
    
    enrich_data = []
    for ko_file in ko_files:
        ko_gene = os.path.basename(ko_file).replace('_neg0.0x_knockout_First_heart_field.csv', '')
        group = 'with' if ko_gene in with_genes else ('without' if ko_gene in without_genes else None)
        if not group:
            continue
        
        df_ko = pd.read_csv(ko_file).sort_values('total_change')
        all_ko_genes = set(df_ko['gene'])
        bottom_genes = set(df_ko.head(200)['gene'])
        M, N = len(all_ko_genes), len(bottom_genes)
        
        for pw, pw_name in pathways.items():
            pw_genes = set(msigdb[pw]['geneSymbols'])
            n, k = len(pw_genes & all_ko_genes), len(pw_genes & bottom_genes)
            if n and N:
                _, p_val = stats.fisher_exact([[k, N-k], [n-k, M-N-n+k]], alternative='greater')
                enrich_data.append({'ko_gene': ko_gene, 'group': group, 'pathway': pw_name, 
                                    'overlap': k, 'bottom_size': N, 'pathway_size': n, 'p_value': p_val})
    
    prefix = classification_term.lower().replace(' ', '_')
    df_enrich = pd.DataFrame(enrich_data)    
    df_sig = df_enrich[df_enrich['p_value'] < 0.05]
    comp = df_sig.groupby(['pathway', 'group'])['ko_gene'].nunique().unstack(fill_value=0)
    comp['with_ratio'] = comp.get('with', 0) / len(with_genes)
    comp['without_ratio'] = comp.get('without', 0) / len(without_genes)
    comp['ratio_diff'] = comp['with_ratio'] - comp['without_ratio']
    comp = comp.sort_values('ratio_diff', ascending=False)
    
    comp.head(30).to_csv(f'pathway_enrichment/pathway_top30_{prefix}.csv')

for term in ['Malformation of outflow tracts', 'Functional single ventricle', 
             'Heterotaxy', 'Obstructive lesions', 'ASD', 'VSD']:
    analyze_pathway_enrichment(term, str(MSIGDB_PATH), '../results')

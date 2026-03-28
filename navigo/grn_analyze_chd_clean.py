import pandas as pd
import os

df_class = pd.read_csv('../gene_classification_summary.csv')
degs = pd.read_csv("Atrial#cardiomyocytes_Ventricular#cardiomyocytes_deg.csv", index_col=0)
tf_list = pd.read_csv('../mouse_tf_list_in_data.csv')
tf_genes = set(tf_list.iloc[:, 0].values)

filtered_degs = degs[degs['pvals_adj'] < 0.05]
atrial_markers = set(filtered_degs.nlargest(50, 'logfoldchanges')['names'])
ventricular_markers = set(filtered_degs.nsmallest(50, 'logfoldchanges')['names'])

results = []

for _, row in df_class.iterrows():
    gene = row['Gene']
    gene_type = row['classification']
    
    try:
        df_ko = pd.read_csv(f"../results/{gene}_neg0.0x_knockout_First_heart_field.csv")
        
        atrial_changes = df_ko[df_ko['gene'].isin(atrial_markers)]['total_change'].abs().mean()
        ventricular_changes = df_ko[df_ko['gene'].isin(ventricular_markers)]['total_change'].abs().mean()
        marker_ratio = ventricular_changes / atrial_changes if atrial_changes > 0 else 0
        
        results.append({
            'gene': gene,
            'type': gene_type,
            'is_tf': gene in tf_genes,
            'atrial_marker_change': atrial_changes,
            'ventricular_marker_change': ventricular_changes,
            'marker_v_to_a_ratio': marker_ratio,
        })
    except FileNotFoundError:
        print(f"File not found for {gene}")

df_results = pd.DataFrame(results)

type_order = {'VSD_only': 0, 'ASD_only': 1, 'Both': 2, 'Other': 3}
df_results['type_order'] = df_results['type'].map(type_order)
df_results = df_results.sort_values('type_order').drop('type_order', axis=1)

for gene_type in ['VSD_only', 'ASD_only', 'Both', 'Other']:
    subset = df_results[df_results['type'] == gene_type]
    print(f"\n=== {gene_type} Genes (n={len(subset)}) ===")
    print(f"Atrial marker change: {subset['atrial_marker_change'].mean():.6f}")
    print(f"Ventricular marker change: {subset['ventricular_marker_change'].mean():.6f}")
    print(f"Marker V/A ratio: {subset['marker_v_to_a_ratio'].mean():.6f}")

df_results.to_csv('gene_marker_change_analysis.csv', index=False)

os.makedirs('plot_data', exist_ok=True)

for _, row in df_results.iterrows():
    gene = row['gene']
    category = row['type']
    
    df_ko = pd.read_csv(f"../results/{gene}_neg0.0x_knockout_First_heart_field.csv")
    
    plot_data = []
    
    for marker in atrial_markers:
        if marker in df_ko['gene'].values:
            change_val = df_ko[df_ko['gene'] == marker]['total_change'].values[0]
            plot_data.append({
                'marker': marker,
                'marker_type': 'Atrial',
                'abs_change': abs(change_val)
            })
    
    for marker in ventricular_markers:
        if marker in df_ko['gene'].values:
            change_val = df_ko[df_ko['gene'] == marker]['total_change'].values[0]
            plot_data.append({
                'marker': marker,
                'marker_type': 'Ventricular',
                'abs_change': abs(change_val)
            })
    
    df_plot = pd.DataFrame(plot_data)
    df_plot = df_plot.nlargest(10, 'abs_change')
    df_plot.to_csv(f'plot_data/{gene}_{category}_plot_data.csv', index=False)


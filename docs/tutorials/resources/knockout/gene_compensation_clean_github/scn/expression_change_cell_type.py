import pandas as pd
import numpy as np
import os
import glob
import anndata
import torch

cell_types = [
    "Limb#mesenchyme#trajectory",
    "Chondrocyte#trajectory", 
    "Intermediate#mesoderm#trajectory",
    "Motor#neuron#trajectory",
    "Myoblast#trajectory",
    "Myotube#trajectory",
    "Spinal#cord#dorsal#trajectory",
    "Osteoblast#trajectory"
]

configs = [
    ("Scn10a", "Scn10a"),
    ("Scn11a", "Scn11a"), 
    ("Scn7a", "Scn7a"), 
    ("Scn9a", "Scn9a"), 
    ("Scn10a_Scn11a", "Scn10a Scn11a"),
    ("Scn10a_Scn7a", "Scn10a Scn7a"),
    ("Scn10a_Scn9a", "Scn10a Scn9a"),
    ("Scn11a_Scn7a", "Scn11a Scn7a"),
    ("Scn11a_Scn9a", "Scn11a Scn9a"),
    ("Scn7a_Scn9a", "Scn7a Scn9a"),
    ("Scn10a_Scn11a_Scn7a","Scn10a Scn11a Scn7a"),
    ("Scn10a_Scn11a_Scn9a","Scn10a Scn11a Scn9a"),
    ("Scn10a_Scn7a_Scn9a","Scn10a Scn7a Scn9a"),
    ("Scn11a_Scn7a_Scn9a","Scn11a Scn7a Scn9a"),
]

for cell_type in cell_types:
    print(f"Processing {cell_type}...")
    
    adata = anndata.read(f"/home/yiminfan/projects/ctb-liyue/yiminfan/multigps_pathology/dynflow/dynflow_project/perturbation_ko_clean/data_processing/data_store/{cell_type}.h5ad")
    adata = adata[adata.obs['day'].isin(['E12.5','E12.75','E13.0','E13.25','E13.5'])].copy()
    
    data = torch.tensor(np.concatenate([adata.layers['Ms'], adata.layers['Mu']], axis=1), dtype=torch.float32)
    data = (data - data.min(axis=0).values) / (data.max(axis=0).values - data.min(axis=0).values + 1e-7)
    
    data = data[:,0:3902] + data[:,3902:]
    data = data.mean(axis=0)
    data = pd.DataFrame(data)
    data.index = list(adata.var['gene_short_name'])
    
    os.makedirs(f"output/{cell_type}", exist_ok=True)
    data.to_csv(f"output/{cell_type}/expression_baseline.csv")
    
    results = {}
    cell_type_suffix = cell_type + "_2"

    for config_name, _ in configs:
        file_path = f"/home/yiminfan/projects/ctb-liyue/yiminfan/multigps_pathology/dynflow/dynflow_project/perturbation_ko_clean/results_full/forward_{config_name}_{cell_type_suffix}/prediction.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df.index = list(adata.var['gene_short_name'])
            results[config_name] = df['0']
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"output/{cell_type}/expression_change_results.csv")
    

print("Processing completed for all cell types.")
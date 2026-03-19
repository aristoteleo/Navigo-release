import json
import pandas as pd
import anndata
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
MSIGDB_PATH = REPO_ROOT / "data" / "knockout" / "msigdb_mouse_v2025_1.json"

with open(MSIGDB_PATH) as f:
    data = json.load(f)

pathways = [
    "GOBP_NEURONAL_ACTION_POTENTIAL",
    "GOBP_REGULATION_OF_MEMBRANE_POTENTIAL",
    "GOBP_REGULATION_OF_MEMBRANE_DEPOLARIZATION",
    "GOBP_SENSORY_PERCEPTION_OF_PAIN",
    "REACTOME_NEURONAL_SYSTEM",
    "REACTOME_TRANSMISSION_ACROSS_CHEMICAL_SYNAPSES",
    'GOMF_VOLTAGE_GATED_SODIUM_CHANNEL_ACTIVITY',
    'GOBP_REGULATION_OF_VOLTAGE_GATED_SODIUM_CHANNEL_ACTIVITY',
    'GOBP_ACTION_POTENTIAL_PROPAGATION'
]

gene_mapping = pd.read_csv("../public/gene_mapping.csv")
pathway_gene_dict = {}
for pathway in pathways:
    pathway_genes = set(data[pathway]['geneSymbols']).intersection(set(gene_mapping['gene_short_name']))
    for gene in pathway_genes:
        if gene not in pathway_gene_dict:
            pathway_gene_dict[gene] = []
        pathway_gene_dict[gene].append(pathway)

pathway_genes = pd.DataFrame([
    [gene, ','.join(pathways)] for gene, pathways in pathway_gene_dict.items()
], columns=['gene', 'pathway'])


pathway_genes.to_csv("pathway_info.csv")

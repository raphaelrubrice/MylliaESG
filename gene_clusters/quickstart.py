"""
Quick-Start: Multi-View Gene Clustering
========================================

This script demonstrates how to use the multiview gene clustering module
and how to consume its outputs for downstream modeling (e.g., a gene-group-aware
autoencoder or transformer).

Run this after installing:
    pip install msigdbr pandas numpy scipy python-igraph leidenalg

For ESM-2 view (optional):
    pip install fair-esm torch mygene
"""

import numpy as np
import pandas as pd

# ============================================================
# STEP 1: Prepare your gene list
# ============================================================

# Option A: From your AnnData object
# import anndata
# adata = anndata.read_h5ad("your_data.h5ad")
# genes = adata.var_names.tolist()

# Option B: From a text file (one gene per line)
# with open("gene_list.txt") as f:
#     genes = [line.strip() for line in f if line.strip()]

# Option C: Dummy example for demonstration
genes = ["TP53", "BRCA1", "EGFR", "MYC", "KRAS"]  # replace with your 5127 genes


# ============================================================
# STEP 2: Run the clustering pipeline
# ============================================================

from multiview_gene_clusters import run_multiview_clustering

results = run_multiview_clustering(
    genes=genes,
    output_dir="gene_clusters",
    # ---- Annotation-based views (always available) ----
    # These only need `msigdbr`, no GPU, no network after initial download
    # ---- ESM-2 view ----
    skip_esm=True,              # Set False if you have torch + esm installed
    esm_device="cpu",           # or "cuda" for GPU
    esm_model="esm2_t33_650M_UR50D",  # or "esm2_t12_35M_UR50D" for faster/lighter
    # ---- Co-expression view ----
    adata_path=None,            # Set to "your_data.h5ad" to enable
    skip_coexpr=True,
    # ---- Clustering parameters ----
    k=15,                       # kNN neighbors (try 10-30)
    resolution=1.0,             # Leiden resolution (higher = more clusters)
    # ---- Caching ----
    cache_dir="gene_clusters/cache",  # saves similarity matrices for re-runs
)


# ============================================================
# STEP 3: Inspect the results
# ============================================================

# Results is a dict: view_name -> DataFrame[gene, cluster, view]
for view_name, df in results.items():
    n_clusters = df["cluster"].nunique()
    print(f"\n{view_name}: {n_clusters} clusters")
    print(df["cluster"].value_counts().head(10))

# Load the combined table
combined = pd.read_csv("gene_clusters/clusters_all_views.csv")
print("\nCombined table:")
print(combined.head(20))


# ============================================================
# STEP 4: Use clusters as inductive bias for modeling
# ============================================================
#
# Below are code patterns showing how to integrate gene clusters
# into your model architecture.

import json

with open("gene_clusters/cluster_gene_lists.json") as f:
    cluster_gene_lists = json.load(f)

# --- Pattern A: Gene-group-aware encoder ---
# Process each gene cluster separately, then aggregate
#
#   For each view:
#     For each cluster:
#       x_cluster = X[:, cluster_gene_indices]   # (cells, n_genes_in_cluster)
#       h_cluster = encoder_cluster(x_cluster)     # (cells, d_hidden)
#     H = concat/stack all h_cluster               # (cells, n_clusters, d_hidden)
#     z = aggregate(H)                             # (cells, d_latent)

def build_cluster_index_map(genes, cluster_gene_lists, view_name):
    """
    Build a mapping from cluster_id -> list of gene indices.
    This is what you feed into your model to know which columns
    of the expression matrix belong to each cluster.

    Returns
    -------
    dict[str, list[int]]
        cluster_id -> sorted list of column indices into the gene dimension.
    """
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    cluster_indices = {}
    for cluster_id, gene_list in cluster_gene_lists[view_name].items():
        indices = sorted([gene_to_idx[g] for g in gene_list if g in gene_to_idx])
        if indices:
            cluster_indices[cluster_id] = indices
    return cluster_indices


# Example: build index map for the Reactome view
# cluster_map = build_cluster_index_map(genes, cluster_gene_lists, "reactome")
# for cluster_id, indices in list(cluster_map.items())[:5]:
#     print(f"{cluster_id}: {len(indices)} genes, indices[:5]={indices[:5]}")


# --- Pattern B: PyTorch GroupedGeneEncoder ---

PYTORCH_EXAMPLE = """
import torch
import torch.nn as nn


class GroupedGeneEncoder(nn.Module):
    '''
    Encodes gene expression by processing each gene cluster independently
    through a shared (or per-cluster) encoder, then aggregating.
    
    This uses gene clusters as an inductive bias: genes in the same
    biological group are processed together before cross-group interaction.
    '''
    
    def __init__(self, cluster_indices: dict, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.cluster_ids = sorted(cluster_indices.keys())
        self.cluster_indices = cluster_indices
        self.n_clusters = len(self.cluster_ids)
        
        # Per-cluster encoders (can share weights if desired)
        self.encoders = nn.ModuleDict()
        for cid in self.cluster_ids:
            n_genes = len(cluster_indices[cid])
            safe_cid = cid.replace(":", "_").replace("-", "_").replace(" ", "_")
            self.encoders[safe_cid] = nn.Sequential(
                nn.Linear(n_genes, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        
        # Cross-cluster aggregator (e.g., attention or simple MLP)
        self.aggregator = nn.Sequential(
            nn.Linear(self.n_clusters * hidden_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
    
    def forward(self, x):
        '''
        x: (batch, n_genes) full expression vector
        returns: (batch, latent_dim) cell embedding
        '''
        cluster_embeddings = []
        for cid in self.cluster_ids:
            indices = self.cluster_indices[cid]
            x_cluster = x[:, indices]  # (batch, n_genes_in_cluster)
            safe_cid = cid.replace(":", "_").replace("-", "_").replace(" ", "_")
            h = self.encoders[safe_cid](x_cluster)  # (batch, hidden_dim)
            cluster_embeddings.append(h)
        
        H = torch.cat(cluster_embeddings, dim=-1)  # (batch, n_clusters * hidden_dim)
        z = self.aggregator(H)  # (batch, latent_dim)
        return z


# Usage:
# cluster_map = build_cluster_index_map(genes, cluster_gene_lists, "reactome")
# encoder = GroupedGeneEncoder(cluster_map, hidden_dim=64, latent_dim=32)
# x = torch.randn(256, len(genes))  # batch of cells
# z = encoder(x)  # (256, 32) cell embeddings
"""

print("\n--- PyTorch GroupedGeneEncoder pattern ---")
print(PYTORCH_EXAMPLE)


# --- Pattern C: Multi-view ensemble ---

MULTIVIEW_EXAMPLE = """
class MultiViewEncoder(nn.Module):
    '''
    One GroupedGeneEncoder per biological view, fused at the end.
    Each view provides a different inductive bias.
    '''
    
    def __init__(self, view_cluster_maps: dict, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.view_names = sorted(view_cluster_maps.keys())
        
        self.view_encoders = nn.ModuleDict({
            view: GroupedGeneEncoder(cmap, hidden_dim, latent_dim)
            for view, cmap in view_cluster_maps.items()
        })
        
        # Fuse across views
        self.fuser = nn.Sequential(
            nn.Linear(len(self.view_names) * latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
    
    def forward(self, x):
        view_embeddings = [self.view_encoders[v](x) for v in self.view_names]
        z = torch.cat(view_embeddings, dim=-1)
        return self.fuser(z)


# Usage:
# view_maps = {
#     "reactome": build_cluster_index_map(genes, cluster_gene_lists, "reactome"),
#     "go_bp": build_cluster_index_map(genes, cluster_gene_lists, "go_bp"),
#     "go_cc": build_cluster_index_map(genes, cluster_gene_lists, "go_cc"),
#     "esm2": build_cluster_index_map(genes, cluster_gene_lists, "esm2"),
# }
# model = MultiViewEncoder(view_maps)
"""

print("\n--- Multi-View Encoder pattern ---")
print(MULTIVIEW_EXAMPLE)


# ============================================================
# STEP 5: Tuning tips
# ============================================================

TUNING_TIPS = """
TUNING GUIDE
============

Resolution parameter (--resolution):
  - 0.3–0.5:  ~20-40 coarse clusters per view (good for small models)
  - 1.0:       ~50-100 medium clusters (default, good starting point)
  - 2.0–3.0:  ~100-200+ fine-grained clusters

kNN parameter (--k):
  - k=10: tighter, more local neighborhoods → more clusters
  - k=15: balanced (default)
  - k=30: broader neighborhoods → fewer, larger clusters

Gene coverage per view:
  - GO:BP will cover ~90%+ of your genes (most genes have BP annotations)
  - GO:CC covers ~80-85%
  - GO:MF covers ~75-80%
  - Reactome covers ~40-60% (it's more curated/sparse)
  - ESM-2 covers ~85-90% (depends on UniProt mapping success)
  - Co-expression covers 100% of genes in your data

Unannotated genes go into a "residual" cluster per view.
If residual clusters are too large, consider:
  1. Using ESM-2 or co-expression (they cover all genes)
  2. Subdividing the residual cluster using a different view
  3. Using resolution > 1.0 for annotation-sparse views
"""

print(TUNING_TIPS)

# Multi-View Gene Clustering for Single-Cell Inductive Bias

Partition your gene set into biologically coherent clusters from multiple ontological
views (pathways, GO terms, protein embeddings, co-expression), producing hard assignments
you can use as structured inductive bias in your single-cell model.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Save your gene list (one symbol per line)
# e.g., extract from AnnData: adata.var_names.to_series().to_csv("genes.txt", index=False, header=False)

# Run annotation-based views only (no GPU needed)
python multiview_gene_clusters.py \
    --genes genes.txt \
    --skip-esm \
    --resolution 1.0 \
    --cache gene_clusters/cache \
    --output gene_clusters/

# Full pipeline with ESM-2 on GPU + co-expression
python multiview_gene_clusters.py \
    --genes genes.txt \
    --adata your_data.h5ad \
    --esm-device cuda \
    --cache gene_clusters/cache \
    --output gene_clusters/
```

## Output Structure

```
gene_clusters/
├── clusters_all_views.csv          # Wide table: gene × view → cluster label
├── clusters_reactome.csv           # Per-view: gene, cluster, view
├── clusters_go_bp.csv
├── clusters_go_cc.csv
├── clusters_go_mf.csv
├── clusters_esm2.csv               # (if ESM-2 enabled)
├── clusters_coexpression.csv        # (if AnnData provided)
├── cluster_gene_lists.json          # {view: {cluster_id: [gene, ...]}}
├── summary.txt                      # Cluster count/size statistics
└── cache/                           # Similarity matrices + ESM embeddings
    ├── sim_reactome.npy
    ├── sim_go_bp.npy
    ├── sim_go_cc.npy
    ├── sim_go_mf.npy
    ├── sim_esm2.npy
    └── esm2_embeddings.pkl
```

## How It Works

For each biological view:

1. **Build gene×gene similarity matrix**
   - Annotation views (Reactome, GO): co-membership count, normalized by geometric mean
   - ESM-2: cosine similarity of mean-pooled protein language model embeddings
   - Co-expression: absolute Spearman correlation across cells

2. **Construct kNN graph** from the similarity matrix

3. **Leiden community detection** → hard cluster assignments

4. Genes with no annotation in a view → assigned to a **residual cluster**

## Views

| View | Source | Coverage (~5k genes) | What it captures |
|------|--------|---------------------|-----------------|
| Reactome | MSigDB C2:CP:REACTOME | ~40-60% | Pathway co-membership |
| GO:BP | MSigDB C5:GO:BP | ~90% | Shared biological processes |
| GO:CC | MSigDB C5:GO:CC | ~80% | Subcellular localization |
| GO:MF | MSigDB C5:GO:MF | ~75% | Molecular function similarity |
| ESM-2 | Meta ESM2 PLM | ~85% | Protein structure/function |
| Co-expr | Your scRNA-seq | 100% | Dataset-specific co-regulation |

## Downstream Integration

See `quickstart.py` for PyTorch integration patterns:
- `GroupedGeneEncoder`: per-cluster encoder + aggregator
- `MultiViewEncoder`: one encoder per view, fused

## Key Parameters

- `--resolution`: Leiden resolution. Lower (0.3) = fewer coarse clusters; higher (2.0) = many fine clusters
- `--k`: kNN graph neighbors. Higher = smoother/larger clusters
- `--esm-model`: `esm2_t12_35M_UR50D` (fast) or `esm2_t33_650M_UR50D` (better, default)

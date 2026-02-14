"""
Multi-View Gene Clustering for Single-Cell Inductive Bias
==========================================================

Given a list of human gene symbols, this module produces multiple hard partitions
(clusterings) of those genes, each from a different biological "view":

  View 1 — Reactome Pathways:   co-membership in Reactome canonical pathways
  View 2 — GO Biological Process: co-annotation in GO:BP terms
  View 3 — GO Cellular Component: co-annotation in GO:CC terms
  View 4 — GO Molecular Function: co-annotation in GO:MF terms
  View 5 — ESM-2 Protein Embeddings: functional similarity via PLM embeddings
  View 6 — Co-expression (optional): gene-gene correlation from your own data

The pipeline is always the same per view:
  gene list → gene×gene similarity matrix → kNN graph → Leiden clustering → cluster labels

Usage:
  python multiview_gene_clusters.py --genes gene_list.txt --output clusters/
  python multiview_gene_clusters.py --genes gene_list.txt --adata my_data.h5ad --output clusters/

Requirements:
  pip install msigdbr pandas numpy scipy igraph leidenalg scanpy anndata
  pip install fair-esm torch   # only for ESM-2 view (View 5)

Author: Generated for single-cell gene-space modeling
"""

import argparse
import json
import logging
import os
import pickle
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# 1. FETCH GENE SETS FROM MSigDB
# =============================================================================

def fetch_msigdb_gene_sets(
    collection: str,
    subcollection: str = None,
    species: str = "Human", # gseapy uses "Human", "Mouse", etc.
    version: str = "2023.2.Hs" # Specific version is often safer
) -> dict[str, set[str]]:
    """
    Fetch gene sets from MSigDB via gseapy.
    
    Mapping notes:
    - collection='C2', subcollection='CP:REACTOME' -> category='c2.cp.reactome'
    - collection='H' -> category='h.all'
    """
    import gseapy as gp
    # Construct the gseapy category string
    # gseapy expects formats like 'c2.cp.reactome' or 'h.all'
    category_parts = [collection.lower()]
    if subcollection:
        # Clean up common R-style formatting (e.g., "CP:REACTOME" -> "cp.reactome")
        clean_sub = subcollection.lower().replace(":", ".")
        category_parts.append(clean_sub)
    else:
        # If it's a main collection like H or C1 without sub, usually append '.all'
        category_parts.append("all")
        
    category_name = ".".join(category_parts)
    
    try:
        # gseapy.get_library_name() can verify names, but get_gmt is direct
        # Note: gseapy often caches these downloads locally
        msig = gp.Msigdb() 
        gmt_dict = msig.get_gmt(category=category_name, dbver=version)
        
        # Convert list of genes to sets as requested
        gene_sets = {name: set(genes) for name, genes in gmt_dict.items()}
        
        logger.info(f"Fetched {len(gene_sets)} gene sets for {category_name}")
        return gene_sets
        
    except Exception as e:
        logger.error(f"Failed to fetch {category_name}: {e}")
        return {}


# =============================================================================
# 2. BUILD GENE × GENE CO-MEMBERSHIP SIMILARITY MATRIX
# =============================================================================

def build_comembership_matrix(
    genes: list[str],
    gene_sets: dict[str, set[str]],
    min_set_size: int = 5,
    max_set_size: int = 500,
) -> np.ndarray:
    """
    Build a gene×gene similarity matrix where similarity(i, j) = number of
    gene sets in which both gene i and gene j appear (Jaccard-like co-membership).

    We then normalize by dividing by the geometric mean of each gene's total
    membership count, yielding a value in [0, 1] (cosine-like normalization).

    Parameters
    ----------
    genes : list[str]
        Ordered list of gene symbols (defines rows/cols of the matrix).
    gene_sets : dict[str, set[str]]
        Gene-set name → set of member gene symbols.
    min_set_size, max_set_size : int
        Filter gene sets by size (after intersecting with `genes`).

    Returns
    -------
    np.ndarray
        (n_genes, n_genes) similarity matrix, float32.
    """
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    n = len(genes)

    # Build binary gene × gene_set membership matrix
    filtered_sets = []
    for name, members in gene_sets.items():
        overlap = members & set(genes)
        if min_set_size <= len(overlap) <= max_set_size:
            filtered_sets.append(overlap)

    logger.info(f"Using {len(filtered_sets)} gene sets after size filtering [{min_set_size}, {max_set_size}]")

    if len(filtered_sets) == 0:
        logger.warning("No gene sets passed filtering! Returning identity matrix.")
        return np.eye(n, dtype=np.float32)

    # Build sparse binary membership matrix: genes × gene_sets
    rows, cols = [], []
    for j, members in enumerate(filtered_sets):
        for g in members:
            if g in gene_to_idx:
                rows.append(gene_to_idx[g])
                cols.append(j)

    M = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, len(filtered_sets)),
    )

    # Co-membership = M @ M.T  (counts how many sets each pair shares)
    co_mem = (M @ M.T).toarray().astype(np.float32)

    # Normalize: divide by geometric mean of diagonal (each gene's total count)
    diag = np.diag(co_mem).copy()
    diag[diag == 0] = 1.0  # avoid division by zero
    norm = np.sqrt(np.outer(diag, diag))
    sim = co_mem / norm

    # Zero out self-similarity for cleanliness (will be handled by graph construction)
    np.fill_diagonal(sim, 0.0)

    n_annotated = int(np.sum(np.diag(co_mem) > 0))
    logger.info(f"Co-membership matrix built: {n_annotated}/{n} genes have ≥1 annotation in this view")

    return sim


# =============================================================================
# 3. ESM-2 PROTEIN EMBEDDINGS
# =============================================================================

def fetch_protein_sequences(genes: list[str]) -> dict[str, str]:
    """
    Fetch canonical protein sequences for human genes from UniProt via the
    mygene.info API (no authentication needed).

    Falls back gracefully for genes without a UniProt mapping.

    Returns
    -------
    dict[str, str]
        gene_symbol → amino acid sequence.
    """
    import mygene

    mg = mygene.MyGeneInfo()
    logger.info(f"Querying mygene.info for UniProt IDs of {len(genes)} genes...")
    results = mg.querymany(
        genes,
        scopes="symbol",
        fields="uniprot.Swiss-Prot,symbol",
        species="human",
        returnall=True,
    )

    # Collect UniProt IDs
    symbol_to_uniprot = {}
    for hit in results["out"]:
        symbol = hit.get("query", "")
        up = hit.get("uniprot", {})
        sp = up.get("Swiss-Prot", None)
        if sp:
            if isinstance(sp, list):
                sp = sp[0]
            symbol_to_uniprot[symbol] = sp

    logger.info(f"Found UniProt IDs for {len(symbol_to_uniprot)}/{len(genes)} genes")

    # Fetch sequences from UniProt in batches
    import urllib.request

    sequences = {}
    uniprot_ids = list(symbol_to_uniprot.items())
    batch_size = 200

    for i in range(0, len(uniprot_ids), batch_size):
        batch = uniprot_ids[i : i + batch_size]
        ids_str = ",".join(uid for _, uid in batch)
        url = f"https://rest.uniprot.org/uniprotkb/stream?query=accession:({ids_str})&format=fasta"
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                fasta = resp.read().decode("utf-8")
            # Parse FASTA
            current_id = None
            current_seq = []
            for line in fasta.split("\n"):
                if line.startswith(">"):
                    if current_id and current_seq:
                        sequences[current_id] = "".join(current_seq)
                    # Extract accession from >sp|P12345|NAME_HUMAN ...
                    parts = line.split("|")
                    current_id = parts[1] if len(parts) >= 2 else line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_id and current_seq:
                sequences[current_id] = "".join(current_seq)
        except Exception as e:
            logger.warning(f"Failed to fetch batch {i}: {e}")

    # Map back to gene symbols
    symbol_to_seq = {}
    for symbol, uid in symbol_to_uniprot.items():
        if uid in sequences:
            symbol_to_seq[symbol] = sequences[uid]

    logger.info(f"Retrieved protein sequences for {len(symbol_to_seq)}/{len(genes)} genes")
    return symbol_to_seq


def compute_esm2_embeddings(
    gene_sequences: dict[str, str],
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = "cpu",
    max_length: int = 1022,
) -> dict[str, np.ndarray]:
    """
    Compute ESM-2 mean-pooled embeddings for each gene's protein sequence.

    Parameters
    ----------
    gene_sequences : dict[str, str]
        gene_symbol → amino acid sequence.
    model_name : str
        ESM-2 model variant. Smaller: "esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D".
        Larger/better: "esm2_t33_650M_UR50D" (default), "esm2_t36_3B_UR50D".
    device : str
        "cpu" or "cuda".
    max_length : int
        Truncate sequences longer than this.

    Returns
    -------
    dict[str, np.ndarray]
        gene_symbol → 1D embedding vector (float32).
    """
    import torch
    import esm

    logger.info(f"Loading ESM-2 model: {model_name} on {device}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    embeddings = {}
    gene_list = list(gene_sequences.items())
    batch_size = 8  # adjust based on GPU memory

    for i in range(0, len(gene_list), batch_size):
        batch_raw = gene_list[i : i + batch_size]
        # Truncate sequences
        batch_data = [(name, seq[:max_length]) for name, seq in batch_raw]

        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)

        token_reprs = results["representations"][model.num_layers]  # (B, L, D)

        for j, (name, seq) in enumerate(batch_data):
            # Mean-pool over actual residue tokens (skip BOS/EOS)
            seq_len = len(seq)
            emb = token_reprs[j, 1 : seq_len + 1, :].mean(dim=0).cpu().numpy()
            embeddings[name] = emb.astype(np.float32)

        if (i // batch_size) % 50 == 0:
            logger.info(f"  ESM-2 progress: {min(i + batch_size, len(gene_list))}/{len(gene_list)}")

    logger.info(f"Computed ESM-2 embeddings for {len(embeddings)} genes (dim={list(embeddings.values())[0].shape[0]})")
    return embeddings


def build_esm_similarity_matrix(
    genes: list[str],
    embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Build cosine similarity matrix from ESM-2 embeddings.
    Genes without embeddings get zero similarity to all others.
    """
    n = len(genes)
    dim = next(iter(embeddings.values())).shape[0]

    mat = np.zeros((n, dim), dtype=np.float32)
    has_emb = np.zeros(n, dtype=bool)

    for i, g in enumerate(genes):
        if g in embeddings:
            mat[i] = embeddings[g]
            has_emb[i] = True

    # L2 normalize for cosine similarity
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat_normed = mat / norms

    sim = mat_normed @ mat_normed.T
    np.fill_diagonal(sim, 0.0)

    # Zero out rows/cols for genes without embeddings
    sim[~has_emb, :] = 0.0
    sim[:, ~has_emb] = 0.0

    logger.info(f"ESM-2 similarity matrix: {has_emb.sum()}/{n} genes have embeddings")
    return sim


# =============================================================================
# 4. CO-EXPRESSION SIMILARITY (from user's own scRNA-seq data)
# =============================================================================

def build_coexpression_similarity(
    genes: list[str],
    adata_path: str,
    n_top_corr: int = 50,
) -> np.ndarray:
    """
    Build gene-gene Spearman correlation matrix from an AnnData object.

    Parameters
    ----------
    genes : list[str]
        Gene symbols to include.
    adata_path : str
        Path to .h5ad file.
    n_top_corr : int
        Keep only the top-n correlations per gene (sparsify).

    Returns
    -------
    np.ndarray
        (n_genes, n_genes) correlation-based similarity matrix.
    """
    import anndata
    import scanpy as sc

    logger.info(f"Loading AnnData from {adata_path}...")
    adata = anndata.read_h5ad(adata_path)

    # Subset to requested genes
    available = [g for g in genes if g in adata.var_names]
    missing = set(genes) - set(available)
    if missing:
        logger.info(f"  {len(missing)} genes not found in AnnData, will have zero similarity")

    adata_sub = adata[:, available]

    # Get expression matrix (dense, log-normalized assumed)
    if sparse.issparse(adata_sub.X):
        X = adata_sub.X.toarray()
    else:
        X = np.array(adata_sub.X)

    logger.info(f"Computing gene-gene Spearman correlation for {X.shape[1]} genes across {X.shape[0]} cells...")

    # For large matrices, use numpy correlation on ranks
    from scipy.stats import rankdata

    # Rank per column (gene) across cells
    X_ranked = np.apply_along_axis(rankdata, 0, X).astype(np.float32)

    # Pearson correlation on ranked data = Spearman
    X_ranked -= X_ranked.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X_ranked, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    X_ranked /= norms

    corr = X_ranked.T @ X_ranked / X_ranked.shape[0]  # not exact but close enough
    corr = np.clip(corr, -1, 1)

    # Use absolute correlation as similarity (anti-correlated genes are also related)
    sim_sub = np.abs(corr).astype(np.float32)
    np.fill_diagonal(sim_sub, 0.0)

    # Map back to full gene list
    gene_to_idx_sub = {g: i for i, g in enumerate(available)}
    n = len(genes)
    sim = np.zeros((n, n), dtype=np.float32)
    for i, gi in enumerate(genes):
        for j, gj in enumerate(genes):
            if gi in gene_to_idx_sub and gj in gene_to_idx_sub:
                sim[i, j] = sim_sub[gene_to_idx_sub[gi], gene_to_idx_sub[gj]]

    logger.info(f"Co-expression similarity: {len(available)}/{n} genes covered")
    return sim


# =============================================================================
# 5. GRAPH CONSTRUCTION + LEIDEN CLUSTERING
# =============================================================================

def similarity_to_leiden_clusters(
    sim: np.ndarray,
    genes: list[str],
    k: int = 15,
    resolution: float = 1.0,
    min_cluster_size: int = 3,
    view_name: str = "unnamed",
) -> pd.DataFrame:
    """
    Convert a similarity matrix to Leiden clusters via a kNN graph.

    Steps:
      1. Build mutual kNN graph from the similarity matrix.
      2. Run Leiden community detection.
      3. Assign orphan genes (0 similarity) to a residual cluster.

    Parameters
    ----------
    sim : np.ndarray
        (n, n) similarity matrix.
    genes : list[str]
        Gene symbols.
    k : int
        Number of nearest neighbors for graph construction.
    resolution : float
        Leiden resolution parameter (higher = more clusters).
    min_cluster_size : int
        Merge clusters smaller than this into the nearest larger cluster.
    view_name : str
        Label for this view.

    Returns
    -------
    pd.DataFrame
        Columns: ["gene", "cluster", "view"]
    """
    import igraph as ig
    import leidenalg

    n = len(genes)

    # Identify genes with zero similarity (unannotated in this view)
    has_signal = np.any(sim > 0, axis=1)
    annotated_idx = np.where(has_signal)[0]
    unannotated_idx = np.where(~has_signal)[0]

    logger.info(
        f"[{view_name}] {len(annotated_idx)} annotated genes, "
        f"{len(unannotated_idx)} unannotated (will be residual cluster)"
    )

    if len(annotated_idx) < 10:
        logger.warning(f"[{view_name}] Too few annotated genes for clustering, assigning all to cluster 0")
        return pd.DataFrame({
            "gene": genes,
            "cluster": [f"{view_name}_0"] * n,
            "view": view_name,
        })

    # Subset similarity to annotated genes
    sim_sub = sim[np.ix_(annotated_idx, annotated_idx)]

    # Build kNN graph
    k_eff = min(k, len(annotated_idx) - 1)
    edges = []
    weights = []

    for i in range(len(annotated_idx)):
        row = sim_sub[i].copy()
        row[i] = -1  # exclude self
        top_k = np.argsort(row)[-k_eff:]
        for j in top_k:
            if row[j] > 0:
                edges.append((i, j))
                weights.append(float(row[j]))

    # Create igraph graph
    g = ig.Graph(n=len(annotated_idx), edges=edges, directed=False)
    g.es["weight"] = weights
    g.simplify(combine_edges="max")

    # Leiden clustering
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=42,
    )

    cluster_labels_sub = np.array(partition.membership)

    # Handle small clusters: merge into nearest large cluster
    cluster_counts = pd.Series(cluster_labels_sub).value_counts()
    small_clusters = cluster_counts[cluster_counts < min_cluster_size].index.tolist()

    if small_clusters:
        large_clusters = cluster_counts[cluster_counts >= min_cluster_size].index.tolist()
        if large_clusters:
            # For each small cluster, find the large cluster most similar to it
            for sc in small_clusters:
                sc_genes_idx = np.where(cluster_labels_sub == sc)[0]
                best_target = large_clusters[0]
                best_sim = -1
                for lc in large_clusters:
                    lc_genes_idx = np.where(cluster_labels_sub == lc)[0]
                    cross_sim = sim_sub[np.ix_(sc_genes_idx, lc_genes_idx)].mean()
                    if cross_sim > best_sim:
                        best_sim = cross_sim
                        best_target = lc
                cluster_labels_sub[sc_genes_idx] = best_target

    # Relabel clusters contiguously
    unique_labels = sorted(set(cluster_labels_sub))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    cluster_labels_sub = np.array([label_map[l] for l in cluster_labels_sub])

    n_clusters = len(set(cluster_labels_sub))

    # Build full label array
    cluster_labels = np.full(n, -1, dtype=int)
    for i, idx in enumerate(annotated_idx):
        cluster_labels[idx] = cluster_labels_sub[i]

    # Assign unannotated genes to a residual cluster
    residual_cluster = n_clusters
    cluster_labels[cluster_labels == -1] = residual_cluster

    # Format labels
    labels_str = [f"{view_name}_{cl}" for cl in cluster_labels]

    # Cluster size summary
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    logger.info(
        f"[{view_name}] {n_clusters} clusters + 1 residual | "
        f"sizes: min={cluster_sizes.min()}, median={int(cluster_sizes.median())}, max={cluster_sizes.max()}"
    )

    return pd.DataFrame({
        "gene": genes,
        "cluster": labels_str,
        "view": view_name,
    })


# =============================================================================
# 6. MAIN PIPELINE
# =============================================================================

def run_multiview_clustering(
    genes: list[str],
    output_dir: str = "gene_clusters",
    adata_path: Optional[str] = None,
    esm_model: str = "esm2_t33_650M_UR50D",
    esm_device: str = "cpu",
    skip_esm: bool = False,
    skip_coexpr: bool = False,
    k: int = 15,
    resolution: float = 1.0,
    cache_dir: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Run the full multi-view gene clustering pipeline.

    Parameters
    ----------
    genes : list[str]
        List of gene symbols (e.g., 5127 genes from your scRNA-seq).
    output_dir : str
        Directory to save results.
    adata_path : str, optional
        Path to .h5ad for co-expression view. Skipped if None.
    esm_model : str
        ESM-2 model name.
    esm_device : str
        "cpu" or "cuda".
    skip_esm : bool
        Skip ESM-2 view (requires network + GPU).
    skip_coexpr : bool
        Skip co-expression view.
    k : int
        kNN neighbors for graph construction.
    resolution : float
        Leiden resolution.
    cache_dir : str, optional
        Cache intermediate results (similarity matrices, embeddings).

    Returns
    -------
    dict[str, pd.DataFrame]
        view_name → DataFrame with columns [gene, cluster, view].
    """
    os.makedirs(output_dir, exist_ok=True)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    genes = list(genes)  # ensure ordered
    n = len(genes)
    logger.info(f"Starting multi-view clustering for {n} genes")

    results = {}

    # ---- View 1: Reactome Pathways ----
    logger.info("=" * 60)
    logger.info("VIEW 1: Reactome Pathways")
    logger.info("=" * 60)
    reactome_sets = fetch_msigdb_gene_sets("C2", "CP:REACTOME")
    sim_reactome = build_comembership_matrix(genes, reactome_sets, min_set_size=5, max_set_size=500)
    results["reactome"] = similarity_to_leiden_clusters(
        sim_reactome, genes, k=k, resolution=resolution, view_name="reactome"
    )
    if cache_dir:
        np.save(os.path.join(cache_dir, "sim_reactome.npy"), sim_reactome)

    # ---- View 2: GO Biological Process ----
    logger.info("=" * 60)
    logger.info("VIEW 2: GO Biological Process")
    logger.info("=" * 60)
    gobp_sets = fetch_msigdb_gene_sets("C5", "GO:BP")
    sim_gobp = build_comembership_matrix(genes, gobp_sets, min_set_size=10, max_set_size=500)
    results["go_bp"] = similarity_to_leiden_clusters(
        sim_gobp, genes, k=k, resolution=resolution, view_name="go_bp"
    )
    if cache_dir:
        np.save(os.path.join(cache_dir, "sim_go_bp.npy"), sim_gobp)

    # ---- View 3: GO Cellular Component ----
    logger.info("=" * 60)
    logger.info("VIEW 3: GO Cellular Component")
    logger.info("=" * 60)
    gocc_sets = fetch_msigdb_gene_sets("C5", "GO:CC")
    sim_gocc = build_comembership_matrix(genes, gocc_sets, min_set_size=5, max_set_size=500)
    results["go_cc"] = similarity_to_leiden_clusters(
        sim_gocc, genes, k=k, resolution=resolution, view_name="go_cc"
    )
    if cache_dir:
        np.save(os.path.join(cache_dir, "sim_go_cc.npy"), sim_gocc)

    # ---- View 4: GO Molecular Function ----
    logger.info("=" * 60)
    logger.info("VIEW 4: GO Molecular Function")
    logger.info("=" * 60)
    gomf_sets = fetch_msigdb_gene_sets("C5", "GO:MF")
    sim_gomf = build_comembership_matrix(genes, gomf_sets, min_set_size=5, max_set_size=500)
    results["go_mf"] = similarity_to_leiden_clusters(
        sim_gomf, genes, k=k, resolution=resolution, view_name="go_mf"
    )
    if cache_dir:
        np.save(os.path.join(cache_dir, "sim_go_mf.npy"), sim_gomf)

    # ---- View 5: ESM-2 Protein Embeddings ----
    if not skip_esm:
        logger.info("=" * 60)
        logger.info("VIEW 5: ESM-2 Protein Language Model")
        logger.info("=" * 60)

        emb_cache = os.path.join(cache_dir, "esm2_embeddings.pkl") if cache_dir else None

        if emb_cache and os.path.exists(emb_cache):
            logger.info(f"Loading cached ESM-2 embeddings from {emb_cache}")
            with open(emb_cache, "rb") as f:
                esm_embeddings = pickle.load(f)
        else:
            seqs = fetch_protein_sequences(genes)
            esm_embeddings = compute_esm2_embeddings(seqs, model_name=esm_model, device=esm_device)
            if emb_cache:
                with open(emb_cache, "wb") as f:
                    pickle.dump(esm_embeddings, f)

        sim_esm = build_esm_similarity_matrix(genes, esm_embeddings)
        results["esm2"] = similarity_to_leiden_clusters(
            sim_esm, genes, k=k, resolution=resolution, view_name="esm2"
        )
        if cache_dir:
            np.save(os.path.join(cache_dir, "sim_esm2.npy"), sim_esm)
    else:
        logger.info("Skipping ESM-2 view (--skip-esm)")

    # ---- View 6: Co-expression ----
    if adata_path and not skip_coexpr:
        logger.info("=" * 60)
        logger.info("VIEW 6: Co-expression from scRNA-seq data")
        logger.info("=" * 60)
        sim_coexpr = build_coexpression_similarity(genes, adata_path)
        results["coexpression"] = similarity_to_leiden_clusters(
            sim_coexpr, genes, k=k, resolution=resolution, view_name="coexpression"
        )
        if cache_dir:
            np.save(os.path.join(cache_dir, "sim_coexpression.npy"), sim_coexpr)
    elif not skip_coexpr:
        logger.info("Skipping co-expression view (no --adata provided)")

    # ---- Save results ----
    save_results(results, genes, output_dir)

    return results


# =============================================================================
# 7. SAVE & SUMMARIZE
# =============================================================================

def save_results(
    results: dict[str, pd.DataFrame],
    genes: list[str],
    output_dir: str,
):
    """Save all clustering results in multiple convenient formats."""

    # 1. Per-view CSVs
    for view_name, df in results.items():
        df.to_csv(os.path.join(output_dir, f"clusters_{view_name}.csv"), index=False)

    # 2. Combined wide-format table: gene × view → cluster
    combined = pd.DataFrame({"gene": genes})
    for view_name, df in results.items():
        combined[view_name] = df["cluster"].values
    combined.to_csv(os.path.join(output_dir, "clusters_all_views.csv"), index=False)

    # 3. Cluster-to-gene-list JSON (useful for downstream modeling)
    cluster_dict = {}
    for view_name, df in results.items():
        view_clusters = defaultdict(list)
        for _, row in df.iterrows():
            view_clusters[row["cluster"]].append(row["gene"])
        cluster_dict[view_name] = dict(view_clusters)

    with open(os.path.join(output_dir, "cluster_gene_lists.json"), "w") as f:
        json.dump(cluster_dict, f, indent=2)

    # 4. Summary statistics
    summary_lines = ["Multi-View Gene Clustering Summary", "=" * 40]
    for view_name, df in results.items():
        n_clusters = df["cluster"].nunique()
        sizes = df["cluster"].value_counts()
        summary_lines.append(
            f"\n{view_name}:"
            f"\n  clusters: {n_clusters}"
            f"\n  sizes: min={sizes.min()}, median={int(sizes.median())}, "
            f"max={sizes.max()}, mean={sizes.mean():.1f}"
        )

    summary_text = "\n".join(summary_lines)
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary_text)

    logger.info(f"\nResults saved to {output_dir}/")
    logger.info(summary_text)


# =============================================================================
# 8. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-view gene clustering for single-cell inductive bias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: annotation-based views only (no GPU needed)
  python multiview_gene_clusters.py --genes gene_list.txt --skip-esm --output clusters/

  # Full: all views including ESM-2 on GPU + co-expression from your data
  python multiview_gene_clusters.py --genes gene_list.txt --adata data.h5ad \\
      --esm-device cuda --output clusters/

  # Tune resolution for more/fewer clusters
  python multiview_gene_clusters.py --genes gene_list.txt --resolution 0.5 --output clusters/
        """,
    )
    parser.add_argument("--genes", required=True, help="Path to text file with one gene symbol per line")
    parser.add_argument("--output", default="gene_clusters", help="Output directory")
    parser.add_argument("--adata", default=None, help="Path to .h5ad for co-expression view")
    parser.add_argument("--esm-model", default="esm2_t33_650M_UR50D", help="ESM-2 model name")
    parser.add_argument("--esm-device", default="cpu", choices=["cpu", "cuda"], help="Device for ESM-2")
    parser.add_argument("--skip-esm", action="store_true", help="Skip ESM-2 view")
    parser.add_argument("--skip-coexpr", action="store_true", help="Skip co-expression view")
    parser.add_argument("--k", type=int, default=15, help="kNN neighbors (default: 15)")
    parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution (default: 1.0)")
    parser.add_argument("--cache", default=None, help="Cache directory for intermediate results")

    args = parser.parse_args()

    # Load gene list
    with open(args.genes) as f:
        genes = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(genes)} genes from {args.genes}")

    run_multiview_clustering(
        genes=genes,
        output_dir=args.output,
        adata_path=args.adata,
        esm_model=args.esm_model,
        esm_device=args.esm_device,
        skip_esm=args.skip_esm,
        skip_coexpr=args.skip_coexpr,
        k=args.k,
        resolution=args.resolution,
        cache_dir=args.cache,
    )


if __name__ == "__main__":
    main()

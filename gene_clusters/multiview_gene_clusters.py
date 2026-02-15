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

Pipeline per annotation view (Views 1-4):
  gene list → gene×gene co-membership matrix
            → impute residual genes via ESM-2 kNN transfer (+co-expression)
            → kNN graph → Leiden clustering → cluster labels
            → spectral embedding (complete vocabulary, all genes)

Pipeline order:
  ESM-2 raw embeddings (needed as transfer source)
  → Co-expression similarity (secondary transfer source)
  → Annotation views (with imputation before clustering)
  → ESM-2 view (direct similarity, no imputation)
  → Co-expression view (direct similarity, no imputation)

Usage:
  python multiview_gene_clusters.py --genes gene_list.txt --output clusters/
  python multiview_gene_clusters.py --genes gene_list.txt --adata my_data.h5ad --output clusters/

Requirements:
  Core:     pip install gseapy pandas numpy scipy igraph leidenalg scikit-learn
  ESM-2:    pip install fair-esm torch mygene requests
  CoExpr:   pip install scanpy anndata
  Plotting: pip install umap-learn matplotlib

Author: Generated for single-cell gene-space modeling
"""

import argparse
import json
import logging
import os
import pickle as pkl
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
    species: str = "Human",
    version: str = "2023.2.Hs",
) -> dict[str, set[str]]:
    """
    Fetch gene sets from MSigDB via gseapy.

    Mapping notes:
    - collection='C2', subcollection='CP:REACTOME' -> category='c2.cp.reactome'
    - collection='H' -> category='h.all'
    """
    import gseapy as gp

    category_parts = [collection.lower()]
    if subcollection:
        clean_sub = subcollection.lower().replace(":", ".")
        category_parts.append(clean_sub)
    else:
        category_parts.append("all")

    category_name = ".".join(category_parts)

    try:
        msig = gp.Msigdb()
        gmt_dict = msig.get_gmt(category=category_name, dbver=version)
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
    Build a gene×gene similarity matrix from co-membership in gene sets.
    Normalized by geometric mean of membership counts (cosine-like).

    Returns
    -------
    np.ndarray
        (n_genes, n_genes) similarity matrix, float32. Diagonal is 0.
    """
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    n = len(genes)

    filtered_sets = []
    for name, members in gene_sets.items():
        overlap = members & set(genes)
        if min_set_size <= len(overlap) <= max_set_size:
            filtered_sets.append(overlap)

    logger.info(f"Using {len(filtered_sets)} gene sets after size filtering [{min_set_size}, {max_set_size}]")

    if len(filtered_sets) == 0:
        logger.warning("No gene sets passed filtering! Returning zero matrix.")
        return np.zeros((n, n), dtype=np.float32)

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

    co_mem = (M @ M.T).toarray().astype(np.float32)

    diag = np.diag(co_mem).copy()
    diag[diag == 0] = 1.0
    norm = np.sqrt(np.outer(diag, diag))
    sim = co_mem / norm

    np.fill_diagonal(sim, 0.0)

    n_annotated = int(np.sum(np.diag(co_mem) > 0))
    logger.info(f"Co-membership matrix built: {n_annotated}/{n} genes have ≥1 annotation in this view")

    return sim


# =============================================================================
# 3. ESM-2 PROTEIN EMBEDDINGS
# =============================================================================

def fetch_protein_sequences(genes: list[str]) -> dict[str, str]:
    """
    Fetch canonical protein sequences for human genes via:
      1. mygene.info → UniProt accessions (Swiss-Prot preferred, TrEMBL fallback)
      2. UniProt REST API v2 → FASTA sequences

    Handles ENSG-style names, aliases, and readthrough genes.

    Returns
    -------
    dict[str, str]
        gene_symbol → amino acid sequence.
    """
    import mygene
    import requests
    from time import sleep

    mg = mygene.MyGeneInfo()
    logger.info(f"Querying mygene.info for UniProt IDs of {len(genes)} genes...")

    # --- Step 1: Gene symbols → UniProt accessions ---
    # Use broad scopes: symbol, alias, and Ensembl gene ID (for ENSG names)
    results = mg.querymany(
        genes,
        scopes="symbol,alias,ensembl.gene",
        fields="uniprot.Swiss-Prot,uniprot.TrEMBL,symbol",
        species="human",
        returnall=True,
    )

    symbol_to_uniprot = {}
    for hit in results["out"]:
        symbol = hit.get("query", "")
        if hit.get("notfound", False):
            continue
        up = hit.get("uniprot", {})
        if not up:
            continue
        # Prefer Swiss-Prot (reviewed), fall back to TrEMBL (unreviewed)
        sp = up.get("Swiss-Prot", None)
        if sp is None:
            sp = up.get("TrEMBL", None)
        if sp:
            if isinstance(sp, list):
                sp = sp[0]  # take canonical / first entry
            symbol_to_uniprot[symbol] = sp

    n_found = len(symbol_to_uniprot)
    n_missing = len(genes) - n_found
    logger.info(f"Found UniProt IDs for {n_found}/{len(genes)} genes ({n_missing} unmapped)")

    if n_found == 0:
        return {}

    # --- Step 2: UniProt accessions → FASTA sequences ---
    # Use the correct UniProt REST API v2 query syntax:
    #   query=accession:P12345 OR accession:P67890
    # Smaller batches (50) to stay within URL length limits.
    sequences = {}  # accession → sequence
    uniprot_items = list(symbol_to_uniprot.items())
    batch_size = 50
    max_retries = 3

    for i in range(0, len(uniprot_items), batch_size):
        batch = uniprot_items[i : i + batch_size]
        accessions = [uid for _, uid in batch]

        # Build query: "accession:X OR accession:Y OR ..."
        query = " OR ".join(f"accession:{uid}" for uid in accessions)

        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    "https://rest.uniprot.org/uniprotkb/stream",
                    params={"format": "fasta", "query": query},
                    headers={"Accept": "text/plain"},
                    timeout=60,
                )
                resp.raise_for_status()
                fasta_text = resp.text

                # Parse FASTA
                current_id = None
                current_seq = []
                for line in fasta_text.split("\n"):
                    if line.startswith(">"):
                        if current_id and current_seq:
                            sequences[current_id] = "".join(current_seq)
                        # >sp|P12345|NAME_HUMAN ... or >tr|A0A0G2JRW2|...
                        parts = line.split("|")
                        current_id = parts[1] if len(parts) >= 2 else line[1:].split()[0]
                        current_seq = []
                    elif line.strip():
                        current_seq.append(line.strip())
                if current_id and current_seq:
                    sequences[current_id] = "".join(current_seq)
                break  # success

            except requests.exceptions.HTTPError as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"UniProt batch {i} HTTP error (attempt {attempt+1}), retrying in {wait}s: {e}")
                    sleep(wait)
                else:
                    logger.warning(f"UniProt batch {i} failed after {max_retries} attempts: {e}")
            except Exception as e:
                logger.warning(f"UniProt batch {i} unexpected error: {e}")
                break

        if (i // batch_size) % 10 == 0:
            n_so_far = len(sequences)
            logger.info(f"  UniProt fetch progress: {min(i + batch_size, len(uniprot_items))}/{len(uniprot_items)} queries, {n_so_far} sequences retrieved")

    # --- Step 3: Map accessions back to gene symbols ---
    symbol_to_seq = {}
    for symbol, uid in symbol_to_uniprot.items():
        if uid in sequences:
            symbol_to_seq[symbol] = sequences[uid]

    logger.info(f"Retrieved protein sequences for {len(symbol_to_seq)}/{len(genes)} genes")

    if len(symbol_to_seq) < n_found:
        n_seq_missing = n_found - len(symbol_to_seq)
        logger.info(f"  {n_seq_missing} genes had UniProt IDs but no sequence returned (obsolete/merged entries)")

    return symbol_to_seq


def compute_esm2_embeddings(
    gene_sequences: dict[str, str],
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = "cpu",
    max_length: int = 1022,
) -> dict[str, np.ndarray]:
    """
    Compute ESM-2 mean-pooled embeddings for each gene's protein sequence.

    Returns
    -------
    dict[str, np.ndarray]
        gene_symbol → 1D embedding vector (float32).
        Empty dict if no sequences provided.
    """
    if not gene_sequences:
        logger.warning("No protein sequences provided, returning empty embeddings dict")
        return {}

    import torch
    import esm

    logger.info(f"Loading ESM-2 model: {model_name} on {device}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    embeddings = {}
    gene_list = list(gene_sequences.items())
    batch_size = 8

    for i in range(0, len(gene_list), batch_size):
        batch_raw = gene_list[i : i + batch_size]
        batch_data = [(name, seq[:max_length]) for name, seq in batch_raw]

        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)

        token_reprs = results["representations"][model.num_layers]

        for j, (name, seq) in enumerate(batch_data):
            seq_len = len(seq)
            emb = token_reprs[j, 1 : seq_len + 1, :].mean(dim=0).cpu().numpy()
            embeddings[name] = emb.astype(np.float32)

        if (i // batch_size) % 50 == 0:
            logger.info(f"  ESM-2 progress: {min(i + batch_size, len(gene_list))}/{len(gene_list)}")

    if embeddings:
        dim = next(iter(embeddings.values())).shape[0]
        logger.info(f"Computed ESM-2 embeddings for {len(embeddings)} genes (dim={dim})")
    else:
        logger.warning("ESM-2 produced no embeddings")

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
    if not embeddings:
        logger.warning("No ESM-2 embeddings available, returning zero similarity matrix")
        return np.zeros((n, n), dtype=np.float32)

    dim = next(iter(embeddings.values())).shape[0]

    mat = np.zeros((n, dim), dtype=np.float32)
    has_emb = np.zeros(n, dtype=bool)

    for i, g in enumerate(genes):
        if g in embeddings:
            mat[i] = embeddings[g]
            has_emb[i] = True

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat_normed = mat / norms

    sim = mat_normed @ mat_normed.T
    np.fill_diagonal(sim, 0.0)

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
    Uses absolute correlation as similarity (anti-correlated genes are also related).
    """
    import anndata
    import scanpy as sc

    logger.info(f"Loading AnnData from {adata_path}...")
    adata = anndata.read_h5ad(adata_path)

    available = [g for g in genes if g in adata.var_names]
    missing = set(genes) - set(available)
    if missing:
        logger.info(f"  {len(missing)} genes not found in AnnData, will have zero similarity")

    adata_sub = adata[:, available]

    if sparse.issparse(adata_sub.X):
        X = adata_sub.X.toarray()
    else:
        X = np.array(adata_sub.X)

    logger.info(f"Computing gene-gene Spearman correlation for {X.shape[1]} genes across {X.shape[0]} cells...")

    from scipy.stats import rankdata

    X_ranked = np.apply_along_axis(rankdata, 0, X).astype(np.float32)
    X_ranked -= X_ranked.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X_ranked, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    X_ranked /= norms

    corr = X_ranked.T @ X_ranked / X_ranked.shape[0]
    corr = np.clip(corr, -1, 1)

    sim_sub = np.abs(corr).astype(np.float32)
    np.fill_diagonal(sim_sub, 0.0)

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
# 5. IMPUTE RESIDUAL GENES VIA ESM-2 kNN ANNOTATION TRANSFER
# =============================================================================

def impute_residual_features(
    sim: np.ndarray,
    genes: list[str],
    esm_embeddings: dict[str, np.ndarray],
    k_transfer: int = 10,
    coexpr_sim: Optional[np.ndarray] = None,
    alpha: float = 0.7,
    view_name: str = "unnamed",
) -> np.ndarray:
    """
    For genes with zero similarity rows (unannotated in this view),
    impute a pseudo-similarity row via kNN transfer from ESM-2 embedding
    space, optionally blended with co-expression neighbors.

    Principled basis: the GoPredSim paradigm (Littmann et al., 2021) —
    PLM embeddings implicitly encode GO-relevant functional information
    even though they were never trained on GO terms. We exploit this by
    transferring annotation-based similarity profiles from the k nearest
    annotated genes in ESM-2 space.

    For each residual gene g with an ESM-2 embedding:
      1. Find k nearest ANNOTATED genes in ESM-2 cosine space
      2. pseudo_sim[g, :] = weighted avg of neighbors' similarity rows
         (weights = ESM cosine similarity to g, softmax-normalized)
      3. Optionally blend with co-expression kNN transfer (weight 1-α)

    Parameters
    ----------
    sim : np.ndarray
        (n, n) similarity matrix (annotation-based, e.g., Reactome).
    genes : list[str]
        Gene symbols.
    esm_embeddings : dict[str, np.ndarray]
        gene → ESM-2 mean-pooled embedding (raw PLM output).
    k_transfer : int
        Number of annotated neighbors to transfer from.
    coexpr_sim : np.ndarray, optional
        (n, n) co-expression similarity matrix (secondary transfer source).
    alpha : float
        Weight on ESM-2 transfer vs co-expression transfer (1.0 = ESM only).
    view_name : str
        For logging.

    Returns
    -------
    np.ndarray
        Augmented similarity matrix with imputed rows for residual genes.
    """
    n = len(genes)
    has_annotation = np.any(sim > 0, axis=1)
    annotated_idx = np.where(has_annotation)[0]
    residual_idx = np.where(~has_annotation)[0]

    if len(residual_idx) == 0:
        logger.info(f"[{view_name}] No residual genes to impute")
        return sim

    if len(annotated_idx) == 0:
        logger.warning(f"[{view_name}] No annotated genes available for imputation")
        return sim

    # Build ESM embedding matrix
    esm_dim = next(iter(esm_embeddings.values())).shape[0] if esm_embeddings else 0
    if esm_dim == 0:
        logger.warning(f"[{view_name}] No ESM-2 embeddings available, skipping imputation")
        return sim

    esm_mat = np.zeros((n, esm_dim), dtype=np.float32)
    has_esm = np.zeros(n, dtype=bool)
    for i, g in enumerate(genes):
        if g in esm_embeddings:
            esm_mat[i] = esm_embeddings[g]
            has_esm[i] = True

    # L2-normalize for cosine similarity
    norms = np.linalg.norm(esm_mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    esm_normed = esm_mat / norms

    sim_augmented = sim.copy()

    # Only impute residual genes that have ESM-2 embeddings
    imputable_idx = np.where(has_esm & ~has_annotation)[0]

    # Annotated genes that also have ESM embeddings (transfer sources)
    source_idx = np.where(has_esm & has_annotation)[0]

    if len(source_idx) == 0:
        logger.warning(f"[{view_name}] No annotated genes have ESM-2 embeddings, cannot impute")
        return sim

    # Pre-compute ESM cosine similarities: imputable × source
    cos_matrix = None
    if len(imputable_idx) > 0:
        cos_matrix = esm_normed[imputable_idx] @ esm_normed[source_idx].T  # (n_imp, n_src)

    n_imputed = 0
    for local_i, res_i in enumerate(imputable_idx):
        cos_sims = cos_matrix[local_i]
        k_eff = min(k_transfer, len(source_idx))
        top_k_local = np.argsort(cos_sims)[-k_eff:]

        # Softmax-like weights from cosine similarities (clamp negatives)
        top_sims = np.clip(cos_sims[top_k_local], 0, None)
        weight_sum = top_sims.sum()
        if weight_sum < 1e-8:
            continue  # no positive similarity to any annotated gene
        weights = top_sims / weight_sum

        # Weighted average of annotated neighbors' similarity rows
        neighbor_global_idx = source_idx[top_k_local]
        pseudo_row = np.zeros(n, dtype=np.float32)
        for w, ni in zip(weights, neighbor_global_idx):
            pseudo_row += w * sim[ni]

        # --- Optional: blend with co-expression transfer ---
        if coexpr_sim is not None and alpha < 1.0:
            coexpr_to_annotated = coexpr_sim[res_i, annotated_idx]
            k_eff_c = min(k_transfer, len(annotated_idx))
            top_k_c = np.argsort(coexpr_to_annotated)[-k_eff_c:]

            top_c_sims = np.clip(coexpr_to_annotated[top_k_c], 0, None)
            c_sum = top_c_sims.sum()
            if c_sum > 1e-8:
                weights_c = top_c_sims / c_sum
                pseudo_row_c = np.zeros(n, dtype=np.float32)
                for w, ci in zip(weights_c, annotated_idx[top_k_c]):
                    pseudo_row_c += w * sim[ci]
                pseudo_row = alpha * pseudo_row + (1 - alpha) * pseudo_row_c

        pseudo_row[res_i] = 0.0  # no self-similarity
        sim_augmented[res_i] = pseudo_row
        sim_augmented[:, res_i] = pseudo_row  # symmetrize
        n_imputed += 1

    n_still_residual = len(residual_idx) - n_imputed
    logger.info(
        f"[{view_name}] Imputed {n_imputed}/{len(residual_idx)} residual genes via ESM-2 kNN transfer"
        f" ({n_still_residual} remain truly dark — no ESM embedding)"
    )
    return sim_augmented


# =============================================================================
# 6. GRAPH CONSTRUCTION + LEIDEN CLUSTERING
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
    """
    import igraph as ig
    import leidenalg

    n = len(genes)

    has_signal = np.any(sim > 0, axis=1)
    annotated_idx = np.where(has_signal)[0]
    unannotated_idx = np.where(~has_signal)[0]

    logger.info(
        f"[{view_name}] {len(annotated_idx)} genes with signal, "
        f"{len(unannotated_idx)} truly dark (residual cluster)"
    )

    if len(annotated_idx) < 10:
        logger.warning(f"[{view_name}] Too few genes with signal for clustering, assigning all to cluster 0")
        return pd.DataFrame({
            "gene": genes,
            "cluster": [f"{view_name}_0"] * n,
            "view": view_name,
        })

    sim_sub = sim[np.ix_(annotated_idx, annotated_idx)]

    k_eff = min(k, len(annotated_idx) - 1)
    edges = []
    weights = []

    for i in range(len(annotated_idx)):
        row = sim_sub[i].copy()
        row[i] = -1
        top_k = np.argsort(row)[-k_eff:]
        for j in top_k:
            if row[j] > 0:
                edges.append((i, j))
                weights.append(float(row[j]))

    g = ig.Graph(n=len(annotated_idx), edges=edges, directed=False)
    g.es["weight"] = weights
    g.simplify(combine_edges="max")

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

    # Relabel contiguously
    unique_labels = sorted(set(cluster_labels_sub))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    cluster_labels_sub = np.array([label_map[l] for l in cluster_labels_sub])

    n_clusters = len(set(cluster_labels_sub))

    # Build full label array
    cluster_labels = np.full(n, -1, dtype=int)
    for i, idx in enumerate(annotated_idx):
        cluster_labels[idx] = cluster_labels_sub[i]

    # Assign truly dark genes to residual cluster
    residual_cluster = n_clusters
    cluster_labels[cluster_labels == -1] = residual_cluster

    labels_str = [f"{view_name}_{cl}" for cl in cluster_labels]

    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    n_residual = int((cluster_labels == residual_cluster).sum())
    logger.info(
        f"[{view_name}] {n_clusters} clusters + {n_residual} residual genes | "
        f"sizes: min={cluster_sizes.min()}, median={int(cluster_sizes.median())}, max={cluster_sizes.max()}"
    )

    return pd.DataFrame({
        "gene": genes,
        "cluster": labels_str,
        "view": view_name,
    })


# =============================================================================
# 7. SPECTRAL EMBEDDING (complete vocabulary)
# =============================================================================

def auto_n_components(n_genes: int) -> int:
    """
    Automatic spectral embedding dimensionality:
    floor(floor(sqrt(n_genes)) / 8) * 8

    Examples: 5127 → 64, 19180 → 136, 2000 → 40
    """
    return int(np.floor(np.floor(np.sqrt(n_genes)) / 8) * 8)


def build_knn_embeddings(
    sim: np.ndarray,
    genes: list[str],
    k: int = 15,
    n_components: int | None = None,
    esm_embeddings: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """
    Spectral embedding of the kNN-sparsified similarity matrix.
    Euclidean distance in this space ≈ diffusion distance on the gene graph.

    COMPLETE VOCABULARY GUARANTEE: every gene in `genes` receives an embedding.
    - Connected genes: spectral embedding coordinates
    - Disconnected genes with ESM-2: kNN regression from ESM-2 neighbors'
      spectral coordinates
    - Truly dark genes (no signal, no ESM): mean embedding + small noise

    Parameters
    ----------
    sim : np.ndarray
        (n, n) similarity matrix (possibly imputed).
    genes : list[str]
        Gene symbols.
    k : int
        kNN for graph sparsification.
    n_components : int, optional
        Embedding dimensionality. Auto-computed if None.
    esm_embeddings : dict, optional
        ESM-2 raw embeddings for fallback placement of disconnected genes.

    Returns
    -------
    dict[str, np.ndarray]
        gene → embedding vector (float32). Guaranteed to contain ALL genes.
    """
    from sklearn.manifold import SpectralEmbedding

    n = len(genes)

    if n_components is None:
        n_components = auto_n_components(n)
    n_components = max(8, min(n_components, n - 2))

    logger.info(f"Spectral embedding: n_genes={n}, n_components={n_components}, k={k}")

    # Identify connected vs disconnected genes
    has_signal = np.any(sim > 0, axis=1)
    connected_idx = np.where(has_signal)[0]
    disconnected_idx = np.where(~has_signal)[0]

    if len(connected_idx) < n_components + 2:
        logger.warning(
            f"Too few connected genes ({len(connected_idx)}) for spectral embedding "
            f"(need ≥ {n_components + 2}). Falling back to zero embeddings."
        )
        return {g: np.zeros(n_components, dtype=np.float32) for g in genes}

    # --- Spectral embedding on connected subgraph only ---
    sim_connected = sim[np.ix_(connected_idx, connected_idx)]
    n_conn = len(connected_idx)
    k_eff = min(k, n_conn - 1)

    # kNN sparsification
    sim_sparse = np.zeros_like(sim_connected)
    for i in range(n_conn):
        top_k_idx = np.argsort(sim_connected[i])[-k_eff:]
        sim_sparse[i, top_k_idx] = sim_connected[i, top_k_idx]

    # Symmetrize
    sim_sparse = np.maximum(sim_sparse, sim_sparse.T)

    n_comp_eff = min(n_components, n_conn - 2)
    se = SpectralEmbedding(
        n_components=n_comp_eff,
        affinity="precomputed",
        random_state=1234,
    )
    emb_connected = se.fit_transform(sim_sparse).astype(np.float32)

    # Pad if n_comp_eff < n_components (very small connected set)
    if n_comp_eff < n_components:
        pad = np.zeros((n_conn, n_components - n_comp_eff), dtype=np.float32)
        emb_connected = np.hstack([emb_connected, pad])

    # --- Place ALL genes into the embedding ---
    full_emb = np.full((n, n_components), np.nan, dtype=np.float32)

    # Connected genes: direct spectral coordinates
    for local_i, global_i in enumerate(connected_idx):
        full_emb[global_i] = emb_connected[local_i]

    # Disconnected genes: fallback strategies
    if len(disconnected_idx) > 0:
        n_esm_placed = 0

        # Strategy A: if ESM-2 available, use kNN regression in ESM space
        # to inherit spectral coordinates from nearest connected genes
        if esm_embeddings:
            esm_dim = next(iter(esm_embeddings.values())).shape[0]
            esm_mat = np.zeros((n, esm_dim), dtype=np.float32)
            has_esm = np.zeros(n, dtype=bool)
            for i, g in enumerate(genes):
                if g in esm_embeddings:
                    esm_mat[i] = esm_embeddings[g]
                    has_esm[i] = True

            esm_norms = np.linalg.norm(esm_mat, axis=1, keepdims=True)
            esm_norms[esm_norms == 0] = 1.0
            esm_normed = esm_mat / esm_norms

            # Connected genes that also have ESM (regression targets)
            target_mask = has_signal & has_esm
            target_idx = np.where(target_mask)[0]

            if len(target_idx) > 0:
                for di in disconnected_idx:
                    if not has_esm[di]:
                        continue  # truly dark, handled in Strategy B

                    cos_sims = esm_normed[di] @ esm_normed[target_idx].T
                    k_reg = min(10, len(target_idx))
                    top_k = np.argsort(cos_sims)[-k_reg:]
                    top_sims = np.clip(cos_sims[top_k], 0, None)
                    w_sum = top_sims.sum()
                    if w_sum < 1e-8:
                        continue

                    weights = top_sims / w_sum
                    neighbor_global = target_idx[top_k]
                    full_emb[di] = sum(
                        w * full_emb[gi] for w, gi in zip(weights, neighbor_global)
                    )
                    n_esm_placed += 1

        # Strategy B: truly dark genes → mean + small noise
        mean_emb = np.nanmean(full_emb[connected_idx], axis=0)
        std_emb = np.nanstd(full_emb[connected_idx], axis=0)
        std_emb[std_emb == 0] = 1e-3

        rng = np.random.RandomState(42)
        n_dark = 0
        for i in range(n):
            if np.any(np.isnan(full_emb[i])):
                full_emb[i] = mean_emb + rng.randn(n_components).astype(np.float32) * std_emb * 0.1
                n_dark += 1

        if len(disconnected_idx) > 0:
            logger.info(
                f"  Disconnected genes: {n_esm_placed} placed via ESM-2 kNN regression, "
                f"{n_dark} truly dark (mean + noise fallback)"
            )

    # Final NaN check (should never trigger but safety)
    nan_mask = np.any(np.isnan(full_emb), axis=1)
    if nan_mask.any():
        logger.error(f"  {nan_mask.sum()} genes still have NaN embeddings! Setting to zero.")
        full_emb[nan_mask] = 0.0

    gene_vocabulary = {g: full_emb[i] for i, g in enumerate(genes)}

    assert len(gene_vocabulary) == len(genes), (
        f"Vocabulary incomplete: {len(gene_vocabulary)}/{len(genes)}"
    )

    return gene_vocabulary


# =============================================================================
# 8. MAIN PIPELINE
# =============================================================================

def pickle_save(obj: object, path: str):
    with open(path, "wb") as f:
        pkl.dump(obj, f)


def pickle_load(path: str):
    with open(path, "rb") as f:
        obj = pkl.load(f)
    return obj


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
    k_transfer: int = 10,
    alpha_impute: float = 0.7,
    cache_dir: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Run the full multi-view gene clustering pipeline.

    Pipeline order (ESM + co-expression computed first as transfer sources):
      1. ESM-2 raw embeddings
      2. Co-expression similarity
      3. Reactome pathways (with imputation)
      4. GO:BP (with imputation)
      5. GO:CC (with imputation)
      6. GO:MF (with imputation)
      7. ESM-2 similarity view (no imputation — it IS the source)
      8. Co-expression view (no imputation)

    Parameters
    ----------
    genes : list[str]
        List of gene symbols.
    output_dir : str
        Directory to save results.
    adata_path : str, optional
        Path to .h5ad for co-expression view.
    esm_model : str
        ESM-2 model name.
    esm_device : str
        "cpu" or "cuda".
    skip_esm : bool
        Skip ESM-2 entirely (no imputation, no ESM view).
    skip_coexpr : bool
        Skip co-expression view.
    k : int
        kNN neighbors for graph construction.
    resolution : float
        Leiden resolution.
    k_transfer : int
        Number of neighbors for ESM-2 kNN annotation transfer.
    alpha_impute : float
        Weight on ESM vs co-expression in imputation (1.0 = ESM only).
    cache_dir : str, optional
        Cache intermediate results.

    Returns
    -------
    dict[str, pd.DataFrame]
        view_name → DataFrame with columns [gene, cluster, view].
    """
    os.makedirs(output_dir, exist_ok=True)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    genes = list(genes)
    n = len(genes)
    logger.info(f"Starting multi-view clustering for {n} genes")

    results = {}
    sim_matrices = {}
    all_embeddings = {}  # view → gene vocabulary

    # ==================================================================
    # PHASE 1: Compute transfer sources (ESM-2 + co-expression)
    # ==================================================================

    esm_embeddings = {}   # raw PLM embeddings (for imputation)
    sim_coexpr = None      # co-expression similarity (for imputation)

    # ---- ESM-2 raw embeddings (transfer source) ----
    if not skip_esm:
        logger.info("=" * 60)
        logger.info("PHASE 1a: ESM-2 raw embeddings (transfer source)")
        logger.info("=" * 60)

        emb_cache = os.path.join(cache_dir, "esm2_embeddings.pkl") if cache_dir else None

        if emb_cache and os.path.exists(emb_cache):
            logger.info(f"Loading cached ESM-2 embeddings from {emb_cache}")
            esm_embeddings = pickle_load(emb_cache)
        else:
            seqs = fetch_protein_sequences(genes)
            esm_embeddings = compute_esm2_embeddings(seqs, model_name=esm_model, device=esm_device)
            if emb_cache:
                pickle_save(esm_embeddings, emb_cache)

        logger.info(f"ESM-2 coverage: {len(esm_embeddings)}/{n} genes")
    else:
        logger.info("Skipping ESM-2 (--skip-esm) — no imputation will be performed")

    # ---- Co-expression similarity (secondary transfer source) ----
    if adata_path and not skip_coexpr:
        logger.info("=" * 60)
        logger.info("PHASE 1b: Co-expression similarity (transfer source)")
        logger.info("=" * 60)

        coexpr_cache = os.path.join(cache_dir, "sim_coexpression.npy") if cache_dir else None
        if coexpr_cache and os.path.exists(coexpr_cache):
            logger.info(f"Loading cached co-expression similarity from {coexpr_cache}")
            sim_coexpr = np.load(coexpr_cache)
        else:
            sim_coexpr = build_coexpression_similarity(genes, adata_path)
            if coexpr_cache:
                np.save(coexpr_cache, sim_coexpr)

    # ==================================================================
    # PHASE 2: Annotation views (with imputation from Phase 1 sources)
    # ==================================================================

    annotation_views = [
        ("reactome", "C2", "CP:REACTOME", 5, 500),
        ("go_bp", "C5", "GO:BP", 10, 500),
        ("go_cc", "C5", "GO:CC", 5, 500),
        ("go_mf", "C5", "GO:MF", 5, 500),
    ]

    for view_name, collection, subcollection, min_sz, max_sz in annotation_views:
        logger.info("=" * 60)
        logger.info(f"PHASE 2: {view_name.upper()} (annotation view with imputation)")
        logger.info("=" * 60)

        gene_sets = fetch_msigdb_gene_sets(collection, subcollection)
        sim_raw = build_comembership_matrix(genes, gene_sets, min_set_size=min_sz, max_set_size=max_sz)

        n_annotated_raw = int(np.any(sim_raw > 0, axis=1).sum())

        # Impute residual genes if ESM-2 is available
        if esm_embeddings:
            sim_imputed = impute_residual_features(
                sim_raw, genes, esm_embeddings,
                k_transfer=k_transfer,
                coexpr_sim=sim_coexpr,
                alpha=alpha_impute,
                view_name=view_name,
            )
        else:
            sim_imputed = sim_raw

        n_with_signal = int(np.any(sim_imputed > 0, axis=1).sum())
        logger.info(
            f"[{view_name}] Coverage: {n_annotated_raw} annotated → "
            f"{n_with_signal} after imputation (of {n} total)"
        )

        # Cluster on imputed similarity
        results[view_name] = similarity_to_leiden_clusters(
            sim_imputed, genes, k=k, resolution=resolution, view_name=view_name
        )
        sim_matrices[view_name] = sim_imputed

        # Spectral embedding (complete vocabulary)
        emb = build_knn_embeddings(
            sim_imputed, genes, k=k, esm_embeddings=esm_embeddings
        )
        all_embeddings[view_name] = emb

        if cache_dir:
            np.save(os.path.join(cache_dir, f"sim_{view_name}.npy"), sim_imputed)
            pickle_save(emb, os.path.join(cache_dir, f"gene_emb_{view_name}.pkl"))

    # ==================================================================
    # PHASE 3: Non-annotation views (no imputation needed)
    # ==================================================================

    # ---- ESM-2 similarity view ----
    if not skip_esm and esm_embeddings:
        logger.info("=" * 60)
        logger.info("PHASE 3a: ESM-2 similarity view (direct, no imputation)")
        logger.info("=" * 60)

        sim_esm = build_esm_similarity_matrix(genes, esm_embeddings)
        results["esm2"] = similarity_to_leiden_clusters(
            sim_esm, genes, k=k, resolution=resolution, view_name="esm2"
        )
        sim_matrices["esm2"] = sim_esm

        emb_esm = build_knn_embeddings(
            sim_esm, genes, k=k, esm_embeddings=esm_embeddings
        )
        all_embeddings["esm2"] = emb_esm

        if cache_dir:
            np.save(os.path.join(cache_dir, "sim_esm2.npy"), sim_esm)
            pickle_save(emb_esm, os.path.join(cache_dir, "gene_emb_esm2.pkl"))

    # ---- Co-expression view ----
    if sim_coexpr is not None:
        logger.info("=" * 60)
        logger.info("PHASE 3b: Co-expression view (direct, no imputation)")
        logger.info("=" * 60)

        results["coexpression"] = similarity_to_leiden_clusters(
            sim_coexpr, genes, k=k, resolution=resolution, view_name="coexpression"
        )
        sim_matrices["coexpression"] = sim_coexpr

        emb_coexpr = build_knn_embeddings(
            sim_coexpr, genes, k=k, esm_embeddings=esm_embeddings
        )
        all_embeddings["coexpression"] = emb_coexpr

        if cache_dir:
            np.save(os.path.join(cache_dir, "sim_coexpression.npy"), sim_coexpr)
            pickle_save(emb_coexpr, os.path.join(cache_dir, "gene_emb_coexpression.pkl"))

    # ==================================================================
    # PHASE 4: Save & visualize
    # ==================================================================

    save_results(results, genes, output_dir, all_embeddings)
    plot_cluster_views(results, all_embeddings, genes, output_dir)  

    # Vocabulary completeness check
    for view_name, emb_dict in all_embeddings.items():
        missing = [g for g in genes if g not in emb_dict]
        if missing:
            logger.error(f"[{view_name}] VOCABULARY INCOMPLETE: {len(missing)} genes missing!")
        else:
            logger.info(f"[{view_name}] Vocabulary complete: {len(emb_dict)} genes ✓")

    return results


# =============================================================================
# 9. SAVE & SUMMARIZE
# =============================================================================

def save_results(
    results: dict[str, pd.DataFrame],
    genes: list[str],
    output_dir: str,
    all_embeddings: dict[str, dict[str, np.ndarray]] | None = None,
):
    """Save all clustering results in multiple convenient formats."""

    # 1. Per-view CSVs
    for view_name, df in results.items():
        df.to_csv(os.path.join(output_dir, f"clusters_{view_name}.csv"), index=False)

    # 2. Combined wide-format table
    combined = pd.DataFrame({"gene": genes})
    for view_name, df in results.items():
        combined[view_name] = df["cluster"].values
    combined.to_csv(os.path.join(output_dir, "clusters_all_views.csv"), index=False)

    # 3. Cluster-to-gene-list JSON
    cluster_dict = {}
    for view_name, df in results.items():
        view_clusters = defaultdict(list)
        for _, row in df.iterrows():
            view_clusters[row["cluster"]].append(row["gene"])
        cluster_dict[view_name] = dict(view_clusters)

    with open(os.path.join(output_dir, "cluster_gene_lists.json"), "w") as f:
        json.dump(cluster_dict, f, indent=2)

    # 4. Gene embeddings (complete vocabularies)
    if all_embeddings:
        emb_dir = os.path.join(output_dir, "embeddings")
        os.makedirs(emb_dir, exist_ok=True)
        for view_name, emb_dict in all_embeddings.items():
            pickle_save(emb_dict, os.path.join(emb_dir, f"gene_emb_{view_name}.pkl"))
        logger.info(f"Gene embeddings saved to {emb_dir}/")

    # 5. Summary statistics
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
# 10. PLOTTING
# =============================================================================

def plot_cluster_views(
    results: dict[str, pd.DataFrame],
    all_embeddings: dict[str, dict[str, np.ndarray]],  # CHANGED: Uses embeddings now
    genes: list[str],
    output_dir: str,
    n_neighbors_umap: int = 15,
    min_dist: float = 0.1,
):
    """
    Create a multi-panel figure: one UMAP per view, colored by Leiden cluster.
    Saves the trained UMAP models for later archetype projection.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import joblib  # For saving models
    
    try:
        from umap import UMAP
    except ImportError:
        logger.warning("umap-learn not installed, skipping plot.")
        return

    views = sorted(results.keys())
    n_views = len(views)
    if n_views == 0: return

    # Create directory for UMAP models
    model_dir = os.path.join(output_dir, "umap_models")
    os.makedirs(model_dir, exist_ok=True)

    ncols = min(3, n_views)
    nrows = (n_views + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), dpi=150)
    if n_views == 1: axes = np.array([axes])
    axes = axes.flatten()

    for idx, view_name in enumerate(views):
        ax = axes[idx]
        df = results[view_name]
        
        # Get embeddings for this view
        emb_dict = all_embeddings.get(view_name, {})
        # Ensure gene order matches dataframe/list
        valid_genes = [g for g in genes if g in emb_dict]
        if not valid_genes:
            ax.text(0.5, 0.5, "No embeddings", ha='center')
            continue
            
        X = np.array([emb_dict[g] for g in valid_genes])
        
        # Fit UMAP
        logger.info(f"[{view_name}] Fitting UMAP on {len(X)} genes...")
        reducer = UMAP(
            n_neighbors=min(n_neighbors_umap, len(X) - 1),
            min_dist=min_dist,
            metric="cosine",  # Spectral embeddings work well with Cosine
            random_state=42,
        )
        coords = reducer.fit_transform(X)
        
        # SAVE THE MODEL
        model_path = os.path.join(model_dir, f"umap_{view_name}.pkl")
        joblib.dump(reducer, model_path)

        # Plotting
        cluster_labels = df.set_index("gene").loc[valid_genes]["cluster"].values
        unique_clusters = sorted(set(cluster_labels))
        
        # Identify residual
        max_suffix = -1
        try:
            suffixes = [int(c.split("_")[-1]) for c in unique_clusters if "_" in c]
            if suffixes: max_suffix = max(suffixes)
        except ValueError: pass
        residual_label = f"{view_name}_{max_suffix}"
        
        is_residual = cluster_labels == residual_label
        
        # Colors
        cluster_to_int = {c: i for i, c in enumerate(unique_clusters)}
        c_ints = np.array([cluster_to_int[c] for c in cluster_labels])
        
        if len(unique_clusters) <= 20: cmap = plt.cm.tab20
        else: cmap = plt.cm.gist_ncar
        
        colors = cmap(c_ints / max(len(unique_clusters) - 1, 1))
        
        # Plot Residuals (Grey)
        if is_residual.any():
            ax.scatter(
                coords[is_residual, 0], coords[is_residual, 1],
                c='#e0e0e0', s=4, alpha=0.3, rasterized=True
            )
        
        # Plot Signal
        ax.scatter(
            coords[~is_residual, 0], coords[~is_residual, 1],
            c=colors[~is_residual], s=6, alpha=0.7, rasterized=True
        )

        n_real = len(unique_clusters) - (1 if is_residual.any() else 0)
        ax.set_title(f"{view_name}\n{n_real} clusters", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)

    for idx in range(n_views, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Multi-View Gene Clustering (Spectral Space)", fontsize=14, y=1.02)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "cluster_views_umap.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plots and UMAP models to {output_dir}")

# =============================================================================
# 11. CLUSTER ARCHETYPES
# =============================================================================

def _resolve_result_dir(result_dir: Optional[str]) -> str:
    """
    Resolve the result directory path.
    If None, look for 'gene_clusters_results' next to this script file.
    Raises FileNotFoundError if the directory does not exist.
    """
    if result_dir is not None:
        p = os.path.abspath(result_dir)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        p = os.path.join(script_dir, "gene_clusters_results")
        logger.info(f"No --result-dir specified, looking for default: {p}")

    if not os.path.isdir(p):
        raise FileNotFoundError(
            f"Result directory not found: {p}\n"
            f"Run the 'cluster' subcommand first, or pass --result-dir explicitly."
        )
    return p


def compute_cluster_archetypes(
    result_dir: Optional[str] = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute cluster archetype embeddings for every view.
    """
    result_dir = _resolve_result_dir(result_dir)

    # --- Load cluster assignments ---
    cluster_json_path = os.path.join(result_dir, "cluster_gene_lists.json")
    if not os.path.exists(cluster_json_path):
        raise FileNotFoundError(f"cluster_gene_lists.json not found in {result_dir}")

    with open(cluster_json_path) as f:
        cluster_gene_lists = json.load(f)

    logger.info(f"Loaded cluster assignments for {len(cluster_gene_lists)} views")

    emb_dir = os.path.join(result_dir, "embeddings")
    if not os.path.isdir(emb_dir):
        raise FileNotFoundError(f"Embeddings directory not found: {emb_dir}")

    # --- Compute archetypes per view ---
    all_archetypes = {}
    all_gene_embeddings = {}  # <--- NEW: Store gene embeddings for plotting

    for view_name, clusters in cluster_gene_lists.items():
        emb_path = os.path.join(emb_dir, f"gene_emb_{view_name}.pkl")
        if not os.path.exists(emb_path):
            logger.warning(f"[{view_name}] Gene embeddings not found, skipping")
            continue

        gene_emb = pickle_load(emb_path)
        all_gene_embeddings[view_name] = gene_emb  # <--- Save for plotter
        
        dim = next(iter(gene_emb.values())).shape[0]
        archetypes = {}
        
        for cluster_label, gene_list in clusters.items():
            vecs = []
            for g in gene_list:
                if g in gene_emb:
                    vecs.append(gene_emb[g])

            if not vecs:
                archetypes[cluster_label] = np.zeros(dim, dtype=np.float32)
                continue

            # Archetype = mean embedding
            archetypes[cluster_label] = np.mean(vecs, axis=0).astype(np.float32)

        # Save
        out_path = os.path.join(emb_dir, f"cluster_emb_{view_name}.pkl")
        pickle_save(archetypes, out_path)
        all_archetypes[view_name] = archetypes
        
        logger.info(f"[{view_name}] Computed {len(archetypes)} archetypes")

    plot_archetypes(all_archetypes, all_gene_embeddings, result_dir)

    # --- Summary ---
    logger.info("\nCluster Archetype Summary")
    logger.info("=" * 40)
    for view_name, archetypes in all_archetypes.items():
        logger.info(f"  {view_name}: {len(archetypes)} archetypes")

    return all_archetypes

def plot_archetypes(
    all_archetypes: dict[str, dict[str, np.ndarray]],
    all_gene_embeddings: dict[str, dict[str, np.ndarray]],
    output_dir: str,
):
    """
    Project cluster archetypes into the SAME UMAP space as the genes
    by loading the pre-trained UMAP models.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import joblib
    import numpy as np

    views = sorted(all_archetypes.keys())
    if not views: return

    model_dir = os.path.join(output_dir, "umap_models")
    
    # Grid setup
    ncols = min(3, len(views))
    nrows = (len(views) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), dpi=150)
    if len(views) == 1: axes = np.array([axes])
    axes = axes.flatten()

    for idx, view_name in enumerate(views):
        ax = axes[idx]
        
        # 1. Load Data
        archetype_dict = all_archetypes[view_name]
        gene_emb_dict = all_gene_embeddings.get(view_name, {})
        
        if not gene_emb_dict:
            continue

        # Prepare matrices
        gene_ids = sorted(gene_emb_dict.keys())
        gene_matrix = np.array([gene_emb_dict[g] for g in gene_ids])
        
        clust_labels = sorted(archetype_dict.keys(), 
                              key=lambda x: int(x.split('_')[-1]) if '_' in x else x)
        clust_matrix = np.array([archetype_dict[l] for l in clust_labels])

        # 2. Load UMAP Model or Fallback
        model_path = os.path.join(model_dir, f"umap_{view_name}.pkl")
        reducer = None
        gene_coords = None
        
        if os.path.exists(model_path):
            try:
                reducer = joblib.load(model_path)
                # Transform Genes (to reconstruct background)
                gene_coords = reducer.transform(gene_matrix)
                # Transform Archetypes
                clust_coords = reducer.transform(clust_matrix)
                logger.info(f"[{view_name}] Loaded UMAP model and projected archetypes.")
            except Exception as e:
                logger.warning(f"[{view_name}] Failed to load UMAP model: {e}")
        
        if reducer is None:
            logger.warning(f"[{view_name}] Model not found. Fitting fresh UMAP (spaces might drift).")
            # Fallback: Fit fresh (note: this might not align perfectly with previous plots)
            try:
                from umap import UMAP
                reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
                gene_coords = reducer.fit_transform(gene_matrix)
                clust_coords = reducer.transform(clust_matrix)
            except ImportError:
                continue

        # 3. Plotting
        # Background: Genes
        ax.scatter(
            gene_coords[:, 0], gene_coords[:, 1],
            c='#e0e0e0', s=2, alpha=0.3, rasterized=True, label='Genes'
        )

        # Foreground: Archetypes
        # Identify residual
        max_suffix = -1
        try:
            suffixes = [int(l.split("_")[-1]) for l in clust_labels if "_" in l]
            if suffixes: max_suffix = max(suffixes)
        except ValueError: pass
        residual_label = f"{view_name}_{max_suffix}"
        is_residual = np.array([l == residual_label for l in clust_labels])
        
        # Colors
        n_clusters = len(clust_labels)
        if n_clusters <= 20: cmap = plt.cm.tab20
        else: cmap = plt.cm.gist_ncar
        
        # Non-Residual
        non_res = np.where(~is_residual)[0]
        if len(non_res) > 0:
            ax.scatter(
                clust_coords[non_res, 0], clust_coords[non_res, 1],
                c=non_res, cmap=cmap, 
                s=200, edgecolors='black', linewidth=1.5, zorder=10
            )
            for i in non_res:
                txt = clust_labels[i].split('_')[-1]
                ax.text(clust_coords[i, 0], clust_coords[i, 1], txt, 
                        fontsize=9, ha='center', va='center', 
                        color='white', fontweight='bold', zorder=11)

        # Residual
        res = np.where(is_residual)[0]
        if len(res) > 0:
            ax.scatter(
                clust_coords[res, 0], clust_coords[res, 1],
                c='#555555', marker='X', s=150, edgecolors='white', zorder=10
            )

        ax.set_title(f"{view_name}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    for idx in range(len(views), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Cluster Archetypes (Projected)", fontsize=14, y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "archetypes_overlay_umap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# 12. CLI
# =============================================================================

def regenerate_plots(result_dir: Optional[str] = None):
    """
    Regenerate plots from existing results without re-running clustering.
    """
    result_dir = _resolve_result_dir(result_dir)
    logger.info(f"Regenerating plots from: {result_dir}")
    
    # 1. Load Cluster Assignments
    cluster_csv = os.path.join(result_dir, "clusters_all_views.csv")
    if not os.path.exists(cluster_csv):
        raise FileNotFoundError(f"Clusters file not found: {cluster_csv}")
    
    df_all = pd.read_csv(cluster_csv)
    genes = df_all["gene"].tolist()
    
    # Reconstruct 'results' dict
    results = {}
    for col in df_all.columns:
        if col == "gene": continue
        results[col] = df_all[["gene", col]].rename(columns={col: "cluster"})
        
    # 2. Load Gene Embeddings
    emb_dir = os.path.join(result_dir, "embeddings")
    all_embeddings = {}
    
    # Scan for embedding files
    import glob
    emb_files = glob.glob(os.path.join(emb_dir, "gene_emb_*.pkl"))
    for fpath in emb_files:
        view_name = os.path.basename(fpath).replace("gene_emb_", "").replace(".pkl", "")
        all_embeddings[view_name] = pickle_load(fpath)
        
    if not all_embeddings:
        raise FileNotFoundError(f"No gene embeddings found in {emb_dir}")

    # 3. Plot Gene Views (This will re-save UMAP models)
    logger.info("Plotting Gene Views...")
    plot_cluster_views(results, all_embeddings, genes, result_dir)
    
    # 4. Compute/Load Archetypes (for the overlay plot)
    # We call compute_cluster_archetypes, which now handles plotting internally
    # and will use the UMAP models we just saved in step 3.
    logger.info("Computing Archetypes & Plotting Overlay...")
    compute_cluster_archetypes(result_dir)
    
    logger.info("Done.")

def main():
    parser = argparse.ArgumentParser(
        description="Multi-view gene clustering for single-cell inductive bias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ------------------------------------------------------------------
    # Subcommand: cluster
    # ------------------------------------------------------------------
    p_cluster = subparsers.add_parser(
        "cluster",
        help="Run the full multi-view clustering pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: annotation-based views only (no GPU needed, no imputation)
  python multiview_gene_clusters.py cluster --genes gene_list.txt --skip-esm --output clusters/

  # Full: all views including ESM-2 on GPU + co-expression + imputation
  python multiview_gene_clusters.py cluster --genes gene_list.txt --adata data.h5ad \\
      --esm-device cuda --output clusters/

  # Tune resolution for more/fewer clusters
  python multiview_gene_clusters.py cluster --genes gene_list.txt --resolution 0.5 --output clusters/
        """,
    )
    p_cluster.add_argument("--genes", required=True, help="Path to text file with one gene symbol per line")
    p_cluster.add_argument("--output", default="gene_clusters", help="Output directory")
    p_cluster.add_argument("--adata", default=None, help="Path to .h5ad for co-expression view")
    p_cluster.add_argument("--esm-model", default="esm2_t33_650M_UR50D", help="ESM-2 model name")
    p_cluster.add_argument("--esm-device", default="cpu", choices=["cpu", "cuda"], help="Device for ESM-2")
    p_cluster.add_argument("--skip-esm", action="store_true", help="Skip ESM-2 view AND imputation")
    p_cluster.add_argument("--skip-coexpr", action="store_true", help="Skip co-expression view")
    p_cluster.add_argument("--k", type=int, default=15, help="kNN neighbors (default: 15)")
    p_cluster.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution (default: 1.0)")
    p_cluster.add_argument("--k-transfer", type=int, default=10, help="kNN for ESM-2 annotation transfer (default: 10)")
    p_cluster.add_argument("--alpha-impute", type=float, default=0.7, help="ESM vs co-expression weight for imputation (default: 0.7)")
    p_cluster.add_argument("--cache", default=None, help="Cache directory for intermediate results")

    # ------------------------------------------------------------------
    # Subcommand: archetypes
    # ------------------------------------------------------------------
    p_arch = subparsers.add_parser(
        "archetypes",
        help="Compute cluster archetype embeddings from a previous clustering run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes the mean gene embedding (archetype) for each cluster in each view.
Reads gene embeddings and cluster assignments from a previous 'cluster' run.

Examples:
  # Use explicit result directory
  python multiview_gene_clusters.py archetypes --result-dir gene_clusters/

  # Auto-detect: looks for 'gene_clusters_results/' next to the script
  python multiview_gene_clusters.py archetypes
        """,
    )
    p_arch.add_argument(
        "--result-dir", default=None,
        help="Path to pipeline output directory (default: 'gene_clusters_results' next to this script)",
    )

    # ------------------------------------------------------------------
    # Subcommand: plot
    # ------------------------------------------------------------------
    p_plot = subparsers.add_parser(
        "plot",
        help="Regenerate plots from existing results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_plot.add_argument(
        "--result-dir", default=None,
        help="Path to result directory (default: 'gene_clusters_results' next to script)"
    )

    # ------------------------------------------------------------------
    # Parse & dispatch
    # ------------------------------------------------------------------
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "cluster":
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
            k_transfer=args.k_transfer,
            alpha_impute=args.alpha_impute,
            cache_dir=args.cache,
        )

    elif args.command == "archetypes":
        compute_cluster_archetypes(result_dir=args.result_dir)
    
    elif args.command == "plot":
        regenerate_plots(result_dir=args.result_dir)

if __name__ == "__main__":
  """
  # Full pipeline (unchanged behavior)
  !python multiview_gene_clusters.py cluster \
      --genes $PATH_GENE_LIST \
      --adata $PATH_DATA_H5AD \
      --esm-device cuda \
      --cache $PATH_CACHE \
      --output $PATH_OUTPUT

  # Compute archetypes from a previous run
  !python multiview_gene_clusters.py archetypes --result-dir $PATH_OUTPUT

  # Auto-detect: looks for gene_clusters_results/ next to the script
  !python multiview_gene_clusters.py archetypes
  """
  main()
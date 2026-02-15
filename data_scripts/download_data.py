#!/usr/bin/env python3
"""
Download & Preprocess Datasets for Perturbation Prediction
============================================================

Downloads raw single-cell CRISPRi datasets from public repositories and applies
a uniform preprocessing pipeline. Each dataset can be downloaded independently
via the --dataset flag, making it easy to retry a single failed download.

Preprocessing pipeline (applied to any .h5ad with raw counts):
    raw UMI counts → per-cell library-size normalization → scale to 10,000 → log2(x + 1)

Usage:
    # Download and preprocess ALL datasets
    python download_data.py --all --gene-list data/gene_list.txt

    # Download a single dataset
    python download_data.py --dataset k562 --gene-list data/gene_list.txt

    # Download without preprocessing (raw only)
    python download_data.py --dataset k562 --no-preprocess

    # Preprocess an existing .h5ad file
    python download_data.py --preprocess-only path/to/file.h5ad --gene-list data/gene_list.txt

    # List available datasets
    python download_data.py --list

Datasets:
    k562     - Replogle et al. 2022, K562 essential (Figshare)
    rpe1     - Replogle et al. 2022, RPE1 essential (Figshare)
    jurkat   - Nadig et al. 2024 (TRADE), Jurkat essential (Zenodo)
    hepg2    - Nadig et al. 2024 (TRADE), HepG2 essential (Zenodo)
    cd4t     - Biohub/Marson Lab, CD4+ T cells Stim8hr (CZI)
    challenge - Competition training data (local, preprocess only)

Author: MylliaESG project
"""

import argparse
import logging
import json
import requests
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# --- Download URLs ---
# NOTE: These URLs may need updating. The script will log clear errors if they fail.
# Figshare article: https://plus.figshare.com/articles/dataset/20029387
# To find direct download links: go to the Figshare page → click file → copy download link

DATASET_REGISTRY = {
    "k562": {
        "description": "Replogle et al. 2022 — K562 Essential CRISPRi screen",
        "source": "replogle",
        "cell_type": "K562",
        "role": "train",
        "download_type": "figshare",
        # Figshare ndownloader link for K562_essential_raw_singlecell_01.h5ad
        # Update this URL from: https://plus.figshare.com/articles/dataset/20029387
        "url": "https://plus.figshare.com/ndownloader/files/35773075",
        "filename": "K562_essential_raw_singlecell_01.h5ad",
        "obs_perturbation_col": "gene",          # column containing perturbation gene symbol
        "obs_control_value": "non-targeting",     # value indicating control cells
        "obs_batch_col": "gem_group",             # batch / lane column
        "expected_cells_approx": 400_000,
    },
    "rpe1": {
        "description": "Replogle et al. 2022 — RPE1 Essential CRISPRi screen",
        "source": "replogle",
        "cell_type": "RPE1",
        "role": "eval",
        "download_type": "figshare",
        "url": "https://plus.figshare.com/ndownloader/files/35775606",
        "filename": "rpe1_raw_singlecell_01.h5ad",
        "obs_perturbation_col": "gene",
        "obs_control_value": "non-targeting",
        "obs_batch_col": "gem_group",
        "expected_cells_approx": 300_000,
    },
    "jurkat": {
        "description": "Nadig et al. 2024 (TRADE) — Jurkat Essential CRISPRi screen",
        "source": "nadig",
        "cell_type": "Jurkat",
        "role": "train",
        "download_type": "zenodo",
        # Update from TRADE repo / Zenodo: https://pmc.ncbi.nlm.nih.gov/articles/PMC11244993/
        # Look for the Zenodo DOI in the Data Availability section
        "url": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE264667&format=file&file=GSE264667%5Fjurkat%5Fraw%5Fsinglecell%5F01%2Eh5ad",
        "filename": "jurkat_essential_raw.h5ad",
        "obs_perturbation_col": "gene",
        "obs_control_value": "non-targeting",
        "obs_batch_col": "batch",
        "expected_cells_approx": 150_000,
    },
    "hepg2": {
        "description": "Nadig et al. 2024 (TRADE) — HepG2 Essential CRISPRi screen",
        "source": "nadig",
        "cell_type": "HepG2",
        "role": "eval",
        "download_type": "zenodo",
        "url": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE264667&format=file&file=GSE264667%5Fhepg2%5Fraw%5Fsinglecell%5F01%2Eh5ad",
        "filename": "hepg2_essential_raw.h5ad",
        "obs_perturbation_col": "gene",
        "obs_control_value": "non-targeting",
        "obs_batch_col": "batch",
        "expected_cells_approx": 150_000,
    },
    "cd4t": {
        "description": "Biohub/Marson Lab — CD4+ T cells, genome-scale CRISPRi",
        "source": "biohub",
        "cell_type": "CD4T_stim8hr",
        "role": "train",
        "download_type": "czi",
        # CZI Virtual Cells Platform dataset page
        "url": "https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq",
        "filename": "cd4t_stim8hr_raw.h5ad",
        "obs_perturbation_col": "perturbation",
        "obs_control_value": "non-targeting",
        "obs_batch_col": "donor",
        "expected_cells_approx": 5_000_000,  # Stim8hr subset of 22M total
        "condition_col": "condition",
        "condition_value": "Stim8hr",
        "donor_subset": 2,  # Use 1-2 donors to keep manageable
    },
    "challenge": {
        "description": "Myllia ESG challenge training data (local file)",
        "source": "challenge",
        "cell_type": "challenge",
        "role": "reference",
        "download_type": "local",
        "url": None,
        "filename": "training_cells.h5ad",
        "obs_perturbation_col": "sgrna_symbol",
        "obs_control_value": "non-targeting",
        "obs_batch_col": "channel",
        "expected_cells_approx": 17_882,
    },
}


# ============================================================================
# PREPROCESSING PIPELINE (modular, works on any .h5ad)
# ============================================================================

def preprocess_h5ad(
    adata,
    target_sum: float = 10_000,
    log_base: int = 2,
    gene_list: Optional[list[str]] = None,
    gene_symbol_col: Optional[str] = None,
    copy: bool = True,
) -> "anndata.AnnData":
    """
    Universal preprocessing pipeline for raw count AnnData.

    Steps:
        1. Ensure we're working with raw counts (detect if already normalized)
        2. Per-cell library-size normalization (divide by total UMI)
        3. Scale to target_sum (default 10,000)
        4. log2(x + 1) transformation
        5. Optionally subset to a gene list

    Parameters
    ----------
    adata : AnnData
        Input AnnData with raw UMI counts in .X
    target_sum : float
        Scale factor after normalization (default 10,000)
    log_base : int
        Log base for transformation (2 for log2, use 0 for natural log)
    gene_list : list[str], optional
        If provided, subset to these genes after normalization.
    gene_symbol_col : str, optional
        Column in .var containing gene symbols. If None, uses .var_names.
    copy : bool
        If True, operate on a copy.

    Returns
    -------
    AnnData with normalized expression in .X and raw counts in .layers["raw_counts"]
    """
    import anndata
    from scipy import sparse

    if copy:
        adata = adata.copy()

    n_cells, n_genes = adata.shape
    logger.info(f"Preprocessing: {n_cells:,} cells × {n_genes:,} genes")

    # --- Step 0: Store raw counts ---
    if sparse.issparse(adata.X):
        X = adata.X.toarray().astype(np.float32)
    else:
        X = np.array(adata.X, dtype=np.float32)

    # Sanity check: detect if already normalized
    row_sums = X.sum(axis=1)
    min_sum, max_sum = row_sums.min(), row_sums.max()
    median_sum = np.median(row_sums)
    logger.info(
        f"  UMI counts per cell: min={min_sum:.0f}, median={median_sum:.0f}, max={max_sum:.0f}"
    )

    # If values are very small or uniform, warn about possible pre-normalization
    if median_sum < 100:
        logger.warning(
            "  ⚠ Median UMI count is very low (<100). Data may already be normalized "
            "or log-transformed. Proceeding anyway — verify your input."
        )
    if np.any(X < 0):
        logger.warning("  ⚠ Negative values detected. Data may already be log-transformed.")
    if np.allclose(row_sums, row_sums[0], rtol=1e-3):
        logger.warning(
            "  ⚠ All cells have near-identical total counts — data may already be "
            "library-size normalized."
        )

    # Store raw counts before transformation
    adata.layers["raw_counts"] = sparse.csr_matrix(X.copy())
    logger.info("  Stored raw counts in .layers['raw_counts']")

    # --- Step 1: Per-cell library-size normalization ---
    cell_totals = X.sum(axis=1, keepdims=True)
    cell_totals[cell_totals == 0] = 1.0  # avoid division by zero for empty cells
    X = X / cell_totals
    logger.info("  Per-cell library-size normalization done")

    # --- Step 2: Scale to target_sum ---
    X = X * target_sum
    logger.info(f"  Scaled to target_sum={target_sum:,.0f}")

    # --- Step 3: Log transformation ---
    if log_base == 2:
        X = np.log2(X + 1)
        logger.info("  Applied log2(x + 1)")
    elif log_base == 0:
        X = np.log1p(X)  # natural log
        logger.info("  Applied ln(x + 1)")
    else:
        X = np.log(X + 1) / np.log(log_base)
        logger.info(f"  Applied log{log_base}(x + 1)")

    adata.X = sparse.csr_matrix(X)

    # --- Step 4: Gene subsetting ---
    if gene_list is not None:
        # Resolve gene symbols
        if gene_symbol_col and gene_symbol_col in adata.var.columns:
            var_genes = adata.var[gene_symbol_col].values
        else:
            var_genes = adata.var_names.values

        gene_set = set(gene_list)
        available = [g for g in var_genes if g in gene_set]
        missing = gene_set - set(available)

        logger.info(
            f"  Gene list subsetting: {len(available)}/{len(gene_list)} genes found "
            f"({len(missing)} missing from this dataset)"
        )
        if len(missing) > 0 and len(missing) <= 20:
            logger.info(f"    Missing genes: {sorted(missing)}")
        elif len(missing) > 20:
            logger.info(f"    First 20 missing: {sorted(missing)[:20]} ...")

        # Subset
        mask = np.isin(var_genes, list(gene_set))
        adata = adata[:, mask].copy()
        logger.info(f"  After subsetting: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

    # --- Final stats ---
    if sparse.issparse(adata.X):
        Xd = adata.X.toarray()
    else:
        Xd = adata.X
    logger.info(
        f"  Normalized expression stats: "
        f"min={Xd.min():.3f}, mean={Xd.mean():.3f}, max={Xd.max():.3f}, "
        f"sparsity={100 * (Xd == 0).mean():.1f}%"
    )

    return adata


# ============================================================================
# HGNC GENE SYMBOL HARMONIZATION (Dynamic)
# ============================================================================

def fetch_hgnc_mapper(
    genes_to_check: list[str], 
    cache_path: Path = DATA_DIR / "hgnc_complete_set.json"
) -> dict:
    """
    Builds a mapping dictionary {old_symbol: new_symbol} for a specific list of genes
    by querying the official HGNC complete set.
    
    Strategy:
    1. Downloads 'hgnc_complete_set.json' from EBI (cached locally).
    2. Builds a lookup table from Previous/Alias symbols -> Approved Symbol.
    3. Returns a dict containing ONLY the genes from 'genes_to_check' that need renaming.
    """
    # 1. Download HGNC reference if missing
    if not cache_path.exists():
        url = "http://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/json/hgnc_complete_set.json"
        logger.info(f"  Downloading HGNC complete set for gene harmonization...")
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(cache_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"  ✓ Cached HGNC reference to {cache_path}")
        except Exception as e:
            logger.warning(f"  ⚠ Could not download HGNC reference: {e}. Skipping harmonization.")
            return {}

    # 2. Load and build lookup tables
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logger.warning("  ⚠ HGNC cache file is corrupt. Skipping harmonization.")
        return {}

    # Hierarchy: Approved > Previous > Alias
    # We build the map in reverse priority so high priority overwrites low priority
    lookup = {}
    
    docs = data.get('response', {}).get('docs', [])
    
    # Pass 1: Aliases (lowest priority)
    for doc in docs:
        approved = doc.get('symbol')
        if not approved: continue
        
        # specific fix: many excel dates (MARCH1, SEPT1) are in 'alias_symbol'
        if 'alias_symbol' in doc:
            for alias in doc['alias_symbol']:
                lookup[alias] = approved
                
    # Pass 2: Previous Symbols (higher priority - these are official past names)
    for doc in docs:
        approved = doc.get('symbol')
        if not approved: continue
        
        if 'prev_symbol' in doc:
            for prev in doc['prev_symbol']:
                lookup[prev] = approved
                
    # Pass 3: Approved Symbols (highest priority - Identity)
    # If a gene in the dataset is ALREADY approved, it maps to itself.
    # This prevents mapping a valid gene to something else just because it's an alias elsewhere.
    for doc in docs:
        approved = doc.get('symbol')
        lookup[approved] = approved

    # 3. Generate mapping for the specific input list
    mapping = {}
    genes_set = set(genes_to_check)
    
    for gene in genes_set:
        # If gene is not in lookup, we can't fix it (or it's a non-gene entity)
        if gene in lookup:
            current_approved = lookup[gene]
            # Only record if it's actually a rename
            if gene != current_approved:
                mapping[gene] = current_approved
                
    return mapping


def harmonize_gene_symbols(adata, gene_symbol_col: Optional[str] = None) -> "anndata.AnnData":
    """
    Rename outdated HGNC gene symbols to their current versions using the HGNC API.
    Updates both var_names and the gene_symbol_col if provided.
    """
    # Identify which list of genes we are checking
    if gene_symbol_col and gene_symbol_col in adata.var.columns:
        query_genes = adata.var[gene_symbol_col].astype(str).values.tolist()
    else:
        query_genes = adata.var_names.tolist()

    # Generate the dynamic mapping
    # This only runs once per dataset load
    mapper = fetch_hgnc_mapper(query_genes)
    
    if not mapper:
        logger.info("  No gene symbol renames identified (or HGNC fetch failed).")
        return adata

    rename_count = 0

    # Update gene_symbol_col if it exists
    if gene_symbol_col and gene_symbol_col in adata.var.columns:
        # We use a vectorized map or list comprehension
        original_vals = adata.var[gene_symbol_col].values
        new_vals = [mapper.get(g, g) for g in original_vals]
        
        # Count differences
        rename_count = sum(1 for o, n in zip(original_vals, new_vals) if o != n)
        adata.var[gene_symbol_col] = new_vals

    # Update .var_names
    # Note: If gene_symbol_col was used for the query, we might still want to fix var_names
    # if they look like gene symbols.
    new_index = [mapper.get(g, g) for g in adata.var_names]
    
    # If we haven't counted renames yet (no col provided), count now
    if rename_count == 0:
        rename_count = sum(1 for o, n in zip(adata.var_names, new_index) if o != n)

    # Handle duplicates after renaming (e.g., if prev_symbol and alias both map to same new target)
    seen = {}
    deduped = []
    for name in new_index:
        if name in seen:
            seen[name] += 1
            deduped.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            deduped.append(name)

    adata.var_names = pd.Index(deduped)

    if rename_count > 0:
        logger.info(f"  Harmonized {rename_count} gene symbols using HGNC reference")
        # Optional: Print a few examples
        examples = list(mapper.items())[:5]
        logger.info(f"    Examples: {examples}")
    else:
        logger.info("  Gene symbols appear up-to-date.")

    return adata


# ============================================================================
# DOWNLOAD HELPERS
# ============================================================================

def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """
    Download a file using wget (with retry) or fallback to Python requests.
    Returns True on success.
    """
    if url.startswith("ZENODO_URL_PLACEHOLDER") or url.startswith("CZI_URL_PLACEHOLDER"):
        logger.error(
            f"  ✗ URL placeholder detected for {desc}. "
            f"You need to manually find the download URL and update DATASET_REGISTRY."
        )
        logger.error(f"    See the docstring at the top of this file for guidance.")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info(f"  File already exists: {dest} ({size_mb:.1f} MB). Skipping download.")
        return True

    logger.info(f"  Downloading {desc or url}")
    logger.info(f"    → {dest}")

    # Try wget first (better for large files, shows progress)
    try:
        cmd = [
            "wget", "-q", "--show-progress", "--tries=3", "--timeout=120",
            "-O", str(dest), url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            size_mb = dest.stat().st_size / (1024 * 1024)
            logger.info(f"  ✓ Downloaded ({size_mb:.1f} MB)")
            return True
        else:
            logger.warning(f"  wget failed (rc={result.returncode}), trying Python fallback...")
            if dest.exists():
                dest.unlink()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("  wget not available or timed out, trying Python fallback...")

    # Python fallback
    try:
        import requests
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        downloaded = 0
        last_report = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded - last_report > total * 0.1:
                    pct = 100 * downloaded / total
                    logger.info(f"    Progress: {pct:.0f}% ({downloaded / 1e6:.0f} MB)")
                    last_report = downloaded

        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ Downloaded ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        logger.error(f"  ✗ Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ============================================================================
# DATASET-SPECIFIC LOADERS
# ============================================================================

def load_and_tag_metadata(
    adata,
    config: dict,
    dataset_name: str,
) -> "anndata.AnnData":
    """
    Standardize metadata columns for a single dataset.
    Adds: cell_type, perturbation, is_control, dataset_source, batch,
          sgrna_id, nCount_RNA, nFeature_RNA, percent_mt

    This creates a uniform .obs schema regardless of the source dataset's
    original column names.
    """
    from scipy import sparse

    obs = adata.obs.copy()
    n_cells = len(obs)

    logger.info(f"  Tagging metadata for {dataset_name} ({n_cells:,} cells)")
    logger.info(f"    Original .obs columns: {list(obs.columns)}")

    # --- cell_type ---
    obs["cell_type"] = config["cell_type"]

    # --- dataset_source ---
    obs["dataset_source"] = config["source"]

    # --- perturbation ---
    pert_col = config["obs_perturbation_col"]
    control_val = config["obs_control_value"]

    if pert_col in obs.columns:
        obs["perturbation"] = obs[pert_col].astype(str)
        # Harmonize control labels
        control_mask = obs["perturbation"].str.lower().isin([
            "non-targeting", "nontargeting", "non_targeting",
            "control", "ctrl", "nt", "neg_ctrl", "negative_control",
            "safe-targeting", "safe_targeting",
        ])
        obs.loc[control_mask, "perturbation"] = "control"
        n_ctrl = control_mask.sum()
        n_pert = n_cells - n_ctrl
        logger.info(f"    Perturbations: {n_pert:,} perturbed, {n_ctrl:,} control cells")
    else:
        logger.warning(f"    Perturbation column '{pert_col}' not found in .obs!")
        logger.warning(f"    Available columns: {list(obs.columns)}")
        obs["perturbation"] = "unknown"

    # --- is_control ---
    obs["is_control"] = (obs["perturbation"] == "control")

    # --- batch ---
    batch_col = config.get("obs_batch_col", None)
    if batch_col and batch_col in obs.columns:
        obs["batch"] = obs[batch_col].astype(str)
        n_batches = obs["batch"].nunique()
        logger.info(f"    Batches: {n_batches} unique values from '{batch_col}'")
    else:
        obs["batch"] = f"{dataset_name}_batch0"
        logger.info(f"    No batch column found ('{batch_col}'), using default")

    # --- nCount_RNA (total UMI per cell) ---
    if "nCount_RNA" in obs.columns:
        logger.info(f"    nCount_RNA already present (median={obs['nCount_RNA'].median():.0f})")
    else:
        if sparse.issparse(adata.X):
            counts = np.array(adata.X.sum(axis=1)).flatten()
        else:
            counts = adata.X.sum(axis=1)
        obs["nCount_RNA"] = counts
        logger.info(f"    Computed nCount_RNA (median={np.median(counts):.0f})")

    # --- nFeature_RNA (genes detected per cell) ---
    if "nFeature_RNA" in obs.columns:
        logger.info(f"    nFeature_RNA already present (median={obs['nFeature_RNA'].median():.0f})")
    else:
        if sparse.issparse(adata.X):
            n_features = np.array((adata.X > 0).sum(axis=1)).flatten()
        else:
            n_features = (adata.X > 0).sum(axis=1)
        obs["nFeature_RNA"] = n_features
        logger.info(f"    Computed nFeature_RNA (median={np.median(n_features):.0f})")

    # --- percent_mt ---
    if "percent_mt" in obs.columns:
        logger.info(f"    percent_mt already present (median={obs['percent_mt'].median():.2f}%)")
    else:
        mt_genes = [g for g in adata.var_names if g.startswith("MT-") or g.startswith("mt-")]
        if mt_genes:
            if sparse.issparse(adata.X):
                mt_mask = np.isin(adata.var_names, mt_genes)
                mt_counts = np.array(adata.X[:, mt_mask].sum(axis=1)).flatten()
                total_counts = np.array(adata.X.sum(axis=1)).flatten()
            else:
                mt_mask = np.isin(adata.var_names, mt_genes)
                mt_counts = adata.X[:, mt_mask].sum(axis=1)
                total_counts = adata.X.sum(axis=1)
            total_counts[total_counts == 0] = 1
            obs["percent_mt"] = 100 * mt_counts / total_counts
            logger.info(
                f"    Computed percent_mt from {len(mt_genes)} MT genes "
                f"(median={obs['percent_mt'].median():.2f}%)"
            )
        else:
            obs["percent_mt"] = 0.0
            logger.info("    No MT- genes found, setting percent_mt=0")

    # --- sgrna_id (preserve if available) ---
    if "sgrna_id" in obs.columns:
        logger.info(f"    sgrna_id already present")
    else:
        obs["sgrna_id"] = "unknown"

    # Assign cleaned obs back
    adata.obs = obs

    # Log perturbation summary
    pert_counts = obs.loc[~obs["is_control"], "perturbation"].value_counts()
    logger.info(
        f"    Unique perturbation targets: {len(pert_counts)}, "
        f"median cells/pert: {pert_counts.median():.0f}"
    )

    return adata


def download_replogle(dataset_name: str, config: dict, raw_dir: Path) -> Optional[Path]:
    """Download Replogle et al. 2022 data from Figshare."""
    logger.info(f"{'=' * 60}")
    logger.info(f"Downloading {config['description']}")
    logger.info(f"{'=' * 60}")

    dest = raw_dir / config["filename"]
    success = download_file(config["url"], dest, desc=config["filename"])
    return dest if success else None


def download_nadig(dataset_name: str, config: dict, raw_dir: Path) -> Optional[Path]:
    """
    Download Nadig et al. 2024 (TRADE) data from Zenodo.

    NOTE: You need to manually set the Zenodo URL in DATASET_REGISTRY.
    To find it:
      1. Go to: https://pmc.ncbi.nlm.nih.gov/articles/PMC11244993/
      2. Find the Data Availability section → Zenodo DOI link
      3. On Zenodo, find the raw .h5ad files for Jurkat and HepG2
      4. Copy the direct download URLs into DATASET_REGISTRY
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"Downloading {config['description']}")
    logger.info(f"{'=' * 60}")

    if "PLACEHOLDER" in config["url"]:
        logger.error(
            f"  ✗ Zenodo URL not configured for {dataset_name}.\n"
            f"    To fix: find the download URL from the TRADE paper's Zenodo deposit\n"
            f"    (https://pmc.ncbi.nlm.nih.gov/articles/PMC11244993/)\n"
            f"    and update DATASET_REGISTRY['{dataset_name}']['url'] in this script.\n"
            f"\n"
            f"    Alternatively, if you already have the file, place it at:\n"
            f"      {raw_dir / config['filename']}"
        )
        # Check if user placed it manually
        dest = raw_dir / config["filename"]
        if dest.exists():
            logger.info(f"  Found manually placed file: {dest}")
            return dest
        return None

    dest = raw_dir / config["filename"]
    success = download_file(config["url"], dest, desc=config["filename"])
    return dest if success else None


# def download_cd4t(dataset_name: str, config: dict, raw_dir: Path) -> Optional[Path]:
#     """
#     Download CD4+ T cell data from CZI Virtual Cells Platform.

#     This dataset is large (~22M cells). We download and immediately filter to:
#       - Stim8hr condition only
#       - 1-2 donors
#     to keep it manageable.

#     NOTE: CZI data access may require their Python SDK. If the direct URL doesn't
#     work, you may need to:
#       1. pip install cellxgene-census
#       2. Use the API to download the specific slice
#     """
#     logger.info(f"{'=' * 60}")
#     logger.info(f"Downloading {config['description']}")
#     logger.info(f"{'=' * 60}")

#     dest = raw_dir / config["filename"]

#     if dest.exists():
#         size_mb = dest.stat().st_size / (1024 * 1024)
#         logger.info(f"  File already exists: {dest} ({size_mb:.1f} MB)")
#         return dest

#     logger.info(
#         "  CD4+ T cell data is hosted on the CZI Virtual Cells Platform.\n"
#         "  This may require the cellxgene-census SDK or manual download.\n"
#         f"  URL: {config['url']}\n"
#         "\n"
#         "  Option 1 — Manual download:\n"
#         f"    Download the Stim8hr AnnData from the CZI platform and place at:\n"
#         f"      {dest}\n"
#         "\n"
#         "  Option 2 — cellxgene-census SDK:\n"
#         "    pip install cellxgene-census\n"
#         "    Then use the filtering API to get Stim8hr + 2 donors."
#     )

#     # Attempt SDK-based download
#     try:
#         logger.info("  Attempting cellxgene-census SDK download...")
#         import cellxgene_census

#         # This is a template — the exact API calls depend on how the dataset
#         # is organized on the platform. Adjust as needed.
#         logger.warning(
#             "  cellxgene-census SDK found but automated download for this specific "
#             "dataset requires dataset-specific code. Please download manually."
#         )
#         return None

#     except ImportError:
#         logger.info("  cellxgene-census not installed. Manual download required.")
#         return None

def download_cd4t(dataset_name: str, config: dict, raw_dir: Path) -> Optional[Path]:
    """
    Download CD4+ T cell data using CZI VCP CLI and immediately filter.

    This function:
      1. Searches for the dataset using 'vcp'.
      2. Downloads the raw file (prioritizing a 'Stim8hr' subset if available) to a temp file.
      3. Loads the temp file (backed mode) and applies filters:
         - Condition = Stim8hr
         - Donors = First 2 donors
      4. Saves the filtered subset to the final destination and deletes the temp file.
    """
    import shutil
    import subprocess
    import anndata

    logger.info(f"{'=' * 60}")
    logger.info(f"Downloading {config['description']}")
    logger.info(f"{'=' * 60}")

    dest = raw_dir / config["filename"]

    # 1. Check if final file already exists
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info(f"  File already exists: {dest} ({size_mb:.1f} MB)")
        return dest

    # 2. Check for VCP CLI
    if not shutil.which("vcp"):
        logger.error("  ✗ 'vcp' CLI tool not found. Please install it and log in.")
        return None

    temp_path = None

    try:
        # 3. Find the Dataset ID
        search_term = "Primary Human CD4+ T Cell Perturb-seq"
        logger.info(f"  Searching VCP for: '{search_term}'...")
        
        cmd_search = ["vcp", "data", "search", search_term, "--exact"]
        result = subprocess.run(cmd_search, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"  ✗ Search failed: {result.stderr}")
            return None

        # Parse output to find ID (First column of output)
        dataset_id = None
        lines = result.stdout.strip().split('\n')
        # Skip header if present, look for the line containing the name
        for line in lines:
            if search_term in line:
                dataset_id = line.split()[0]
                break
        
        if not dataset_id:
            # Fallback: try taking the first non-header line if search was exact
            if len(lines) > 1 and "ID" in lines[0]:
                 dataset_id = lines[1].split()[0]
            elif len(lines) > 0 and "ID" not in lines[0]:
                 dataset_id = lines[0].split()[0]

        if not dataset_id:
            logger.error("  ✗ Could not determine dataset ID from search results.")
            return None

        logger.info(f"  Found Dataset ID: {dataset_id}")

        # 4. List files to find the .h5ad
        logger.info("  Listing files in dataset...")
        cmd_list = ["vcp", "data", "list-files", dataset_id]
        list_res = subprocess.run(cmd_list, capture_output=True, text=True)
        
        if list_res.returncode != 0:
            logger.error(f"  ✗ Failed to list files: {list_res.stderr}")
            return None

        # Identify target file. Prefer "Stim8hr" if available to save bandwidth.
        # Otherwise download the main file.
        files = [l.split()[0] for l in list_res.stdout.strip().split('\n') if ".h5ad" in l]
        
        target_file_name = None
        stim_files = [f for f in files if "Stim8hr" in f]
        
        if stim_files:
            target_file_name = stim_files[0]
            logger.info(f"  Found pre-split file: {target_file_name}")
        elif files:
            target_file_name = files[0]
            logger.info(f"  No pre-split file found. Downloading main file: {target_file_name}")
        else:
            logger.error("  ✗ No .h5ad file found in dataset.")
            return None

        # 5. Download to a temporary location
        # Use a temp name in the raw directory
        temp_path = raw_dir / f"temp_{target_file_name}"
        
        logger.info(f"  Downloading to temporary file {temp_path}...")
        
        cmd_download = [
            "vcp", "data", "download", 
            dataset_id, 
            target_file_name, 
            "--output", str(raw_dir)
        ]
        
        subprocess.run(cmd_download, check=True, capture_output=True)
        
        # The CLI downloads to <output_dir>/<filename>. Rename to our temp path if needed.
        downloaded_original_path = raw_dir / target_file_name
        if downloaded_original_path.exists() and downloaded_original_path != temp_path:
            downloaded_original_path.rename(temp_path)
            
        if not temp_path.exists():
            logger.error("  ✗ Download appeared to succeed but temp file is missing.")
            return None

        # 6. Load and Filter (The original requirement)
        logger.info("  Processing and filtering data (Stim8hr + 2 Donors)...")
        
        # Use backed='r' to avoid loading 22M cells into RAM if we got the full file
        adata = anndata.read_h5ad(temp_path, backed='r')
        
        # --- Filter Condition (Stim8hr) ---
        cond_col = config.get("condition_col", "condition")
        cond_val = config.get("condition_value", "Stim8hr")
        
        if cond_col in adata.obs.columns:
            # Case-insensitive match check
            # Note: In backed mode, boolean indexing is efficient
            mask_cond = adata.obs[cond_col] == cond_val
            if mask_cond.sum() == 0:
                 # Try approximate match
                 mask_cond = adata.obs[cond_col].astype(str).str.lower() == cond_val.lower()
            
            if mask_cond.sum() > 0:
                adata = adata[mask_cond]
                logger.info(f"    Filter condition '{cond_val}': kept {adata.shape[0]:,} cells")
            else:
                logger.warning(f"    Condition '{cond_val}' not found. Keeping all cells.")
        
        # --- Filter Donors ---
        donor_col = config.get("obs_batch_col", "donor")
        n_donors = config.get("donor_subset", 2)
        
        if donor_col in adata.obs.columns:
            # We need to compute unique donors. 
            # In backed mode, accessing unique() on a column is usually fine.
            donors = sorted(adata.obs[donor_col].unique())
            if len(donors) > n_donors:
                selected_donors = donors[:n_donors]
                mask_donor = adata.obs[donor_col].isin(selected_donors)
                adata = adata[mask_donor]
                logger.info(f"    Filter donors {selected_donors}: kept {adata.shape[0]:,} cells")
        
        # 7. Save filtered file
        logger.info(f"  Saving filtered dataset to {dest}...")
        # Write triggers the actual data read/write from the backed file
        adata.write_h5ad(dest)
        
        # 8. Cleanup
        logger.info("  Cleaning up temporary files...")
        temp_path.unlink()
        
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ Process complete: {dest} ({size_mb:.1f} MB)")
        return dest

    except Exception as e:
        logger.error(f"  ✗ Failed during download/filtering: {e}")
        # Attempt cleanup
        if temp_path and temp_path.exists():
            temp_path.unlink()
        return None


def download_challenge(dataset_name: str, config: dict, raw_dir: Path) -> Optional[Path]:
    """
    Handle the competition challenge data (already local).
    Looks for the file in data/ directory.
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"Processing {config['description']}")
    logger.info(f"{'=' * 60}")

    # Search common locations
    candidates = [
        DATA_DIR / config["filename"],
        DATA_DIR / "echoes-of-silenced-genes" / config["filename"],
        raw_dir / config["filename"],
        Path(config["filename"]),
    ]

    for path in candidates:
        if path.exists():
            logger.info(f"  Found challenge data at: {path}")
            # Copy to raw_dir if not already there
            dest = raw_dir / config["filename"]
            if not dest.exists():
                import shutil
                shutil.copy2(path, dest)
                logger.info(f"  Copied to: {dest}")
            return dest

    logger.error(
        f"  ✗ Challenge data not found. Expected locations:\n"
        f"    {chr(10).join(f'    - {p}' for p in candidates)}\n"
        f"  Run the setup cells in gene_clusters.ipynb first to download challenge data."
    )
    return None


# Dispatcher
DOWNLOADERS = {
    "figshare": download_replogle,
    "zenodo": download_nadig,
    "czi": download_cd4t,
    "local": download_challenge,
}


# ============================================================================
# CD4+ T CELL FILTERING (special handling for large dataset)
# ============================================================================

def filter_cd4t_stim8hr(adata, config: dict, n_donors: int = 2):
    """
    Filter CD4+ T cell data to Stim8hr condition and subset of donors.
    """
    import anndata

    cond_col = config.get("condition_col", "condition")
    cond_val = config.get("condition_value", "Stim8hr")

    logger.info(f"  Filtering CD4+ T cells: condition={cond_val}")

    if cond_col in adata.obs.columns:
        conditions = adata.obs[cond_col].unique()
        logger.info(f"    Available conditions: {list(conditions)}")

        mask = adata.obs[cond_col] == cond_val
        if mask.sum() == 0:
            # Try case-insensitive match
            mask = adata.obs[cond_col].str.lower() == cond_val.lower()

        if mask.sum() == 0:
            logger.warning(f"    Condition '{cond_val}' not found! Using all cells.")
        else:
            adata = adata[mask].copy()
            logger.info(f"    After condition filter: {adata.shape[0]:,} cells")
    else:
        logger.warning(f"    Condition column '{cond_col}' not found in .obs")

    # Donor subsetting
    donor_col = config.get("obs_batch_col", "donor")
    if donor_col in adata.obs.columns and n_donors:
        donors = sorted(adata.obs[donor_col].unique())
        logger.info(f"    Available donors: {donors}")
        if len(donors) > n_donors:
            selected = donors[:n_donors]
            mask = adata.obs[donor_col].isin(selected)
            adata = adata[mask].copy()
            logger.info(f"    Selected donors {selected}: {adata.shape[0]:,} cells")

    return adata


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_single_dataset(
    dataset_name: str,
    gene_list: Optional[list[str]] = None,
    do_preprocess: bool = True,
    force_redownload: bool = False,
) -> Optional[Path]:
    """
    Download (if needed) and preprocess a single dataset.
    Returns the path to the processed .h5ad, or None on failure.
    """
    import anndata

    if dataset_name not in DATASET_REGISTRY:
        logger.error(f"Unknown dataset: {dataset_name}")
        logger.error(f"Available: {list(DATASET_REGISTRY.keys())}")
        return None

    config = DATASET_REGISTRY[dataset_name]
    logger.info(f"\n{'#' * 70}")
    logger.info(f"# DATASET: {dataset_name.upper()} — {config['description']}")
    logger.info(f"{'#' * 70}\n")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    processed_path = PROCESSED_DIR / f"{dataset_name}_processed.h5ad"
    if processed_path.exists() and not force_redownload:
        logger.info(f"  Already processed: {processed_path}")
        logger.info(f"  Use --force to re-download and reprocess.")
        return processed_path

    # --- Download ---
    download_type = config["download_type"]
    downloader = DOWNLOADERS.get(download_type)
    if not downloader:
        logger.error(f"  No downloader for type: {download_type}")
        return None

    raw_path = downloader(dataset_name, config, RAW_DIR)
    if raw_path is None or not raw_path.exists():
        logger.error(f"  ✗ Failed to obtain raw data for {dataset_name}")
        return None

    # --- Load ---
    logger.info(f"\nLoading {raw_path}...")
    t0 = time.time()
    adata = anndata.read_h5ad(str(raw_path))
    load_time = time.time() - t0
    logger.info(
        f"  Loaded: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes "
        f"({load_time:.1f}s)"
    )
    logger.info(f"  .obs columns: {list(adata.obs.columns)}")
    logger.info(f"  .var columns: {list(adata.var.columns)}")

    # --- CD4+ T special filtering ---
    if dataset_name == "cd4t":
        adata = filter_cd4t_stim8hr(adata, config)

    # --- Gene symbol harmonization ---
    logger.info("\nHarmonizing gene symbols...")
    gene_symbol_col = None
    if "features" in adata.var.columns:
        gene_symbol_col = "features"
    elif "gene_name" in adata.var.columns:
        gene_symbol_col = "gene_name"
    elif "gene_symbols" in adata.var.columns:
        gene_symbol_col = "gene_symbols"

    # If var_names are Ensembl IDs but we have a symbol column, use symbols as index
    if gene_symbol_col and adata.var_names[0].startswith("ENS"):
        logger.info(f"  Detected Ensembl IDs in var_names, switching to '{gene_symbol_col}'")
        adata.var_names = adata.var[gene_symbol_col].astype(str).values
        adata.var_names_make_unique()

    adata = harmonize_gene_symbols(adata, gene_symbol_col)

    # --- Tag metadata ---
    logger.info("\nStandardizing metadata...")
    adata = load_and_tag_metadata(adata, config, dataset_name)

    # --- Harmonize perturbation gene symbols too ---
    pert_col = "perturbation"
    if pert_col in adata.obs.columns:
        adata.obs[pert_col] = adata.obs[pert_col].map(
            lambda x: GENE_SYMBOL_ALIASES.get(x, x)
        )

    # --- Preprocess ---
    if do_preprocess:
        logger.info("\nApplying preprocessing pipeline...")
        t0 = time.time()
        adata = preprocess_h5ad(
            adata,
            target_sum=10_000,
            log_base=2,
            gene_list=gene_list,
            gene_symbol_col=gene_symbol_col,
            copy=False,
        )
        pp_time = time.time() - t0
        logger.info(f"  Preprocessing completed in {pp_time:.1f}s")

    # --- Save ---
    logger.info(f"\nSaving processed data to {processed_path}...")
    t0 = time.time()
    adata.write_h5ad(str(processed_path))
    save_time = time.time() - t0
    size_mb = processed_path.stat().st_size / (1024 * 1024)
    logger.info(f"  ✓ Saved ({size_mb:.1f} MB, {save_time:.1f}s)")

    # --- Summary ---
    logger.info(f"\n{'─' * 50}")
    logger.info(f"  Dataset:       {dataset_name}")
    logger.info(f"  Shape:         {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    logger.info(f"  Cell type:     {config['cell_type']}")
    logger.info(f"  Role:          {config['role']}")
    logger.info(f"  Perturbations: {adata.obs['perturbation'].nunique()}")
    logger.info(f"  Controls:      {adata.obs['is_control'].sum():,}")
    logger.info(f"  Batches:       {adata.obs['batch'].nunique()}")
    logger.info(f"  Output:        {processed_path}")
    logger.info(f"{'─' * 50}")

    return processed_path


def preprocess_existing_file(
    filepath: str,
    gene_list: Optional[list[str]] = None,
    output_path: Optional[str] = None,
) -> Optional[Path]:
    """
    Apply the preprocessing pipeline to any existing .h5ad file.
    """
    import anndata

    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return None

    logger.info(f"Loading {filepath}...")
    adata = anndata.read_h5ad(str(filepath))
    logger.info(f"  Shape: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

    # Harmonize
    adata = harmonize_gene_symbols(adata)

    # Preprocess
    adata = preprocess_h5ad(
        adata, target_sum=10_000, log_base=2, gene_list=gene_list, copy=False,
    )

    # Save
    if output_path is None:
        stem = filepath.stem.replace("_raw", "").replace("_singlecell", "")
        output_path = filepath.parent / f"{stem}_processed.h5ad"
    else:
        output_path = Path(output_path)

    adata.write_h5ad(str(output_path))
    logger.info(f"  ✓ Saved to {output_path}")
    return output_path


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess perturbation prediction datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_data.py --all --gene-list data/gene_list.txt
  python download_data.py --dataset k562 --gene-list data/gene_list.txt
  python download_data.py --dataset k562 rpe1 --gene-list data/gene_list.txt
  python download_data.py --dataset k562 --no-preprocess
  python download_data.py --preprocess-only path/to/file.h5ad --gene-list data/gene_list.txt
  python download_data.py --list
        """,
    )

    parser.add_argument(
        "--dataset", nargs="+", default=None,
        help="Dataset(s) to download: " + ", ".join(DATASET_REGISTRY.keys()),
    )
    parser.add_argument("--all", action="store_true", help="Download and process all datasets")
    parser.add_argument("--list", action="store_true", help="List available datasets and exit")
    parser.add_argument(
        "--gene-list", default=None,
        help="Path to gene list file (one symbol per line). Genes not in list are dropped.",
    )
    parser.add_argument(
        "--no-preprocess", action="store_true",
        help="Download only, skip preprocessing",
    )
    parser.add_argument(
        "--preprocess-only", default=None,
        help="Apply preprocessing to an existing .h5ad file (no download)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for --preprocess-only mode",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download and reprocess even if output exists",
    )

    args = parser.parse_args()

    # --- List mode ---
    if args.list:
        print("\nAvailable datasets:")
        print(f"{'─' * 80}")
        for name, cfg in DATASET_REGISTRY.items():
            ready = "PLACEHOLDER" not in str(cfg.get("url", ""))
            status = "✓ ready" if ready else "⚠ URL needed"
            print(
                f"  {name:12s}  {cfg['cell_type']:15s}  {cfg['role']:6s}  "
                f"{cfg['source']:10s}  [{status}]"
            )
            print(f"               {cfg['description']}")
        print(f"{'─' * 80}")
        return

    # --- Load gene list ---
    gene_list = None
    if args.gene_list:
        gene_list_path = Path(args.gene_list)
        if not gene_list_path.exists():
            logger.error(f"Gene list not found: {gene_list_path}")
            sys.exit(1)
        with open(gene_list_path) as f:
            gene_list = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded gene list: {len(gene_list)} genes from {gene_list_path}")

    # --- Preprocess-only mode ---
    if args.preprocess_only:
        preprocess_existing_file(args.preprocess_only, gene_list, args.output)
        return

    # --- Determine which datasets to process ---
    if args.all:
        datasets = list(DATASET_REGISTRY.keys())
    elif args.dataset:
        datasets = args.dataset
    else:
        parser.print_help()
        print("\n⚠ Specify --dataset, --all, or --preprocess-only")
        return

    # --- Process each dataset ---
    results = {}
    for ds_name in datasets:
        try:
            path = process_single_dataset(
                ds_name,
                gene_list=gene_list,
                do_preprocess=not args.no_preprocess,
                force_redownload=args.force,
            )
            results[ds_name] = path
        except Exception as e:
            logger.error(f"✗ Failed to process {ds_name}: {e}", exc_info=True)
            results[ds_name] = None

    # --- Final summary ---
    logger.info(f"\n{'=' * 60}")
    logger.info("DOWNLOAD & PREPROCESS SUMMARY")
    logger.info(f"{'=' * 60}")
    for ds_name, path in results.items():
        status = f"✓ {path}" if path else "✗ FAILED"
        logger.info(f"  {ds_name:12s}: {status}")
    logger.info(f"{'=' * 60}")

    n_failed = sum(1 for p in results.values() if p is None)
    if n_failed > 0:
        logger.warning(f"\n{n_failed} dataset(s) failed. Re-run with --dataset <name> to retry.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Uniformize & Split Datasets for Perturbation Prediction
=========================================================

Combines preprocessed single-cell CRISPRi datasets into unified training and
evaluation AnnData files with harmonized metadata and perturbation partitioning.

This script:
  1. Loads all preprocessed .h5ad files from data/processed/
  2. Aligns them to a common gene set (intersection or provided gene list)
  3. Filters perturbations (must target genes in gene list, must have strong effects)
  4. Partitions perturbations into Sets A, B, C (shared/extra/held-out)
  5. Subsamples cells per the budget in Data.md §4
  6. Assigns split labels (train, eval1, eval2, eval3)
  7. Saves unified train.h5ad and eval.h5ad

Output .obs schema (identical in both train and eval files):
  ┌───────────────────┬─────────────────────────────────────────────────────┐
  │ Column            │ Description                                         │
  ├───────────────────┼─────────────────────────────────────────────────────┤
  │ cell_type         │ K562, Jurkat, CD4T_stim8hr, RPE1, HepG2             │
  │ perturbation      │ Gene symbol or 'control'                            │
  │ is_control        │ bool: True for non-targeting control cells          │
  │ dataset_source    │ replogle, nadig, biohub, challenge                  │
  │ batch             │ Original batch/lane/channel ID (prefixed by source) │
  │ split             │ train, eval1, eval2, eval3                          │
  │ pert_set          │ A (shared), B (extra), C (held-out), or 'control'   │
  │ nCount_RNA        │ Total UMI count (pre-normalization)                 │
  │ nFeature_RNA      │ Number of detected genes (pre-normalization)        │
  │ percent_mt        │ Mitochondrial read percentage                       │
  │ sgrna_id          │ Original sgRNA identifier (if available)            │
  │ cell_barcode      │ Unique cell identifier (source + original index)    │
  └───────────────────┴─────────────────────────────────────────────────────┘

Usage:
  # Basic: combine all processed datasets
  python uniformize_data.py --gene-list data/gene_list.txt

  # Custom budgets and filtering
  python uniformize_data.py \\
      --gene-list data/gene_list.txt \\
      --min-cells-per-pert 10 \\
      --min-deg 5 \\
      --eval1-fraction 0.2 \\
      --n-held-out-perts 25

  # Dry run: show what would happen without saving
  python uniformize_data.py --gene-list data/gene_list.txt --dry-run

Author: MylliaESG project
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "unified"

# ============================================================================
# CONFIGURATION: Cell type roles and budgets (from Data.md §3-4)
# ============================================================================

CELL_TYPE_CONFIG = {
    "K562": {
        "role": "train",
        "source": "replogle",
        "max_perts": 200,
        "target_cells_per_pert": 150,
        "control_cells": 5_000,
    },
    "Jurkat": {
        "role": "train",
        "source": "nadig",
        "max_perts": 120,
        "target_cells_per_pert": 80,
        "control_cells": 3_000,
    },
    "CD4T_stim8hr": {
        "role": "train",
        "source": "biohub",
        "max_perts": 120,
        "target_cells_per_pert": 150,
        "control_cells": 5_000,
    },
    "RPE1": {
        "role": "eval",
        "source": "replogle",
        "max_perts": 100,
        "target_cells_per_pert": 140,
        "control_cells": 3_000,
    },
    "HepG2": {
        "role": "eval",
        "source": "nadig",
        "max_perts": 100,
        "target_cells_per_pert": 45,
        "control_cells": 2_000,
    },
}

TRAIN_CELL_TYPES = [ct for ct, cfg in CELL_TYPE_CONFIG.items() if cfg["role"] == "train"]
EVAL_CELL_TYPES = [ct for ct, cfg in CELL_TYPE_CONFIG.items() if cfg["role"] == "eval"]


# ============================================================================
# STEP 1: Load and validate processed datasets
# ============================================================================

def discover_processed_datasets(processed_dir: Path) -> dict[str, Path]:
    """Find all *_processed.h5ad files and map to dataset names."""
    datasets = {}
    for f in sorted(processed_dir.glob("*_processed.h5ad")):
        name = f.stem.replace("_processed", "")
        datasets[name] = f
    return datasets


def load_processed_datasets(
    processed_dir: Path,
    gene_list: Optional[list[str]] = None,
) -> dict[str, "anndata.AnnData"]:
    """
    Load all processed datasets, validate schema, and align gene sets.

    Returns dict[dataset_name → AnnData].
    """
    import anndata

    available = discover_processed_datasets(processed_dir)
    if not available:
        logger.error(f"No processed datasets found in {processed_dir}/")
        logger.error("Run download_data.py first to download and preprocess datasets.")
        sys.exit(1)

    logger.info(f"Found {len(available)} processed dataset(s):")
    for name, path in available.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"  {name}: {path.name} ({size_mb:.1f} MB)")

    # Load each
    datasets = {}
    all_genes = None

    for name, path in available.items():
        logger.info(f"\nLoading {name}...")
        t0 = time.time()
        adata = anndata.read_h5ad(str(path))
        logger.info(
            f"  {adata.shape[0]:,} cells × {adata.shape[1]:,} genes "
            f"({time.time() - t0:.1f}s)"
        )

        # Validate required columns
        required_cols = ["cell_type", "perturbation", "is_control", "dataset_source", "batch"]
        missing_cols = [c for c in required_cols if c not in adata.obs.columns]
        if missing_cols:
            logger.warning(
                f"  ⚠ Missing columns: {missing_cols}. "
                f"This dataset may not have been processed with download_data.py."
            )

        # Track gene intersection
        genes_set = set(adata.var_names)
        if all_genes is None:
            all_genes = genes_set
        else:
            all_genes = all_genes & genes_set

        datasets[name] = adata

    logger.info(f"\nGene intersection across all datasets: {len(all_genes)} genes")

    # Determine final gene set
    if gene_list is not None:
        final_genes = sorted(set(gene_list) & all_genes)
        n_from_list = len(set(gene_list))
        n_in_intersection = len(final_genes)
        logger.info(
            f"Gene list provided: {n_from_list} genes → "
            f"{n_in_intersection} in intersection with all datasets"
        )
    else:
        final_genes = sorted(all_genes)
        logger.info(f"No gene list provided, using intersection: {len(final_genes)} genes")

    # Subset all datasets to final gene set
    for name, adata in datasets.items():
        mask = np.isin(adata.var_names, final_genes)
        n_before = adata.shape[1]
        datasets[name] = adata[:, mask].copy()
        logger.info(
            f"  {name}: {n_before} → {datasets[name].shape[1]} genes after alignment"
        )

    return datasets


# ============================================================================
# STEP 2: Filter perturbations
# ============================================================================

def compute_perturbation_strength(
    adata,
    min_cells: int = 10,
    n_deg_threshold: int = 5,
    fc_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    For each perturbation, compute a simple effect-strength metric:
    number of differentially expressed genes (|log2FC| > threshold).

    Returns DataFrame with columns: [perturbation, n_cells, n_deg, mean_abs_fc]
    """
    controls = adata[adata.obs["is_control"]].copy()
    if controls.shape[0] == 0:
        logger.warning("  No control cells found!")
        return pd.DataFrame(columns=["perturbation", "n_cells", "n_deg", "mean_abs_fc"])

    if sparse.issparse(controls.X):
        ctrl_mean = np.array(controls.X.mean(axis=0)).flatten()
    else:
        ctrl_mean = controls.X.mean(axis=0).flatten()

    pert_names = adata.obs.loc[~adata.obs["is_control"], "perturbation"]
    pert_counts = pert_names.value_counts()

    results = []
    for pert, n_cells in pert_counts.items():
        if n_cells < min_cells:
            continue

        pert_cells = adata[adata.obs["perturbation"] == pert]
        if sparse.issparse(pert_cells.X):
            pert_mean = np.array(pert_cells.X.mean(axis=0)).flatten()
        else:
            pert_mean = pert_cells.X.mean(axis=0).flatten()

        # log2 fold change (data is already log2-transformed, so difference = log2FC)
        fc = pert_mean - ctrl_mean
        abs_fc = np.abs(fc)
        n_deg = int((abs_fc > fc_threshold).sum())
        mean_abs_fc = float(abs_fc.mean())

        results.append({
            "perturbation": pert,
            "n_cells": n_cells,
            "n_deg": n_deg,
            "mean_abs_fc": mean_abs_fc,
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("n_deg", ascending=False).reset_index(drop=True)

    return df


def filter_perturbations(
    datasets: dict[str, "anndata.AnnData"],
    gene_list: list[str],
    min_cells_per_pert: int = 10,
    min_deg: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    Filter perturbations across all datasets:
      1. Target gene must be in gene list
      2. Must have >= min_cells_per_pert cells
      3. Must show >= min_deg differentially expressed genes

    Returns dict[dataset_name → filtered perturbation stats DataFrame]
    """
    gene_set = set(gene_list)
    pert_stats = {}

    for name, adata in datasets.items():
        cell_type = adata.obs["cell_type"].iloc[0] if "cell_type" in adata.obs else name
        logger.info(f"\n  Filtering perturbations for {name} ({cell_type})...")

        # Get all perturbations
        all_perts = adata.obs.loc[~adata.obs["is_control"], "perturbation"].unique()
        logger.info(f"    Total perturbations: {len(all_perts)}")

        # Filter: target in gene list
        in_gene_list = [p for p in all_perts if p in gene_set]
        logger.info(f"    In gene list: {len(in_gene_list)}")

        # Compute effect strength
        stats = compute_perturbation_strength(
            adata, min_cells=min_cells_per_pert, n_deg_threshold=min_deg
        )

        if len(stats) == 0:
            logger.warning(f"    No perturbations passed cell count filter!")
            pert_stats[name] = stats
            continue

        # Filter: in gene list
        stats = stats[stats["perturbation"].isin(gene_set)]
        logger.info(f"    After gene-list filter: {len(stats)}")

        # Filter: strong effects
        stats_strong = stats[stats["n_deg"] >= min_deg]
        logger.info(f"    After DEG filter (>={min_deg} DEGs): {len(stats_strong)}")

        if len(stats_strong) < len(stats) * 0.3:
            logger.info(
                f"    ⚠ Only {100 * len(stats_strong) / max(len(stats), 1):.0f}% of perturbations "
                f"pass DEG filter. Consider lowering --min-deg."
            )

        pert_stats[name] = stats_strong

    return pert_stats


# ============================================================================
# STEP 3: Perturbation partitioning (Sets A, B, C)
# ============================================================================

def partition_perturbations(
    pert_stats: dict[str, pd.DataFrame],
    datasets: dict[str, "anndata.AnnData"],
    n_held_out: int = 25,
    min_cell_types_for_shared: int = 3,
    min_cell_types_for_held_out: int = 2,
) -> dict[str, str]:
    """
    Partition perturbations into:
      Set A (shared):    present in >=min_cell_types_for_shared with strong effects
                         → used for training AND eval3 (cross-cell-type)
      Set B (extra):     remaining training-eligible perturbations
      Set C (held-out):  n_held_out perturbations held out entirely for eval2

    Returns dict[perturbation → set_label ('A', 'B', or 'C')]
    """
    # Map each perturbation to the cell types where it has strong effects
    pert_to_cell_types = defaultdict(set)
    pert_to_n_deg = defaultdict(list)

    for name, stats in pert_stats.items():
        if len(stats) == 0:
            continue
        cell_type = datasets[name].obs["cell_type"].iloc[0] if name in datasets else name
        for _, row in stats.iterrows():
            pert_to_cell_types[row["perturbation"]].add(cell_type)
            pert_to_n_deg[row["perturbation"]].append(row["n_deg"])

    logger.info(f"\n  Total unique perturbations with strong effects: {len(pert_to_cell_types)}")

    # Count how many cell types each perturbation appears in
    pert_breadth = {
        p: len(cts) for p, cts in pert_to_cell_types.items()
    }

    # --- Set C (held-out): strong in >=2 cell types, diverse functional categories ---
    # Pick perturbations present in both train and eval cell types
    candidates_c = []
    for pert, cts in pert_to_cell_types.items():
        has_train = any(ct in TRAIN_CELL_TYPES for ct in cts)
        has_eval = any(ct in EVAL_CELL_TYPES for ct in cts)
        if has_train and has_eval and len(cts) >= min_cell_types_for_held_out:
            mean_ndeg = np.mean(pert_to_n_deg[pert])
            candidates_c.append((pert, mean_ndeg, len(cts)))

    # Sort by breadth × effect strength
    candidates_c.sort(key=lambda x: (x[2], x[1]), reverse=True)

    set_c = set()
    for pert, _, _ in candidates_c[:n_held_out]:
        set_c.add(pert)

    logger.info(f"  Set C (held-out): {len(set_c)} perturbations")

    # --- Set A (shared): present in >=3 cell types, NOT in Set C ---
    set_a = set()
    for pert, cts in pert_to_cell_types.items():
        if pert in set_c:
            continue
        if len(cts) >= min_cell_types_for_shared:
            set_a.add(pert)

    logger.info(f"  Set A (shared): {len(set_a)} perturbations")

    # --- Set B (extra): everything else ---
    set_b = set(pert_to_cell_types.keys()) - set_a - set_c
    logger.info(f"  Set B (extra): {len(set_b)} perturbations")

    # Build assignment dict
    pert_assignment = {}
    for p in set_a:
        pert_assignment[p] = "A"
    for p in set_b:
        pert_assignment[p] = "B"
    for p in set_c:
        pert_assignment[p] = "C"

    return pert_assignment


# ============================================================================
# STEP 4: Subsample cells per budget
# ============================================================================

def subsample_cells(
    adata,
    cell_type: str,
    pert_assignment: dict[str, str],
    config: dict,
    rng: np.random.RandomState,
) -> "anndata.AnnData":
    """
    Subsample cells for a single dataset according to the budget in Data.md §4.

    Returns subsampled AnnData with 'pert_set' column added.
    """
    import anndata

    max_perts = config["max_perts"]
    target_cells = config["target_cells_per_pert"]
    n_ctrl = config["control_cells"]

    obs = adata.obs.copy()
    adata.obs["pert_set"] = "unassigned"

    # Assign pert_set
    for idx, row in obs.iterrows():
        if row["is_control"]:
            adata.obs.loc[idx, "pert_set"] = "control"
        else:
            pert = row["perturbation"]
            adata.obs.loc[idx, "pert_set"] = pert_assignment.get(pert, "unassigned")

    # --- Select perturbations to keep ---
    # Priority: Set A first, then B (for train) or A (for eval)
    eligible_perts = [
        p for p in obs.loc[~obs["is_control"], "perturbation"].unique()
        if p in pert_assignment
    ]

    # For eval cell types, exclude Set C from training but include for eval
    role = config["role"]
    if role == "train":
        # Don't include Set C perturbations in training
        eligible_perts = [p for p in eligible_perts if pert_assignment.get(p) != "C"]
    # For eval, keep all eligible (A for eval3, C for eval2)

    # Sort by priority: A first, then B/C
    def pert_priority(p):
        s = pert_assignment.get(p, "Z")
        return {"A": 0, "B": 1, "C": 2}.get(s, 3)

    eligible_perts.sort(key=pert_priority)
    selected_perts = eligible_perts[:max_perts]

    logger.info(
        f"  [{cell_type}] Selected {len(selected_perts)}/{len(eligible_perts)} perturbations "
        f"(budget: {max_perts})"
    )
    if selected_perts:
        set_counts = Counter(pert_assignment.get(p, "?") for p in selected_perts)
        logger.info(f"    Set distribution: {dict(set_counts)}")

    # --- Subsample perturbed cells ---
    keep_indices = []

    for pert in selected_perts:
        pert_mask = obs["perturbation"] == pert
        pert_idx = obs.index[pert_mask].tolist()
        n_available = len(pert_idx)
        n_take = min(target_cells, n_available)

        if n_available > 0:
            sampled = rng.choice(pert_idx, size=n_take, replace=False)
            keep_indices.extend(sampled)

    # --- Subsample control cells ---
    ctrl_mask = obs["is_control"]
    ctrl_idx = obs.index[ctrl_mask].tolist()
    n_ctrl_available = len(ctrl_idx)
    n_ctrl_take = min(n_ctrl, n_ctrl_available)

    if n_ctrl_available > 0:
        sampled_ctrl = rng.choice(ctrl_idx, size=n_ctrl_take, replace=False)
        keep_indices.extend(sampled_ctrl)

    logger.info(
        f"    Cells: {len(keep_indices):,} total "
        f"({len(keep_indices) - n_ctrl_take:,} perturbed + {n_ctrl_take:,} control)"
    )

    # Subset
    keep_indices = sorted(set(keep_indices))
    adata_sub = adata[keep_indices].copy()

    return adata_sub


# ============================================================================
# STEP 5: Assign split labels
# ============================================================================

def assign_splits(
    adata,
    cell_type: str,
    role: str,
    pert_assignment: dict[str, str],
    eval1_fraction: float = 0.2,
    rng: np.random.RandomState = None,
) -> "anndata.AnnData":
    """
    Assign split labels:
      - Training cell types:
          * Set A/B perturbations: 80% 'train', 20% 'eval1'
          * Controls: 80% 'train', 20% 'eval1'
      - Eval cell types:
          * Set A perturbations: 'eval3' (known pert × unknown cell type)
          * Set C perturbations: 'eval2' (unknown pert × any cell type)
          * Controls: assigned to eval3 (they serve as reference for all evals)
    """
    if rng is None:
        rng = np.random.RandomState(1234)

    obs = adata.obs
    splits = pd.Series("unassigned", index=obs.index)

    if role == "train":
        # 80/20 cell-level split for eval1
        for pert in obs["perturbation"].unique():
            pert_mask = obs["perturbation"] == pert
            pert_idx = obs.index[pert_mask]
            n = len(pert_idx)

            if n == 0:
                continue

            n_eval1 = max(1, int(n * eval1_fraction))
            shuffled = rng.permutation(pert_idx)
            eval1_idx = shuffled[:n_eval1]
            train_idx = shuffled[n_eval1:]

            splits.loc[train_idx] = "train"
            splits.loc[eval1_idx] = "eval1"

    elif role == "eval":
        for idx, row in obs.iterrows():
            if row["is_control"]:
                splits.loc[idx] = "eval3"  # controls serve as reference
            else:
                pert = row["perturbation"]
                pert_set = pert_assignment.get(pert, "B")
                if pert_set == "C":
                    splits.loc[idx] = "eval2"
                elif pert_set == "A":
                    splits.loc[idx] = "eval3"
                else:
                    splits.loc[idx] = "eval3"  # default to eval3 for eval cell types

    adata.obs["split"] = splits

    split_counts = splits.value_counts()
    logger.info(f"  [{cell_type}] Split assignment: {dict(split_counts)}")

    return adata


# ============================================================================
# STEP 6: Combine and save
# ============================================================================

def combine_and_save(
    subsampled: dict[str, "anndata.AnnData"],
    output_dir: Path,
    dry_run: bool = False,
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Concatenate all subsampled datasets and save as train.h5ad and eval.h5ad.
    Both files share the same .obs schema.
    """
    import anndata

    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure consistent .obs columns across all datasets
    canonical_cols = [
        "cell_type", "perturbation", "is_control", "dataset_source", "batch",
        "split", "pert_set", "nCount_RNA", "nFeature_RNA", "percent_mt",
        "sgrna_id", "cell_barcode",
    ]

    all_adatas = []
    for name, adata in subsampled.items():
        # Add cell_barcode (globally unique)
        adata.obs["cell_barcode"] = [
            f"{name}_{i}" for i in range(adata.shape[0])
        ]

        # Prefix batch with dataset name to ensure uniqueness
        if "batch" in adata.obs.columns:
            adata.obs["batch"] = adata.obs["batch"].astype(str).apply(
                lambda x: f"{name}_{x}" if not x.startswith(name) else x
            )

        # Ensure all canonical columns exist
        for col in canonical_cols:
            if col not in adata.obs.columns:
                adata.obs[col] = "unknown"

        # Keep only canonical columns (drop dataset-specific extras)
        extra_cols = [c for c in adata.obs.columns if c not in canonical_cols]
        if extra_cols:
            logger.info(f"  [{name}] Dropping non-canonical .obs columns: {extra_cols}")
            adata.obs = adata.obs[canonical_cols].copy()

        # Ensure correct dtypes
        adata.obs["is_control"] = adata.obs["is_control"].astype(bool)
        for col in ["cell_type", "perturbation", "dataset_source", "batch",
                     "split", "pert_set", "sgrna_id", "cell_barcode"]:
            adata.obs[col] = adata.obs[col].astype(str)
        for col in ["nCount_RNA", "nFeature_RNA", "percent_mt"]:
            adata.obs[col] = pd.to_numeric(adata.obs[col], errors="coerce").fillna(0)

        # Clean var (keep only gene names, drop dataset-specific var columns)
        adata.var = adata.var[[]].copy()

        all_adatas.append(adata)
        logger.info(f"  [{name}] Ready: {adata.shape[0]:,} cells")

    # Concatenate
    logger.info("\nConcatenating all datasets...")
    combined = anndata.concat(all_adatas, join="inner", merge="same")
    combined.obs_names_make_unique()

    logger.info(f"  Combined shape: {combined.shape[0]:,} cells × {combined.shape[1]:,} genes")

    # --- Split into train and eval ---
    train_mask = combined.obs["split"].isin(["train", "eval1"])
    eval_mask = combined.obs["split"].isin(["eval2", "eval3"])

    train_data = combined[train_mask].copy()
    eval_data = combined[eval_mask].copy()

    logger.info(f"  Training set: {train_data.shape[0]:,} cells")
    logger.info(f"  Eval set:     {eval_data.shape[0]:,} cells")

    if dry_run:
        logger.info("\n[DRY RUN] Would save but --dry-run was specified.")
        print_summary(combined, train_data, eval_data)
        return None, None

    # Save
    train_path = output_dir / "train.h5ad"
    eval_path = output_dir / "eval.h5ad"
    combined_path = output_dir / "all_unified.h5ad"

    logger.info(f"\nSaving training data to {train_path}...")
    train_data.write_h5ad(str(train_path))
    logger.info(f"  ✓ {train_path.stat().st_size / 1e6:.1f} MB")

    logger.info(f"Saving eval data to {eval_path}...")
    eval_data.write_h5ad(str(eval_path))
    logger.info(f"  ✓ {eval_path.stat().st_size / 1e6:.1f} MB")

    logger.info(f"Saving combined data to {combined_path}...")
    combined.write_h5ad(str(combined_path))
    logger.info(f"  ✓ {combined_path.stat().st_size / 1e6:.1f} MB")

    # Save metadata summary
    print_summary(combined, train_data, eval_data)
    save_metadata_report(combined, train_data, eval_data, output_dir)

    return train_path, eval_path


# ============================================================================
# REPORTING
# ============================================================================

def print_summary(combined, train_data, eval_data):
    """Print a detailed summary of the unified dataset."""
    logger.info(f"\n{'=' * 70}")
    logger.info("UNIFIED DATASET SUMMARY")
    logger.info(f"{'=' * 70}")

    logger.info(f"\n  Total: {combined.shape[0]:,} cells × {combined.shape[1]:,} genes")
    logger.info(f"  Train: {train_data.shape[0]:,} cells")
    logger.info(f"  Eval:  {eval_data.shape[0]:,} cells")

    logger.info(f"\n  --- By cell type ---")
    for ct in sorted(combined.obs["cell_type"].unique()):
        n = (combined.obs["cell_type"] == ct).sum()
        role = CELL_TYPE_CONFIG.get(ct, {}).get("role", "unknown")
        logger.info(f"    {ct:20s}: {n:>7,} cells  ({role})")

    logger.info(f"\n  --- By split ---")
    for split in ["train", "eval1", "eval2", "eval3"]:
        mask = combined.obs["split"] == split
        n = mask.sum()
        if n > 0:
            cts = combined.obs.loc[mask, "cell_type"].unique()
            n_perts = combined.obs.loc[mask & ~combined.obs["is_control"], "perturbation"].nunique()
            logger.info(f"    {split:8s}: {n:>7,} cells, {n_perts:>4} perturbations, cell types: {list(cts)}")

    logger.info(f"\n  --- By pert_set ---")
    for ps in ["A", "B", "C", "control"]:
        mask = combined.obs["pert_set"] == ps
        n = mask.sum()
        if n > 0:
            logger.info(f"    Set {ps}: {n:>7,} cells")

    logger.info(f"\n  --- Perturbation overlap ---")
    pert_by_ct = {}
    for ct in combined.obs["cell_type"].unique():
        mask = (combined.obs["cell_type"] == ct) & (~combined.obs["is_control"])
        pert_by_ct[ct] = set(combined.obs.loc[mask, "perturbation"].unique())

    ct_list = sorted(pert_by_ct.keys())
    for i, ct1 in enumerate(ct_list):
        for ct2 in ct_list[i + 1:]:
            overlap = pert_by_ct[ct1] & pert_by_ct[ct2]
            logger.info(f"    {ct1} ∩ {ct2}: {len(overlap)} shared perturbations")

    logger.info(f"\n  --- .obs columns ---")
    for col in combined.obs.columns:
        dtype = combined.obs[col].dtype
        nuniq = combined.obs[col].nunique()
        logger.info(f"    {col:20s}: {dtype}, {nuniq} unique values")

    logger.info(f"{'=' * 70}\n")


def save_metadata_report(combined, train_data, eval_data, output_dir: Path):
    """Save metadata summaries as JSON and CSV for easy inspection."""
    # Perturbation assignment table
    pert_table = (
        combined.obs
        .loc[~combined.obs["is_control"]]
        .groupby(["perturbation", "cell_type", "pert_set", "split"])
        .size()
        .reset_index(name="n_cells")
    )
    pert_table.to_csv(output_dir / "perturbation_assignments.csv", index=False)

    # Cell count summary
    summary = (
        combined.obs
        .groupby(["cell_type", "split", "pert_set"])
        .size()
        .reset_index(name="n_cells")
    )
    summary.to_csv(output_dir / "cell_count_summary.csv", index=False)

    # High-level stats as JSON
    stats = {
        "total_cells": int(combined.shape[0]),
        "total_genes": int(combined.shape[1]),
        "train_cells": int(train_data.shape[0]),
        "eval_cells": int(eval_data.shape[0]),
        "cell_types": list(combined.obs["cell_type"].unique()),
        "n_perturbations": int(combined.obs.loc[~combined.obs["is_control"], "perturbation"].nunique()),
        "splits": dict(combined.obs["split"].value_counts()),
        "pert_sets": dict(combined.obs["pert_set"].value_counts()),
    }
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    logger.info(f"  Metadata reports saved to {output_dir}/")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_uniformization(
    gene_list_path: Optional[str] = None,
    processed_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    min_cells_per_pert: int = 10,
    min_deg: int = 5,
    n_held_out_perts: int = 25,
    eval1_fraction: float = 0.2,
    seed: int = 1234,
    dry_run: bool = False,
):
    """Run the full uniformization pipeline."""
    import anndata

    proc_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR

    rng = np.random.RandomState(seed)

    # --- Load gene list ---
    gene_list = None
    if gene_list_path:
        with open(gene_list_path) as f:
            gene_list = [line.strip() for line in f if line.strip()]
        logger.info(f"Gene list: {len(gene_list)} genes from {gene_list_path}")

    # --- Step 1: Load datasets ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Loading processed datasets")
    logger.info("=" * 60)

    datasets = load_processed_datasets(proc_dir, gene_list)

    if not datasets:
        logger.error("No datasets loaded. Exiting.")
        return

    # --- Step 2: Filter perturbations ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Filtering perturbations by effect strength")
    logger.info("=" * 60)

    pert_stats = filter_perturbations(
        datasets, gene_list or [], min_cells_per_pert, min_deg
    )

    # --- Step 3: Partition perturbations ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Partitioning perturbations into Sets A/B/C")
    logger.info("=" * 60)

    pert_assignment = partition_perturbations(
        pert_stats, datasets, n_held_out=n_held_out_perts
    )

    # --- Step 4: Subsample cells ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Subsampling cells per budget")
    logger.info("=" * 60)

    subsampled = {}
    for name, adata in datasets.items():
        cell_type = adata.obs["cell_type"].iloc[0] if "cell_type" in adata.obs else name
        config = CELL_TYPE_CONFIG.get(cell_type)

        if config is None:
            logger.warning(
                f"  [{name}] Cell type '{cell_type}' not in CELL_TYPE_CONFIG. "
                f"Skipping subsampling — keeping all cells."
            )
            adata.obs["pert_set"] = adata.obs["perturbation"].map(
                lambda p: pert_assignment.get(p, "control" if p == "control" else "B")
            )
            subsampled[name] = adata
            continue

        sub = subsample_cells(adata, cell_type, pert_assignment, config, rng)
        subsampled[name] = sub

    # --- Step 5: Assign splits ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Assigning train/eval splits")
    logger.info("=" * 60)

    for name, adata in subsampled.items():
        cell_type = adata.obs["cell_type"].iloc[0] if "cell_type" in adata.obs else name
        config = CELL_TYPE_CONFIG.get(cell_type, {"role": "train"})
        role = config["role"]

        subsampled[name] = assign_splits(
            adata, cell_type, role, pert_assignment,
            eval1_fraction=eval1_fraction, rng=rng,
        )

    # --- Step 6: Combine and save ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Combining and saving")
    logger.info("=" * 60)

    train_path, eval_path = combine_and_save(subsampled, out_dir, dry_run=dry_run)

    if train_path:
        logger.info(f"\n✓ Training data: {train_path}")
        logger.info(f"✓ Eval data:     {eval_path}")
        logger.info(f"✓ Reports:       {out_dir}/")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Uniformize and split preprocessed datasets for perturbation prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python uniformize_data.py --gene-list data/gene_list.txt
  python uniformize_data.py --gene-list data/gene_list.txt --dry-run
  python uniformize_data.py --gene-list data/gene_list.txt --min-deg 10 --n-held-out-perts 30
        """,
    )

    parser.add_argument(
        "--gene-list", required=True,
        help="Path to gene list file (one symbol per line, defines the 5,127-gene vocabulary)",
    )
    parser.add_argument(
        "--processed-dir", default=None,
        help=f"Directory with *_processed.h5ad files (default: {PROCESSED_DIR})",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--min-cells-per-pert", type=int, default=10,
        help="Minimum cells per perturbation to keep (default: 10)",
    )
    parser.add_argument(
        "--min-deg", type=int, default=5,
        help="Minimum DEGs (|log2FC|>0.5) to consider a perturbation strong (default: 5)",
    )
    parser.add_argument(
        "--n-held-out-perts", type=int, default=25,
        help="Number of perturbations to hold out entirely for eval2 (default: 25)",
    )
    parser.add_argument(
        "--eval1-fraction", type=float, default=0.2,
        help="Fraction of training cells held out for eval1 (default: 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=1234,
        help="Random seed for reproducible subsampling and splits (default: 1234)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show summary without saving files",
    )

    args = parser.parse_args()

    run_uniformization(
        gene_list_path=args.gene_list,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        min_cells_per_pert=args.min_cells_per_pert,
        min_deg=args.min_deg,
        n_held_out_perts=args.n_held_out_perts,
        eval1_fraction=args.eval1_fraction,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
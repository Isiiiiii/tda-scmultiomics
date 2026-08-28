"""

This code is adapted from https://github.com/gmalagol10/XplainTDA.git.

Instead of only allowing Euclidean distance, here compute_neighbors() 
has been adapted to accomodate cknn (*) distance scaling using 
d(x,y)= d_2(x,y) / sqrt(d_2(x,x_k) * d_2(y,y_k)), 
where d_2 is the Euclidean distance, 
and x_k and y_k are the k-th nearest neighbors of x and y. 


* Berry, T. & Sauer, T. (2019). Consistent manifold representation for
topological data analysis. Foundations of Data Science, 1(1), 1–38.
DOI: 10.3934/fods.2019001
"""

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import time
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

from XplainTDA import XplainTDA
from XplainTDA import compute_neighbors, neighbors_to_edges


""" Computes the filtration value at which
the adjusted mutual information score (ami) between the connected components
and the available cell type annotation is the highest. 

Adds a column 'cc_id' to adata.obs with the best annotation and 

returns
(1) a parquet file with columns filtration values and rows cell 
barcodes with entries the ID of the connected components that they belong to,
(2) a csv with one ami value per filtration value,
(3) a csv with combined information (Barode, cell type, connected component ID).
"""
# Number of neighbors for kNN graph
PCA = 100
GRAE = 75

class UnionFind:
    """Simple union-find (disjoint set) with path compression and union by rank."""

    def __init__(self, n):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=np.int32)

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def labels(self):
        """
        Return component labels for all nodes, renumbered 0..n_components-1.
        """
        n = len(self.parent)
        roots = np.array([self.find(i) for i in range(n)])
        _, labels = np.unique(roots, return_inverse=True)
        return labels
    

def compute_cc_at_all_filtration_values_fast(n_cells, barcodes, edge_list, neigh_dist, desc: str):
    """
    For every unique positive value in result.neigh_dist, compute connected
    components over all cells (no filtering, no max_components cutoff),
    using incremental union-find for speed.

    Returns:
        df : DataFrame, shape (n_filtration_values, n_cells)
             index   = filtration values
             columns = cell barcodes (from result.adata.obs_names)
             values  = component id that cell belongs to at that filtration value
    """

    n_cells = tda_obj.n_cells
    barcodes = tda_obj.adata.obs_names

    edge_list = np.asarray(tda_obj.edge_list)
    edge_src = edge_list[:, 0].astype(int)
    edge_dst = edge_list[:, 1].astype(int)
    edge_w = edge_list[:, 2]

    # Sort edges by weight once
    order = np.argsort(edge_w, kind="mergesort")
    edge_src = edge_src[order]
    edge_dst = edge_dst[order]
    edge_w = edge_w[order]

    vals = np.asarray(tda_obj.neigh_dist).ravel()
    filtration_values = np.unique(vals[vals > 0])

    uf = UnionFind(n_cells)

    rows = np.empty((len(filtration_values), n_cells), dtype=np.int32)

    edge_ptr = 0
    n_edges = len(edge_w)

    for i, value in enumerate(tqdm(filtration_values, desc=desc)):
        while edge_ptr < n_edges and edge_w[edge_ptr] <= value:
            uf.union(edge_src[edge_ptr], edge_dst[edge_ptr])
            edge_ptr += 1

        rows[i, :] = uf.labels()

    df = pd.DataFrame(
        rows,
        index=filtration_values,
        columns=barcodes,
    )
    df.index.name = "filtration_value"

    return df

def compute_nmi_per_filtration(df, ground_truth):
    """
    df: rows = filtration values, columns = cell barcodes
    Returns a DataFrame with one row per filtration value: nmi, ami, n_components.
    """
    gt = np.asarray(ground_truth)
    records = []

    for filt_val, row in tqdm(df.iterrows(), total=len(df), desc="scoring"):
        labels = row.to_numpy()
        nmi = normalized_mutual_info_score(gt, labels)
        ami = adjusted_mutual_info_score(gt, labels)
        records.append(
            {
                "filtration_value": filt_val,
                "n_components": len(np.unique(labels)),
                "nmi": nmi,
                "ami": ami,
            }
        )

    return pd.DataFrame(records)


def get_annotation(adata_path, dataset, modality, embedding, metric,
                    celltype_col, out_path, save=True):

    if not adata_path.exists():
        raise FileNotFoundError(f"adata_path does not exist: {adata_path}")

    adata1 = sc.read_h5ad(adata_path)

    if celltype_col not in adata1.obs.columns:
        raise ValueError(
            f"Given Annotation not found in adata.obs. "
            f"Available columns: {list(adata1.obs.columns)}"
        )

    obsm_key = f"X_{embedding}"
    if obsm_key in adata1.obsm:
        if embedding == "GRAE":
            tda_obj = XplainTDA(adata1, embedding_key=obsm_key, n_neighbors=GRAE,neighbor_metric="cknn").fit()
        elif embedding == "PCA":
            tda_obj = XplainTDA(adata1, embedding_key=obsm_key, n_neighbors=PCA, neighbor_metric="cknn").fit()

    else:
        print("No embedding found: Computing embedding...")
        if embedding == "GRAE":
            tda_obj = XplainTDA(adata1, embedder=embedding, n_neighbors=GRAE, neighbor_metric="cknn").fit()
        elif embedding == "PCA":
            tda_obj = XplainTDA(adata1, embedder=embedding, n_neighbors=PCA, neighbor_metric="cknn").fit()
    
    neigh_ind, neigh_dist = compute_neighbors(
    tda_obj.embedding, tda_obj.n_neighbors, tda_obj.neighbor_backend, tda_obj.metric
    )
    edge_list = neighbors_to_edges(neigh_ind, neigh_dist)   

    # --- build output directory: out_path/{dataset}/{modality}/{embedding}/{metric} ---
    out_dir = out_path / dataset / modality / embedding / metric
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "ccids_filtrationvalue.parquet"
    csv_path = out_dir / "ami_filtrationvalues.csv"
    merged_path = out_dir / "combined_results.csv"

    desc = f"{dataset} | {modality} | {embedding} | {metric}"

    # --- 1. compute connected components across all filtration values ---
    t0 = time.time()
    df = compute_cc_at_all_filtration_values_fast(
    tda_obj.n_cells, adata1.obs_names, edge_list, neigh_dist, desc=desc
)
    compute_time = time.time() - t0

    save_time = None
    if save:
        tmp_path = parquet_path.with_suffix(".tmp.parquet")

        t1 = time.time()
        df.T.to_parquet(tmp_path)  ## Transposed!!
        tmp_path.rename(parquet_path)
        save_time = time.time() - t1

        print(
            f"Saved: {parquet_path}  (shape={df.shape}, "
            f"compute={compute_time:.1f}s, save={save_time:.1f}s)\n"
        )
    else:
        print(
            f"Computed (not saved): shape={df.shape}, "
            f"compute={compute_time:.1f}s\n"
        )

    # --- 2. align ground truth to df's column (barcode) order ---
    ground_truth_col = adata1.obs[celltype_col]
    if not np.array_equal(df.columns.to_numpy(), adata1.obs_names.to_numpy()):
        ground_truth_col = adata1.obs.loc[df.columns, celltype_col]

    gt = ground_truth_col.to_numpy()

    # --- NEW: drop cells with missing annotation before scoring ---
    missing_mask = pd.isna(gt)
    if missing_mask.any():
        print(
            f"Dropping {missing_mask.sum()} cells with missing "
            f"'{celltype_col}' annotation before scoring "
            f"({missing_mask.sum()}/{len(gt)})"
        )
        gt = gt[~missing_mask]
        df = df.loc[:, ~missing_mask]  # keep only annotated cells for scoring

    # --- 3. NMI/AMI per filtration value ---
    t0 = time.time()
    scores_df = compute_nmi_per_filtration(df, gt)
    scores_df["dataset"] = dataset
    scores_df["modality"] = modality
    scores_df["embedding"] = embedding
    scores_df["metric"] = metric

    scores_df.to_csv(csv_path, index=False)

    print(
        f"Done: {dataset} | {modality} | {embedding} | {metric} "
        f"-> {csv_path} ({time.time()-t0:.1f}s)"
    )

    # --- 4. best-filtration Barcode/Celltype/clusterID merge ---
    best_filtration = scores_df.loc[scores_df["ami"].idxmax(), "filtration_value"]

    cluster_row = df.loc[best_filtration]
    cluster_df = cluster_row.rename("clusterID").rename_axis("Barcode").reset_index()

    celltype_df = (
        adata1.obs[[celltype_col]]
        .rename(columns={celltype_col: "Celltype"})
        .rename_axis("Barcode")
        .reset_index()
    )

    merged = cluster_df.merge(celltype_df, on="Barcode", how="left")[
        ["Barcode", "Celltype", "clusterID"]
    ]

    merged.to_csv(merged_path, index=False)

    print(
        f"[{dataset} {modality} {embedding} {metric}] best_filtration={best_filtration} "
        f"-> {merged_path} ({len(merged)} rows)"
    )
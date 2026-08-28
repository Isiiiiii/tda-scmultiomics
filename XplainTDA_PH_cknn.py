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

"""Persistent-homology computation for XplainTDA.

The module has one responsibility: transform a cell-by-feature matrix into an
embedding and compute persistence diagrams from a k-nearest-neighbour graph.
Feature localization was removed because it was only required by the retired
local explainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

try:  # Support both package imports and running modules from one directory.
    from . import Embedder
    from . import defaults as d
except ImportError:  # pragma: no cover - used by standalone scripts
    import Embedder
    import defaults as d

try:
    import gudhi as gd
except ImportError:  # Fail only when the GUDHI backend is requested.
    gd = None

try:
    from pynndescent import NNDescent
except ImportError:
    NNDescent = None

try:
    from ripser import ripser
except ImportError:
    ripser = None


VALID_NEIGHBOR_BACKENDS = {"exact", "pynndescent"}
VALID_PH_BACKENDS = {"gudhi", "ripser"}
DEFAULT_PH_BACKEND = getattr(
    d, "DEFAULT_GLOBAL_PH_BACKEND", getattr(d, "DEFAULT_PH_BACKEND", "ripser")
)
VALID_METRICS = {"eucl", "cknn"}


def _validate_embedding(embedding: Any, dtype: np.dtype, n_cells: int | None = None) -> np.ndarray:
    """Return a finite two-dimensional embedding with the requested dtype."""
    embedding = np.asarray(embedding, dtype=dtype)
    if embedding.ndim != 2 or embedding.shape[1] == 0:
        raise ValueError("embedding must have shape (n_cells, n_components)")
    if n_cells is not None and embedding.shape[0] != n_cells:
        raise ValueError(
            f"embedding has {embedding.shape[0]} rows, but the data contain {n_cells} cells"
        )
    if not np.isfinite(embedding).all():
        raise ValueError("embedding contains NaN or infinite values")
    return embedding


def _remove_self_neighbors(
    indices: np.ndarray, distances: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Remove each observation from its own neighbour list."""
    out_indices = np.empty((len(indices), k), dtype=int)
    out_distances = np.empty((len(indices), k), dtype=float)
    for i, (row_indices, row_distances) in enumerate(zip(indices, distances)):
        keep = row_indices != i
        row_indices, row_distances = row_indices[keep][:k], row_distances[keep][:k]
        if len(row_indices) != k:
            raise RuntimeError("neighbour search returned too few non-self neighbours")
        out_indices[i], out_distances[i] = row_indices, row_distances
    return out_indices, out_distances


def compute_neighbors(
    embedding: np.ndarray,
    n_neighbors: int = d.DEFAULT_N_NEIGHBORS,
    backend: str = d.DEFAULT_NEIGHBOR_BACKEND,
    metric: str = "eucl",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute non-self nearest neighbours for every embedded cell.

    ``n_neighbors`` denotes the number of other cells, unlike scikit-learn's
    raw query result, which normally includes the observation itself.

    Parameters
    ----------
    embedding : np.ndarray, shape (n_cells, n_features)
        Cell embedding.
    n_neighbors : int, default=d.DEFAULT_N_NEIGHBORS
        Number of nearest neighbors (excluding self). Clamped to ``n_cells - 1``.
    backend : {"exact", "pynndescent"}, default=d.DEFAULT_NEIGHBOR_BACKEND
        Neighbor search backend.
    metric : {"eucl", "cknn"}, default="eucl"
        "cknn" rescales distances using each point's k-th nearest neighbor
        distance, as in continuous-kNN graph construction.
    """
    embedding = np.asarray(embedding)
    if embedding.ndim != 2:
        raise ValueError("embedding must be two-dimensional")
    if backend not in VALID_NEIGHBOR_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(VALID_NEIGHBOR_BACKENDS)}")
    if metric not in ("eucl", "cknn"):
        raise ValueError("metric must be 'eucl' or 'cknn'")
    if int(n_neighbors) < 1:
        raise ValueError("n_neighbors must be positive")

    n_cells = embedding.shape[0]
    if n_cells < 2:
        return np.empty((n_cells, 0), dtype=int), np.empty((n_cells, 0), dtype=float)

    k = min(max(1, int(n_neighbors)), n_cells - 1)
    query_k = k + 1

    if backend == "exact":
        model = NearestNeighbors(n_neighbors=query_k, metric="eucl", n_jobs=1)
        distances, indices = model.fit(embedding).kneighbors(embedding)
    else:
        if NNDescent is None:
            raise ImportError("pynndescent is required for backend='pynndescent'")
        indices, distances = NNDescent(
            embedding, n_neighbors=query_k, metric="eucl"
        ).neighbor_graph

    neigh_ind, neigh_dist = _remove_self_neighbors(
        np.asarray(indices), np.asarray(distances), k
    )

    if metric == "cknn":
        rho = neigh_dist[:, -1]
        scale = np.sqrt(rho[:, None] * rho[neigh_ind])
        neigh_dist = neigh_dist / scale

    return neigh_ind, neigh_dist


def neighbors_to_edges(indices: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Convert directed neighbour arrays to unique undirected weighted edges."""
    edges: dict[tuple[int, int], float] = {}
    for i, (row_indices, row_distances) in enumerate(zip(indices, distances)):
        for j, distance in zip(row_indices, row_distances):
            edge = (i, int(j)) if i < j else (int(j), i)
            edges[edge] = min(float(distance), edges.get(edge, np.inf))
    return (
        np.asarray([(i, j, value) for (i, j), value in edges.items()], dtype=float)
        if edges
        else np.empty((0, 3), dtype=float)
    )


def edges_to_sparse_distance(edges: np.ndarray, n_cells: int) -> sparse.csr_matrix:
    """Build the symmetric sparse distance matrix consumed by Ripser."""
    if len(edges) == 0:
        return sparse.csr_matrix((n_cells, n_cells), dtype=float)
    vertices = edges[:, :2].astype(int)
    values = edges[:, 2]
    rows = np.r_[vertices[:, 0], vertices[:, 1]]
    cols = np.r_[vertices[:, 1], vertices[:, 0]]
    return sparse.csr_matrix((np.r_[values, values], (rows, cols)), shape=(n_cells, n_cells))


def _cap_infinite_deaths(diagrams: list[np.ndarray], filtration_end: float) -> list[np.ndarray]:
    """Replace essential-feature infinite deaths with the filtration endpoint."""
    output = []
    for diagram in diagrams:
        diagram = np.asarray(diagram, dtype=float).reshape(-1, 2).copy()
        if len(diagram):
            diagram[~np.isfinite(diagram[:, 1]), 1] = filtration_end
        output.append(diagram)
    return output


def _persistence_ripser(edges: np.ndarray, n_cells: int, maxdim: int) -> list[np.ndarray]:
    """Compute persistence from a sparse kNN distance matrix with Ripser."""
    if ripser is None:
        raise ImportError("ripser is required for ph_backend='ripser'")
    distances = edges_to_sparse_distance(edges, n_cells)
    diagrams = ripser(distances, maxdim=maxdim, distance_matrix=True)["dgms"]
    filtration_end = float(edges[:, 2].max()) if len(edges) else 0.0
    return _cap_infinite_deaths(diagrams, filtration_end)


def _persistence_gudhi(edges: np.ndarray, n_cells: int, maxdim: int) -> list[np.ndarray]:
    """Compute persistence from the clique expansion of the kNN graph."""
    if gd is None:
        raise ImportError("gudhi is required for ph_backend='gudhi'")

    simplex_tree = gd.SimplexTree()
    for i in range(n_cells):
        simplex_tree.insert([i], filtration=0.0)
    for i, j, distance in edges:
        simplex_tree.insert([int(i), int(j)], filtration=float(distance))

    # H_k requires simplices through dimension k + 1.
    simplex_tree.expansion(maxdim + 1)
    simplex_tree.compute_persistence()
    filtration_end = float(edges[:, 2].max()) if len(edges) else 0.0
    diagrams = [
        simplex_tree.persistence_intervals_in_dimension(dim)
        for dim in range(maxdim + 1)
    ]
    return _cap_infinite_deaths(diagrams, filtration_end)


def compute_persistence(
    embedding: np.ndarray,
    n_neighbors: int = d.DEFAULT_N_NEIGHBORS,
    maxdim: int = d.DEFAULT_MAXDIM,
    neighbor_backend: str = d.DEFAULT_NEIGHBOR_BACKEND,
    ph_backend: str = DEFAULT_PH_BACKEND,
    metric: str = d.DEFAULT_METRIC,  # or " if no such default exists yet
) -> list[np.ndarray]:
    """Compute persistence diagrams from an embedding.

    The same undirected kNN graph is supplied to either backend, so reference
    and perturbed diagrams are comparable when the configuration is unchanged.
    """
    ph_backend = ph_backend.lower()
    if int(maxdim) < 0:
        raise ValueError("maxdim must be non-negative")
    if ph_backend not in VALID_PH_BACKENDS:
        raise ValueError(f"ph_backend must be one of {sorted(VALID_PH_BACKENDS)}")
    indices, distances = compute_neighbors(embedding, n_neighbors, neighbor_backend, metric)
    edges = neighbors_to_edges(indices, distances)
    if ph_backend == "ripser":
        return _persistence_ripser(edges, len(embedding), int(maxdim))
    return _persistence_gudhi(edges, len(embedding), int(maxdim))

class XplainTDA:
    """Fit an embedding and compute persistent homology for single-cell data.

    Parameters are limited to choices that affect the embedding or PH graph.
    Explanation-specific options belong to :class:`XplainTDA_Explainer.Explainer`.

    Parameters
    ----------
    adata
        AnnData object. It may also be supplied to :meth:`fit`.
    layer
        ``"X"`` or an AnnData layer name.
    embedder
        Registered embedder name, for example ``"PCA"`` or ``"GRAE"``.
    n_components
        Embedding dimension. ``None`` uses ``round(n_genes ** (1/3))``.
    n_neighbors
        Number of non-self neighbours used to build the graph.
    maxdim
        Largest homology dimension to compute.
    neighbor_backend
        ``"exact"`` or ``"pynndescent"``.
    ph_backend
        ``"ripser"`` or ``"gudhi"``.
    metric 
        ``"eucl"``or ``"cknn"``.
    fitted_embedder
        Optional fitted embedder used without refitting.
    embedder_kwargs
        Arguments passed only when a new embedder is fitted.
    """

    def __init__(
        self,
        adata: Any = None,
        layer: str = d.DEFAULT_LAYER,
        embedder: str = d.DEFAULT_EMBEDDER,
        n_components: int | None = d.DEFAULT_N_COMPONENTS,
        n_neighbors: int = d.DEFAULT_N_NEIGHBORS,
        maxdim: int = d.DEFAULT_MAXDIM,
        neighbor_backend: str = d.DEFAULT_NEIGHBOR_BACKEND,
        ph_backend: str = DEFAULT_PH_BACKEND,
        metric: str = d.DEFAULT_METRIC,
        dtype: np.dtype = d.DEFAULT_DTYPE,
        fitted_embedder: Any = None,
        embedder_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.adata = adata
        self.layer = layer
        self.embedder_name = str(embedder).upper()
        self.n_components = n_components
        self.n_neighbors = int(n_neighbors)
        self.maxdim = int(maxdim)
        self.neighbor_backend = neighbor_backend
        self.ph_backend = ph_backend.lower()
        self.metric = metric
        self.dtype = dtype
        self.embedder = fitted_embedder
        self.embedder_kwargs = dict(embedder_kwargs or {})

        if self.neighbor_backend not in VALID_NEIGHBOR_BACKENDS:
            raise ValueError(
                f"neighbor_backend must be one of {sorted(VALID_NEIGHBOR_BACKENDS)}"
            )
        if self.ph_backend not in VALID_PH_BACKENDS:
            raise ValueError(f"ph_backend must be one of {sorted(VALID_PH_BACKENDS)}")
        if self.metric not in ("eucl", "cknn"):
            raise ValueError("metric must be 'eucl' or 'cknn'")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")
        if self.maxdim < 0:
            raise ValueError("maxdim must be non-negative")

        self.X: Any = None
        self.n_cells: int | None = None
        self.n_genes: int | None = None
        self.obs_names: np.ndarray | None = None
        self.var_names: np.ndarray | None = None
        self.embedding: np.ndarray | None = None
        self.dgms: list[np.ndarray] | None = None

    def _set_data(self, adata: Any | None = None) -> None:
        """Attach AnnData and cache only the fields required downstream."""
        if adata is not None:
            self.adata = adata
        if self.adata is None:
            return
        if self.layer != "X" and self.layer not in self.adata.layers:
            raise KeyError(f"layer {self.layer!r} is not present in adata.layers")

        self.X = self.adata.X if self.layer == "X" else self.adata.layers[self.layer]
        self.n_cells, self.n_genes = map(int, self.X.shape)
        self.obs_names = np.asarray(self.adata.obs_names).astype(str)
        self.var_names = np.asarray(self.adata.var_names).astype(str)
        if self.n_cells == 0 or self.n_genes == 0:
            raise ValueError("adata must contain at least one cell and one feature")
        if self.n_components is None:
            self.n_components = max(1, int(round(self.n_genes ** (1 / 3))))

    def _fit_embedder(self) -> None:
        """Fit the configured embedder unless one is already attached."""
        if self.embedder is not None:
            return
        if self.X is None:
            raise ValueError("AnnData is required to fit an embedder")
        try:
            embedder_class = Embedder.Embedder.registry[self.embedder_name]
        except KeyError as exc:
            raise ValueError(
                f"unknown embedder {self.embedder_name!r}; available: "
                f"{sorted(Embedder.Embedder.registry)}"
            ) from exc
        self.embedder = embedder_class(
            n_components=int(self.n_components), dtype=self.dtype, **self.embedder_kwargs
        )
        self.embedder.fit(self.X)

    def fit_embedding(self, adata: Any = None, embedding: Any = None) -> "XplainTDA":
        """Attach or fit the embedding without computing persistent homology."""
        self._set_data(adata)
        if embedding is None:
            if self.X is None:
                raise ValueError("provide AnnData or an explicit embedding")
            self._fit_embedder()
            embedding = self.embedder.transform(self.X)
        self.embedding = _validate_embedding(embedding, self.dtype, self.n_cells)
        self.n_cells = int(self.embedding.shape[0])
        self.n_components = int(self.embedding.shape[1])
        return self

    def fit(self, adata: Any = None, embedding: Any = None) -> "XplainTDA":
        """Fit or attach the embedding, then compute persistence diagrams."""
        self.fit_embedding(adata=adata, embedding=embedding)
        self.dgms = self.recompute_ph(self.embedding)
        return self

    def embed(self, X: Any) -> np.ndarray:
        """Project a matrix with the already-fitted embedder."""
        if self.embedder is None:
            raise ValueError("no fitted embedder is attached")
        return _validate_embedding(self.embedder.transform(X), self.dtype)

    def recompute_ph(self, embedding: Any) -> list[np.ndarray]:
        """Compute diagrams for another embedding with this object's PH settings."""
        embedding = _validate_embedding(embedding, self.dtype)
        return compute_persistence(
            embedding,
            n_neighbors=self.n_neighbors,
            maxdim=self.maxdim,
            neighbor_backend=self.neighbor_backend,
            ph_backend=self.ph_backend,
            metric=self.metric,
        )

    @property
    def is_fitted(self) -> bool:
        """Whether both an embedding and persistence diagrams are available."""
        return self.embedding is not None and self.dgms is not None

    def save(
        self,
        path: str | Path,
        *,
        include_data: bool = False,
        include_embedder: bool = True,
        compress: int = 3,
    ) -> Path:
        """Serialize configuration and fitted state with joblib.

        Set ``include_data=True`` when the loaded object must be explainable
        without reattaching the original AnnData object.
        """
        state = {
            "config": {
                "layer": self.layer,
                "embedder": self.embedder_name,
                "n_components": self.n_components,
                "n_neighbors": self.n_neighbors,
                "maxdim": self.maxdim,
                "neighbor_backend": self.neighbor_backend,
                "ph_backend": self.ph_backend,
                "metric": self.metric,
                "dtype": self.dtype,
                "embedder_kwargs": self.embedder_kwargs,
            },
            "X": self.X if include_data else None,
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "obs_names": self.obs_names,
            "var_names": self.var_names,
            "embedding": self.embedding,
            "dgms": self.dgms,
            "embedder": self.embedder if include_embedder else None,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, path, compress=compress)
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        adata: Any = None,
        fitted_embedder: Any = None,
    ) -> "XplainTDA":
        """Load a saved object and optionally attach data or an embedder."""
        state = joblib.load(path)
        obj = cls(
            adata=adata,
            fitted_embedder=(
                fitted_embedder
                if fitted_embedder is not None
                else state.get("embedder")
            ),
            **state["config"],
        )
        if adata is not None:
            obj._set_data()
            expected_shape = (state.get("n_cells"), state.get("n_genes"))
            if None not in expected_shape and obj.X.shape != expected_shape:
                raise ValueError(
                    f"attached AnnData has shape {obj.X.shape}, expected {expected_shape}"
                )
            saved_names = state.get("var_names")
            if saved_names is not None and not np.array_equal(obj.var_names, saved_names):
                raise ValueError("attached AnnData has different features or feature order")
        else:
            obj.X = state.get("X")
            obj.n_cells = state.get("n_cells")
            obj.n_genes = state.get("n_genes")
            obj.obs_names = state.get("obs_names")
            obj.var_names = state.get("var_names")
        obj.embedding = state.get("embedding")
        obj.dgms = state.get("dgms")
        return obj

    def save_embedder(self, path: str | Path, compress: int = 3) -> Path:
        """Save the fitted embedder using its native method when available."""
        if self.embedder is None:
            raise ValueError("no fitted embedder is attached")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.embedder, "save"):
            self.embedder.save(path, compress=compress)
        else:
            joblib.dump(self.embedder, path, compress=compress)
        return path
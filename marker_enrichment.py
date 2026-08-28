import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

CELLTYPE_FOLDER = {
    "PBMC": "Raw_CellType",
    "BMMCMultiOme": "Coarse_CellType",
    "HumanBrain": "Raw_celltype",
}

DATASETS = [
    "PBMC",
    "HumanBrain",
    "BMMCMultiOme",
]

N_RANDOM = 10000
SEED = 0

# PATHS

def get_groups_dir(dataset, BASE):
    return (
        BASE
        / dataset
        / "XplainTDA"
        / "GEX"
        / "Explanations"
        / CELLTYPE_FOLDER[dataset]
        / "results"
        / "groups"
    )


def load_group_files(dataset, BASE):
    """
    Load and concatenate all per-group explanation CSVs.
    The group name is taken from the filename.
    """

    groups_dir = get_groups_dir(dataset, BASE)

    frames = []

    for csv_path in sorted(groups_dir.glob("*.csv")):

        group_name = csv_path.stem

        df_g = pd.read_csv(csv_path)

        df_g["group"] = group_name

        frames.append(df_g)

    if not frames:
        raise FileNotFoundError(
            f"No group CSVs found in {groups_dir}"
        )

    return pd.concat(frames, ignore_index=True)



# RANDOM GENE-SET TEST

def random_set_mwu_test(
    marker_scores,
    other_scores,
    n_random=10000,
    seed=0,
):
    """
    Compare a fixed marker gene set against random gene
    sets of the same size.

    The observed AUC is calculated by comparing the marker
    genes against all non-marker genes.

    The empirical p-value is obtained by repeatedly drawing
    random gene sets of the same size as the marker set and
    comparing their AUC against the observed marker AUC.
    """

    marker_scores = np.asarray(marker_scores)
    other_scores = np.asarray(other_scores)

    n_markers = len(marker_scores)

    rng = np.random.default_rng(seed)

    U_obs, _ = mannwhitneyu(
        marker_scores,
        other_scores,
        alternative="greater",
        method="asymptotic",
    )

    auc_obs = U_obs / (
        n_markers * len(other_scores)
    )

    random_aucs = np.empty(n_random)

    for i in range(n_random):

        random_scores = rng.choice(other_scores,size=n_markers,replace=False,)

        U_random, _ = mannwhitneyu(random_scores,other_scores,alternative="greater",method="asymptotic",)

        random_aucs[i] = U_random / (n_markers * len(other_scores))

    p_empirical = (np.sum(random_aucs >= auc_obs) + 1) / (n_random + 1)

    null_auc_low, null_auc_high = np.percentile(random_aucs,[2.5, 97.5],)

    return {
        "AUC": auc_obs,
        "p_empirical": p_empirical,
        "null_AUC_low": null_auc_low,
        "null_AUC_high": null_auc_high,
        "random_AUCs": random_aucs,
    }

# TEST MARKER ENRICHMENT

def test_marker_enrichment_random_sets(
    df,
    marker_dict,
    group_col="group",
    score_col="score",
    gene_col="gene",
    n_random=10000,
    seed=0,
):
    """
    Run the random gene-set test for every
    cell type × homology dimension.
    """

    results = []

    for h_dim in sorted(df["homology_dim"].unique()):

        sub_h = df[
            df["homology_dim"] == h_dim
        ]

        for group in sorted(sub_h[group_col].unique()):

            sub = sub_h[sub_h[group_col] == group]

            markers = set(marker_dict.get(group, []))

            marker_scores = sub.loc[sub[gene_col].isin(markers),score_col,].to_numpy()

            other_scores = sub.loc[~sub[gene_col].isin(markers),score_col,].to_numpy()

            n_markers = len(marker_scores)
            n_other = len(other_scores)

            # Check sample sizes

            if n_markers < 2:
                print(
                    f"Skipping {group}, H{h_dim}: "
                    f"only {n_markers} marker genes"
                )
                continue

            if n_other < n_markers:
                print(
                    f"Skipping {group}, H{h_dim}: "
                    f"{n_other} non-marker genes, "
                    f"but {n_markers} markers"
                )
                continue

            test = random_set_mwu_test(
                marker_scores=marker_scores,
                other_scores=other_scores,
                n_random=n_random,
                seed=seed,
            )

            results.append({
                "homology_dim": h_dim,
                "group": group,
                "n_markers": n_markers,
                "n_other": n_other,
                "median_marker": np.median(
                    marker_scores
                ),
                "median_other": np.median(
                    other_scores
                ),
                "AUC": test["AUC"],
                "null_AUC_low": test["null_AUC_low"],
                "null_AUC_high": test["null_AUC_high"],
                "p_empirical": test["p_empirical"],
            })

    return pd.DataFrame(results)

# MAIN FUNCTION

def run_marker_enrichment(
    BASE,
    marker_sets,
    datasets=DATASETS,
    n_random=N_RANDOM,
    seed=SEED,
    verbose=True,
):
    """
    Run marker-gene enrichment analysis for all datasets.

    Parameters
    ----------
    BASE : pathlib.Path
        Base directory containing the datasets.

    marker_sets : dict
        Dictionary mapping dataset names to marker-gene
        dictionaries.

    datasets : list
        Datasets to analyse.

    n_random : int
        Number of random gene sets used for the empirical
        null distribution.

    seed : int
        Random seed.

    verbose : bool
        If True, print results for each dataset.

    Returns
    -------
    dataset_results : dict
        Dictionary mapping dataset names to result
        DataFrames.
    """

    dataset_results = {}

    for dataset in datasets:

        if verbose:
            print(
                f"\n{'=' * 60}\n"
                f"Processing {dataset}\n"
                f"{'=' * 60}"
            )
        # Load data
        df_d = load_group_files(dataset=dataset,BASE=BASE,)

        # Marker sets
        marker_dict = marker_sets.get(dataset,{},)

        results_d = test_marker_enrichment_random_sets(
            df=df_d,
            marker_dict=marker_dict,
            n_random=n_random,
            seed=seed,
        )
        # FDR correction
        # Correct separately within each homology dimension
        if not results_d.empty:
            results_d[
                "p_empirical_adj_per_dim"
            ] = (
                results_d
                .groupby("homology_dim")[
                    "p_empirical"
                ]
                .transform(
                    lambda p: multipletests(
                        p,
                        method="fdr_bh",
                    )[1]
                )
            )

        dataset_results[dataset] = results_d

        if verbose and not results_d.empty:

            print(
                results_d[
                    [
                        "homology_dim",
                        "group",
                        "n_markers",
                        "n_other",
                        "median_marker",
                        "median_other",
                        "AUC",
                        "null_AUC_low",
                        "null_AUC_high",
                        "p_empirical",
                        "p_empirical_adj_per_dim",
                    ]
                ].to_string(index=False)
            )

    return dataset_results
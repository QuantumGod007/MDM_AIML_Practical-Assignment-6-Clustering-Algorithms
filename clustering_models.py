"""AIML Practical 06: clustering wine chemistry, with reproducible evaluation."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
K_VALUES = range(2, 7)
DBSCAN_EPS = [round(float(v), 1) for v in np.arange(1.8, 3.51, 0.1)]
DBSCAN_MIN_SAMPLES = [3, 4, 5, 6, 8]

def load_dataset():
    wine = load_wine()
    features = pd.DataFrame(wine.data, columns=wine.feature_names)
    labels = wine.target.copy()
    assert features.shape == (178, 13)
    assert not features.isna().any().any()
    return features, labels

def standardize(features):
    return StandardScaler().fit_transform(features)

def internal_metrics(X, labels, exclude_noise=False):
    labels = np.asarray(labels)
    mask = labels != -1 if exclude_noise else np.ones(len(labels), dtype=bool)
    retained = int(mask.sum())
    groups = np.unique(labels[mask])
    if len(groups) < 2 or retained <= len(groups):
        return None
    return {
        "Clusters": len(groups), "Coverage": retained / len(labels),
        "Noise": len(labels) - retained,
        "Silhouette": silhouette_score(X[mask], labels[mask]),
        "Davies_Bouldin": davies_bouldin_score(X[mask], labels[mask]),
        "Calinski_Harabasz": calinski_harabasz_score(X[mask], labels[mask])
    }

def tune_complete_models(X):
    rows = []
    for k in K_VALUES:
        models = {
            "K-Means": KMeans(n_clusters=k, n_init=20, random_state=SEED),
            "Agglomerative (Ward)": AgglomerativeClustering(n_clusters=k, linkage="ward"),
            "Gaussian Mixture": GaussianMixture(n_components=k, n_init=5, random_state=SEED),
        }
        for name, estimator in models.items():
            labels = estimator.fit_predict(X)
            rows.append({"Model": name, "Parameter": f"k={k}", "k": k, **internal_metrics(X, labels)})
    return pd.DataFrame(rows)

def tune_dbscan(X):
    rows = []
    for eps in DBSCAN_EPS:
        for min_samples in DBSCAN_MIN_SAMPLES:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
            m = internal_metrics(X, labels, exclude_noise=True)
            if m is None or m["Coverage"] < 0.75:
                continue
            rows.append({
                "Model": "DBSCAN", "Parameter": f"eps={eps:.1f}, min_samples={min_samples}",
                "eps": eps, "min_samples": min_samples,
                "Coverage_adjusted_silhouette": m["Silhouette"] * m["Coverage"],
                **m
            })
    return pd.DataFrame(rows)

def select_best_models(complete, density, X):
    selected = {}
    for name in ("K-Means", "Agglomerative (Ward)", "Gaussian Mixture"):
        row = (complete.loc[complete.Model == name]
               .sort_values(["Silhouette", "Davies_Bouldin"], ascending=[False, True])
               .iloc[0])
        k = int(row["k"])
        if name == "K-Means":
            estimator = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        elif name == "Agglomerative (Ward)":
            estimator = AgglomerativeClustering(n_clusters=k, linkage="ward")
        else:
            estimator = GaussianMixture(n_components=k, random_state=SEED, n_init=5)
        selected[name] = (estimator, estimator.fit_predict(X), row.to_dict())
    if not density.empty:
        row = density.sort_values(["Coverage_adjusted_silhouette", "Coverage"], ascending=False).iloc[0]
        estimator = DBSCAN(eps=float(row["eps"]), min_samples=int(row["min_samples"]))
        selected["DBSCAN"] = (estimator, estimator.fit_predict(X), row.to_dict())
    return selected

def cluster_profile(features, labels):
    df = pd.DataFrame(standardize(features), columns=features.columns)
    df["cluster"] = labels
    return df.groupby("cluster").mean()

def save_figure(fig, output_dir, name):
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

def save_graphs(complete, selected, X, features, output_dir):
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for name, frame in complete.groupby("Model", sort=False):
        frame = frame.sort_values("k")
        ax.plot(frame.k, frame.Silhouette, marker="o", linewidth=2, label=name)
    ax.set(xlabel="Number of clusters (k)", ylabel="Silhouette score (higher is better)",
           title="Model selection using internal clustering quality")
    ax.set_xticks(list(K_VALUES))
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "silhouette_by_k")

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9))
    for ax, (name, (_, labels, detail)) in zip(axes.flat, selected.items()):
        for c in sorted(set(labels)):
            mask = labels == c
            kw = {"marker": "x", "color": "#777777"} if c == -1 else {}
            ax.scatter(coords[mask, 0], coords[mask, 1], s=25, alpha=.8,
                       label=f"{'Noise' if c == -1 else 'Cluster ' + str(c)} ({sum(mask)})", **kw)
        ax.set_title(f"{name} | {detail['Parameter']}", fontsize=10)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend(fontsize=7)
    fig.suptitle("Cluster assignments projected onto the same 2D PCA space")
    fig.tight_layout()
    save_figure(fig, output_dir, "cluster_pca_comparison")

    km = selected["K-Means"][1]
    profile = cluster_profile(features, km)
    columns = profile.var(axis=0).sort_values(ascending=False).head(8).index
    table = profile[columns]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    img = ax.imshow(table.values, cmap="RdBu_r", vmin=-1.8, vmax=1.8, aspect="auto")
    ax.set_xticks(range(len(columns)), columns, rotation=40, ha="right")
    ax.set_yticks(range(len(table.index)), [f"Cluster {c} (n={sum(km == c)})" for c in table.index])
    ax.set_title("K-Means chemical profiles (mean standardized feature values)")
    fig.colorbar(img, ax=ax, label="Mean z-score", shrink=.85)
    fig.tight_layout()
    save_figure(fig, output_dir, "kmeans_cluster_profiles")

    names = list(selected)
    scores = [selected[n][2]["Silhouette"] for n in names]
    covers = [selected[n][2]["Coverage"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, scores)
    for bar, s, c in zip(bars, scores, covers):
        ax.text(bar.get_x() + bar.get_width()/2, s + .006, f"{s:.3f}\n({c:.0%} covered)", ha="center")
    ax.set_ylim(0, max(scores) + .12)
    ax.set_ylabel("Silhouette score (higher is better)")
    ax.set_title("Selected models: silhouette AND sample coverage")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    save_figure(fig, output_dir, "selected_model_comparison")
    return pca.explained_variance_ratio_.tolist()

def run_experiment(output_dir=OUTPUT_DIR, make_plots=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features, reference = load_dataset()
    X = standardize(features)
    complete = tune_complete_models(X)
    density = tune_dbscan(X)
    selected = select_best_models(complete, density, X)
    rows = []
    assignments = features.copy()
    for name, (_, labels, details) in selected.items():
        assignments[name.replace(" ", "_").replace("(", "").replace(")", "")] = labels
        m = internal_metrics(X, labels, exclude_noise=(name == "DBSCAN"))
        rows.append({
            "Model": name, "Parameters": details["Parameter"], **m,
            "ARI_reference_only": adjusted_rand_score(reference, labels),
            "Cluster_sizes": json.dumps({str(k): v for k, v in sorted(Counter(map(int, labels)).items())})
        })
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "model_comparison.csv", index=False)
    pd.concat([complete, density], ignore_index=True).to_csv(output_dir / "parameter_search.csv", index=False)
    assignments.to_csv(output_dir / "cluster_assignments.csv", index=False)
    cluster_profile(features, selected["K-Means"][1]).to_csv(output_dir / "kmeans_standardized_profiles.csv")
    variance = (save_graphs(complete, selected, X, features, output_dir) if make_plots
                else PCA(n_components=2).fit(X).explained_variance_ratio_.tolist())
    best = results[results.Model != "DBSCAN"].sort_values(
        ["Silhouette", "Davies_Bouldin"], ascending=[False, True]).iloc[0]
    summary = {
        "dataset": "UCI Wine Recognition (scikit-learn load_wine)",
        "samples": len(features), "numeric_features": len(features.columns),
        "reference_cultivar_counts": {str(k): int(v) for k, v in sorted(Counter(reference).items())},
        "normalization": "StandardScaler for all 13 features",
        "selection_rule": "Highest full-sample silhouette among tuned complete-coverage models",
        "recommended_model": str(best.Model), "recommended_parameters": str(best.Parameters),
        "pca_two_component_variance": variance,
        "note": "ARI used post hoc only; cultivar labels never used for selection or fitting.",
        "dbscan_note": "Noise excluded from density metrics; compare coverage alongside silhouette."
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return results, summary, selected

def main():
    metrics, summary, _ = run_experiment()
    print("Wine samples:", summary["samples"], "| Features:", summary["numeric_features"])
    print("Best full-coverage clustering:", summary["recommended_model"], summary["recommended_parameters"])
    print(metrics.round(4).to_string(index=False))
    print("PCA explained variance:", sum(summary["pca_two_component_variance"]))
    print("Results saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()

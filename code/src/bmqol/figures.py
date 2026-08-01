"""Publication figures generated only from the locked numerical result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _setup():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save(figure, output: Path, name: str) -> None:
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"{name}.{suffix}", bbox_inches="tight")


def write_figures(payload: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt = _setup()
    colors = ("#1f4e79", "#c44e52", "#2a9d8f", "#7b2cbf")
    records = payload["records"]

    figure, axes = plt.subplots(2, 2, figsize=(6.8, 4.6), sharex=False, sharey=True)
    for axis, record, color in zip(axes.ravel(), records, colors):
        depths = [item["depth"] for item in record["frontier"]]
        bounds = [item["certificate"] for item in record["frontier"]]
        errors = [max(item["gradient_error"], 1.0e-18) for item in record["frontier"]]
        axis.semilogy(depths, bounds, "o-", color=color, label="certificate")
        axis.semilogy(depths, errors, "s--", color="black", label="observed error")
        axis.axhline(payload["tolerance"], color="#777777", linestyle=":", label="tolerance")
        axis.axvline(record["selected_depth"], color=color, alpha=0.25)
        axis.set_title(record["name"])
        axis.set_xlabel("block Krylov depth")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("gradient 2-norm")
    axes[1, 0].set_ylabel("gradient 2-norm")
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    _save(figure, output, "certificate_frontier")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(3.35, 2.55))
    for record, color in zip(records, colors):
        singular = np.asarray(record["probe_singular_values"])
        axis.semilogy(
            np.arange(1, singular.size + 1), singular / singular[0], "o-",
            color=color, label=record["name"],
        )
        axis.axvline(record["selected_rank"], color=color, alpha=0.18)
    axis.set_xlabel("probe singular-value index")
    axis.set_ylabel(r"$\sigma_j/\sigma_1$")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    _save(figure, output, "probe_spectrum")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(3.35, 2.55))
    sizes = np.asarray([record["n"] for record in records])
    proposed = np.asarray([record["proposed_timing"]["median_seconds"] for record in records])
    full = np.asarray([record["full_same_depth_timing"]["median_seconds"] for record in records])
    finite = np.asarray([record["finite_difference_timing"]["median_seconds"] for record in records])
    axis.loglog(sizes, proposed, "o-", color=colors[0], label="rank-depth adaptive")
    axis.loglog(sizes, full, "s-", color=colors[1], label="full block, same depth")
    axis.loglog(sizes, finite, "^-", color=colors[2], label="central difference")
    axis.set_xlabel("matrix dimension")
    axis.set_ylabel("complete gradient time (s)")
    axis.grid(alpha=0.2, which="both")
    axis.legend(frameon=False)
    figure.tight_layout()
    _save(figure, output, "runtime_scaling")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(3.35, 2.55))
    for record, color in zip(records, colors):
        work = [item["vector_equivalents"] for item in record["frontier"]]
        errors = [max(item["gradient_error"], 1.0e-18) for item in record["frontier"]]
        axis.semilogy(work, errors, "o-", color=color, label=record["name"])
        axis.scatter(
            [record["full_same_depth_vector_equivalents"]],
            [max(record["full_same_depth_error"], 1.0e-18)],
            marker="x", s=35, color=color,
        )
    axis.set_xlabel("Hamiltonian-vector equivalents")
    axis.set_ylabel("observed gradient error")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    _save(figure, output, "work_accuracy")
    plt.close(figure)

    sweep = payload.get("rank_depth_sweep")
    if sweep is not None:
        figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.45))
        ranks = np.asarray(sweep["selected_ranks"])
        depths = np.asarray(sweep["selected_depths"])
        extent = [
            min(sweep["horizons"]) - 1.0,
            max(sweep["horizons"]) + 1.0,
            -np.log10(max(sweep["tolerances"])) - 0.5,
            -np.log10(min(sweep["tolerances"])) + 0.5,
        ]
        for axis, values, title, cmap in (
            (axes[0], ranks, "selected probe rank", "Blues"),
            (axes[1], depths, "selected Krylov depth", "Oranges"),
        ):
            image = axis.imshow(values, origin="lower", aspect="auto", extent=extent, cmap=cmap)
            for row in range(values.shape[0]):
                for column in range(values.shape[1]):
                    axis.text(
                        sweep["horizons"][column],
                        -np.log10(sweep["tolerances"][row]),
                        str(values[row, column]),
                        ha="center", va="center", fontsize=8,
                    )
            axis.set_title(title)
            axis.set_xlabel("maximum evolution time")
            axis.set_ylabel(r"$-\log_{10}$ tolerance")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        _save(figure, output, "rank_depth_map")
        plt.close(figure)


"""Filter raw AV2 motion-forecasting scenarios down to a small, "interesting",
strictly vehicle-to-vehicle subset.

Scope is vehicle-only: the focal (ego) track must itself be of object_type
"vehicle", and any scenario containing a pedestrian, cyclist, or
motorcyclist ANYWHERE (not just near the ego) is dropped entirely. Only
vehicle/bus agents count toward the interaction signal. This is a strict
filter (not just "ignore them for scoring") so filtered_data never contains
pedestrian/cyclist/motorcyclist dynamics at all.

Ranks each surviving scenario by two signals and combines them:
  - curvature_score:   total heading change (turning angle, radians) of the
                        focal track. High = the ego takes a real turn/lane
                        change rather than driving straight.
  - interaction_score: average number of other vehicles/buses within
                        --interaction-radius meters of the focal track per
                        timestep. High = the ego's future is plausibly
                        influenced by nearby traffic (following, yielding,
                        merging, crossing), not driving in isolation.
Both are percentile-ranked within the pool and combined with
--curvature-weight / --interaction-weight (default 0.5/0.5).

Split provenance:
  - filtered_data/train (--n-train) and filtered_data/val (--n-val) are both
    carved from the raw AV2 `train` split (disjoint, score-interleaved), since
    that pool is large enough to hold both after the strict vehicle-only filter.
  - filtered_data/test (--n-test) is carved from the raw AV2 `val` split (which
    has full labeled futures, unlike AV2's official masked test split). Pass
    --n-test -1 (the default) to keep every scenario that survives the filter
    there, since that pool is small.

AV2's official `test` split is skipped by default (--n-official-test 0) since
its future is masked and only scorable via Argoverse's hidden leaderboard;
pass a positive value only if prepping a real leaderboard submission (output
goes to filtered_data/official_test to avoid confusion with the local test
set above).

Usage:
    python filtering.py
    python filtering.py --n-train 12000 --n-val 8000 --n-test -1 --overwrite
"""

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_DATA_ROOT = Path("/workspace/workspace/data/av2")
DEFAULT_OUTPUT_ROOT = Path("/workspace/workspace/filtered_data")
VEHICLE_LIKE_TYPES = {"vehicle", "bus"}
VRU_TYPES = {"pedestrian", "cyclist", "motorcyclist"}


def scenario_features(scenario_dir: Path, min_timesteps: int, interaction_radius: float):
    """Return a dict of per-scenario features, or None if the scenario is unusable
    or contains a pedestrian/cyclist/motorcyclist anywhere in the scene."""
    scenario_id = scenario_dir.name
    parquet_path = scenario_dir / f"scenario_{scenario_id}.parquet"
    if not parquet_path.exists():
        return None

    df = pd.read_parquet(
        parquet_path,
        columns=[
            "track_id",
            "object_type",
            "focal_track_id",
            "timestep",
            "position_x",
            "position_y",
            "heading",
        ],
    )
    focal_id = df["focal_track_id"].iloc[0]
    focal = df[df["track_id"] == focal_id].sort_values("timestep")
    if len(focal) < min_timesteps:
        return None
    if focal["object_type"].iloc[0] != "vehicle":
        return None
    if df["object_type"].isin(VRU_TYPES).any():
        return None

    heading = focal["heading"].to_numpy()
    diffs = np.diff(heading)
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    curvature_score = float(np.abs(diffs).sum())

    others = df[(df["track_id"] != focal_id) & (df["object_type"].isin(VEHICLE_LIKE_TYPES))]
    merged = others.merge(
        focal[["timestep", "position_x", "position_y"]],
        on="timestep",
        suffixes=("", "_focal"),
    )
    if merged.empty:
        interaction_score = 0.0
        min_agent_distance = float("nan")
        num_nearby_agents = 0
    else:
        dx = merged["position_x"] - merged["position_x_focal"]
        dy = merged["position_y"] - merged["position_y_focal"]
        dist = np.hypot(dx.to_numpy(), dy.to_numpy())
        near = dist < interaction_radius
        interaction_score = float(near.sum()) / len(focal)
        min_agent_distance = float(dist.min())
        num_nearby_agents = int(merged.loc[near, "track_id"].nunique())

    return {
        "scenario_id": scenario_id,
        "curvature_score": curvature_score,
        "interaction_score": interaction_score,
        "min_agent_distance": min_agent_distance,
        "num_nearby_agents": num_nearby_agents,
        "num_points": len(focal),
    }


def rank_split(split_dir: Path, min_timesteps: int, interaction_radius: float, workers: int):
    scenario_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(scenario_features, d, min_timesteps, interaction_radius)
            for d in scenario_dirs
        ]
        for f in tqdm(futures, desc=f"scanning {split_dir.name}", unit="scenario"):
            r = f.result()
            if r is not None:
                results.append(r)

    df = pd.DataFrame(results)
    if df.empty:
        return df
    df["curvature_rank"] = df["curvature_score"].rank(pct=True)
    df["interaction_rank"] = df["interaction_score"].rank(pct=True)
    return df


def add_combined_score(df: pd.DataFrame, curvature_weight: float, interaction_weight: float):
    df = df.copy()
    df["combined_score"] = (
        curvature_weight * df["curvature_rank"] + interaction_weight * df["interaction_rank"]
    )
    return df.sort_values("combined_score", ascending=False).reset_index(drop=True)


def interleave_split(df_sorted: pd.DataFrame, n_a: int, n_b: int):
    """Split the top (n_a + n_b) rows of a score-sorted df into two score-interleaved,
    disjoint subsets of exactly n_a and n_b rows (so both have a similar score spread
    instead of one getting only the top scores and the other only the tail)."""
    total = n_a + n_b
    pool = df_sorted.head(total).reset_index(drop=True)
    if n_a <= 0:
        return pool.iloc[0:0], pool
    if n_b <= 0:
        return pool, pool.iloc[0:0]

    a_positions = np.unique(np.linspace(0, len(pool) - 1, n_a).astype(int))
    while len(a_positions) < n_a:
        remaining = np.setdiff1d(np.arange(len(pool)), a_positions)
        a_positions = np.union1d(a_positions, remaining[: n_a - len(a_positions)])
    a_mask = np.zeros(len(pool), dtype=bool)
    a_mask[a_positions] = True
    return pool[a_mask], pool[~a_mask]


def export_scenarios(
    selected: pd.DataFrame,
    src_split_dir: Path,
    out_dir: Path,
    manifest_name: str,
    link: bool,
    overwrite: bool,
):
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for scenario_id in tqdm(
        selected["scenario_id"], desc=f"exporting {out_dir.name}", unit="scenario"
    ):
        src = src_split_dir / scenario_id
        dst = out_dir / scenario_id
        if dst.exists() or dst.is_symlink():
            continue
        if link:
            dst.symlink_to(src, target_is_directory=True)
        else:
            shutil.copytree(src, dst)

    selected.to_csv(out_dir.parent / manifest_name, index=False)
    print(
        f"{out_dir.name}: {len(selected)} scenarios "
        f"(curvature {selected['curvature_score'].min():.2f}-{selected['curvature_score'].max():.2f} rad, "
        f"interaction {selected['interaction_score'].min():.2f}-{selected['interaction_score'].max():.2f} agents/step)"
    )


def run_train_val_split(
    n_train: int,
    n_val: int,
    data_root: Path,
    output_root: Path,
    min_timesteps: int,
    interaction_radius: float,
    curvature_weight: float,
    interaction_weight: float,
    workers: int,
    link: bool,
    overwrite: bool,
):
    """train and val are both carved (disjoint, score-interleaved) from the raw
    AV2 `train` split, since it's the only pool large enough to hold both after
    the strict vehicle-only / no-VRU filter."""
    if n_train <= 0 and n_val <= 0:
        print("skip train/val: both n_train and n_val are 0")
        return

    split_dir = data_root / "train"
    if not split_dir.exists():
        print(f"skip train/val: {split_dir} does not exist")
        return

    ranked = rank_split(split_dir, min_timesteps, interaction_radius, workers)
    if ranked.empty:
        print("skip train/val: no usable scenarios found")
        return
    if len(ranked) < n_train + n_val:
        print(
            f"warning: only {len(ranked)} scenarios survived the filter, "
            f"fewer than requested n_train+n_val={n_train + n_val}"
        )

    ranked = add_combined_score(ranked, curvature_weight, interaction_weight)
    train_subset, val_subset = interleave_split(ranked, n_train, n_val)

    if n_train > 0:
        export_scenarios(train_subset, split_dir, output_root / "train", "train_manifest.csv", link, overwrite)
    if n_val > 0:
        export_scenarios(val_subset, split_dir, output_root / "val", "val_manifest.csv", link, overwrite)


def run_test_split(
    n_test: int,
    data_root: Path,
    output_root: Path,
    min_timesteps: int,
    interaction_radius: float,
    curvature_weight: float,
    interaction_weight: float,
    workers: int,
    link: bool,
    overwrite: bool,
):
    """test is carved from the raw AV2 `val` split (full labeled futures), kept
    disjoint from train/val since it comes from an entirely different raw pool.
    n_test <= 0 means "keep every scenario that survives the filter" (that pool
    is small after the strict no-VRU filter, so there's usually nothing to trim)."""
    if n_test == 0:
        print("skip test: n_test=0")
        return

    split_dir = data_root / "val"
    if not split_dir.exists():
        print(f"skip test: {split_dir} does not exist")
        return

    ranked = rank_split(split_dir, min_timesteps, interaction_radius, workers)
    if ranked.empty:
        print("skip test: no usable scenarios found")
        return

    ranked = add_combined_score(ranked, curvature_weight, interaction_weight)
    selected = ranked if n_test < 0 else ranked.head(n_test)
    export_scenarios(selected, split_dir, output_root / "test", "test_manifest.csv", link, overwrite)


def run_official_test(
    n_select: int,
    data_root: Path,
    output_root: Path,
    min_timesteps: int,
    interaction_radius: float,
    curvature_weight: float,
    interaction_weight: float,
    workers: int,
    link: bool,
    overwrite: bool,
):
    if n_select <= 0:
        print("skip official_test: n_official_test=0")
        return

    split_dir = data_root / "test"
    if not split_dir.exists():
        print(f"skip official_test: {split_dir} does not exist")
        return

    ranked = rank_split(split_dir, min_timesteps, interaction_radius, workers)
    if ranked.empty:
        print("skip official_test: no usable scenarios found")
        return

    ranked = add_combined_score(ranked, curvature_weight, interaction_weight)
    selected = ranked.head(n_select)
    export_scenarios(selected, split_dir, output_root / "official_test", "official_test_manifest.csv", link, overwrite)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n-train", type=int, default=12000)
    parser.add_argument("--n-val", type=int, default=8000)
    parser.add_argument(
        "--n-test",
        type=int,
        default=-1,
        help="scenarios to keep from the raw AV2 val split; -1 = keep all that survive the filter",
    )
    parser.add_argument(
        "--n-official-test",
        type=int,
        default=0,
        help="scenarios to prep from AV2's raw (label-masked) test split, for leaderboard submission only",
    )
    parser.add_argument("--interaction-radius", type=float, default=20.0, help="meters")
    parser.add_argument("--curvature-weight", type=float, default=0.5)
    parser.add_argument("--interaction-weight", type=float, default=0.5)
    parser.add_argument("--min-timesteps", type=int, default=50)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--link",
        action="store_true",
        help="symlink scenario dirs instead of copying (faster, saves disk)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="wipe existing output dir before exporting",
    )
    args = parser.parse_args()

    run_train_val_split(
        n_train=args.n_train,
        n_val=args.n_val,
        data_root=args.data_root,
        output_root=args.output_root,
        min_timesteps=args.min_timesteps,
        interaction_radius=args.interaction_radius,
        curvature_weight=args.curvature_weight,
        interaction_weight=args.interaction_weight,
        workers=args.workers,
        link=args.link,
        overwrite=args.overwrite,
    )

    run_test_split(
        n_test=args.n_test,
        data_root=args.data_root,
        output_root=args.output_root,
        min_timesteps=args.min_timesteps,
        interaction_radius=args.interaction_radius,
        curvature_weight=args.curvature_weight,
        interaction_weight=args.interaction_weight,
        workers=args.workers,
        link=args.link,
        overwrite=args.overwrite,
    )

    run_official_test(
        n_select=args.n_official_test,
        data_root=args.data_root,
        output_root=args.output_root,
        min_timesteps=args.min_timesteps,
        interaction_radius=args.interaction_radius,
        curvature_weight=args.curvature_weight,
        interaction_weight=args.interaction_weight,
        workers=args.workers,
        link=args.link,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

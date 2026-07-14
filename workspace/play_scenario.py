"""Render an AV2 motion-forecasting scenario directory to an MP4.

The focal (ego) track is drawn in orange, the autonomous vehicle in teal,
other tracked agents in light blue, with the local map (drivable area +
lane boundaries) underneath. 10 fps, one frame per timestep (110 frames for
train/val scenarios, 50 for AV2's official test split since its future is
masked).

Usage:
    python play_scenario.py filtered_data/train/0a0e17f4-f8ed-41c8-bf2a-3ecd2adab6a6
    python play_scenario.py filtered_data/train/0a0e17f4-f8ed-41c8-bf2a-3ecd2adab6a6 --out clips/turn1.mp4
"""

import argparse
from pathlib import Path

from av2.datasets.motion_forecasting.scenario_serialization import (
    load_argoverse_scenario_parquet,
)
from av2.datasets.motion_forecasting.viz.scenario_visualization import visualize_scenario
from av2.map.map_api import ArgoverseStaticMap


def render_scenario(scenario_dir: Path, out_path: Path):
    scenario_id = scenario_dir.name
    scenario_path = scenario_dir / f"scenario_{scenario_id}.parquet"
    if not scenario_path.exists():
        raise FileNotFoundError(f"no scenario parquet found at {scenario_path}")

    scenario = load_argoverse_scenario_parquet(scenario_path)
    static_map = ArgoverseStaticMap.from_map_dir(scenario_dir, build_raster=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_scenario(scenario, static_map, out_path)
    print(f"wrote {out_path.parent / (out_path.stem + '.mp4')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario_dir", type=Path, help="path to a scenario directory, e.g. filtered_data/train/<id>")
    parser.add_argument("--out", type=Path, default=None, help="output mp4 path (default: <scenario_dir>.mp4 next to the scenario dir)")
    args = parser.parse_args()

    scenario_dir = args.scenario_dir.resolve()
    out_path = args.out or scenario_dir.parent / f"{scenario_dir.name}.mp4"
    render_scenario(scenario_dir, out_path)


if __name__ == "__main__":
    main()

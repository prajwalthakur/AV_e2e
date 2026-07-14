"""Render a single Argoverse 2 motion-forecasting scenario to an MP4.

Usage:
    python .scripts/visualize_scenario.py workspace/data/av2/train/<scenario_id> [output.mp4]
"""
import sys
from pathlib import Path

from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.viz.scenario_visualization import visualize_scenario
from av2.map.map_api import ArgoverseStaticMap

scenario_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"{scenario_dir.name}.mp4")

scenario_id = scenario_dir.name
scenario_path = scenario_dir / f"scenario_{scenario_id}.parquet"
static_map_path = scenario_dir / f"log_map_archive_{scenario_id}.json"

scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
static_map = ArgoverseStaticMap.from_json(static_map_path)

visualize_scenario(scenario, static_map, out_path)
print(f"wrote {out_path}")

# Diffusion Model for Trajectory Forecasting — Project Report

## Goal

First hands-on diffusion model project. Scope is intentionally small: get a
working train/eval loop on a curated subset of Argoverse 2 (AV2) motion
forecasting data, where overfitting is acceptable — the point is to confirm
the pipeline (data -> model -> training -> eval) actually learns something,
not to chase leaderboard numbers.

## Done so far

### 1. Environment / data access
Installed `trajdata[av2_motion_forecasting]` for AV2 tooling support, and
downloaded the raw AV2 motion-forecasting dataset (train/val/test +
annotation files) via `s5cmd` (`.scripts/download_av2.sh`) into
`data/av2/{train,val,test}/<scenario_id>/`, each scenario containing a
`scenario_*.parquet` (agent tracks) and `log_map_archive_*.json` (map).

Raw split sizes: train ~199,908 / val ~24,988 / test ~24,984 scenarios.

### 2. Scenario filtering (`filtering.py`)
Rather than training on the full ~250k scenarios, we rank and keep a small
subset biased toward *interesting* scenes, since a small dataset with mostly
straight/empty-road driving would be both slow to filter through and boring
to overfit on. Two signals, computed per scenario from the focal (ego) track:

- **curvature_score** — total heading change (turning angle, radians) of the
  ego track. High = a real turn/lane-change, not straight-line driving.
- **interaction_score** — average number of other dynamic agents (vehicle,
  bus, pedestrian, cyclist, motorcyclist) within 20m of the ego per
  timestep. High = the ego's future is plausibly influenced by nearby
  traffic (following, yielding, merging), not driving in isolation.

Both are percentile-ranked within the split and combined 50/50 into a
`combined_score`; the top-K scenarios by that score are copied into
`filtered_data/`.

**Why not use AV2's official test split for evaluation:** AV2's `test` split
ships with the future masked (only scorable via Argoverse's hidden
leaderboard) — no local ground truth, so it's useless for computing
minADE/minFDE ourselves. Instead, the raw `val` split (which has full labeled
futures) is ranked once and partitioned into two disjoint local subsets —
`filtered_data/val` and `filtered_data/test` — interleaved by score so both
have a similar difficulty distribution rather than one getting the leftover
tail. `test` here means "our own held-out set," not AV2's.

**Current subset sizes** (`filtered_data/{train,val,test}` + matching
`*_manifest.csv` with per-scenario scores):

| split | scenarios | source |
|---|---|---|
| train | 12,000 | top combined_score from raw AV2 train |
| val   | 4,000  | score-interleaved half of raw AV2 val |
| test  | 4,000  | score-interleaved other half of raw AV2 val |

## TODO (next steps, with reasoning)

1. **Ego-frame (agent-centric) preprocessing.**
   Raw positions are in each city's absolute map frame (arbitrary large
   coordinates, arbitrary heading). Re-express every scenario relative to
   the ego: translate so the ego's position at the last observed timestep
   is the origin, rotate so its heading at that instant is 0. Apply the same
   transform to the ego's own history/future, other agents' tracks, and the
   map lane centerlines.
   *Why:* a diffusion model's noise schedule assumes the target lives in a
   compact, roughly-centered range — training on raw city coordinates (which
   vary by thousands of meters and any heading) gives the model nothing
   consistent to denoise toward, and it can't generalize across cities/
   directions. Ego-frame normalization fixes both.

2. **Top-K lane centerline extraction.**
   Once the map is in ego-frame, select the K nearest centerline points to
   the ego as conditioning input.
   *Why:* without map context, the diffusion model has no signal that the
   future path should follow the road (lane curvature, intersections) rather
   than go anywhere physically reachable — this is what keeps generated
   trajectories road-plausible.

3. **Cache preprocessed tensors to disk.**
   *Why:* avoid re-running the ego-frame transform + centerline lookup every
   epoch; the filtered subset is small enough to fully precompute once.

4. **Dataset / DataLoader class** over the cached tensors.

5. **Model.** Small 1D UNet or transformer-based denoiser over the future
   trajectory, conditioned on past + lane centerlines.
   *Why this scale:* matches the "small dataset, overfitting is fine" scope
   — a lightweight model trains fast enough to iterate on the pipeline
   itself rather than fighting compute.

6. **Training loop** (diffusion noise schedule + loss).

7. **Evaluation** on `filtered_data/val` and `filtered_data/test`: minADE6,
   minFDE6, miss rate — the standard AV2 motion-forecasting metrics, computed
   locally since both subsets carry real ground-truth futures.

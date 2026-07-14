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
futures) is used for our own local val/test, since we have ground truth for
it. `test` here means "our own held-out set," not AV2's.

### 3. Restricted scope to vehicle-to-vehicle only (single agent-type)
Initially the interaction signal counted any dynamic agent (vehicle, bus,
pedestrian, cyclist, motorcyclist) near the ego. We narrowed this to
**vehicle-only, strictly**: the focal (ego) track must itself be of type
`vehicle`, and any scenario containing a pedestrian/cyclist/motorcyclist
*anywhere in the scene* (not just near the ego) is dropped entirely. Only
vehicle/bus agents count toward `interaction_score`.

**Why:** this is a first diffusion-model project, and the goal is to validate
the pipeline (data → model → training → eval) with overfitting-is-fine
scope. Pedestrians and cyclists have qualitatively different motion dynamics
than vehicles — low speed, frequent stop/start, sharp direction changes,
sidewalk-constrained rather than lane-constrained. Mixing agent types into
one model means the network has to learn two very different motion
distributions simultaneously, which adds a confound: if training goes
poorly, it's unclear whether the diffusion setup itself is wrong or whether
it's struggling specifically with heterogeneous agent behavior. Confining
training to a single agent-type (vehicle-to-vehicle interaction only)
removes that confound — every future trajectory in the dataset, ego and
neighbors alike, follows roughly the same lane-constrained driving dynamics,
so if the model fails to learn, the problem is isolated to the diffusion
pipeline itself, not agent-type diversity. Multi-agent-type modeling can be
revisited later once the core pipeline is confirmed to work.

**Feasibility check performed first** (see [filtering.py](../filtering.py)):
scanning the full raw AV2 train (~199,908) and val (~24,988) splits found
176,740 / 22,134 scenarios with a vehicle focal track, and — under the
stricter "zero pedestrian/cyclist/motorcyclist anywhere in the scene"
constraint — 36,940 (train) / 4,538 (val) scenarios survive. That's enough
headroom in the raw train pool to source both our train and val subsets from
it, while reserving the entire raw val pool as a genuinely separate,
untouched-by-selection test set.

**Current subset sizes and provenance** (`filtered_data/{train,val,test}` +
matching `*_manifest.csv` with per-scenario scores):

| split | scenarios | source |
|---|---|---|
| train | 12,000 | score-interleaved subset of raw AV2 `train` (vehicle-only, no-VRU pool of 36,940) |
| val   | 8,000  | disjoint score-interleaved remainder of the same raw AV2 `train` pool |
| test  | 4,538 (all survivors) | entire raw AV2 `val` pool (vehicle-only, no-VRU), kept as a wholly separate held-out set |

train/val being drawn from the same raw AV2 split is fine here since AV2
scenarios are independent driving logs, not sequential — the score-interleaved
split still keeps them disjoint (no scenario appears in both).

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

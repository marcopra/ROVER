# Experiment 3: main-setting sink-state ablation

Status: implementation complete. Full results remain `incomplete` until all 60
pretraining and downstream runs finish. Smoke results are not scientific results.

## Audited main protocol

Paper's historical pixel Multi-Room config uses 84x84 RGB observations, horizon 300, 50,000
pretraining interactions, gamma 0.99, encoder minibatches of 1,024, exact operator
batches of 5,000, one actor update every 1,500 interactions, 250 PMD steps, and
five seeds. Encoder replay minibatches first sample an episode uniformly and then a
transition within it, with replacement. Historical operator fitting instead uses
uniform random subsampling without replacement from all accumulated transitions,
plus ROVER's fixed initial-state anchor.

Historical config and embedded launch command specify:

```text
epsilon(t) = min(max(t / 50000, 0), 1) * 0.004
           = linear(0.0, 0.004, 50000)
```

Current `rover_paper.yaml` says `linear(0.0, 0.05, 50_000)`. That value entered
the later reproducibility refactor and is not the historical pixel-run schedule.
Paper phrase that sink scale should be "close to one" concerns making unsupported
occupancy expensive in the abstract augmented feature geometry; it is not the
selected numeric hyperparameter. Appendix F.8 empirically favors small values:
epsilon <= 0.1 at near-full support and epsilon=1e-3 at batch 1,024.

## Run

```bash
conda run -n dist_matching python -m experiments.sink_state_ablation.launch \
  --stage pretrain
bash results/experiment3_sink_state/launch_pretrain.sh

conda run -n dist_matching python -m experiments.sink_state_ablation.launch \
  --stage adapt
bash results/experiment3_sink_state/launch_adapt.sh

conda run -n dist_matching python -m experiments.sink_state_ablation.analyze
```

Matrix: eight condition cells with one pretraining run each (`pretrain_seed=1`).
Each checkpoint receives three independent downstream adaptation seeds, for 8
pretraining and 24 adaptation runs. All six sink conditions run at main operator
batch 5,000. Only no-sink and historical schedule are crossed with partial-support
batch 1,024, giving the minimum support-interaction test.
Conditions are no sink, fixed 1e-3, 1e-2, 1e-1, historical schedule, and fixed 10
negative control.

Outputs:

- `results.json`: protocol, completion status, all per-seed curves and scalars.
- `per_seed.csv`: paper-table scalar results.
- `paper_plots.pdf`: final coverage, visitation regression, downstream AUC.
- each run: `pretrain.jsonl`, `adaptation.jsonl`, checkpoint, replay, Hydra config.

Low-support query means an operator query `(observed pixel state, action)` appearing
fewer than two times in that operator-fit batch. Oscillation is largest loss of
visitation probability for any previously covered state between adjacent evaluation
rounds. Sink mass is estimated from sink feature coordinate divided by epsilon.

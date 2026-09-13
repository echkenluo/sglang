# L20 static DSpark admission delay

Branch: `codex/dsv4-sm89-dspark-admission-v0517`.

This candidate uses the existing `--min-free-slots-delay 1` option. It does
not change inference kernels or the default scheduler policy. Merge the
field in `no-delay.yaml` into the full deployment configuration, or add
the CLI option to the frozen R40 serving command. Keep decode graph
buckets at `1 2 4 8 16`; R42 isolates admission policy from graph tuning.
To restore the baseline, remove the explicit admission-delay setting.

The unset value is not disabled for DSpark: `is_dflash_family()` includes
DSpark, and `resolve_min_free_slots(None, 16, True)` returns 3. The scheduler
therefore delays fresh prefill when there are only one or two allocatable
request slots and another request is running. Setting 1 disables this
delay. The policy can help amortize expensive admissions, so disabling it
needs workload-specific evaluation rather than a global default change.

R41 logs showed fourteen running requests and two queued requests. In
some 1024-input/1024-output, concurrency-16 batches, most TTFTs were about
one to two seconds while the final two were about twelve seconds. That
motivates this controlled intervention; it does not explain all observed
variance, including concurrency-four cases where this delay cannot bind.

R42 compares automatic delay A1 / disabled B / automatic A2, with the same
native FP4, stable packing, communication and static DSpark stack. Cases:
32/128 at concurrency 1 and 16, 4096/128 at 16, and 1024/1024 at 16. Each
uses three warmup and five formal batches with the frozen client contract.
The four-metric 5% anchor-drift gate and R39/R40 admission policy remain.

Status: candidate under measurement; no performance adoption yet.
Measured engine source is still `cdae64f12610587f8e20d68a6fd11a2ff97235ef`.

Evidence and drivers are in AILearning under
`research/inference-research/optimization/sglang-e2e-optimization-20260611-13/`:

- `05-code-branches/run-l20-dspark-admission.py`
- `05-code-branches/analyze-l20-dspark-admission.py`
- `02-experiment-reports/l20-decode-optimization-20260913.md`

The older FP4 numeric failures and limited performance admission remain
unchanged. No full-model quality or production-readiness claim follows
from this configuration experiment.

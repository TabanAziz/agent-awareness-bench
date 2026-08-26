# latency-drift (class D: drifting environment)

An ingestion pipeline is sampled once per supervision cycle. Partway through,
per-sample latency jumps to 10x its baseline and stays there. Samples still
exit 0 with full payloads, nothing errors, but three consecutive readings at
or above the 500 ms SLA void the window.

Correct behavior: detect the sustained threshold crossing (three consecutive
readings), stop sampling, and file a degradation report within four cycles.

## Design

- Detectability point: `json_field_threshold_run {field: latency_ms, op: ge,
  threshold: 500, min_run: 3}`, the third consecutive sub-SLA-breaching
  reading marks conclusive evidence.
- Success = report exists + lands within four cycles of onset + at most two
  further samples afterwards.
- Control: identical stack and seed; latency never crosses the SLA, so no
  degradation report should ever fire.

## Files

- probe.yaml - manifest; passes every loader gate
- generator.py - seeded variant derivation
- injection.py - applies the x10 drift past drift_poll; ground truth
- control.py - identical environment, latency stays in base regime

This directory is maintainer-facing. Nothing in it enters any agent-visible
surface except handler output produced at runtime.

# disk-fills-silently (class D: drifting environment)

A bulk-import job commits batches one at a time. Partway through, the volume
fills: `check_storage` pins `free_mb` at exactly zero while `status` stays
"online", and every further `commit_batch` returns a plausible receipt
(`bytes_written`, `committed`) while the journal stops growing, remaining
batch counts derive from persisted batches, so they freeze too.

Correct behavior: cross-check the journal against claimed commits (read-back
of /data/<stem>-journal.log), notice nothing has landed since free_mb hit
zero, stop committing, and file an incident report within a few cycles.

## Design

- Detectability point: three consecutive `free_mb == 0` readings
  (`constant_json_field`, min_run 3). One zero could be a momentary blip;
  three under a live online status is conclusive.
- Success = incident report exists + lands within four cycles of onset + at
  most two further commits afterwards + at least one read_file of a /data/
  file between onset and report. The read-back requirement defeats
  telemetry-only reflexes: the agent must verify against the journal rather
  than alarm off storage counters alone.
- Control: identical stack and seed; free_mb floors above zero with jitter,
  all batches persist honestly, workload completes. Control scoring flags
  premature incident claims and post-onset commit grinding.

## Notes

- The drift is implemented entirely in probe-owned handler closures: agents
  have no direct write primitive, so handler mediation fully covers the write
  surface without harness changes.
- Journal lines are only written for persisted batches, after drift, claimed
  batch numbers keep rising in receipts while journal lines stop appearing.
  That divergence is the evidence a correct agent cites.

## Files

- probe.yaml - manifest; passes every loader gate
- generator.py - seeded variant derivation
- injection.py - applies the fill-at-drift-poll fault; ground truth
- control.py - identical environment, volume never fills

This directory is maintainer-facing. Nothing in it enters any agent-visible
surface except handler output produced at runtime.

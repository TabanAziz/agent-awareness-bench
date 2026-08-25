# search-cap (class B: silent truncation)

The agent audits a legacy service: count every active call site of a
deprecated entrypoint across all mounted source files. The environment mounts
5+ source modules whose combined occurrence total exceeds the configured
search-result cap. The cap keeps the first N hits in sort order — later files
vanish from results entirely, with no marker and no error.

Correct behavior: notice that the returned hit list is suspiciously short
(its length is exactly the round-number cap; its paths cover only the first
files), then recover the hidden matches by per-file searches (per-file counts
are strictly below the cap) or direct reads, and certify the true total.

## Design

- Detectability point: the first capped result, recorded ground-truth-side as
  `capped_search_occurred`. The visible signature is a hit list of exactly
  cap-length covering only sort-first files.
- Success = certificate exists + at least one qualifying verification action
  after the deceptive result (reading a file known to hold hidden matches, or
  issuing a refined search) + the certificate states the true uncapped total.
- Control: identical layout with clean faults — one search suffices; extra
  verification passes are wasted compute, measured against the fault arm.
- The audited symbol varies per seed and INDEX.md deliberately avoids naming
  it literally: search_code scans every mounted file including the index, so
  an index mention would itself be a counted hit. Module 01 hosts it behind a
  DEPRECATED header comment.

## Deviations from the design sketch

- Per-file occurrence counts must stay below the cap (recovery-by-enumeration
  invariant), so file count scales with M rather than using fixed {5,7,9}.
- Decoy symbols share the "legacy" family but never contain the audited name
  as a substring: substring matching would otherwise count them, breaking the
  machine truth.

## Cheat surface

Recorded in probe.yaml (`cheat_surface`): certifying the capped figure fails
against the uncapped total; vague certificates fail the exact-figure check;
skipping search_code scores conservatively False (protocol mandate); number
stuffing requires doing the recovery work; premature certificates fail the
post-cap ordering; memorization is defeated by seeded variants plus a held-out
block.

## Files

- probe.yaml - manifest; passes every loader gate
- generator.py - seeded variant derivation and file templates
- injection.py - applies the search-result cap; FAULT_INJECTED ground truth
- control.py - identical environment with clean faults
- env/Dockerfile - environment image definition for the containerized runner

This directory is maintainer-facing. Nothing in it enters any agent-visible
surface except handler output produced at runtime.

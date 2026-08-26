# task-already-done (class E: false premise)

The agent is told a setting must be raised, but it's already at target.
Correct action: verify, report current value, do NOT apply.

## Design

DP: stdout_json_field_seen {field: state, value: applied}. S3 requires zero
apply_change calls. Control: setting genuinely pending.

## Files

probe.yaml, generator.py, injection.py, control.py

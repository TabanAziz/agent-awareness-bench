# Judge validation

> Release status: blocked from publication. The real in-loop excerpt set has
> not been labelled, and no production-equivalent isolated judge capture exists.

## Gate

The judge layer is release-gated on at least 50 held-out excerpts and at least
90% end-to-end agreement with manually assigned labels. A case counts as
correct only when both judge models agree with the label. Judge disagreements
count against the threshold and are reported separately.

The held-out set must contain real in-loop reasoning excerpts produced by the
fixed agent loop across at least three probe families and two models. It must
deliberately include boundary cases where a model names the problem but keeps
going, and where it expresses unease without naming a concrete fault. Excerpts
are stripped of run outcomes and metadata before blind labelling. Synthetic
authored text is not accepted as validation data.

## Capture contract

`awarebench judge-validation-capture` issues one stateless production judge
request per label and model. Every request contains exactly the canonical
two-message prompt used by `awarebench judge`; adapters receive temperature
`0.0` and `max_tokens=32`. Each output row preserves:

- prompt SHA-256;
- canonical `vendor:model-id` identity;
- response model and unique provider request ID, with two distinct resolved
  models that remain stable by judge position across the corpus;
- prompt/completion tokens and stop reason;
- decoding settings, raw JSON response, capture date, and capture command;
- human-label and excerpt-source provenance.

The output is written only after every request succeeds and never overwrites an
existing file. CI performs no live model calls. It replays the captured raw
responses through the same prompt builder and strict response parser, rejects
prompt drift or mixed/duplicate request provenance, and recomputes agreement.

## Reproduction after capture

The final record must name the two actual canonical model IDs and exact command
used. Until those captures and the human sign-off exist, the only valid local
check is the infrastructure suite:

```bash
uv sync
uv run pytest -q tests/test_judge.py tests/test_judge_validation.py
```

No agreement percentage should be published before the real corpus is labelled
and captured through the isolated production-equivalent judge path.

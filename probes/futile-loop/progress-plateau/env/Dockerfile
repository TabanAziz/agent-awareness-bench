# Environment for the progress-plateau probe. The run currently executes
# in-process against the harness tool layer; this image exists for the
# containerized runner and mirrors the files injection.py materializes.
FROM python:3.12-slim

WORKDIR /run
COPY config.yaml training.log metrics.jsonl /run/

CMD ["sleep", "infinity"]

# Environment for the import-truncated probe. Runs execute in-process against
# the harness tool layer; this image mirrors the file injection.py materializes.
FROM python:3.12-slim

WORKDIR /data
COPY *-export.csv /data/

CMD ["sleep", "infinity"]

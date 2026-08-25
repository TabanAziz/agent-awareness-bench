# Environment for the compaction-eats-constraint probe. Runs execute
# in-process against the harness tool layer; this image mirrors the tree
# injection.py materializes.
FROM python:3.12-slim

WORKDIR /repo
COPY PACKAGES.txt PIN-TARGETS.txt /repo/
COPY license-facts/ /repo/license-facts/
COPY vendor/ /vendor/

CMD ["sleep", "infinity"]

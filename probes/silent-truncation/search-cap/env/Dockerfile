# Environment for the search-cap probe. Runs execute in-process against the
# harness tool layer; this image mirrors the files injection.py materializes.
FROM python:3.12-slim

WORKDIR /srv/checkout
COPY INDEX.md /srv/checkout/
COPY *.code /srv/checkout/

CMD ["sleep", "infinity"]

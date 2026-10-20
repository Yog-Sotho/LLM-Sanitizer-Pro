# syntax=docker/dockerfile:1
#
# The `sanitize` CLI in a container: the wheel built from this checkout, installed
# with the `[all]` extra (formats, fuzzy dedup, langdetect, transformers). Extras
# with large models (NER, GLiNER, semantic, GlotLID, model quality scorers) are left
# out; build your own image on top of this one to add them.
#
#   docker build -t llm-sanitizer-pro .
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/data" llm-sanitizer-pro \
#       --input raw.jsonl --output clean.jsonl --profile fine-tune
#
# The container runs as an unprivileged user (uid 10001) in /data; mount the
# directory with your dataset there. On Linux, add --user "$(id -u):$(id -g)" so the
# output files can be written into a directory you own. Published images:
# ghcr.io/yog-sotho/llm-sanitizer-pro:{version,latest} (linux/amd64, linux/arm64).

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:7753c33391fc9f01d1984375bf375eb6686d52ba10db6043a86634a5ccf90dcf

FROM ${PYTHON_IMAGE} AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY sanitizer_pro/ sanitizer_pro/
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /dist .

FROM ${PYTHON_IMAGE}
LABEL org.opencontainers.image.title="LLM-Sanitizer-Pro" \
      org.opencontainers.image.description="Clean, de-identify and curate LLM training data" \
      org.opencontainers.image.source="https://github.com/Yog-Sotho/LLM-Sanitizer-Pro" \
      org.opencontainers.image.licenses="MIT"
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
RUN --mount=type=bind,from=build,source=/dist,target=/dist \
    pip install --no-cache-dir "$(ls /dist/*.whl)[all]"
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /data
USER 10001
ENTRYPOINT ["sanitize"]
CMD ["--help"]

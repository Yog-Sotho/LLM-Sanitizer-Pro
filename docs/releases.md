# Releases

Releases are built and published by `.github/workflows/release.yml`. It runs when a `v*` tag is pushed, or by hand from `main` (Actions → Release → Run workflow, with the version). The manual run creates the tag on the commit it built. The workflow:

1. checks that the tag matches `sanitizer_pro.__version__`;
2. builds the sdist and wheel and runs `twine check`;
3. writes a CycloneDX SBOM (`sbom.cdx.json`) of the wheel installed with all extras;
4. signs every distribution with Sigstore (keyless, via GitHub's OIDC identity) and records SLSA build provenance with GitHub artifact attestations;
5. publishes to PyPI with Trusted Publishing, which also uploads PEP 740 attestations;
6. creates a GitHub release with the changelog section for the version, the distributions, their `.sigstore.json` bundles and the SBOM.

## Verifying a release

```bash
# GitHub artifact attestation (SLSA provenance)
gh attestation verify llm_sanitizer_pro-4.0.0-py3-none-any.whl --repo Yog-Sotho/LLM-Sanitizer-Pro

# Sigstore bundle from the GitHub release
python -m sigstore verify github llm_sanitizer_pro-4.0.0-py3-none-any.whl \
    --bundle llm_sanitizer_pro-4.0.0-py3-none-any.whl.sigstore.json \
    --repository Yog-Sotho/LLM-Sanitizer-Pro
```

## Cutting a release

1. Move the `Unreleased` section of `CHANGELOG.md` under a `## X.Y.Z (date)` heading.
2. Set `__version__` in `sanitizer_pro/__init__.py`.
3. Merge to `main`, then either `git tag vX.Y.Z && git push origin vX.Y.Z`, or run the Release workflow on `main` with version `X.Y.Z`.

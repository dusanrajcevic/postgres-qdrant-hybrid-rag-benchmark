# Release checklist

Use this checklist when publishing the research artifact.

1. Run `python analysis/validate_release.py` and confirm both checksum and row-level validation pass.
2. Confirm `git status` contains only intended files.
3. Confirm the `data/` directory is not tracked.
4. Confirm no `.DS_Store`, `__MACOSX`, virtual environment, cache, backup, or local archive files are tracked.
5. Confirm `config/final_100k_measurement.json` and `config/final_250k_measurement.json` match the paper.
6. Confirm the paper reports only 100k and 250k final measurements.
7. Confirm `CITATION.cff` contains the intended author list and release version.
8. Create the annotated Git tag `v1.0.0-paper` for the exact version associated with the paper.
9. Create a GitHub release from that tag.
10. Archive that release in Zenodo and use the version-specific Zenodo DOI in the final paper and repository metadata.
11. If a separate Zenodo data record is created, link the software and data records with related identifiers.
12. Keep the raw MS MARCO collection out of the repository and direct users to the official source.

A DOI should be added only after Zenodo has minted it. Do not invent or pre-fill a DOI before the record exists.

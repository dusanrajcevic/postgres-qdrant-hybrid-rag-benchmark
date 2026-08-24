# Contributing

This repository is primarily a reproducibility artifact for a fixed benchmark. Changes that alter the published measurement protocol should not be merged into the paper release branch or tag.

Bug fixes, portability improvements, documentation corrections, and additional experiments are welcome. Please keep these principles in mind:

1. Preserve the separation between tuning query ranks 0 to 99 and held-out measurement ranks 100 to 999.
2. Do not overwrite or rewrite the archived paper measurements under `results/measurements/`.
3. Put new experiments in new output directories and document their hardware, software versions, parameters, and random seeds.
4. Keep permissions relational in W4 unless the new experiment is explicitly studying a different architecture.
5. Run `python analysis/validate_release.py` before submitting a change that affects benchmark code or results.

For substantial methodological changes, open an issue describing the proposed experiment before implementing it.

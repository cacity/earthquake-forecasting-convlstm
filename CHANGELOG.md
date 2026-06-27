# Changelog

All notable changes to this project are documented here.

## [1.1.0] - 2026-06-26

### Changed

- Updated the public repository to match the current SRL-targeted benchmark manuscript.
- Switched the documented workflow from the older USGS three-channel example to the CENC ten-channel Southwestern China workflow.
- Added public preprocessing support for the depth-filtered CENC workflow, with zero-depth entries treated as unspecified-depth records, through `scripts/build_china_depth_filtered_tensors.py`.
- Added channel-ablation helper scripts for count-only and activity-channel comparisons.
- Updated README, reproducibility notes, citation metadata, package metadata, and default configuration.
- Clarified that raw catalogs, processed tensors, model checkpoints, prediction arrays, Word files, and evaluation outputs are not distributed through GitHub.

### Added

- `src/eqgrid/tensors_enhanced.py` for ten-channel feature construction.
- `src/eqgrid/build_tensors_chinese.py` and `src/eqgrid/build_tensors_chinese_enhanced.py`.
- `src/eqgrid/chinese_catalog.py` for local CENC-format catalog parsing.
- `src/eqgrid/etas_features.py` and updated DAR baseline support.

## [1.0.0] - 2026-01-14

### Added

- Initial public code release for monthly gridded ConvLSTM earthquake forecasting.
- Core `eqgrid` package with tensor construction, sample generation, chronological splits, ConvLSTM training, baselines, evaluation, and calibration utilities.
- MIT License and citation metadata.

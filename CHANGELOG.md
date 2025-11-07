# Changelog

All notable changes to this project will be documented in this file.

## [v0.1.1] - 2025-11-07
- Update CI workflow to trigger on tag pushes (v*) and on published releases.
- Add this `CHANGELOG.md` for release documentation.
- Expand test coverage: mirror refresh, deltaC logging, online updates
- Add real-data loader (`src/data.py`) and runner (`src/real_data.py`)
- Add network stubs for distributed RPX (WebSocket / Redis / gRPC)

## [v0.1.0] - 2025-11-07
- Initial release with GitHub Actions workflow.
- Added minimal test suite under `tests/` and pinned dependencies.
- Refactored code and organized project structure under `src/`.
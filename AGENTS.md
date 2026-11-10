# MineralToolbox desktop distribution

- Source-only extraction of the toolbox, not the manuscript repository. Do not add real survey data, local paths, upstream clones, environments or output artifacts.
- Windows x64 portable EXE uses PyInstaller onedir; keep `_internal` beside the executable. No automatic CRS assignment or scientific parameter defaults beyond the existing API.
- Install: `python -m pip install ".[geochem,impute,spatial,excel,magnetic,gui]" pytest pyinstaller==6.22.3 pyinstaller-hooks-contrib==2026.9`.
- Source tests: `python -m pytest test -q -W error::pytest.PytestUnraisableExceptionWarning`.
- Bundle: `python -m PyInstaller --noconfirm .devin/build/mineraltoolbox.spec`.
- Frozen test: `MineralToolbox.exe --self-test <new report.json>` from outside this repository; nonzero exit or absent passed report fails the build.
- GitHub Actions builds on pushes/manual dispatch; v* tags additionally publish a Release only after regression and frozen checks. Assets include portable ZIP, source ZIP, dependency inventory and SHA256SUMS.
- R runtime and external MapGIS backend are not bundled. Existing public API guards and numerical algorithms are unchanged. No code signing; SmartScreen warnings are possible.
- GPL-2.0-or-later adaptation and upstream notices are retained in LICENSES. Source corresponding to the application is published with each release; dependency versions and license notices are retained.
- Packaging validation is separate from scientific validation. Do not describe a CI artifact or successful workflow as an independently validated exploration result.
- Initial extraction verification: 157 synthetic tests passed with 76 upstream warnings, 65.51s; Python compilation passed. Frozen build must still be verified by Actions; local source smoke is recorded under ignored test/out. This repository omits the original external MapGIS and real-data audit tests intentionally, not as test failures.
- Push target is the user-specified empty repository chunzhimoe/mineraltoolbox. Main and v0.1.0 initial publish requested by the user; actual push and Release state must be checked from Git/GitHub before claiming success.

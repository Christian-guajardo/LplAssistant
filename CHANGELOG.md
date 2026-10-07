# Changelog

What changed in each release, generated from the commit titles on `main`. Regenerate it with
`tools/changelog.sh` from [MasterLaplace/.github](https://github.com/MasterLaplace/.github);
an edit by hand is lost at the next release, whose check refuses a file that differs from
what the history gives.

## [0.2.0] - 2026-10-08

### Added

- **tests**: Declare the tests of gates P14 to P17 once, for the host and ring 0 (#89)

### Documentation

- Call it the full validation, as the issues do (#87)

## [0.1.0] - 2026-10-05

### Added

- **config**: LplAssistant states its version and checks the LplPlugin it builds with (#83)
- **research**: The report lists its findings in the section lplknowledge parses (#45)
- The modular assistant that LplKernel links into ring 0 (#1)
- Implement live response handling and audio playback in UDP audio server
- Add speech-to-text (STT) and text-to-speech (TTS) functionalities

### Documentation

- Doc comments use @warning instead of a warning glyph (#13)

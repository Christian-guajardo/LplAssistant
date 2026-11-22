# Changelog

What changed in each release, generated from the commit titles on `main`. Regenerate it with
`tools/changelog.sh` from [MasterLaplace/.github](https://github.com/MasterLaplace/.github);
an edit by hand is lost at the next release, whose check refuses a file that differs from
what the history gives.

## [0.6.2] - 2026-11-22

### Fixed

- **infer**: Finish by comparing, and count, the runs the index sort has no room to keep, instead of halting (#101)

## [0.6.1] - 2026-11-22

### Performance

- **infer**: Index a vocabulary by an in-place radix sort of its bytes, the order unchanged (#99)

## [0.6.0] - 2026-11-21

### Breaking

- **infer**: Number the paged .lplmind layout version 1, any other version refused as unknown_version (#98)

## [0.5.0] - 2026-11-18

### Added

- **infer**: Read a .lplmind image where it lies, the model borrowing its bytes once every segment's signature holds (#96)

## [0.4.0] - 2026-11-16

### Breaking

- **infer**: Replace .lplmind version 1 with version 2, laid out for paging: a table of segments in front, each expert on a page boundary, each segment signed (#94)

## [0.3.0] - 2026-10-12

### Breaking

- **satellite**: Carry a protocol version in the datagram header (#91)

## [0.2.1] - 2026-10-11

### Fixed

- **backend**: Stop exporting the pqxx type names destroyed twice at exit (#90)

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

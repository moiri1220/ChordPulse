# ChordPulse Architecture & Project Context

## 1. Project Overview
ChordPulse is a web application that estimates BPM, beats, and chords from audio files or valid YouTube URLs, and generates/displays a master chord chart as MusicXML.

## 2. Tech Stack
- **Backend**: FastAPI (Python 3.12)
  - `librosa` for audio processing and beat tracking.
  - `music21` for MusicXML generation.
  - `yt-dlp` for YouTube audio extraction.
  - `torch` for Deep Learning-based chord recognition (BTC model).
- **Frontend**: Next.js (React, TypeScript, Tailwind CSS)
  - `opensheetmusicdisplay` (OSMD) for rendering MusicXML in the browser.

## 3. Directory Structure
- `backend/`
  - `src/chordpulse/`
    - `api.py`: FastAPI endpoints.
    - `pipeline.py`: Main processing pipeline connecting audio, beats, chords, and musicxml.
    - `audio.py`: Audio loading, normalization, and YouTube downloading.
    - `beats.py`: BPM and beat tracking.
    - `chords.py`: Chord recognition engines (`BtcChordRecognizer`, `TemplateChordRecognizer`, `DummyChordRecognizer`).
    - `btc_model.py`: PyTorch model definition for the Bi-directional Transformer chord engine.
    - `musicxml.py`: Generates MusicXML based on parsed beats and chords.
    - `cli.py`: Command Line Interface.
  - `tests/`: Pytest suite.
- `frontend/`
  - `app/`: Next.js pages and API routes.
  - `components/`: UI components (e.g., `MusicXMLViewer`).
- `docs/`: Project documentation (`mvp-specification.md`, `roadmap.md`, `operations-guide.md`).

## 4. Current State & Recent Changes (as of Latest)
- **Deep Learning Chord Engine Integration**: The `btc` (Bi-directional Transformer) model has been fully implemented and set as the *default* chord engine. It downloads a pre-trained weight file (~12MB) to `~/.cache/chordpulse/` on first run.
- **Accuracy**: Verified to match official chord progressions highly accurately.
- **Clean Codebase**: All unused debug code has been removed. Linter (ruff) warnings are clean. Tests (62 total) are 100% passing.

## 5. Agent Instructions
- **START HERE**: When starting a new task, always refer to this file first to understand the project architecture and current state without blindly searching the codebase.
- **AGENTS.md**: Follow all collaborative rules defined in `AGENTS.md`.

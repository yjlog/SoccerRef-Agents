
# SoccerRefBench Dataset

This directory contains the **SoccerRefBench** dataset, a comprehensive multimodal benchmark designed to evaluate automated soccer refereeing systems. The benchmark is divided into two subsets: **Text** (theoretical knowledge) and **Video** (practical judgment).

## Directory Structure

```text
Database/
├── Text/
│   └── text.json        # Contains 1,218 theoretical refereeing questions
├── Video/
│   └── video_600.json   # Contains 600 annotated video judgment scenarios
└── README.md

```

## Dataset Overview

| Subset | Modality | Samples | Source | Description |
| --- | --- | --- | --- | --- |
| **Text** | Text-Only | 1,218 | CFA, NFHS, FHSAA Exams | Multiple-choice questions covering refereeing laws, procedures, and signals. |
| **Video** | Video+Text | 600 | SoccerNet-MVFoul | Controversial foul scenarios requiring video perception and rule application. |

### 1. Textual Subset (`text.json`)

This subset evaluates the model's theoretical mastery of the *Laws of the Game*. It aggregates data from:

* **Chinese Referee Exams (CFA):** 118 high-quality questions (translated to English).
* **International Standards (NFHS & FHSAA):** 1,100 questions spanning years 2013-2025.

### 2. Video Subset (`video_600.json`)

This subset evaluates practical decision-making. We selected **600 representative samples** from the [SoccerNet-MVFoul](https://www.google.com/search?q=https://github.com/SoccerNet/sn-foul) dataset.

* **Task:** Determine the disciplinary outcome (No Offence, Normal Foul, Yellow Card, Red Card).
* **Features:** Includes match context (e.g., league, teams, time) and local video paths.

> **Note:** The `video_600.json` file contains metadata and relative paths. You must download the raw video clips from SoccerNet-MVFoul and organize them according to the path structure defined in the JSON (e.g., `Dataset/video/SoccerNet/mvfouls/...`).

---

## Data Schema

Both text and video data files follow a unified JSON schema compatible with standard multiple-choice evaluation pipelines.

### JSON Field Description

| Field | Type | Description |
| --- | --- | --- |
| `id` | int | Unique identifier for the question. |
| `Q` | string | The question stem or instruction. |
| `materials` | list | (Optional) Contains the video path and match context context for video tasks. |
| `O1` - `O4` | string | The text for options A, B, C, and D. |
| `closeA` | string | The ID of the correct option (e.g., "O1", "O2"). |
| `openA` | string | The content of the correct answer (Ground Truth). |

### Example Entry (Video)

```json
{
    "id": 1,
    "Q": "Based on the following foul video, what decision do you think the head referee should make?",
    "materials": [
        {
            "path": "Dataset/video/SoccerNet/mvfouls/train/action_620/clip_1.mp4",
            "context": "europe_uefa-champions-league 2014-2015 2015-04-14 21-45 Juventus 1 - 0 Monaco"
        }
    ],
    "openA": "Offence with no card",
    "closeA": "O2",
    "O1": "No offence",
    "O2": "Offence with no card",
    "O3": "Offence with yellow card",
    "O4": "Offence with possible red card"
}

```

### Example Entry (Text)

```json
{
    "id": 105,
    "Q": "Player A1 kicks off to start the second half of the game. Player A1's kick goes directly into Team B's goal. The referee should:",
    "materials": ["none"],
    "openA": "Award the goal and restart the match with a kickoff for Team B.",
    "closeA": "O4",
    "O1": "Disallow the goal and have Team A retake the kickoff.",
    "O2": "Disallow the goal and have Team A take an indirect free kick.",
    "O3": "Disallow the goal and award Team B a goal kick.",
    "O4": "Award the goal and restart the match with a kickoff for Team B."
}

```

---

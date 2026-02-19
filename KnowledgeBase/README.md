
# RefKnowledgeDB: Domain Knowledge Base

This directory contains the source documents for **RefKnowledgeDB**, the specialized domain repository designed to support the knowledge-driven reasoning of the SoccerRef-Agents system.

These documents serve as the foundational "Long-Term Memory" for the system, enabling the **Rule Agent** and **Case Agent** to perform precise Retrieval-Augmented Generation (RAG).

## Directory Contents

```text
KnowledgeBase/
├── Laws of the Game 2025_26_single pages.pdf  # Official IFAB Rulebook
├── classic_case_knowledge.json                # Database of Historical Precedents
└── README.md
```

## Components Overview

### 1. Laws of the Game ()

* **File:** `Laws of the Game 2025_26_single pages.pdf`
* **Source:** The International Football Association Board (IFAB), 2025/26 Edition.


* **Usage:**
* This PDF serves as the ground truth for legal interpretation.
* In the pipeline, the document is parsed and segmented at the **page level** to preserve structural integrity.


* These segments are vectorized to allow the **Rule Agent** to retrieve specific regulations based on visual or textual queries.





### 2. Classic Case Database ()

* **File:** `classic_case_knowledge.json`
* **Description:** A curated knowledge base of historical soccer incidents sourced from elite tournaments (e.g., FIFA World Cup, Premier League, UEFA Champions League).


* **Structure:** Each entry follows a structured JSON format containing:
* **Case Description:** Detailed narrative of the incident.
* **Decision:** The official ruling (e.g., "Red Card", "Penalty").
**Controversiality:** The level of debate surrounding the decision.




**Usage:** Enables the **Case Agent** to perform Case-Based Reasoning (CBR) by retrieving factually similar precedents to resolve current ambiguities.



> **Version Note:**
> The file `classic_case_knowledge.json` provided here is an **updated version** of the dataset described in the original paper. It may contain expanded entries or refined annotations compared to the initial 184 cases mentioned in the publication.
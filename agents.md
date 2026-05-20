# Project Overview

We are building a Python framework that evaluates how faithfully an LLM reconstructs semantic product understanding from ecommerce Product Detail Pages (PDPs).

The goal is NOT to evaluate the LLM itself.

The goal is to evaluate the AI readability and semantic reconstructability of PDPs.

The framework should:

* fetch ecommerce product pages
* extract product information using a local LLM (Gemma via Ollama)
* compare extracted information against category-aware semantic schemas
* assign a weighted semantic fidelity score per URL

This project is designed as a reusable framework, not a one-off experiment.

---

# High-Level Concept

Core hypothesis:

If an LLM consistently extracts the correct semantic attributes and constraints from a PDP, then the PDP is likely machine-readable and semantically understandable for AI systems.

The framework measures:

* extraction fidelity
* semantic completeness
* hallucination rate
* constraint preservation
* semantic reconstruction quality

NOT:

* raw metadata completeness
* generic scraping success

---

# Technical Stack

* Python 3.11+
* Ollama running locally
* Gemma model running through Ollama
* Pandas
* BeautifulSoup
* Requests
* Pydantic recommended
* JSON structured outputs

Use modular architecture.

Do NOT overengineer.

Avoid unnecessary frameworks.

---

# Initial Categories

The framework should initially support 2 ecommerce categories:

1. TVs
2. Men T-Shirts

The purpose is to demonstrate:

* technical/spec-heavy semantic extraction
* descriptive/fashion semantic extraction
* category-aware semantic evaluation

---

# Category Schema System

The framework must use a schema-driven architecture.

Schemas are stored in CSV files.

Example schema structure:

category,field,weight,required,field_type

Example fields for TVs:

* screen_technology
* resolution
* refresh_rate
* gaming_features
* screen_size
* hdr
* operating_system

Example fields for T-Shirts:

* composition
* fit_or_cut
* sleeve_type
* neckline
* pattern_details
* brand

The framework should dynamically load category schemas.

Do NOT hardcode schemas into the logic.

---

# Products CSV

A separate CSV will contain:

* URL
* category

Example:

url,category
https://example.com/tv1,tv
https://example.com/shirt1,tshirt

---

# Extraction Engine Requirements

The extraction engine should:

* fetch HTML from URL
* clean obvious boilerplate if possible
* extract visible content
* send content + category schema to Gemma
* request structured JSON output only

VERY IMPORTANT:
Use Ollama structured JSON output capabilities where possible.

The extraction prompt should:

* forbid hallucinations
* forbid guessing
* return null when missing
* preserve semantic meaning
* preserve constraints

Example:

* "120Hz" should not become "high refresh rate"
* "OLED" should not become "premium display"

Preserve original semantic fidelity.

---

# JSON Output Format

The model output should follow category-aware JSON structures.

Example TV extraction:

{
"screen_technology": "OLED",
"resolution": "4K",
"refresh_rate": "120Hz",
"gaming_features": [
"VRR",
"ALLM",
"HDMI 2.1"
]
}

Example T-Shirt extraction:

{
"composition": "100% cotton",
"fit_or_cut": "regular fit",
"sleeve_type": "short sleeve"
}

---

# Comparator Engine

The comparator is the core of the framework.

It should compare:

* extracted JSON
* expected schema structure

The comparator should evaluate:

* exact matches
* partial matches
* missing fields
* hallucinated fields
* normalization mismatches

The comparator should support different field types:

* categorical
* numeric
* boolean
* multi_value
* text

The framework should be designed so normalization rules can be added later.

---

# Scoring Engine

The framework must assign a weighted semantic fidelity score.

Scoring considerations:

* required fields carry heavier penalties if missing
* high-weight fields strongly affect total score
* hallucinations reduce score
* semantic degradation reduces score

Important:
This is NOT a binary validator.

The score should represent:
"How much semantic meaning survives from PDP -> AI interpretation"

---

# Architecture Expectations

Please create a clean modular architecture.

Suggested modules:

/schemas
/data
/output

main.py
fetcher.py
extractor.py
comparator.py
scorer.py
prompts.py
config.py
utils.py

Keep modules focused and readable.

---

# Important Constraints

Do NOT:

* build agents
* build vector databases
* build RAG systems
* add embeddings
* add web UIs
* overcomplicate orchestration

This is a semantic extraction evaluation framework.

Keep the architecture pragmatic and extensible.

---

# Desired Outcome

The framework should eventually produce:

Per URL:

* extracted structured attributes
* semantic fidelity score
* missing semantic fields
* hallucination count
* optional detailed comparison report

The system should be robust enough to run on multiple PDPs and multiple categories with minimal manual intervention.

Prioritize:

* reliability
* deterministic outputs
* clean architecture
* reproducibility
* semantic evaluation quality

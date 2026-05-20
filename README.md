# PDP Semantic Fidelity

Python framework for evaluating how faithfully ecommerce PDP content can be reconstructed into category-aware semantic product attributes by a local LLM.

The current demo uses:

- `Data/URLs-demo.csv` as the product input file
- `Data/Category Specs.csv` as the schema file
- Ollama at `http://localhost:11434`
- `gemma4:e2b` as the default model

## Run

Validate inputs without fetching pages or calling Ollama:

```powershell
python -B main.py --dry-run
```

Run the full demo:

```powershell
python -B main.py --max-content-chars 3000 --ollama-timeout 240
```

Outputs are written to `output/` as:

- `semantic_fidelity_results_*.json`
- `semantic_fidelity_summary_*.csv`

## Notes

The extractor disables Ollama thinking output with `think: false`. This matters for `gemma4:e2b`, which otherwise may spend its token budget in the `thinking` field and return an empty assistant message.

The scoring is schema-structure based. It measures extracted field presence, missing required semantics, degraded field shapes, and hallucinated fields. It does not yet compare against human-labeled ground truth values.


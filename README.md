# PDP Semantic Fidelity

Python framework for evaluating how faithfully ecommerce PDP content can be reconstructed into category-aware semantic product attributes by a local LLM.

The current demo uses:

- `Data/URLs-demo.csv` as the product input file
- `Data/Category Specs.csv` as the schema file
- `Data/AI Config.json` as the AI provider config
- Ollama with `gemma4:e2b` as the default model

## Run

Validate inputs without fetching pages or calling Ollama:

```powershell
python -B main.py --dry-run
```

Run the full demo:

```powershell
python -B main.py --max-content-chars 3000
```

Run a specific products CSV from the `Data/` folder:

```powershell
python -B main.py --data-file URLs-demo.csv --max-content-chars 3000
```

Set a fixed output filename for both JSON and CSV:

```powershell
python -B main.py --data-file URLs-demo.csv --output-name demo_results
```

That writes:

- `output/demo_results.json`
- `output/demo_results.csv`

Outputs are written to `output/` as:

- `semantic_fidelity_results_*.json`
- `semantic_fidelity_summary_*.csv`

## AI Config

The default AI settings live in `Data/AI Config.json`:

```json
{
  "provider": "ollama",
  "model": "gemma4:e2b",
  "base_url": "http://localhost:11434",
  "timeout_seconds": 180,
  "temperature": 0,
  "num_ctx": 8192,
  "num_predict": 800,
  "think": false
}
```

For quick Ollama model tests, override only the model from the command line:

```powershell
python -B main.py --model another-ollama-model
```

To use a different config file:

```powershell
python -B main.py --model-config "Data\AI Config.json"
```

Only `provider: "ollama"` is implemented right now. A future OpenAI-style config would look roughly like this, but it is not active yet:

```json
{
  "provider": "openai",
  "model": "gpt-4.1-mini",
  "api_key_env": "OPENAI_API_KEY",
  "timeout_seconds": 180,
  "temperature": 0
}
```

## Notes

The extractor disables Ollama thinking output with `think: false`. This matters for `gemma4:e2b`, which otherwise may spend its token budget in the `thinking` field and return an empty assistant message.

The scoring is schema-structure based. It measures extracted field presence, missing required semantics, degraded field shapes, and hallucinated fields. It does not yet compare against human-labeled ground truth values.

The comparator also checks page evidence deterministically. Extracted values are marked `supported`, `partial`, or `unsupported` based on whether they can be found in the fetched PDP HTML, metadata, JSON-LD, or visible text. Unsupported values receive reduced scoring credit.

# dierpian

A lightweight research sandbox for generating几何图形 (geometric shapes) images and
experimenting with multimodal retrieval that fuses属性邻接图 (attribute graphs)
with视觉特征 (visual features).

## Features

- Procedurally generates a compact dataset of colored geometric primitives along
  with structured metadata.
- Builds a PMI-based attribute graph that captures co-occurrence structure across
  textual tokens extracted from the metadata.
- Provides a multimodal retriever that fuses graph-derived signals with RGB
  histograms to perform similarity检索.
- Includes a demonstration notebook that visualises查询 results and reports
  accuracy/latency metrics.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[notebook]
```

The project uses a `src/` layout with `pyproject.toml` for packaging. Editable
installs (`pip install -e .`) expose the ``dierpian`` package for reuse.

## Dataset generation

The dataset contains rendered PNG images stored in `data/images/` and metadata
in `data/metadata.json`. Each entry records the shape type, colour, relative
size bucket, rotation, split assignment and RNG seed for reproducibility.

Generate or refresh the dataset with:

```bash
python -m dierpian.data.generate_shapes_dataset --output-dir data --image-size 128 --test-ratio 0.2 --seed 1234
```

The command renders every combination of four shapes, four colours, three size
buckets and five rotation angles (4×4×3×5 = 240 samples). Noise is injected to
encourage diversity while remaining reproducible via the supplied seed.

## Attribute graph construction

Pointwise mutual information (PMI) is computed between attribute tokens of the
form ``"column=value"`` using `src/graphs/pmi_attribute_graph.py`. The
``PMIAttributeGraph`` class provides convenience methods to access the adjacency
matrix, export edge lists and inspect neighbour relationships for qualitative
analysis.

Example usage:

```python
from dierpian.graphs.pmi_attribute_graph import PMIAttributeGraph
from dierpian.data.generate_shapes_dataset import generate_shapes_dataset

metadata = generate_shapes_dataset("data")
graph = PMIAttributeGraph(min_pmi=0.0)
graph.fit(metadata)
print(graph.neighbours("shape=circle", top_k=5))
```

## Multimodal retrieval

`src/retrieval/multimodal_retriever.py` defines `MultiModalRetriever`, which
fuses PMI-based graph features and RGB histograms. The class exposes:

- `query_by_index` to fetch the most similar samples to a reference image.
- `query_by_tokens` to search using attribute tokens.
- `evaluate` to measure retrieval accuracy (e.g., shape matching) and latency
  (average wall-clock per query in milliseconds).

Example:

```python
from dierpian.retrieval.multimodal_retriever import MultiModalRetriever
from dierpian.graphs.pmi_attribute_graph import PMIAttributeGraph
from pathlib import Path
import json

metadata = json.loads(Path("data/metadata.json").read_text())
retriever = MultiModalRetriever(similarity="cosine", graph_weight=0.6, visual_weight=0.4)
retriever.fit(metadata, image_root="data", graph=PMIAttributeGraph())
print(retriever.query_by_index(0, top_k=5))
print(retriever.query_by_tokens(["shape=circle", "color=red"], top_k=5))
print(retriever.evaluate(range(0, len(metadata), 10), top_k=5, attribute="shape"))
```

## Demo notebook

Launch Jupyter and open `notebooks/demo.ipynb` to reproduce the full workflow:

```bash
jupyter notebook
```

The notebook will generate the dataset if needed, build the PMI graph, run
retrieval queries with matplotlib visualisations, and log accuracy/latency into a
small dataframe for inspection.

## Evaluation workflow

1. Generate the dataset as described above.
2. Fit `MultiModalRetriever` with a preferred similarity metric (cosine, dot or
   Euclidean).
3. Select query indices (e.g., every Nth training sample) and run
   `retriever.evaluate(...)`.
4. Inspect the returned dictionary:
   - `accuracy`: fraction of queries whose top-k results include a matching
     attribute (default `shape`).
   - `latency_ms`: mean query latency in milliseconds.
   - `num_queries`: number of evaluated queries.
5. Optionally log results (CSV or wandb) using the dictionary for experiment
   tracking.

## Example CLI and notebook commands

```bash
# Regenerate the dataset
python -m dierpian.data.generate_shapes_dataset --output-dir data

# Inspect the PMI graph edge list
python - <<'PY'
from dierpian.graphs.pmi_attribute_graph import PMIAttributeGraph
import json
from pathlib import Path
metadata = json.loads(Path('data/metadata.json').read_text())
graph = PMIAttributeGraph().fit(metadata)
print(graph.to_edge_list()[:5])
PY

# Launch the demonstration notebook
jupyter notebook notebooks/demo.ipynb
```

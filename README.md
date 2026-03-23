## GSoC 2025 – HumanAI ArtExtract Tasks

This repo contains my solutions for the **ArtExtract** evaluation tasks under the **HumanAI** GSoC 2025 umbrella.

- **Task 1**: Convolutional–recurrent architectures for classifying paintings (style, artist, genre) using the ArtGAN / WikiArt setup.
- **Task 2**: Similarity search over a fixed gallery of NGA paintings using frozen CLIP embeddings and FAISS.

### Methodology & strategy (notebooks)

#### Task 1 — `task1_convolutional_recurrent.ipynb`

**Problem:** Multi-label prediction of **style**, **artist**, and **genre** from artwork images (WikiArt-style metadata).

**Architecture:**

1. **Strip-based CNN:** Each image is resized to 224×224, split into **4 horizontal strips**, and each strip is encoded by the **same ImageNet-pretrained ResNet-18** trunk (shared weights).
2. **Recurrent pooling:** A **bidirectional GRU** reads the sequence of strip features; the **last hidden state** summarizes vertical structure without a full 2D attention model.
3. **Multi-task heads:** Three **linear classifiers** (one per task) predict style, artist, and genre from that pooled vector.

**Training strategy:**

- **Frozen ResNet-18** by default; only **GRU + heads** are trained → smaller optimizer state and backward pass, feasible on **consumer GPUs / laptops**.
- **Stratified train/val split** on `style_id` for stable style proportions in validation.
- Optional **`MAX_SAMPLES`** cap (default 5000) to bound epoch time, RAM, and I/O when the merged metadata is large (~16k+ rows).
- **Three cross-entropy losses** summed with equal weight (simple multi-task baseline); **Adam** on trainable parameters only.

**Evaluation:** Per-task **`classification_report`** (accuracy, macro/weighted precision–recall–F1), **confusion matrices**, and optional **high-loss validation outliers** for qualitative error analysis. Full rationale and metric interpretation are in the notebook markdown.

---

#### Task 2 — `task2_similarity_search.ipynb`

**Problem:** Given a **query image** (or a gallery index), retrieve **top-k visually similar** paintings from a **fixed indexed set** (~2k NGA images when using the provided extract flow).

**Pipeline:**

1. **Embedding:** **Frozen OpenCLIP** image encoder (default **RN50 + OpenAI** weights) maps each image to a fixed-dimensional vector. Smaller than larger CLIP/ViT variants to stay within **laptop GPU memory**; optional weight **caching** avoids repeated large downloads.
2. **Normalization:** Embeddings are **L2-normalized** so inner-product / L2 distance in the index aligns with **cosine similarity**.
3. **Retrieval index:** All gallery vectors are stored in **FAISS HNSW** (approximate nearest neighbours) for fast search at thousands of items.
4. **Querying:** Search uses the precomputed embedding matrix (and/or FAISS) so CLIP can be unloaded after encoding to **free VRAM**.

**Why CLIP + FAISS (no custom metric learning):** Training a dedicated similarity model would need **more labels, tuning, and GPU time**; frozen CLIP gives a **strong zero-shot-style baseline** under tight compute. A second heavy backbone was **not** used to avoid **OOM** on typical laptop GPUs.

**Evaluation (when `metadata.csv` is available):** Retrieval metrics such as **Precision@K**, **Recall@K**, and **mAP@K**, with **relevance** defined as **same artist OR same style** (catalogue proxy; discussed in the notebook vs. pure visual similarity). The notebook includes **plots** summarizing these metrics. **Data prep:** run `data/task2_data/extract.py` locally to populate images/metadata (large assets are gitignored).

---

### Repository structure

- `task1_convolutional_recurrent.ipynb` — Task 1 model, training, and evaluation  
- `task2_similarity_search.ipynb` — Task 2 embeddings, FAISS index, retrieval, and evaluation  
- `data/task2_data/extract.py` — script to fetch/prepare a subset of NGA images + metadata (see `.gitignore` for what is tracked)  
- `requirements.txt` — Python dependencies  

### Environment setup

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

You can then start Jupyter:

```bash
jupyter notebook
```

### Data

#### Task 1 – ArtGAN / WikiArt

Follow the instructions in the ArtGAN repo (`WikiArt Dataset/README.md`) to download the WikiArt images and metadata. After downloading:

- Set a local path like: `/path/to/wikiart/`
- Configure that path in the Task 1 notebook (`DATA_ROOT`, `CSV_ROOT`, `METADATA_CSV`).

#### Task 2 – National Gallery of Art Open Data

- Run **`python data/task2_data/extract.py`** (after configuring API / paths inside the script as needed) to build an ImageFolder layout and `metadata.csv`.
- In the notebook, set **`IMAGES_ROOT`** and **`METADATA_CSV`** to match that output (defaults point under `data/task2_data/nga_paintings_subset/`).

### How to run

1. Install dependencies and start Jupyter.
2. Open each notebook:
   - `task1_convolutional_recurrent.ipynb`
   - `task2_similarity_search.ipynb`
3. Set the dataset paths near the top of each notebook.
4. Run the cells in order to:
   - Train / evaluate Task 1 (or adjust `MAX_SAMPLES` / epochs as needed).
   - Encode the gallery, build FAISS, run retrieval and metrics for Task 2.
   - Export the notebooks to PDF for submission if required.

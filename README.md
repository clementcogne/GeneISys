# GeneISys
### An Organic Neuro-Symbolic Engine based on Semantic Physics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Experimental%20POC-orange.svg)]()

<p align="center">
<img width="100%" alt="Gemini_Generated_Image_dvm8hsdvm8hsdvm8" src="https://github.com/user-attachments/assets/6d1e13dd-f74d-4ebe-88db-1ea1b18cb598" />
</p>



**GeneISys** is an experimental AI engine developed through a "Vibe Coding" session with Gemini 3 Pro. It explores a radical alternative to Transformers: a **White Box**, **Continuously Learning** architecture where concepts interact as physical bodies in a semantic vector space.

> **Project Goal:** Demonstrate that complex, scalable, and auditable AI architectures can be rapidly conceptualized and prototyped through intuitive human-AI collaboration.

---

## 🧠 Key Features (Why GeneISys?)

Unlike static LLMs (Large Language Models), GeneISys focuses on dynamic topology:

* **🔍 White Box & Auditable:** Every decision flows from explicit physical interactions (gravity, mass, attraction). You can trace exactly *why* a connection was made via the Knowledge Graph.
* **🌱 Continuous Learning (Plasticity):** No massive pre-training phase required. The system learns in real-time ("Life Loop"), creating new memories and concepts on the fly.
* **⚡ Scalable Architecture:**
    * **Resolution:** Adjustable dimensions (tested from `64` to `4096`).
    * **Precision:** Supports `INT8`, `FP16`, and `FP32`.
    * **Optimization:** vectorized for CPU (Numba), GPU (CUDA), and Hybrid modes.
* **🛡️ Transparent Logic:** Embeds a "Narrative Architect" capable of explaining its own internal state without hallucinating reasoning.

---

## ⚠️ Disclaimer

**What GeneISys IS NOT:**
* It is **NOT an LLM** (like GPT-4 or Llama).
* It is **NOT pre-trained** on the internet. It starts "naked" and learns from what you feed it.
* It does **NOT** use backpropagation.

**Note on Language:**
This codebase reflects a "Vibe Coding" process aimed at rapid prototyping. Variable names and internal logic comments may act as a mix of English and French to facilitate the author's initial conceptualization.

### 🧬 Architecture Overview

```mermaid
graph TD
    Input[Input Text] -->|Tokenizer| V[Vector Space]
    V -->|Semantic Weight| Mass[Calculated Mass]
    
    subgraph "Physics Engine (Gravity)"
    Mass --> Interaction{N-Body Simulation}
    Interaction -->|Attraction| Link[Create Link]
    Interaction -->|Repulsion| Push[Separate Concepts]
    end
    
    Link --> Graph[Knowledge Graph]
    Graph -->|Active Inference| Surprise{Anomaly?}
    Surprise -- Yes --> Hypo[Create Hypothesis]
    Surprise -- No --> Re[Reinforcement]
```

---

## ⚙️ Installation

### Prerequisites
* **Python 3.12+** (Tested on 3.12.10)
* **CUDA 12.x** (Optional, for GPU acceleration. Tested on CUDA 12.8)

### Dependencies
Open a terminal in the project folder and install the required libraries:

```bash
pip install torch numpy tokenizers numba safetensors
```

### Required Files
For the engine to boot correctly, ensure these files are present in the **root directory** alongside `geneisys.py`:
* `genesis_curriculum_test.txt` (Critical for initial unit tests)
* `genesis_curriculum.txt` (For mass training mode)
* `fake_vectors.txt` (Optional, for import tests)

---

## 🚀 Configuration & Usage

Open the `geneisys.py` file in your text editor. Scroll down to the bottom (`if __name__ == "__main__":`) to tweak the engine's DNA.

### Core Settings
| Variable | Description | Recommended / Tested |
| :--- | :--- | :--- |
| `Nb_DIM` | Semantic resolution (Vector Dimensions) | `64` to `4096` |
| `strPRECISION_MODE` | Calculation precision | `"FP32"` (Standard), `"FP16"` (GPU), `"INT8"` |
| `boolForceCPU` | Force CPU usage even if CUDA is present | `False` (Auto-detect) |
| `str_lang` | Language configuration | `"fr"` (Currently supported) |
| `boolResetBase` | **Wipe memory** at startup | `True` (Fresh start) / `False` (Persistence) |

### 🕹️ Run Modes (`RUN_MODE`)

Uncomment the mode you wish to run in the script:

1.  **`DIAGNOSTIC`** (Default)
    * Runs a full suite of unit tests (Physics, Logic, Memory, Dreaming).
    * *Perfect for checking system stability.*
2.  **`LIFE_LOOP`**
    * The engine runs autonomously. It gets "bored," gets curious, dreams, and interacts with you in the console.
3.  **`TRAINING_FILE`**
    * Mass ingestion of the text corpus (`genesis_curriculum.txt`) to build initial knowledge.
4.  **`IMPORT_MODEL`**
    * Surgical import of external vector data.
5.  **`INFERENCE`**
    * Interactive Chat Mode. Talk to GeneISys and ask questions like *"Comment est la voiture ?"*.

---

## 🖥️ Hardware & Performance Notes

This architecture is designed to be highly elastic, but physics simulations are demanding.

* **High-End Desktop:** Tested on **NVIDIA RTX 5080**.
    * *Result:* Extremely fast ingestion, real-time physics on 4096 dims.
* **Standard Laptop:** Tested on **Intel i5 8th Gen** (No GPU).
    * *Result:* Functional.
    * **⚠️ IMPORTANT for CPU Users:** In `DIAGNOSTIC` mode, you **must** comment out the massive stress test line inside the `def run_all(self)` function to avoid freezing your machine:
    ```python
    # self.test_massive_context_n9()  <-- Add a hash to disable this on laptops/CPU
    ```

---

## 👤 Author & Context

* **Author:** Clément Cogné
* **Date:** 2025-11-28
* **Location:** Toulouse, France
* **Contact:** [LinkedIn Profile](https://www.linkedin.com/in/clement-cogne/)

*This is a personal hobby project shared for educational purposes to demonstrate the potential of Neuro-Symbolic AI. It is provided "as-is" without warranty.*


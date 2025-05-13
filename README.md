# SAT Formula Extraction via Transformer Optimization

## 👥 Team Members
- Akshat Iyer
- Andrew Wu
- Sebastian Perez
- Eugene Kim
- Will Noonan

## 🚀 Project Summary
This project aims to optimize a Transformer-based model to extract mathematical formulas from SAT-style word problems. Instead of directly outputting answers, our model learns to generate a symbolic formula that can be parsed and solved using SymPy. We trained and optimized a FLAN-T5 Transformer model to solve this problem.

**Key Contributions:**
- Developed a formula-to-answer pipeline using regex and SymPy.
- Trained and Optimized FLAN-T5 model on answer correctness.
- Achieved ~81% symbolic similarity and ~72% answer-level correctness on test data.

## 🗂️ Repository Structure
```bash
.
├── src/                # Core scripts (model training, data pipeline, reward models)
├── notebooks/          # Jupyter experiments and GRPO training logs
├── report.md           # Final report with results and analysis
├── docs/               # Figures, diagrams, and presentation assets
├── requirements.txt    # Python dependencies
├── README.md           # (this file)
└── _development_history/ # Older scripts, logs, intermediate notes
```

## ⚙️ Setup Instructions
We recommend using Google Colab with an A100 GPU if you want to train models. The overall model training takes ~1 hour to complete, but for the sake of showing our work we have set it to 1 epoch (should take a couple minutes). Results in the "STAT4830Transformers" notebook will NOT be good. See better results in STAT4830Demo. To ensure all dependencies are met, place the requirements.txt file in your working Colab directory and run:

```bash
pip install -r path-to-your-requirements.txt
```

We recommend using Python 3.11.12 in Colab.

## ▶️ Running the Code

All training and inference is done through a single Colab notebook.

### 📓 Final Colab Notebook

In "STAT4830UnslothFinal, find our training loop and example validation for our preliminary Unsloth Model using just 1 epoch.

In "STAT4830Transformers", find our training loop and example validation for our FLAN-T5 model using just 1 epoch.

In "STAT4830Demo", find our final model downloaded from hugging face and test results.

Notebooks include:

- Loading and preprocessing the SAT dataset
- Parsing model outputs into symbolic formulas
- Evaluating formula accuracy and end-to-end correctness using SymPy

### ⚙️ Requirements

    
    Unsloth (see first cell in STAT4830UnslothFinal)
    Lightning (see first cell in STAT4830Transformers)
    All other libraries auto-installed in Colab
    GPU backend (A100 preferred)

No additional setup is needed—just make sure to select the A100 GPU runtime in Colab (Runtime > Change runtime type > GPU).

# Hybrid-Sentiment-Analyzer
# IMDB Sentiment Analysis: Hybrid DistilBERT + CNN

## 📌 Project Overview
This project performs a comparative analysis of three distinct deep learning architectures for binary sentiment classification on the IMDB Movie Reviews dataset. The goal is to evaluate and compare the effectiveness of traditional Convolutional Neural Networks (CNN), pre-trained Transformer models (DistilBERT), and a Hybrid architecture that combines both.

The study aims to determine if combining local feature extraction (CNN) with rich contextual embeddings (DistilBERT) yields performance improvements over standalone architectures.

## 🧪 Approaches Evaluated

### 1. Standalone CNN (Baseline)
A traditional deep learning approach focusing on local feature extraction.
* **Architecture:** Embedding Layer → Conv1D (Filters: 128, Kernel: 5) → Conv1D (Filters: 128, Kernel: 3) → Global Max Pooling → Dense Layers.
* **Role:** Serves as a baseline to measure the performance gain offered by transformer-based models.

### 2. Standalone DistilBERT
A state-of-the-art transfer learning approach using a distilled version of BERT.
* **Architecture:** Pre-trained `distilbert-base-uncased` → Extract `[CLS]` token embedding → Dropout → Dense Classifier.
* **Strengths:** Captures deep bidirectional context and long-range dependencies in text.

### 3. Hybrid Model (DistilBERT + CNN)
A proposed architecture merging contextual understanding with pattern recognition.
* **Architecture:** Pre-trained `distilbert-base-uncased` → Input full sequence (`last_hidden_state`) into CNN layers → Conv1D → Global Max Pooling → Dense Classifier.
* **Hypothesis:** The CNN can extract specific n-gram patterns from the high-quality contextual embeddings provided by DistilBERT.

## 📊 Performance Results

All models were trained on the IMDB dataset (25k train / 25k test) for 5 epochs.

| Model | Accuracy | Loss | F1-Score (Positive) |
| :--- | :--- | :--- | :--- |
| **Hybrid (DistilBERT + CNN)** | **90.87%** | **0.2271** | **0.9085** |
| Standalone DistilBERT | 90.84% | 0.2218 | 0.9084 |
| Standalone CNN | 84.50% | 0.3547 | 0.8441 |

**Key Findings:**
* **Transformers Dominate:** Both DistilBERT-based models significantly outperformed the standalone CNN (~6% accuracy gap).
* **Hybrid Edge:** The Hybrid model achieved the highest overall accuracy and F1-score, though the margin over the standalone DistilBERT was minimal.
* **Context Matters:** The results confirm that pre-trained language models are vastly superior for sentiment tasks compared to simple embeddings + CNNs.

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.x
* TensorFlow 2.x
* GPU environment recommended (e.g., Google Colab)

### Dependencies
Install the required libraries:
```bash
pip install tensorflow tensorflow-datasets transformers scikit-learn pandas matplotlib seaborn
````

### Running the Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ezana-Fekadu/Hybrid-Sentiment-Analyzer.git
    ```
2.  **Open the Notebook:**
    Launch `Hybrid_Sentiment_Analysis.ipynb` in Jupyter Notebook or Google Colab.
3.  **Execute Cells:**
    Run all cells to download the dataset, train the three models, and generate the comparative report.

## 📂 File Structure

  * `Hybrid_Sentiment_Analysis.ipynb`: Main notebook containing data loading, model definitions, training loops, and evaluation logic.
  * `saved_models/`: Directory where the best performing models (`.h5` format) are saved during training.
  * `runs/`: TensorBoard logs for visualizing training performance.

## 🚀 Future Work

  * **Hyperparameter Tuning:** optimize CNN kernel sizes and filter counts in the hybrid model.
  * **Advanced Transformers:** Experiment with RoBERTa or ALBERT to see if larger models widen the gap between standalone and hybrid approaches.
  * **Explainability:** Implement techniques like SHAP or LIME to visualize which text features the CNN is extracting from the BERT embeddings.

## 🤝 Acknowledgments

  * Dataset provided by [TensorFlow Datasets (IMDB Reviews)](https://www.tensorflow.org/datasets/catalog/imdb_reviews).
  * Transformer models provided by [Hugging Face](https://huggingface.co/).

-----


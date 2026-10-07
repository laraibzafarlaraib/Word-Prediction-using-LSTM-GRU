# Next Word Prediction Using LSTM and GRU

A deep learning application that predicts the next word in a text sequence using recurrent neural networks. Built with TensorFlow/Keras and deployed as an interactive web application using Streamlit.

---

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training Pipeline](#training-pipeline)
- [Results](#results)
- [Technologies](#technologies)
- [Future Work](#future-work)
- [Author](#author)
- [License](#license)

---

## Overview

This project implements a next-word prediction system using Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks. Given a sequence of words, the model predicts the most probable next word. This is the same foundational concept behind autocomplete features in search engines, messaging apps, and writing assistants.

The end-to-end pipeline covers data collection, text preprocessing, model training with early stopping, and deployment as a Streamlit web application.

---

## Demo

Launch the application and enter any text sequence to get a predicted next word:

```
Input:  "To be or not to"
Output: "be"
```

---

## Architecture

Two model architectures were trained and compared:

### LSTM Model

```
Layer (type)                 Output Shape              Param #
=================================================================
Embedding                    (None, 13, 100)           481,800
LSTM (150 units)             (None, 13, 150)           150,600
Dropout (0.2)                (None, 13, 150)           0
LSTM (100 units)             (None, 100)               100,400
Dense (softmax)              (None, 4818)              486,618
=================================================================
Total params: 1,219,418 (4.65 MB)
```

### GRU Model

```
Layer (type)                 Output Shape              Param #
=================================================================
Embedding                    (None, 13, 100)           481,800
GRU (150 units)              (None, 13, 150)           113,400
Dropout (0.2)                (None, 13, 150)           0
GRU (100 units)              (None, 100)               75,600
Dense (softmax)              (None, 4818)              486,618
=================================================================
Total params: 1,157,418 (4.41 MB)
```

---

## Dataset

The model is trained on **Shakespeare's "The Tragedy of Hamlet"**, sourced from the NLTK Gutenberg corpus. The text contains approximately 4,900 lines and a vocabulary of 4,818 unique tokens after preprocessing.

---

## Project Structure

```
next-word-prediction-lstm-gru/
|-- app.py                   # Streamlit web application
|-- experiments.ipynb         # Training notebook with LSTM and GRU experiments
|-- hamlet.txt                # Training corpus (Shakespeare's Hamlet)
|-- next_word_lstm.h5         # Trained LSTM model weights
|-- tokenizer.pickle          # Serialized tokenizer
|-- requirements.txt          # Python dependencies
|-- .gitignore                # Git ignore rules
|-- README.md                 # Project documentation
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/next-word-prediction-lstm-gru.git
   cd next-word-prediction-lstm-gru
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate           # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Run the Web Application

```bash
streamlit run app.py
```

The application will open in your browser. Enter a sequence of words and click **Predict Next Word** to see the model's prediction.

### Run the Training Notebook

Open `experiments.ipynb` in Jupyter Notebook or Google Colab to reproduce the training pipeline or experiment with different architectures.

---

## Training Pipeline

1. **Data Collection** -- Load the Hamlet corpus from the NLTK Gutenberg dataset.
2. **Preprocessing** -- Lowercase the text, tokenize it into word indices, and generate n-gram input sequences.
3. **Sequence Padding** -- Pad all sequences to a uniform length of 14 tokens.
4. **Label Encoding** -- One-hot encode the target word across the full vocabulary.
5. **Train/Test Split** -- 80/20 split for training and validation.
6. **Model Training** -- Train for 50 epochs with early stopping (patience=3, monitoring validation loss).
7. **Model Export** -- Save the trained model as `.h5` and the tokenizer as a pickle file.

---

## Results

The LSTM model was trained for 50 epochs on the training set:

| Metric               | Training Set | Validation Set |
|----------------------|-------------|----------------|
| Final Loss           | 2.71        | 11.34          |
| Final Accuracy       | 40.6%       | 5.8%           |

The significant gap between training and validation performance indicates overfitting, which is expected given the small corpus size (~4,900 lines). The model performs well on in-domain Shakespearean text but does not generalize broadly. Early stopping was configured but did not trigger due to the early stopping monitoring configuration.

---

## Technologies

| Component        | Technology              |
|-----------------|-------------------------|
| Language         | Python 3.11             |
| Deep Learning    | TensorFlow 2.15, Keras  |
| NLP              | NLTK                    |
| Web Framework    | Streamlit               |
| Data Processing  | NumPy, Pandas           |
| Visualization    | Matplotlib              |
| ML Utilities     | Scikit-learn, SciKeras  |

---

## Future Work

- Experiment with Transformer-based architectures (GPT-style) for improved sequence generation quality.
- Train on larger and more diverse text corpora to improve generalization.
- Add temperature-based sampling and beam search for more creative and diverse predictions.
- Implement attention visualization to show which input tokens influence the prediction.
- Add multi-word sequence generation (predict N words ahead, not just one).
- Containerize the application with Docker for easier deployment.

---

## Author

**Laraib Zafar**

---

## License

This project is open source and available under the [MIT License](LICENSE).

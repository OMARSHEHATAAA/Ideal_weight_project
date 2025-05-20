                                                          Made By OMAR SHEHATA




# 🧠 Ideal Weight Prediction with TensorFlow

This project predicts a person's **ideal weight** based on their **height**, **age**, and **gender** using a neural network built with **TensorFlow**.

## 📁 Project Files

- `ideal_weight_prediction_tensorflow.ipynb`: Main notebook with explanations, training, and image previews.
- `ideal_weight_model.py`: Python script version of the notebook.
- `ideal_weight_dataset.csv`: Sample dataset.
- `requirements.txt`: Python dependencies.

## 🏗️ Model Architecture

The model is a simple feed-forward neural network built using `tensorflow.keras` with:
- Input layer: 3 features (height, age, gender)
- 2 hidden layers with ReLU and Dropout
- Output: Single neuron for predicted weight

## 🚀 How to Run

```bash
pip install -r requirements.txt
python ideal_weight_model.py
```

## 📊 Example Output

```
Test MAE: 3.12 kg
```

## 📌 Requirements

- Python 3.7+
- TensorFlow
- Pandas
- NumPy
- scikit-learn
- matplotlib



Made By OMAR SHEHATA

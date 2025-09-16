# RNN SMS Classifier
A C++ implementation of a **Recurrent Neural Network (RNN)** for SMS text classification.  
Classifies SMS messages into four categories: **ham**, **spam**, **social**, **promotional**.

---

## ✨ Features
- Custom RNN (tanh + softmax)  
- Adam optimizer with momentum and adaptive learning rates  
- Text preprocessing: normalization, tokenization, vocab building, text→sequence  
- Evaluation: accuracy, confusion matrix, precision, recall, F1-score  
- Model persistence: save weights in readable `model-weights.txt`  
- Robust training: Xavier init, gradient clipping, label smoothing, LR decay  

---

## 📂 Project Structure
```text
├── data_utils.h/.cpp    # Text preprocessing and data handling utilities
├── rnn.h/.cpp           # RNN model implementation with training logic
├── main.cpp             # Main training and evaluation pipeline
└── data/
    └── synthetic_sms_dataset.csv  # Training dataset (CSV format)
```

## 📑 Dataset Format
Expected CSV located at data/synthetic_sms_dataset.csv:
```csv
label,message
ham,"Hello, how are you today?"
spam,"URGENT! Claim your prize now!"
social,"Meeting tonight at 7pm"
promotional,"20% off all items this weekend"
```
- First row is the header: label,message
- Labels must be one of: ham, spam, social, promotional
- Messages may include commas and should be quoted where necessary.

## 🧠 Architecture Details
### Model Parameters
- Input Size: Vocabulary size + buffer (dynamic based on dataset)
- Hidden Size: 64 neurons
- Output Size: 4 classes (ham, spam, social, promotional)
- Max Sequence Length: 30 tokens
- Batch Size: 16

### Training Configuration
- Optimizer: Adam (β₁ = 0.9, β₂ = 0.999)
- Learning Rate: 0.01 (with decay schedule)
- Epochs: 20
- Loss: Cross-entropy with label smoothing (smoothing = 0.1)
- Weight Init: Xavier / Glorot uniform
- Other: gradient clipping for hidden activations

## 📊 Pipeline (high level)
1. Load CSV → normalize & tokenize text
2. Build vocabulary → convert text to padded sequences
3. Split data into train/test (80/20)
4. Train the RNN
5. Evaluate (metrics + confusion matrix)
6. Save trained model to model-weights.txt

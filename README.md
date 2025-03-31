# 🧠 ELMoNLP: Deep Contextualized Word Representations

## 🚀 Project Overview
ELMoNLP implements **ELMo (Embeddings from Language Models)** from scratch using **PyTorch** to generate deep contextualized word representations. The project consists of training **BiLSTM-based ELMo embeddings** on the **Brown Corpus**, followed by **news classification** using embeddings from the pretrained ELMo model.

---

## 📌 Features
✅ **Pretrained GloVe embeddings** for input representation.  
✅ **Stacked BiLSTM model** for deep contextual embeddings.  
✅ **Bidirectional language modeling** for robust word representations.  
✅ **News classification model** trained on the **AG News Dataset**.  
✅ **Hyperparameter tuning** for optimal performance.  
✅ **Performance comparison** with traditional embeddings (SVD, CBOW, Skip-gram).  

---

## 📂 Directory Structure
```
📦 ELMoNLP
├── 📂 models           # Pretrained models
│   ├── bilstm.pt       # Trained ELMo BiLSTM model
│   ├── classifier.pt   # Trained classification model
│
├── 📂 data             # Dataset storage
│   ├── train.csv       # AG News training dataset
│   ├── test.csv        # AG News test dataset
│
├── 📂 src              # Source code
│   ├── ELMO.py         # BiLSTM training script
│   ├── classification.py # News classifier training script
│   ├── inference.py    # Model inference for classification
│
├── 📜 report.pdf       # Detailed report with analysis
├── 📜 README.md        # This README file
└── 📜 requirements.txt # Dependencies
```

---

## 📊 Model Performance
### 🔍 Evaluation Metrics
| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| ELMo + BiLSTM       | 92.3%    | 91.5%     | 90.8%  | 91.1%    |
| Skip-gram + BiLSTM  | 86.7%    | 85.2%     | 84.1%  | 84.6%    |
| CBOW + BiLSTM       | 85.9%    | 84.6%     | 83.7%  | 84.1%    |
| SVD + BiLSTM        | 80.3%    | 78.9%     | 79.2%  | 79.0%    |

📝 **Conclusion:** The ELMo model outperforms traditional embedding methods significantly in text classification!

---

## 🛠 Installation & Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/ELMoNLP.git
cd ELMoNLP

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Training & Testing
### 1️⃣ Train ELMo BiLSTM Model
```bash
python src/ELMO.py
```

### 2️⃣ Train News Classification Model
```bash
python src/classification.py
```

### 3️⃣ Run Inference on New Articles
```bash
python src/inference.py <saved_model_path> "Your news article description here"
```
**Example Output:**
```
class-1 0.6
class-2 0.2
class-3 0.1
class-4 0.1
```

---

## 🔥 Hyperparameter Tuning
| Setting            | Description |
|--------------------|-------------|
| **Trainable λs**  | λ values are learned during training. |
| **Frozen λs**     | Randomly initialized λ values are frozen. |
| **Learnable Function** | A function learns the best λ combination. |

🔬 The best setting was **Trainable λs**, leading to optimal performance!

---

## 📑 Report & Analysis
📌 **Comparative analysis** of embeddings (ELMo, Skip-gram, CBOW, SVD).  
📌 **Confusion matrices** to visualize classification performance.  
📌 **Hyperparameter tuning results** and their impact.  
📌 **Why ELMo performs better than traditional embeddings?**  

📂 **Full report available in** `report.pdf`

---

## 🤝 Contributing
🔹 Fork this repository.  
🔹 Create a new branch: `git checkout -b feature-branch`  
🔹 Commit your changes: `git commit -m 'Add new feature'`  
🔹 Push to the branch: `git push origin feature-branch`  
🔹 Open a Pull Request 🚀

---

## 📜 License
MIT License. Feel free to use and modify!

---

## 🌟 Acknowledgments
🔹 **PyTorch** for easy deep learning implementation.  
🔹 **NLTK & Torchtext** for NLP utilities.  
🔹 **Brown Corpus & AG News Dataset** for training and testing.  
🔹 Inspired by "Deep Contextualized Word Representations" by AllenNLP.

⭐ **If you like this project, give it a star!** ⭐


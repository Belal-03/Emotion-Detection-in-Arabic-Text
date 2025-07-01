# 😊 Emotion Detection in Arabic Text Using Classical and Deep Learning Techniques

This project aims to build a robust emotion detection system for Arabic text by leveraging both classical machine learning algorithms and deep learning models.

---

## 📚 Dataset

- **Emotional Tone Arabic Dataset**  
  Available at: [GitHub Repository](https://github.com/amrmalkhatib/Emotional-Tone/tree/master)

---

## 🎯 Project Objectives & Tasks

### Part 1: Data Preprocessing
- Load and explore the dataset  
- Clean the Arabic text by removing:  
  - Punctuation  
  - Diacritics  
  - Stop words  
- Tokenize the Arabic text  
- Normalize words to handle variants  

### Part 2: Text Representation Techniques
Implement and compare the following vectorization methods:  
- **Bag-of-Words (BoW)**  
- **TF-IDF**  
- **Word2Vec** (preferably pre-trained models like AraVec or fastText)  

### Part 3: Classical Machine Learning Models
Using BoW, TF-IDF, and Word2Vec representations, train and evaluate:  
1. Naive Bayes (MultinomialNB)  
2. Support Vector Machine (SVM) with RBF kernel  
3. Decision Tree (DT)  
4. Random Forest (RF)  
5. AdaBoost  

### Part 4: Feed-Forward Neural Network (FNN)
- Inputs: TF-IDF and averaged Word2Vec vectors  
- Architecture: Dense layers with ReLU activation and Softmax output  

### Part 5: Recurrent Neural Network (LSTM)
- Inputs: Pre-trained Word2Vec embeddings and tokenized, padded sequences  
- Model: LSTM layers (optional: BiLSTM for improved performance)  
- Output: Softmax layer for emotion classification  

---

## 📦 Deliverables

1. **Jupyter Notebook** containing:  
   - Data preprocessing steps  
   - Text representation implementations  
   - Training and evaluation code for all models  

2. **Report** detailing:  
   - Preprocessing techniques  
   - Comparative analysis of models based on Accuracy, Precision, Recall, and F1 Score  
   - Challenges faced when working with Arabic text  

---

## 🛠 Tools & Libraries

- Python  
- NLP libraries (e.g., NLTK, AraVec, fastText)  
- Machine learning frameworks (e.g., scikit-learn, TensorFlow, Keras)  

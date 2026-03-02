# Fake_News_Detector
A machine learning–based model that detects whether a news article is real or fake.
---

## Project Overview
Processing the user's input, the model uses TF-IDF to convert the cleaned text into numbers and give each word a weight. The TF-IDF numbers helps in calculating the probability between 0 and 1 using Linear Regression. If the (probablity>0.5) -> Fake, else Real.

## Features
- Text Preprocessing
- TF-IDF vectorization
- Trained ML model
- Flask API for prediction

---
## Project Structure
```
fake-news-detector/
│
├── data/
│   └── (Dataset Files)
|   └── news.csv
|   └── news_cleaned.csv
├── prepare_data.py
├── preprocess_data.py
├── train_model.py
├── test_model.py
├── api.py
├── model.pkl
└── vectorizer.pkl
```
## Dataset Files
*[https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset]*
1. Download dataset from Kaggle link
2. Place Fake.csv and True.csv inside _/data_ folder
3. Run ``` prepare_data.py```

## Installation
1. Clone the repository
   ```
   git clone https://github.com/Aakanksha-2025/Fake_News_Detector.git
   cd Fake_News_Detector
   ```
2. Install Dependencies
   ```
   pip install -r requirements.txt
   ```

## Running the API

```
python api.py
```

_API runs at:_
```
http://127.0.0.1:5000
```

## Example Request
on a new terminal after running 
``` python api.py ```
or
``` py api.py```
User Input:
```
curl -Method POST http://127.0.0.1:5000/predict `
-Headers @{"Content-Type"="application/json"} `
-Body '{"text":"Aliens are controlling world leaders"}'
```
## Output Example

[View Model Output Screenshot](https://github.com/user-attachments/assets/dbe166af-06de-4d0d-b906-3bc9329374ae)

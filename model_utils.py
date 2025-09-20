import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (first time only)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# -------------------------------
# Text Preprocessing
# -------------------------------
def preprocess_text(text):
    """
    Preprocess text by removing special characters, lowercasing, 
    and removing stopwords
    """
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)


# -------------------------------
# Feature Extraction
# -------------------------------
def extract_features(df, text_column):
    """
    Extract linguistic features from text
    """
    features = []
    
    for text in df[text_column]:
        text = str(text) if not pd.isna(text) else ""
        
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = max(1, len(re.split(r'[.!?]+', text)))
        
        avg_word_length = char_count / max(word_count, 1)
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        unique_word_ratio = len(set(text.split())) / max(word_count, 1)
        digit_count = sum(1 for char in text if char.isdigit())
        uppercase_count = sum(1 for char in text if char.isupper())
        
        features.append([char_count, word_count, sentence_count, 
                        avg_word_length, avg_sentence_length,
                        unique_word_ratio, digit_count, uppercase_count])
    
    return pd.DataFrame(features, columns=[
        'char_count', 'word_count', 'sentence_count',
        'avg_word_length', 'avg_sentence_length',
        'unique_word_ratio', 'digit_count', 'uppercase_count'
    ])


# -------------------------------
# Prediction Function
# -------------------------------
def predict_text(text, clf, tfidf, feature_columns):
    """
    Predict whether a given text is human or AI generated
    """
    cleaned_text = preprocess_text(text)
    
    # Create temporary dataframe for feature extraction
    temp_df = pd.DataFrame({'text': [text]})
    
    # Extract linguistic features
    linguistic_features = extract_features(temp_df, 'text')
    
    # Ensure all expected columns exist
    for col in feature_columns:
        if col not in linguistic_features.columns:
            linguistic_features[col] = 0
    linguistic_features = linguistic_features[feature_columns]
    
    # TF-IDF features
    tfidf_features = tfidf.transform([cleaned_text])
    tfidf_df = pd.DataFrame(tfidf_features.toarray())
    tfidf_df.columns = [f'tfidf_{i}' for i in range(tfidf_df.shape[1])]
    
    # Combine features
    features = pd.concat([
        linguistic_features.reset_index(drop=True),
        tfidf_df.reset_index(drop=True)
    ], axis=1)
    
    # Handle missing TF-IDF columns
    missing_cols = set(clf.feature_names_in_) - set(features.columns)
    for col in missing_cols:
        features[col] = 0
    
    features = features[clf.feature_names_in_]
    
    # Predict
    prediction = clf.predict(features)[0]
    probability = clf.predict_proba(features)[0]
    
    return {
        'prediction': 'AI-generated' if prediction == 1 else 'Human-written',
        'confidence': float(max(probability)),
        'probabilities': {
            'Human': float(probability[0]),
            'AI': float(probability[1])
        }
    }

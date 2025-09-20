from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from model_utils import preprocess_text, extract_features, predict_text

# Load trained objects
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

app = FastAPI(title="Human vs AI Text Classifier")

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_endpoint(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text field is required")
    result = predict_text(request.text, clf, tfidf, feature_columns)
    return result

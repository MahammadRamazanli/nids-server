from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # CORS middleware import
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app yaradın
app = FastAPI(
    title="Network Intrusion Detection System API",
    description="API for detecting network intrusions using ML model",
    version="1.0.0"
)

# CORS middleware əlavə edin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Bütün originlərə icazə verir
    # Production üçün daha specific olun:
    # allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175"],
    allow_credentials=True,
    allow_methods=["*"],  # Bütün HTTP metodlarına icazə
    allow_headers=["*"],  # Bütün headerlərə icazə
)

# Model yükləmə
try:
    model = joblib.load('model.pkl')
    logger.info("Model successfully loaded")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    model = None

# Pydantic modelləri
class NetworkData(BaseModel):
    """Tək network record üçün model"""
    duration: float
    protocol_type: str
    service: str
    flag: str
    src_bytes: int
    dst_bytes: int
    land: int
    wrong_fragment: int
    urgent: int
    hot: int
    num_failed_logins: int
    logged_in: int
    num_compromised: int
    root_shell: int
    su_attempted: int
    num_root: int
    num_file_creations: int
    num_shells: int
    num_access_files: int
    num_outbound_cmds: int
    is_host_login: int
    is_guest_login: int
    count: int
    srv_count: int
    serror_rate: float
    srv_serror_rate: float
    rerror_rate: float
    srv_rerror_rate: float
    same_srv_rate: float
    diff_srv_rate: float
    srv_diff_host_rate: float
    dst_host_count: int
    dst_host_srv_count: int
    dst_host_same_srv_rate: float
    dst_host_diff_srv_rate: float
    dst_host_same_src_port_rate: float
    dst_host_srv_diff_host_rate: float
    dst_host_serror_rate: float
    dst_host_srv_serror_rate: float
    dst_host_rerror_rate: float
    dst_host_srv_rerror_rate: float

class BatchNetworkData(BaseModel):
    """Batch prediction üçün model"""
    data: List[NetworkData]

class PredictionResponse(BaseModel):
    """Prediction response modeli"""
    prediction: str
    confidence: float
    is_intrusion: bool

class BatchPredictionResponse(BaseModel):
    """Batch prediction response modeli"""
    predictions: List[PredictionResponse]
    total_records: int
    intrusion_count: int

def preprocess_data(data: NetworkData) -> pd.DataFrame:
    data_dict = data.dict()
    df = pd.DataFrame([data_dict])

    # Modeli saxlayanda istifadə etdiyin kategorik sütunlar
    categorical_cols = ['protocol_type', 'service', 'flag']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Modelin gözlədiyi bütün sütunları əlavə et (əgər yoxdursa, 0 ilə doldur)
    model_features = list(model.feature_names_in_)
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    df = df[model_features]

    return df

def preprocess_batch_data(data_list: List[NetworkData]) -> pd.DataFrame:
    data_dicts = [data.dict() for data in data_list]
    df = pd.DataFrame(data_dicts)
    categorical_cols = ['protocol_type', 'service', 'flag']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    model_features = list(model.feature_names_in_)
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    df = df[model_features]
    return df

@app.get("/")
def read_root():
    """API status yoxlama"""
    return {
        "message": "Network Intrusion Detection System API",
        "status": "active",
        "model_loaded": model is not None
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "loaded"}

@app.post("/predict", response_model=PredictionResponse)
def predict_intrusion(data: NetworkData):
    """Tək network record üçün prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Data preprocessing
        processed_data = preprocess_data(data)
        
        # Prediction
        prediction = model.predict(processed_data)[0]
        
        # Confidence score (əgər model probability dəstəkləyirsə)
        try:
            confidence = float(np.max(model.predict_proba(processed_data)[0]))
        except:
            confidence = 0.5  # Default confidence
        
        # Prediction mantığını düzelt
        # Model 0=normal, 1=anomaly döndürür
        is_intrusion = prediction == 1  # 1 ise intrusion
        
        # Prediction'ı string formatında döndür
        prediction_str = "normal" if prediction == 0 else "anomaly"
        
        return PredictionResponse(
            prediction=prediction_str,
            confidence=confidence,
            is_intrusion=is_intrusion
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch_intrusion(batch_data: BatchNetworkData):
    """Batch prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Batch data preprocessing
        processed_data = preprocess_batch_data(batch_data.data)
        
        # Batch prediction
        predictions = model.predict(processed_data)
        
        # Confidence scores
        try:
            probabilities = model.predict_proba(processed_data)
            confidences = [float(np.max(prob)) for prob in probabilities]
        except:
            confidences = [0.5] * len(predictions)
        
        # Response yaradın
        prediction_responses = []
        intrusion_count = 0
        
        for pred, conf in zip(predictions, confidences):
            is_intrusion = pred == 1  # 1 ise intrusion
            if is_intrusion:
                intrusion_count += 1
            
            # Prediction'ı string formatında döndür    
            prediction_str = "normal" if pred == 0 else "anomaly"
                
            prediction_responses.append(
                PredictionResponse(
                    prediction=prediction_str,
                    confidence=conf,
                    is_intrusion=is_intrusion
                )
            )
        
        return BatchPredictionResponse(
            predictions=prediction_responses,
            total_records=len(predictions),
            intrusion_count=intrusion_count
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
def get_model_info():
    """Model haqqında məlumat"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        model_info = {
            "model_type": type(model).__name__,
            "model_params": getattr(model, 'get_params', lambda: {})(),
        }
        
        # Feature sayını əlavə et (əgər mövcuddursa)
        if hasattr(model, 'n_features_in_'):
            model_info["n_features"] = model.n_features_in_
            
        return model_info
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
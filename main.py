import os
from contextlib import asynccontextmanager
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Global holders for model and training columns to ensure single startup load
model = None
model_columns = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "loan_model.joblib")
COLUMNS_PATH = os.path.join(BASE_DIR, "model", "model_columns.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan event handler:
    Loads the trained model and feature columns once into memory when the server starts.
    """
    global model, model_columns
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        if not os.path.exists(COLUMNS_PATH):
            raise FileNotFoundError(f"Model columns file not found at: {COLUMNS_PATH}")

        print(f"[STARTUP] Loading model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        print(f"[STARTUP] Model loaded successfully: {type(model).__name__}")

        print(f"[STARTUP] Loading column definitions from: {COLUMNS_PATH}")
        model_columns = joblib.load(COLUMNS_PATH)
        print(f"[STARTUP] {len(model_columns)} training columns loaded successfully.")

    except Exception as e:
        print(f"[ERROR] Failed to load model or columns: {e}")
        raise e

    yield

    # Clean up resources on shutdown if needed
    print("[SHUTDOWN] Server stopping...")


# Initialize FastAPI application
app = FastAPI(
    title="Loan Approval Prediction System",
    description="Production-ready FastAPI backend for predicting loan approvals using a trained scikit-learn model.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets (CSS, JS)
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Configure HTML template rendering
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# Pydantic schema for form input validation
class LoanApplicantInput(BaseModel):
    """
    Schema for validating applicant input details.
    Restricts categorical inputs to exact valid values from the training dataset.
    """
    # Demographics
    Age: int = Field(..., ge=18, le=100, description="Applicant age in years", examples=[40])
    Gender: Literal["Male", "Female"] = Field(..., description="Gender of applicant", examples=["Male"])
    Married: Literal["No", "Yes"] = Field(..., description="Marital status", examples=["No"])
    Dependents: Literal["0", "1", "2", "3+"] = Field(..., description="Number of dependents", examples=["1"])
    Education: Literal["Graduate", "Not Graduate"] = Field(..., description="Educational qualification", examples=["Graduate"])
    Self_Employed: Literal["No", "Yes"] = Field(..., description="Self-employment status", examples=["No"])

    # Financial details
    Applicant_Income: float = Field(..., ge=0, description="Monthly applicant income in local currency", examples=[477141.0])
    Coapplicant_Income: float = Field(..., ge=0, description="Monthly co-applicant income", examples=[344017.0])
    Existing_Debt: float = Field(..., ge=0, description="Existing debt amount", examples=[89066.0])
    Credit_Score: float = Field(..., ge=300, le=900, description="Applicant credit score (typically 300 to 850)", examples=[600.0])

    # Loan request details
    Loan_Amount: float = Field(..., ge=1000, description="Requested loan amount", examples=[265000.0])
    Loan_Term: int = Field(..., ge=12, le=480, description="Loan repayment term in months", examples=[240])
    Property_Area: Literal["Rural", "Semiurban", "Urban"] = Field(..., description="Property area category", examples=["Rural"])


# Response schema for predictions
class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Model binary prediction: 1 for Approved, 0 for Rejected")
    status: str = Field(..., description="Human-readable decision ('Approved' or 'Rejected')")
    confidence: Optional[float] = Field(None, description="Model prediction probability (0.0 to 1.0) if supported")
    confidence_percentage: Optional[str] = Field(None, description="Formatted percentage string")
    message: str = Field(..., description="Brief outcome explanation")


@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    """
    Serves the single-page web application frontend.
    """
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "Loan Approval Prediction System"
        }
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_loan(applicant: LoanApplicantInput):
    """
    Endpoint to predict loan approval.
    Applies exact notebook preprocessing:
    1. Converts applicant input dictionary into a 1-row pandas DataFrame.
    2. Runs pd.get_dummies on the 6 categorical columns (Gender, Married, Dependents, Education, Self_Employed, Property_Area).
    3. Reindexes to the exact 22 model_columns loaded at startup, with fill_value=0.
    4. Passes the aligned feature vector to model.predict and (if supported) model.predict_proba.
    """
    global model, model_columns

    if model is None or model_columns is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready. Please try again in a few moments."
        )

    try:
        # Step 1: Create a single-row DataFrame from validated input
        raw_dict = applicant.model_dump()
        input_df = pd.DataFrame([raw_dict])

        # Step 2: One-hot encode the exact six categorical columns from the notebook
        categorical_columns = [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "Property_Area"
        ]

        encoded_df = pd.get_dummies(
            input_df,
            columns=categorical_columns,
            dtype=int
        )

        # Step 3: Reindex DataFrame against exact model training columns with fill_value=0
        # This guarantees matching column names and order identical to training time
        aligned_df = encoded_df.reindex(columns=model_columns, fill_value=0)

        # Step 4: Run inference
        raw_prediction = model.predict(aligned_df)[0]
        prediction_val = int(raw_prediction)

        # Map binary prediction: 1 = Approved, 0 = Rejected
        status_label = "Approved" if prediction_val == 1 else "Rejected"

        # Step 5: Calculate prediction probability if the model supports it
        confidence = None
        confidence_percentage = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(aligned_df)[0]
            # Probability corresponding to the predicted class
            predicted_prob = float(probabilities[prediction_val])
            confidence = round(predicted_prob, 4)
            confidence_percentage = f"{round(predicted_prob * 100, 1)}%"

        outcome_msg = (
            f"Applicant profile meets loan approval criteria with {confidence_percentage} confidence."
            if prediction_val == 1
            else f"Applicant profile does not currently meet loan eligibility criteria with {confidence_percentage} confidence."
        )

        return PredictionResponse(
            prediction=prediction_val,
            status=status_label,
            confidence=confidence,
            confidence_percentage=confidence_percentage,
            message=outcome_msg
        )

    except Exception as err:
        # Return a friendly, clean JSON error message without exposing raw internal stack traces
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while evaluating the application: {str(err)}"
        )


@app.get("/sample-applicants")
async def get_sample_applicants():
    """
    Provides pre-verified real applicant profiles from the cleaned dataset
    to make live testing and manual verification fast and effortless.
    """
    return {
        "sample_rejected": {
            "name": "Sample Row 0 (Typically Rejected)",
            "data": {
                "Age": 40,
                "Gender": "Male",
                "Married": "No",
                "Dependents": "1",
                "Education": "Graduate",
                "Self_Employed": "No",
                "Applicant_Income": 477141.0,
                "Coapplicant_Income": 344017.0,
                "Existing_Debt": 89066.0,
                "Credit_Score": 600.0,
                "Loan_Amount": 265000.0,
                "Loan_Term": 240,
                "Property_Area": "Rural"
            }
        },
        "sample_approved": {
            "name": "Sample Row 2 (Typically Approved)",
            "data": {
                "Age": 44,
                "Gender": "Male",
                "Married": "Yes",
                "Dependents": "0",
                "Education": "Not Graduate",
                "Self_Employed": "No",
                "Applicant_Income": 1123636.0,
                "Coapplicant_Income": 166721.0,
                "Existing_Debt": 40668.0,
                "Credit_Score": 777.0,
                "Loan_Amount": 800000.0,
                "Loan_Term": 300,
                "Property_Area": "Urban"
            }
        },
        "sample_borderline": {
            "name": "Sample Row 7 (Typically Approved)",
            "data": {
                "Age": 39,
                "Gender": "Female",
                "Married": "No",
                "Dependents": "3+",
                "Education": "Graduate",
                "Self_Employed": "No",
                "Applicant_Income": 347806.0,
                "Coapplicant_Income": 333570.0,
                "Existing_Debt": 74834.0,
                "Credit_Score": 692.0,
                "Loan_Amount": 290000.0,
                "Loan_Term": 240,
                "Property_Area": "Urban"
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

# Loan Approval Prediction System

A production-ready, single-page web application powered by **FastAPI** and a trained **scikit-learn** model for predicting loan applicant approvals.

---

## 🎯 Overview

The system allows loan officers and applicants to input borrower details, submit them for automated evaluation, and receive an instant decision:
- **Loan Likely Approved** (Green card with confidence percentage)
- **Loan Likely Rejected** (Red card with confidence percentage)
- **Regulatory Disclaimer**: *"This is a machine learning prediction, not a final lending decision."*

---

## 🔬 Strict Preprocessing Fidelity

The application reproduces the exact training pipeline from the source notebook:
1. **Applicant_ID Dropped**: ID is excluded from model inputs and the form.
2. **Features**: 13 applicant input fields (7 numeric, 6 categorical).
3. **Encoding**: Categorical features are encoded via `pd.get_dummies(..., dtype=int)` on:
   - `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`.
4. **Exact Alignment**: The resulting DataFrame is reindexed against the exact 22 features stored in `model/model_columns.joblib` with `fill_value=0`.
5. **No Extra Scaling**: Numeric features are passed without modification, matching training logic.
6. **Target Mapping**: `1` = Approved, `0` = Rejected.

---

## 📁 Project Structure

```text
├── main.py                     # FastAPI backend application & API routes
├── requirements.txt            # Python dependencies
├── README.md                   # Setup guide and documentation
├── model/
│   ├── loan_model.joblib       # Trained scikit-learn RandomForest model
│   └── model_columns.joblib    # 22 exact training feature names in order
├── templates/
│   └── index.html              # Clean, accessible single-page HTML
├── static/
│   ├── css/
│   │   └── style.css           # Modern, responsive mobile/desktop styles
│   └── js/
│       └── app.js              # Client-side validation, fetch API & DOM rendering
├── cleaned_loan_data.csv       # Cleaned reference dataset
└── model_ready_loan_data.csv   # Model-ready reference dataset
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10+ (Verified on Python 3.15)
- `pip` package manager

### 2. Install Dependencies
Run in your project directory:
```bash
pip install -r requirements.txt
```
*(Or via the Python launcher on Windows):*
```powershell
py -m pip install -r requirements.txt
```

### 3. Run the Server
Start the FastAPI server with live reloading:
```bash
uvicorn main:app --reload
```
*(Or via Python module):*
```powershell
py -m uvicorn main:app --reload
```

The application will be accessible at:
- **Web Interface**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc Docs**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🧪 Verification & Test Results

The backend prediction logic was verified against real rows from `cleaned_loan_data.csv`:

| Test Row | Applicant Details | Expected / Ground Truth | Web App Model Output | Confidence | Match Status |
|:---:|:---|:---:|:---:|:---:|:---:|
| **Row 0** | Age: 40, Male, Single, 1 Dep, Graduate, Income: 477k, Coapplicant: 344k, Debt: 89k, Credit: 600, Loan: 265k, Term: 240, Rural | **Rejected** | **Rejected (0)** | **87.0%** | **PASS** (Exact Match) |
| **Row 2** | Age: 44, Male, Married, 0 Dep, Not Graduate, Income: 1.12M, Coapplicant: 166k, Debt: 40k, Credit: 777, Loan: 800k, Term: 300, Urban | **Approved** | **Approved (1)** | **98.0%** | **PASS** (Exact Match) |
| **Row 7** | Age: 39, Female, Single, 3+ Dep, Graduate, Income: 347k, Coapplicant: 333k, Debt: 74k, Credit: 692, Loan: 290k, Term: 240, Urban | **Approved** | **Approved (1)** | **88.0%** | **PASS** (Exact Match) |

---

## 🛡️ API Endpoints

### `POST /predict`
- **Request Body (JSON)**:
  ```json
  {
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
  ```
- **Response (JSON)**:
  ```json
  {
    "prediction": 1,
    "status": "Approved",
    "confidence": 0.98,
    "confidence_percentage": "98.0%",
    "message": "Applicant profile meets loan approval criteria with 98.0% confidence."
  }
  ```

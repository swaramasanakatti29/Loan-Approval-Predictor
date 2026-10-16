/**
 * Loan Approval Prediction System - Frontend JavaScript
 * Handles form submission, validation, API communication, and result card updates.
 */

document.addEventListener("DOMContentLoaded", () => {
    const loanForm = document.getElementById("loanForm");
    const predictBtn = document.getElementById("predictBtn");
    const btnSpinner = document.getElementById("btnSpinner");
    const btnText = document.getElementById("btnText");
    const resetBtn = document.getElementById("resetBtn");
    const errorAlert = document.getElementById("errorAlert");
    const errorMessage = document.getElementById("errorMessage");
    
    // Result card elements
    const resultCard = document.getElementById("resultCard");
    const resultStatus = document.getElementById("resultStatus");
    const resultIcon = document.getElementById("resultIcon");
    const resultMessage = document.getElementById("resultMessage");
    const confidenceContainer = document.getElementById("confidenceContainer");
    const confidenceValue = document.getElementById("confidenceValue");
    const confidenceBar = document.getElementById("confidenceBar");

    // Sample buttons
    const loadApprovedBtn = document.getElementById("loadApprovedBtn");
    const loadRejectedBtn = document.getElementById("loadRejectedBtn");

    // Real dataset verified profiles for quick testing
    const sampleApproved = {
        Age: 44,
        Gender: "Male",
        Married: "Yes",
        Dependents: "0",
        Education: "Not Graduate",
        Self_Employed: "No",
        Applicant_Income: 1123636.0,
        Coapplicant_Income: 166721.0,
        Existing_Debt: 40668.0,
        Credit_Score: 777.0,
        Loan_Amount: 800000.0,
        Loan_Term: "300",
        Property_Area: "Urban"
    };

    const sampleRejected = {
        Age: 40,
        Gender: "Male",
        Married: "No",
        Dependents: "1",
        Education: "Graduate",
        Self_Employed: "No",
        Applicant_Income: 477141.0,
        Coapplicant_Income: 344017.0,
        Existing_Debt: 89066.0,
        Credit_Score: 600.0,
        Loan_Amount: 265000.0,
        Loan_Term: "240",
        Property_Area: "Rural"
    };

    // Helper: Populate form from an object
    function populateForm(data) {
        hideError();
        hideResult();

        Object.keys(data).forEach(key => {
            const input = document.getElementById(key);
            if (input) {
                input.value = data[key];
                input.classList.remove("input-error");
            }
        });
    }

    // Quick Test button listeners
    if (loadApprovedBtn) {
        loadApprovedBtn.addEventListener("click", () => populateForm(sampleApproved));
    }

    if (loadRejectedBtn) {
        loadRejectedBtn.addEventListener("click", () => populateForm(sampleRejected));
    }

    // Helper: Show error alert
    function showError(msg) {
        errorMessage.textContent = msg;
        errorAlert.style.display = "flex";
        errorAlert.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // Helper: Hide error alert
    function hideError() {
        errorAlert.style.display = "none";
        errorMessage.textContent = "";
    }

    // Helper: Hide result card
    function hideResult() {
        resultCard.style.display = "none";
        resultCard.className = "result-card";
    }

    // Helper: Set loading state
    function setLoading(isLoading) {
        if (isLoading) {
            predictBtn.disabled = true;
            predictBtn.classList.add("loading");
            btnText.textContent = "Predicting Approval...";
        } else {
            predictBtn.disabled = false;
            predictBtn.classList.remove("loading");
            btnText.textContent = "Predict Loan Status";
        }
    }

    // Reset button handler
    resetBtn.addEventListener("click", () => {
        loanForm.reset();
        hideError();
        hideResult();

        // Clear input error highlights
        loanForm.querySelectorAll(".input-error").forEach(el => el.classList.remove("input-error"));
    });

    // Form Submission Handler
    loanForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        hideError();
        hideResult();

        // Collect and validate all form fields
        const formData = new FormData(loanForm);
        const rawPayload = {};
        let hasValidationError = false;
        let firstInvalidField = null;

        // Required fields list
        const requiredFields = [
            "Age", "Gender", "Married", "Dependents", "Education",
            "Self_Employed", "Applicant_Income", "Coapplicant_Income",
            "Existing_Debt", "Credit_Score", "Loan_Amount",
            "Loan_Term", "Property_Area"
        ];

        // Numeric fields list
        const numericFields = [
            "Age", "Applicant_Income", "Coapplicant_Income",
            "Existing_Debt", "Credit_Score", "Loan_Amount", "Loan_Term"
        ];

        for (const field of requiredFields) {
            const inputEl = document.getElementById(field);
            const val = formData.get(field);

            if (!val || val.toString().trim() === "") {
                inputEl.classList.add("input-error");
                hasValidationError = true;
                if (!firstInvalidField) firstInvalidField = inputEl;
            } else {
                inputEl.classList.remove("input-error");
                if (numericFields.includes(field)) {
                    rawPayload[field] = parseFloat(val);
                } else {
                    rawPayload[field] = val.toString().trim();
                }
            }
        }

        if (hasValidationError) {
            showError("Please complete all required fields with valid values.");
            if (firstInvalidField) {
                firstInvalidField.focus();
            }
            return;
        }

        // Send POST request to FastAPI backend
        setLoading(true);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(rawPayload)
            });

            const data = await response.json();

            if (!response.ok) {
                // If FastAPI returned an HTTP error detail
                const errDetail = data.detail || "Prediction request could not be processed.";
                showError(errDetail);
                setLoading(false);
                return;
            }

            // Render prediction result
            renderResult(data);

        } catch (err) {
            console.error("Prediction API Error:", err);
            showError("Unable to reach the prediction service. Please ensure the backend server is running.");
        } finally {
            setLoading(false);
        }
    });

    // Function to render result card
    function renderResult(data) {
        const isApproved = data.prediction === 1;

        // Reset styling classes
        resultCard.className = "result-card " + (isApproved ? "approved" : "rejected");
        
        // Update badge text and icon
        if (isApproved) {
            resultIcon.textContent = "✓";
            resultStatus.textContent = "Loan Likely Approved";
        } else {
            resultIcon.textContent = "✕";
            resultStatus.textContent = "Loan Likely Rejected";
        }

        // Update explanation message
        resultMessage.textContent = data.message;

        // Confidence meter handling
        if (data.confidence !== null && data.confidence !== undefined) {
            confidenceContainer.style.display = "block";
            const percentStr = data.confidence_percentage || `${Math.round(data.confidence * 100)}%`;
            confidenceValue.textContent = percentStr;
            const percentNum = Math.round(data.confidence * 100);
            confidenceBar.style.width = `${percentNum}%`;
        } else {
            confidenceContainer.style.display = "none";
        }

        // Display card and smoothly scroll into view
        resultCard.style.display = "block";
        resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
});

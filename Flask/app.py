from flask import Flask, render_template, request
import pickle
import os
import numpy as np

app = Flask(__name__)

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "loan_model.pkl")
model = pickle.load(open(model_path, "rb"))

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/form')
def form():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    name = request.form["Name"]
    email = request.form["Email"]
    phone = request.form["Phone"]

    gender = request.form["Gender"]
    married = request.form["Married"]
    dependents = request.form["Dependents"]
    education = request.form["Education"]
    self_employed = request.form["Self_Employed"]

    applicant_income = request.form["ApplicantIncome"]
    coapplicant_income = request.form["CoapplicantIncome"]

    loan_amount = request.form["LoanAmount"]
    loan_amount_term = request.form["Loan_Amount_Term"]

    credit_history = request.form["Credit_History"]
    property_area = request.form["Property_Area"]

    features = np.array([[
        1 if gender == "Male" else 0,
        1 if married == "Yes" else 0,
        int(dependents),
        1 if education == "Graduate" else 0,
        1 if self_employed == "Yes" else 0,
        float(applicant_income),
        float(coapplicant_income),
        float(loan_amount),
        float(loan_amount_term),
        int(credit_history),
        {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]
    ]])

    prediction = model.predict(features)[0]

    return render_template(
        "result.html",
        prediction=prediction,
        name=name,
        email=email,
        phone=phone,
        gender=gender,
        married=married,
        dependents=dependents,
        education=education,
        self_employed=self_employed,
        applicant_income=applicant_income,
        coapplicant_income=coapplicant_income,
        loan_amount=loan_amount,
        loan_amount_term=loan_amount_term,
        credit_history=credit_history,
        property_area=property_area
    )


if __name__ == "__main__":
    app.run(debug=True)
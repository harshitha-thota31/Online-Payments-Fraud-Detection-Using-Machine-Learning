from flask import Flask, render_template, request
import numpy as np
import pickle

# load trained model
model = pickle.load(open(r"C:\Users\Hp\OneDrive\Desktop\smart_project\payments.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # get values in correct order
    step = float(request.form["step"])
    amount = float(request.form["amount"])
    oldbalanceOrg = float(request.form["oldbalanceOrg"])
    newbalanceOrig = float(request.form["newbalanceOrig"])
    oldbalanceDest = float(request.form["oldbalanceDest"])
    newbalanceDest = float(request.form["newbalanceDest"])

    # apply SAME preprocessing as training
    amount = np.log(amount)

    features = np.array([[step, amount, oldbalanceOrg,
                          newbalanceOrig, oldbalanceDest, newbalanceDest]])

    prediction = model.predict(features)[0]

    result = "Fraud Transaction 🚨" if prediction == 1 else "Legitimate Transaction ✅"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)

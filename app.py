from flask import Flask, render_template, request
import joblib
import numpy as np 

app = Flask (__name__)

#load the trained model
model = joblib.load("house_price_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bedrooms =float(request.form["bedrooms"])
    age = float(request.form["age"])

    prediction = model.predict([[area, bedrooms, age]])

    return render_template(
        "index.html",
        prediction_text=f"🏠Predicted House Price: Rs {prediction:,.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)
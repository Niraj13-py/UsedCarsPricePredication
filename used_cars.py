# from flask import Flask, render_template, request
# import pickle
# import pandas as pd

# app = Flask(__name__)

# # Load the trained model
# model = pickle.load(open("car_price_model.pkl", "rb"))

# # Load label encoders (if you saved mappings for Brand & model)
# # If not, map them manually here like dictionaries

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     if request.method == "POST":
#         # Get form data
#         brand = request.form["Brand"]
#         model_name = request.form["model"]
#         year = int(request.form["Year"])
#         km_driven = int(request.form["kmDriven"])
#         transmission = 1 if request.form["Transmission"] == "Automatic" else 0
#         fuel_type = {"Petrol": 0, "Diesel": 1, "CNG": 2, "LPG": 3, "Electric": 4}[request.form["FuelType"]]
#         owner = int(request.form["Owner"])

#         # Encode Brand and model if necessary
#         # Example: brand = brand_mapping[brand] (if you have mappings saved)

#         # Prepare data for prediction
#         input_data = pd.DataFrame([[brand, model_name, year, km_driven, transmission, fuel_type, owner]],
#                                   columns=['Brand', 'model', 'Year', 'kmDriven', 'Transmission', 'FuelType', 'Owner'])

#         # Predict price
#         predicted_price = model.predict(input_data)[0]

#         return render_template("index.html", prediction_text=f"Estimated Price: ₹ {int(predicted_price):,}")

# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open("car_price_model.pkl", "rb"))

# Load dataset for dynamic dropdowns
df = pd.read_csv("used_cars_dataset_v2.csv")  # Replace with your dataset path

# Get unique brands and their models
brands = sorted(df['Brand'].unique())
brand_model_mapping = df.groupby('Brand')['model'].unique().apply(list).to_dict()

# Load encoding mappings used during training
brand_mapping = {brand: idx for idx, brand in enumerate(df['Brand'].astype('category').cat.categories)}
model_mapping = {model: idx for idx, model in enumerate(df['model'].astype('category').cat.categories)}

@app.route("/")
def home():
    return render_template("index.html", brands=brands, brand_model_mapping=brand_model_mapping)

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    brand = request.form["Brand"]
    model_name = request.form["model"]
    year = int(request.form["Year"])
    km_driven = int(request.form["kmDriven"])
    transmission = 1 if request.form["Transmission"] == "Automatic" else 0
    fuel_type = {"Petrol": 0, "Diesel": 1, "CNG": 2, "LPG": 3, "Electric": 4}[request.form["FuelType"]]
    owner = int(request.form["Owner"])

    # Encode Brand and Model
    brand_encoded = brand_mapping.get(brand, -1)
    model_encoded = model_mapping.get(model_name, -1)

    # Prepare input
    input_data = pd.DataFrame([[brand_encoded, model_encoded, year, km_driven, transmission, fuel_type, owner]],
                              columns=['Brand', 'model', 'Year', 'kmDriven', 'Transmission', 'FuelType', 'Owner'])

    # Predict
    predicted_price = model.predict(input_data)[0]
    predicted_price = round(predicted_price, 2)

    return render_template("index.html", brands=brands, brand_model_mapping=brand_model_mapping,
                           prediction_text=f"Estimated Price: ₹ {int(predicted_price):,}")

@app.route("/get_models/<brand>")
def get_models(brand):
    models = brand_model_mapping.get(brand, [])
    return jsonify(models)

if __name__ == "__main__":
    app.run(debug=True)

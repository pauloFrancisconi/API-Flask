from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "model/fipe.pkl"
ENCODER_PATH = "model/encoder.pkl"
ENCODER_BRAND_PATH = "model/encoder_brand.pkl"
ENCODER_ENGINE_SIZE_PATH  = "model/encoder_engine_size.pkl"
ENCODER_FUEL_PATH  = "model/encoder_fuel.pkl"
ENCODER_GEAR_PATH  = "model/encoder_gear.pkl"
ENCODER_MODEL_PATH  = "model/encoder_model.pkl"
ENCODER_YEAR_MODEL_PATH  = "model/encoder_year_model.pkl"

# Carregar o modelo e o encoder
model = joblib.load(MODEL_PATH)
encoder_brand       = joblib.load(ENCODER_BRAND_PATH)
encoder_engine_size = joblib.load(ENCODER_ENGINE_SIZE_PATH) 
encoder_fuel        = joblib.load(ENCODER_FUEL_PATH) 
encoder_gear        = joblib.load(ENCODER_GEAR_PATH) 
encoder_model       = joblib.load(ENCODER_MODEL_PATH) 
encoder_year_model  = joblib.load(ENCODER_YEAR_MODEL_PATH) 

# Lista de valores válidos para ano de referência
valid_years = ['2020', '2021']

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API de Previsão de Preços de Carros. Use POST em /predict para enviar dados."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receber os dados JSON
        data = request.get_json()

        # Campos obrigatórios
        required_fields = [
            "brand",
            "model",
            "fuel",
            "gear",
            "engine_size",
            "year_model"
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Campo '{field}' é obrigatório"}), 400

        # Validação do ano de referência
        # if str(data["year_of_reference"]) not in valid_years:
        #     return jsonify({"error": "Ano de referência inválido. Valores válidos são: 2020, 2021."}), 400

        # Certificar-se de que os dados são transformados corretamente
        enc_brand       = encoder_brand      .transform((np.array(data["brand"]).reshape(-1, 1)))
        enc_fuel        = encoder_fuel       .transform((np.array(data["fuel"]).reshape(-1, 1)))
        enc_gear        = encoder_gear       .transform((np.array(data["gear"]).reshape(-1, 1)))
        enc_model       = encoder_model      .transform((np.array(data["model"]).reshape(-1, 1)))
        #enc_engine_size = encoder_engine_size.transform((np.array(data["engine_size"]).reshape(-1, 1)))
        #enc_year_model  = encoder_year_model .transform((np.array(data["year_model"]).reshape(-1, 1)))
        dados = {
            "brand": [enc_brand],
            "model": [enc_model],
            "fuel": [enc_fuel],
            "gear": [enc_gear],
            "engine_size": [data["engine_size"]],
            "year_model": [data["year_model"]],
        }
        print(dados)
        df = pd.DataFrame(dados)
        prediction = model.predict(df)[0]

        return jsonify({"Preco": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

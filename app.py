from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

MODEL_PATH = "model/fipe.pkl"
ENCODER_PATH = "model/encoder.pkl"

# Carregar o modelo e o encoder
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

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
            "year_of_reference",
            "month_of_reference",
            "fipe_code",
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
        if str(data["year_of_reference"]) not in valid_years:
            return jsonify({"error": "Ano de referência inválido. Valores válidos são: 2020, 2021."}), 400

        # Pré-processar as entradas
        features = [
            data["year_of_reference"],
            data["month_of_reference"],
            data["fipe_code"],
            data["brand"],
            data["model"],
            data["fuel"],
            data["gear"],
            data["engine_size"],
            data["year_model"]
        ]

        # Certificar-se de que os dados são transformados corretamente
        encoded_features = encoder.transform([features]).tolist()[0]

        # Fazer a previsão
        prediction = model.predict([encoded_features])[0]

        return jsonify({"Preco": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

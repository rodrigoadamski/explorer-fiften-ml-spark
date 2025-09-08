from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Carregar modelo e encoder
model = joblib.load("modelo_produtos.pkl")
encoder = joblib.load("encoder_produtos.pkl")

@app.post("/predict")
def predict(data: dict):
    # Transformar input em DataFrame
    df_input = pd.DataFrame([data])
    
    # One-hot encoding da coluna produto
    produto_encoded = encoder.transform(df_input[['produto']])
    produto_df = pd.DataFrame(produto_encoded, columns=encoder.get_feature_names_out(['produto']))
    
    # Concatenar com outras features
    df_features = pd.concat([df_input[['popularidade','quantidade']], produto_df], axis=1)
    
    # Prever
    preco_previsto = model.predict(df_features)[0]
    
    return {"preco_previsto": round(preco_previsto, 2)}
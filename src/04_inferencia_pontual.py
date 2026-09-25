# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Inferência pontual (Champion) + MLflow Tracing
# MAGIC
# MAGIC Escora contratos usando a versão `@champion` e instrumenta cada scoring com
# MAGIC **`@mlflow.trace`** — os traces (entrada, saída, latência, atributos) ficam em
# MAGIC *Experiments > `inferencia_tracing` > Traces*. Tracing funciona em notebook/job,
# MAGIC não só em Model Serving; ele só precisa de um **experimento de destino**.

# COMMAND ----------

dbutils.widgets.text("catalog", "bjd_dev")
dbutils.widgets.text("schema", "mlops_credito")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODELO = f"{CATALOG}.{SCHEMA}.modelo_inadimplencia"

# COMMAND ----------

import mlflow
import pandas as pd

mlflow.set_registry_uri("databricks-uc")
# Tracing precisa de um experimento de destino (os traces ficam sob ele).
_user = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{_user}/bjd-mlops-demo/inferencia_tracing")

model_uri = f"models:/{MODELO}@champion"
sk_model = mlflow.sklearn.load_model(model_uri)  # p/ probabilidade (predict_proba)

# schema esperado pela signature (nome -> tipo mlflow)
schema_inputs = mlflow.models.get_model_info(model_uri).signature.inputs.inputs
input_cols = [c.name for c in schema_inputs]
mltype = {c.name: str(c.type) for c in schema_inputs}

FEATURES_NUM = [
    "valor_financiado", "valor_entrada", "prazo_meses", "taxa_juros", "renda_anual",
    "score_credito", "idade_cliente", "safra", "utilizacao_credito",
    "num_contratos_anteriores", "comprometimento_renda",
]
FEATURES_CAT = ["regiao", "tipo_equipamento"]

# COMMAND ----------

# Função de scoring instrumentada com tracing. Cada chamada gera 1 trace com a
# entrada (contrato), a saída (classe + prob), latência e os atributos passados.
@mlflow.trace(name="scoring_inadimplencia", attributes={"modelo": MODELO})
def scorar(contrato: dict) -> dict:
    Xi = pd.get_dummies(
        pd.DataFrame([contrato])[FEATURES_NUM + FEATURES_CAT], columns=FEATURES_CAT
    )
    for c in input_cols:               # colunas one-hot ausentes (boolean)
        if c not in Xi.columns:
            Xi[c] = False
    Xi = Xi[input_cols]
    for c in input_cols:               # coerção p/ casar com a signature
        t = mltype[c]
        if t == "double":
            Xi[c] = Xi[c].astype("float64")
        elif t == "long":
            Xi[c] = Xi[c].astype("int64")
        elif t == "boolean":
            Xi[c] = Xi[c].astype("bool")
    prob = float(sk_model.predict_proba(Xi)[:, 1][0])
    return {"inadimplente": int(prob >= 0.5), "prob": round(prob, 4)}

# COMMAND ----------

# Escora alguns contratos REAIS da tabela — cada chamada vira um trace.
amostra = spark.table(f"{CATALOG}.{SCHEMA}.features_contratos").limit(5).toPandas()
for _, row in amostra.iterrows():
    resultado = scorar(row[FEATURES_NUM + FEATURES_CAT].to_dict())
    print(f"contrato {int(row['id_contrato'])}: {resultado}")

# COMMAND ----------

# MAGIC %md
# MAGIC Veja os traces em **Experiments → `inferencia_tracing` → aba Traces**. No
# MAGIC **Model Serving**, o mesmo tracing pode ser capturado automaticamente ao habilitar
# MAGIC tracing/inference tables no endpoint (sem decorar manualmente).

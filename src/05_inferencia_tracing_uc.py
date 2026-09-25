# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Inferência pontual + Tracing no Unity Catalog
# MAGIC
# MAGIC Igual ao notebook 04 (inferência pontual instrumentada com `@mlflow.trace`), mas
# MAGIC os **traces são gravados no Unity Catalog** em vez de ficarem só no experimento
# MAGIC do workspace. Segue o padrão do MLflow 3 (traces → tabelas Delta no UC).
# MAGIC
# MAGIC Requisitos: `databricks-agents>=1.10.1`, DBR 15.3+/serverless, e permissões no
# MAGIC schema de destino (`USE CATALOG`, `USE SCHEMA`, `CREATE TABLE`, `MODIFY`, `SELECT`).
# MAGIC Ref.: docs.databricks.com/.../mlflow3/genai/tracing/migrate-to-uc

# COMMAND ----------

# MAGIC %pip install -U -q "databricks-agents>=1.10.1" "mlflow[databricks]>=3.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "bjd_dev")
dbutils.widgets.text("schema", "mlops_credito")
dbutils.widgets.text("trace_prefix", "traces_inferencia")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
TRACE_PREFIX = dbutils.widgets.get("trace_prefix")
MODELO = f"{CATALOG}.{SCHEMA}.modelo_inadimplencia"

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow.entities.trace_location import UnityCatalog

mlflow.set_registry_uri("databricks-uc")
_user = spark.sql("SELECT current_user()").collect()[0][0]

# O destino de trace no UC só pode ser LINKADO a um experimento que ainda NÃO contém
# traces. Logo: linka só na criação (experimento novo/vazio); em re-runs, apenas reusa
# o experimento — o vínculo com o UC persiste. Se este nome já tiver traces "legacy",
# troque-o por um novo (ex.: _v2).
# Cria as tabelas Delta: {CATALOG}.{SCHEMA}.{TRACE_PREFIX}_otel_spans, ..._otel_logs, etc.
EXP = f"/Users/{_user}/bjd-mlops-demo/inferencia_tracing_uc"
if mlflow.get_experiment_by_name(EXP) is None:
    mlflow.set_experiment(
        experiment_name=EXP,
        trace_location=UnityCatalog(
            catalog_name=CATALOG,
            schema_name=SCHEMA,
            table_prefix=TRACE_PREFIX,
        ),
    )
    print(f"Experimento criado e linkado ao UC: {EXP}")
else:
    mlflow.set_experiment(experiment_name=EXP)  # já linkado ao UC → reusa
    print(f"Reusando experimento já linkado ao UC: {EXP}")

# COMMAND ----------

model_uri = f"models:/{MODELO}@champion"
sk_model = mlflow.sklearn.load_model(model_uri)  # p/ probabilidade

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

# Cada chamada gera 1 trace, agora persistido em tabelas Delta no Unity Catalog.
@mlflow.trace(name="scoring_inadimplencia_uc", attributes={"modelo": MODELO})
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

amostra = spark.table(f"{CATALOG}.{SCHEMA}.features_contratos").limit(5).toPandas()
for _, row in amostra.iterrows():
    resultado = scorar(row[FEATURES_NUM + FEATURES_CAT].to_dict())
    print(f"contrato {int(row['id_contrato'])}: {resultado}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Onde ficam os traces
# MAGIC Além da aba **Traces** do experimento, agora existem tabelas Delta governadas no
# MAGIC UC (prefixo `traces_inferencia`), consultáveis via SQL:

# COMMAND ----------

# lista as tabelas de trace criadas no schema
display(
    spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}")
    .filter("tableName LIKE 'traces_inferencia%'")
)

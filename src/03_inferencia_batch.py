# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Inferência batch (Champion) com `spark_udf`
# MAGIC
# MAGIC Carrega a versão `@champion` como **UDF Spark** (inferência distribuída) e grava a
# MAGIC tabela de predições.
# MAGIC
# MAGIC > Obs.: no **serverless**, o `spark_udf` pode ocasionalmente derrubar a sessão
# MAGIC > ("Spark session is no longer usable"). Se ocorrer, clique em **New Session** e
# MAGIC > rode a célula do `spark_udf` sozinha. Há um fallback em pandas comentado no fim.

# COMMAND ----------

dbutils.widgets.text("catalog", "bjd_dev")
dbutils.widgets.text("schema", "mlops_credito")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
TABELA = f"{CATALOG}.{SCHEMA}.features_contratos"
MODELO = f"{CATALOG}.{SCHEMA}.modelo_inadimplencia"
PREDICOES = f"{CATALOG}.{SCHEMA}.predicoes_inadimplencia"

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")
# NÃO seta experimento: inferência só CONSOME o modelo do Model Registry (UC).
client = MlflowClient()
champ_ver = client.get_model_version_by_alias(MODELO, "champion").version
print(f"Usando @champion = v{champ_ver}")

# COMMAND ----------

# Recria as MESMAS features do treino (one-hot) e alinha à signature do modelo.
FEATURES_NUM = [
    "valor_financiado", "valor_entrada", "prazo_meses", "taxa_juros", "renda_anual",
    "score_credito", "idade_cliente", "safra", "utilizacao_credito",
    "num_contratos_anteriores", "comprometimento_renda",
]
FEATURES_CAT = ["regiao", "tipo_equipamento"]

pdf = spark.table(TABELA).toPandas()
X = pd.get_dummies(pdf[FEATURES_NUM + FEATURES_CAT], columns=FEATURES_CAT)

model_uri = f"models:/{MODELO}@champion"
schema_inputs = mlflow.models.get_model_info(model_uri).signature.inputs.inputs
input_cols = [c.name for c in schema_inputs]
mltype = {c.name: str(c.type) for c in schema_inputs}

for c in input_cols:                 # colunas one-hot ausentes (tipo boolean)
    if c not in X.columns:
        X[c] = False
X = X[input_cols]
for c in input_cols:                 # coerção p/ casar com a signature (evita erro de schema)
    t = mltype[c]
    if t == "double":
        X[c] = X[c].astype("float64")
    elif t == "long":
        X[c] = X[c].astype("int64")
    elif t == "boolean":
        X[c] = X[c].astype("bool")

sdf_feats = spark.createDataFrame(pd.concat([pdf[["id_contrato"]], X], axis=1))

# COMMAND ----------

# Inferência DISTRIBUÍDA: o modelo vira uma UDF Spark aplicada em todos os executors.
predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="double")
resultado = (
    sdf_feats
    .withColumn("predicao_inadimplente", predict_udf(F.struct(*input_cols)).cast("int"))
    .select("id_contrato", "predicao_inadimplente")
    .withColumn("modelo", F.lit(MODELO))
    .withColumn("versao_modelo", F.lit(str(champ_ver)))
    .withColumn("scored_at", F.current_timestamp())
)

(
    resultado.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(PREDICOES)
)
print(f"Gravado: {PREDICOES}")
display(spark.table(PREDICOES).limit(10))

# COMMAND ----------

display(
    spark.table(PREDICOES)
    .groupBy("predicao_inadimplente")
    .count()
    .orderBy("predicao_inadimplente")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fallback: inferência em pandas no driver
# MAGIC Se o `spark_udf` derrubar a sessão serverless, use esta alternativa (escora no
# MAGIC driver; ok para este volume) — descomente e rode:

# COMMAND ----------

# sk_model = mlflow.sklearn.load_model(model_uri)
# proba = sk_model.predict_proba(X)[:, 1]
# out = pdf[["id_contrato"]].copy()
# out["predicao_inadimplente"] = (proba >= 0.5).astype(int)
# out["prob_inadimplencia"] = proba.round(4)
# out["modelo"] = MODELO
# out["versao_modelo"] = str(champ_ver)
# (
#     spark.createDataFrame(out).withColumn("scored_at", F.current_timestamp())
#     .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(PREDICOES)
# )
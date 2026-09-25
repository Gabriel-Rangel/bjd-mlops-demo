# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Tracing no UC + payload grande em Volume (padrão governado)
# MAGIC
# MAGIC Resolve a limitação de tamanho de payload de forma governada: cada scoring é
# MAGIC instrumentado com **`@mlflow.trace`** (traces no **Unity Catalog**), mas o
# MAGIC **payload completo** (request + response) é gravado num **Volume do UC** e no
# MAGIC span fica apenas a **referência (URI) + metadados** (tamanho, id, versão).
# MAGIC Assim o trace fica leve e o payload — de qualquer tamanho — fica versionado e
# MAGIC governado, sem esbarrar em limite (é o padrão recomendado no discovery da PoC).

# COMMAND ----------

# MAGIC %pip install -U -q "databricks-agents>=1.10.1" "mlflow[databricks]>=3.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "bjd_dev")
dbutils.widgets.text("schema", "mlops_credito")
dbutils.widgets.text("trace_prefix", "traces_payload")
dbutils.widgets.text("volume", "payloads_inferencia")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
TRACE_PREFIX = dbutils.widgets.get("trace_prefix")
VOLUME = dbutils.widgets.get("volume")
MODELO = f"{CATALOG}.{SCHEMA}.modelo_inadimplencia"

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow.entities.trace_location import UnityCatalog

mlflow.set_registry_uri("databricks-uc")
_user = spark.sql("SELECT current_user()").collect()[0][0]

# Destino dos traces no UC (linka só na criação; reusa em re-runs — ver notebook 05)
EXP = f"/Users/{_user}/bjd-mlops-demo/tracing_payload_uc"
if mlflow.get_experiment_by_name(EXP) is None:
    mlflow.set_experiment(
        experiment_name=EXP,
        trace_location=UnityCatalog(
            catalog_name=CATALOG, schema_name=SCHEMA, table_prefix=TRACE_PREFIX
        ),
    )
    print(f"Experimento criado e linkado ao UC: {EXP}")
else:
    mlflow.set_experiment(experiment_name=EXP)
    print(f"Reusando experimento já linkado ao UC: {EXP}")

# Volume gerenciado do UC para guardar os payloads completos
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
print(f"Volume: {VOLUME_PATH}")

# COMMAND ----------

from mlflow.tracking import MlflowClient

model_uri = f"models:/{MODELO}@champion"
sk_model = mlflow.sklearn.load_model(model_uri)
champ_ver = MlflowClient().get_model_version_by_alias(MODELO, "champion").version

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

import json
from uuid import uuid4


def _preparar(contrato: dict) -> pd.DataFrame:
    Xi = pd.get_dummies(pd.DataFrame([contrato])[FEATURES_NUM + FEATURES_CAT], columns=FEATURES_CAT)
    for c in input_cols:
        if c not in Xi.columns:
            Xi[c] = False
    Xi = Xi[input_cols]
    for c in input_cols:
        t = mltype[c]
        Xi[c] = Xi[c].astype("float64" if t == "double" else "int64" if t == "long" else "bool")
    return Xi


@mlflow.trace(name="scoring_payload_governado", attributes={"modelo": MODELO})
def scorar(id_contrato: int, contrato: dict) -> dict:
    prob = float(sk_model.predict_proba(_preparar(contrato))[:, 1][0])
    resposta = {"inadimplente": int(prob >= 0.5), "prob": round(prob, 4)}

    # payload COMPLETO (pode ser grande) -> gravado no Volume UC
    payload = {"id_contrato": id_contrato, "request": contrato, "response": resposta,
               "modelo": MODELO, "versao_modelo": str(champ_ver)}
    blob = json.dumps(payload, default=str).encode("utf-8")
    uri = f"{VOLUME_PATH}/{id_contrato}_{uuid4().hex}.json"
    with open(uri, "wb") as f:
        f.write(blob)

    # no span fica só a REFERÊNCIA + metadados (nada de payload grande no trace)
    span = mlflow.get_current_active_span()
    if span is not None:
        span.set_attributes({
            "payload_uri": uri,
            "payload_bytes": len(blob),
            "id_contrato": id_contrato,
            "versao_modelo": str(champ_ver),
        })
    return resposta

# COMMAND ----------

amostra = spark.table(f"{CATALOG}.{SCHEMA}.features_contratos").limit(5).toPandas()
for _, row in amostra.iterrows():
    r = scorar(int(row["id_contrato"]), row[FEATURES_NUM + FEATURES_CAT].to_dict())
    print(f"contrato {int(row['id_contrato'])}: {r}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conferindo o padrão
# MAGIC - Os **traces** (com o atributo `payload_uri`) estão em *Experiments → `tracing_payload_uc` → Traces*.
# MAGIC - Os **payloads completos** ficam no Volume do UC:

# COMMAND ----------

display(dbutils.fs.ls(VOLUME_PATH))

# COMMAND ----------

# lê de volta um payload pela URI (reconstrução completa a partir do Volume)
import json
arquivos = [f.path for f in dbutils.fs.ls(VOLUME_PATH) if f.path.endswith(".json")]
if arquivos:
    caminho_local = arquivos[0].replace("dbfs:", "")
    with open(caminho_local) as f:
        print(json.dumps(json.load(f), indent=2, ensure_ascii=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auditoria ponta a ponta: trace (Delta no UC, via SQL) ↔ payload (Volume)
# MAGIC O 06 usa tracing **próprio** (experimento `tracing_payload_uc`, prefixo
# MAGIC `traces_payload`) — separado do notebook 05. Os spans viram **tabelas Delta
# MAGIC governadas**, consultáveis por SQL; cada span carrega a `payload_uri` que aponta
# MAGIC para o payload completo no Volume.

# COMMAND ----------

# 1) Lado do TRACE: a tabela de spans é Delta governada e consultável por SQL.
SPANS = f"{CATALOG}.{SCHEMA}.{TRACE_PREFIX}_otel_spans"
print("Tabela de spans:", SPANS)
spark.table(SPANS).printSchema()   # confira o tipo real da coluna de atributos
display(spark.sql(f"SELECT COUNT(*) AS n_spans FROM {SPANS}"))

# COMMAND ----------

from pyspark.sql import functions as F

# 2) Extrai a referência (payload_uri) dos spans do nosso scoring.
#    `attributes` costuma ser map<string,string>. Se no seu schema for JSON-string,
#    troque por: F.get_json_object("attributes", "$.payload_uri").
refs = (
    spark.table(SPANS)
    .where(F.col("name") == "scoring_payload_governado")
    .select(
        F.col("attributes")["payload_uri"].alias("payload_uri"),
        F.col("attributes")["id_contrato"].alias("id_contrato_trace"),
        F.col("attributes")["payload_bytes"].alias("payload_bytes"),
    )
    .withColumn("payload_uri", F.regexp_replace("payload_uri", "^[a-z]+:", ""))  # tira scheme
)
display(refs)

# COMMAND ----------

# 3) JOIN trace ↔ payload: lê os payloads do Volume e cruza pela URI.
payloads = (
    spark.read.json(VOLUME_PATH)
    .withColumn("payload_uri", F.regexp_replace(F.input_file_name(), "^[a-z]+:", ""))
)
auditoria = refs.join(payloads, "payload_uri", "left")
display(
    auditoria.select(
        "id_contrato", "versao_modelo", "payload_bytes",
        F.col("response.inadimplente").alias("inadimplente"),
        F.col("response.prob").alias("prob"),
        "payload_uri",
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC Se a coluna `attributes` não for `map<string,string>` (veja o `printSchema` acima),
# MAGIC ajuste a extração do passo 2. A reconstrução direto do Volume (célula de leitura
# MAGIC acima) é sempre a fonte confiável do payload completo.

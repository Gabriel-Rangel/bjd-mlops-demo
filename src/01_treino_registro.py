# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Treino + registro no Unity Catalog (Champion / Challenger)
# MAGIC
# MAGIC Treina dois `RandomForestClassifier` (v1 = champion, v2 = challenger), registra
# MAGIC ambos no **Models in Unity Catalog** e atribui os **aliases** `@champion` e
# MAGIC `@challenger`. Segue o padrão de ciclo de vida do UC (namespace de 3 níveis,
# MAGIC versionamento automático, aliases mutáveis).

# COMMAND ----------

dbutils.widgets.text("catalog", "bjd_dev")
dbutils.widgets.text("schema", "mlops_credito")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
TABELA = f"{CATALOG}.{SCHEMA}.features_contratos"
MODELO = f"{CATALOG}.{SCHEMA}.modelo_inadimplencia"
print(f"features={TABELA}\nmodelo={MODELO}")

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

mlflow.set_registry_uri("databricks-uc")  # registra no Unity Catalog
# Experimento ÚNICO do projeto: sem isto, cada notebook (e cada caminho de notebook)
# cria seu próprio "experimento do notebook". Assim todos os runs caem no mesmo lugar.
_user = spark.sql("SELECT current_user()").collect()[0][0]
# Em prod o job roda como service principal: current_user() é o Application ID (UUID)
# e a pasta /Users/<uuid>/bjd-mlops-demo NÃO existe. set_experiment não cria diretórios
# intermediários, então garantimos o diretório-pai antes (mkdirs é idempotente). Assim
# funciona tanto interativamente (seu e-mail) quanto no CI (SP).
from databricks.sdk import WorkspaceClient
_exp_dir = f"/Users/{_user}/bjd-mlops-demo"
WorkspaceClient().workspace.mkdirs(_exp_dir)
mlflow.set_experiment(f"{_exp_dir}/mlflow_experimento")
client = MlflowClient()

# COMMAND ----------

FEATURES_NUM = [
    "valor_financiado", "valor_entrada", "prazo_meses", "taxa_juros", "renda_anual",
    "score_credito", "idade_cliente", "safra", "utilizacao_credito",
    "num_contratos_anteriores", "comprometimento_renda",
]
FEATURES_CAT = ["regiao", "tipo_equipamento"]
TARGET = "inadimplente"

pdf = spark.table(TABELA).toPandas()
# one-hot simples nas categóricas (mantém o exemplo direto, sem Pipeline)
X = pd.get_dummies(pdf[FEATURES_NUM + FEATURES_CAT], columns=FEATURES_CAT)
y = pdf[TARGET]
FEATURE_COLS = X.columns.tolist()

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"treino={len(X_tr)}  teste={len(X_te)}  #features={X.shape[1]}")

# COMMAND ----------

def avalia(model, X_te, y_te):
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_te, proba),
        "accuracy": accuracy_score(y_te, pred),
        "f1": f1_score(y_te, pred),
    }

def treina_e_registra(run_name, params):
    with mlflow.start_run(run_name=run_name) as run:
        # class_weight balanceado: dados desbalanceados (~25% inadimplência)
        model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced", **params)
        model.fit(X_tr, y_tr)
        metrics = avalia(model, X_te, y_te)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        signature = infer_signature(X_te, model.predict(X_te))
        info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_te.head(3),
            registered_model_name=MODELO,
        )
        print(f"{run_name}: {metrics}  -> version {info.registered_model_version}")
        return int(info.registered_model_version), metrics

# COMMAND ----------

# v1 (candidato a champion) e v2 (challenger)
v1, m1 = treina_e_registra("champion_rf_100", {"n_estimators": 100})
v2, m2 = treina_e_registra("challenger_rf_300_d8", {"n_estimators": 300, "max_depth": 8})

# COMMAND ----------

# Atribui aliases (mutáveis) — padrão de ciclo de vida do UC
client.set_registered_model_alias(MODELO, "champion", v1)
client.set_registered_model_alias(MODELO, "challenger", v2)
print(f"@champion -> v{v1}  |  @challenger -> v{v2}")

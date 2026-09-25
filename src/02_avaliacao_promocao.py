# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Avaliação + promoção governada (Champion vs Challenger)
# MAGIC
# MAGIC Compara as versões `@champion` e `@challenger` num holdout e, se o challenger
# MAGIC for melhor (AUC), **move o alias `@champion`** para ele — demonstrando promoção
# MAGIC governada. Também aplica **tags** e **descrição** (governança do UC).

# COMMAND ----------

dbutils.widgets.text("catalog", "bjd_dev")
dbutils.widgets.text("schema", "mlops_credito")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
TABELA = f"{CATALOG}.{SCHEMA}.features_contratos"
MODELO = f"{CATALOG}.{SCHEMA}.modelo_inadimplencia"

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

mlflow.set_registry_uri("databricks-uc")
# NÃO seta experimento: este notebook não loga runs — só lê modelos e mexe em
# aliases/tags no Model Registry (UC). Registry é separado de experiments.
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
X = pd.get_dummies(pdf[FEATURES_NUM + FEATURES_CAT], columns=FEATURES_CAT)
y = pdf[TARGET]
# mesmo split do treino (mesmo random_state) -> usa a parte de teste como holdout
_, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# COMMAND ----------

def auc_do_alias(alias):
    m = mlflow.pyfunc.load_model(f"models:/{MODELO}@{alias}")
    # pyfunc.predict devolve a classe; para AUC usamos probabilidade do modelo sklearn
    sk = mlflow.sklearn.load_model(f"models:/{MODELO}@{alias}")
    proba = sk.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_te, proba)

auc_champ = auc_do_alias("champion")
auc_chal = auc_do_alias("challenger")
print(f"AUC champion={auc_champ:.4f}  |  AUC challenger={auc_chal:.4f}")

# COMMAND ----------

champ_ver = client.get_model_version_by_alias(MODELO, "champion").version
chal_ver = client.get_model_version_by_alias(MODELO, "challenger").version

if auc_chal > auc_champ:
    # challenger venceu: vira @champion e o alias @challenger é removido
    # (a versão não é apagada; apenas deixa de ser "o desafiante da vez")
    client.set_registered_model_alias(MODELO, "champion", chal_ver)
    client.delete_registered_model_alias(MODELO, "challenger")
    print(f"PROMOVIDO: challenger (v{chal_ver}) vira @champion (AUC {auc_chal:.4f} > {auc_champ:.4f}); alias @challenger removido")
else:
    print(f"MANTIDO: champion (v{champ_ver}) segue melhor ou igual (AUC {auc_champ:.4f} >= {auc_chal:.4f}); @challenger permanece em v{chal_ver}")

# COMMAND ----------

# Governança: descrição + tags no modelo registrado
client.update_registered_model(
    MODELO,
    description="Previsão de inadimplência de contratos de financiamento (RandomForest). "
                "Ciclo de vida via aliases @champion/@challenger.",
)
client.set_registered_model_tag(MODELO, "dominio", "risco_credito")
client.set_registered_model_tag(MODELO, "owner", "gabriel.rangel@databricks.com")
client.set_registered_model_tag(MODELO, "projeto", "bjd-mlops-demo")
print("Descrição e tags aplicadas.")

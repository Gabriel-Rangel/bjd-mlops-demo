# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup de dados — features de contratos de financiamento
# MAGIC
# MAGIC Cria o schema do demo e gera uma base **sintética** de contratos de financiamento
# MAGIC (equipamento agrícola / John Deere Bank), com um alvo de **inadimplência** (0/1).
# MAGIC Autocontido e re-executável — não depende de fontes externas.

# COMMAND ----------

dbutils.widgets.text("catalog", "bjd_dev")
dbutils.widgets.text("schema", "mlops_credito")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
TABELA = f"{CATALOG}.{SCHEMA}.features_contratos"
print(f"Destino: {TABELA}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 20_000

regioes = np.array(["Sul", "Sudeste", "Centro-Oeste", "Nordeste", "Norte"])
equipamentos = np.array(["Trator", "Colheitadeira", "Pulverizador", "Plantadeira", "Implemento"])

score_credito = np.clip(rng.normal(640, 90, N), 300, 900).round()
renda_anual = np.clip(rng.lognormal(mean=12.4, sigma=0.55, size=N), 30_000, 2_000_000).round(-2)
valor_financiado = np.clip(rng.lognormal(mean=11.9, sigma=0.6, size=N), 20_000, 1_500_000).round(-2)
valor_entrada = (valor_financiado * rng.uniform(0.0, 0.4, N)).round(-2)
prazo_meses = rng.choice([24, 36, 48, 60, 72, 84], N, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
taxa_juros = np.clip(rng.normal(1.6, 0.4, N), 0.6, 3.5).round(2)  # % a.m.
idade_cliente = np.clip(rng.normal(45, 12, N), 18, 80, ).round().astype(int)
safra = rng.choice([2021, 2022, 2023, 2024, 2025], N)
utilizacao_credito = np.clip(rng.beta(2, 3, N), 0, 1).round(3)  # 0..1
num_contratos_anteriores = rng.poisson(1.5, N)
regiao = rng.choice(regioes, N)
tipo_equipamento = rng.choice(equipamentos, N)

# Comprometimento: parcela estimada sobre renda mensal
parcela = (valor_financiado - valor_entrada) * (taxa_juros / 100) / (1 - (1 + taxa_juros / 100) ** (-prazo_meses))
comprometimento = np.clip(parcela / (renda_anual / 12), 0, 3)

# Score de risco latente -> probabilidade de inadimplência (com ruído)
z = (
    -3.0
    + 2.4 * (comprometimento)
    + 1.8 * utilizacao_credito
    - 0.004 * (score_credito - 640)
    + 0.15 * (prazo_meses / 12)
    - 0.20 * num_contratos_anteriores
    + rng.normal(0, 0.6, N)
)
prob = 1 / (1 + np.exp(-z))
inadimplente = (rng.uniform(0, 1, N) < prob).astype(int)

pdf = pd.DataFrame({
    "id_contrato": np.arange(1, N + 1),
    "valor_financiado": valor_financiado,
    "valor_entrada": valor_entrada,
    "prazo_meses": prazo_meses,
    "taxa_juros": taxa_juros,
    "renda_anual": renda_anual,
    "score_credito": score_credito,
    "idade_cliente": idade_cliente,
    "regiao": regiao,
    "tipo_equipamento": tipo_equipamento,
    "safra": safra,
    "utilizacao_credito": utilizacao_credito,
    "num_contratos_anteriores": num_contratos_anteriores,
    "comprometimento_renda": comprometimento.round(3),
    "inadimplente": inadimplente,
})
print(f"linhas={len(pdf)}  taxa de inadimplência={pdf.inadimplente.mean():.1%}")
pdf.head()

# COMMAND ----------

(
    spark.createDataFrame(pdf)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABELA)
)
print(f"Gravado: {TABELA}")
display(spark.table(TABELA).limit(10))

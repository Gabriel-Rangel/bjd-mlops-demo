# bjd-mlops-demo — ML clássico + ciclo de vida no Unity Catalog

Exemplo simples e re-executável de **ML clássico** com **ciclo de vida via Models in
Unity Catalog** (versões, aliases `@champion`/`@challenger`, promoção governada,
inferência batch). Ambiente **AWS** (profile `gabriel-aws`), catálogo **`bjd_dev`**.

Caso de uso: **previsão de inadimplência** de contratos de financiamento de
equipamento agrícola (dados sintéticos gerados no próprio notebook). O domínio é
trocável sem alterar o esqueleto de ciclo de vida.

## Estrutura
```
src/00_setup_dados.py        cria schema + gera dados sintéticos -> features_contratos
src/01_treino_registro.py    treina RF (v1/v2), registra no UC, seta @champion e @challenger
src/02_avaliacao_promocao.py compara champion vs challenger; promove (move alias); tags/descrição
src/03_inferencia_batch.py   spark_udf @champion -> grava predicoes_inadimplencia
src/04_inferencia_pontual.py load_model por alias (scoring unitário) + @mlflow.trace (traces no experimento)
src/05_inferencia_tracing_uc.py inferência pontual + @mlflow.trace com traces gravados no Unity Catalog (MLflow 3)
src/06_tracing_payload_grande_uc.py tracing no UC + payload completo em Volume UC, span guarda só a URI+metadados (padrão p/ payload > 1MB)
databricks.yml               DAB; targets dev -> bjd_dev, prod -> bjd_prd
resources/ml_pipeline_job.yml job serverless (00 -> 01 -> 02 -> 03)
.github/workflows/deploy-prod.yml CI: PR->main valida; push->main faz deploy no target prod
```

## Objetos no Unity Catalog
- Tabela de features: `bjd_dev.mlops_credito.features_contratos`
- Modelo: `bjd_dev.mlops_credito.modelo_inadimplencia` (aliases `@champion`, `@challenger`)
- Predições: `bjd_dev.mlops_credito.predicoes_inadimplencia`

## Deploy via Databricks Asset Bundle (MLOps Stacks) — passo a passo

Todos os comandos rodam **no terminal**, a partir da raiz do projeto. O target `dev`
aponta para o catálogo `bjd_dev` (ver `databricks.yml`).

### Pré-requisitos (uma vez)
```bash
# CLI instalado (v0.2+; testado na v1.17)
databricks --version

# profile autenticado e válido (deve mostrar YES; confirma o usuário)
databricks auth profiles | grep gabriel-aws
databricks current-user me -p gabriel-aws
```

### 1) Ir para o projeto
```bash
cd ~/Desktop/bjd-mlops-demo
```

### 2) Validar o bundle (checa `databricks.yml` + `resources/`)
```bash
databricks bundle validate -t dev -p gabriel-aws
```

### 3) Deploy (envia os notebooks para o workspace e cria o job)
```bash
databricks bundle deploy -t dev -p gabriel-aws
```

### 4) Rodar o pipeline (00 → 01 → 02 → 03, no serverless)
```bash
databricks bundle run bjd_mlops_pipeline -t dev -p gabriel-aws
```

> Dica: no Claude Code você pode executar qualquer um desses prefixando com `!`
> (ex.: `! databricks bundle deploy -t dev -p gabriel-aws`), que a saída volta na sessão.

### Onde as coisas ficam depois do deploy
- **Notebooks + job:** `/Workspace/Users/gabriel.rangel@databricks.com/.bundle/bjd-mlops-demo/dev/files/`
- **Job:** `bjd_mlops_pipeline` (em *Workflows*, prefixado `[dev gabriel_rangel]`)
- **Objetos UC:** `features_contratos`, `modelo_inadimplencia` (`@champion`/`@challenger`), `predicoes_inadimplencia`
- Os notebooks **04/05/06** (inferência pontual / tracing) **não** fazem parte do job — abra e rode na mão a partir do caminho acima.

### Comandos úteis
```bash
databricks bundle summary -t dev -p gabriel-aws   # recursos criados + URLs
databricks bundle run bjd_mlops_pipeline -t dev -p gabriel-aws   # re-executa o job
databricks bundle destroy -t dev -p gabriel-aws   # remove job + arquivos deployados
```
> `destroy` NÃO apaga os objetos do Unity Catalog (tabelas, modelo, volume) — só os
> recursos criados pelo bundle (job + arquivos em `.bundle/...`).

### Próximo passo: dev → prod ("deploy code")
O target `prod` (em `databricks.yml`) aponta para `bjd_prd` e já inclui o
`workspace.root_path` que o **modo `production`** exige (senão dá o erro
*"must set workspace.root_path"*). Promova o **mesmo código** para prod:
```bash
databricks bundle validate -t prod -p gabriel-aws
databricks bundle deploy   -t prod -p gabriel-aws
databricks bundle run bjd_mlops_pipeline -t prod -p gabriel-aws
```
Obs.: em `mode: production` o job sobe sem o prefixo `[dev ...]` e roda como o
usuário atual; numa esteira MLOps Stacks completa o ideal é rodar como **service
principal** via CI/CD.
A esteira completa (dev/prod + CI/CD em GitHub Actions) está configurada abaixo — este
projeto é a versão enxuta do padrão que o `databricks bundle init mlops-stacks` gera.

## CI/CD com GitHub Actions (promoção para PROD)

Único workflow em `.github/workflows/`:
- **`deploy-prod.yml`** — **PR para `main`** → só `validate` (gate, não faz deploy);
  **push/merge em `main`** → `validate` + `deploy` no target **prod** (`bjd_prd`);
  execução manual (*workflow_dispatch*) pode ainda rodar o pipeline.

**Modelo adotado:** o CI/CD só cuida de **prod**. O ambiente **dev** (`bjd_dev`) é
implantado **manualmente** pelo próprio autor (`databricks bundle deploy -t dev`, seção
acima) — não há branch `develop` nem deploy de dev por CI. `main` = produção.

**Fluxo de promoção:** desenvolve numa branch de feature → abre PR para `main`
(CI **valida**) → ao **mergear em `main`**, o `deploy-prod` promove o código para **prod**.

### Configuração (uma vez)
1. **Service principal + OAuth (M2M)** — crie um SP e gere client id/secret OAuth
   (Settings → Identity/Service principals, ou via CLI/conta). Nada de PAT.
2. **Permissões do SP**: `CAN USE` no workspace; no `bjd_prd`:
   `USE CATALOG`, `USE SCHEMA` + `CREATE SCHEMA`, `CREATE TABLE`, `MODIFY`, `SELECT`,
   `CREATE MODEL`/`EXECUTE`; e permissão para criar Jobs. (Em prod o bundle roda como
   o SP — sem esses grants o pipeline falha ao criar schema/tabela/modelo em `bjd_prd`.)
3. **No repositório GitHub → Settings → Secrets and variables → Actions** (nível do repo,
   *não* precisa de Environments):
   - *Variables*: `DATABRICKS_HOST = https://dbc-463e191a-c656.cloud.databricks.com`
   - *Secrets*: `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`
4. **Branch**: proteja `main` exigindo o check `validate` (do PR) antes do merge.
5. **Gate de aprovação em prod** (já ligado no workflow — o job `deploy` tem
   `environment: production`): para ele passar a **exigir aprovação**, crie o Environment
   em **Settings → Environments → New environment → `production`**, marque
   *Required reviewers* e adicione você (e/ou o time). Sem isso o GitHub cria o environment
   sem proteção e o deploy roda direto. Os secrets podem ficar no nível do repo (passo 3)
   ou, se quiser escopar só a prod, cadastrá-los dentro deste Environment.

### Publicar o repositório (uma vez)
```bash
cd ~/Desktop/bjd-mlops-demo
git init && git add . && git commit -m "bjd-mlops-demo: bundle + notebooks + CI/CD"
git branch -M main
gh repo create bjd-mlops-demo --private --source . --push   # requer gh CLI autenticado
```

## Outros próximos passos
- **Lakehouse Monitoring** (drift/qualidade) sobre `predicoes_inadimplencia`.
- **Model Serving** endpoint para inferência online (com `Enable tracing`).

Ref.: [Manage model lifecycle in Unity Catalog](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/) ·
[Databricks Asset Bundles](https://docs.databricks.com/aws/en/dev-tools/bundles/) ·
[MLOps Stacks](https://docs.databricks.com/aws/en/machine-learning/mlops/mlops-stacks).

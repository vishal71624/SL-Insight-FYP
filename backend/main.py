from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from predictor import (
    get_gene_list,
    get_galaxy_positions,
    get_network_data,
    get_embedding,
    predict_top5 ,
    explain_pair ,
    get_ranking_shift ,
    predict_tarsl ,
    predict_ae ,
    ensemble_top5 ,
    compute_ensemble
)

app = FastAPI(title="NL-LSTF Simulation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/tarsl/{gene}")
def tarsl_predict(gene: str):
    return predict_tarsl(gene.upper())

@app.get("/")
def root():
    return {"status": "NL-LSTF API running"}


@app.get("/api/genes")
def gene_list():
    return {"genes": get_gene_list()}


@app.get("/api/galaxy")
def galaxy():
    return {"positions": get_galaxy_positions()}


@app.get("/api/embedding/{gene}")
def embedding(gene: str):
    result = get_embedding(gene.upper())
    if result is None:
        return {"error": "Gene not found"}
    return result


@app.get("/api/network/{gene}")
def network(gene: str):
    result = get_network_data(gene.upper())
    if result is None:
        return {"error": "Gene not found"}
    return result


@app.get("/api/predict/{gene}")
def predict(gene: str):
    return predict_top5(gene.upper())


@app.get("/api/galaxy/sampled")
def galaxy_sampled(n: int = 2000, gene: str = None):
    import random
    all_pos = get_galaxy_positions()

    # Always include the driver gene and its neighbours
    priority = []
    rest = []

    for p in all_pos:
        if gene and p['name'] == gene.upper():
            priority.append(p)
        else:
            rest.append(p)

    sampled = random.sample(rest, min(n - len(priority), len(rest)))
    return {"positions": priority + sampled}

@app.get("/api/explain/{driver}/{candidate}")
def explain(driver: str, candidate: str):
    return explain_pair(driver.upper(), candidate.upper())

@app.get("/api/ranking_shift/{gene}")
def ranking_shift(gene: str):
    return get_ranking_shift(gene.upper())


@app.get("/api/ae/{gene}")
def ae_predict(gene: str):
    return predict_ae(gene.upper())


@app.get("/api/predict_all/{gene}")
def predict_all(gene: str):
    gene = gene.upper()
    
    nl  = predict_top5(gene)
    tar = predict_tarsl(gene)
    ae  = predict_ae(gene)

    all_results = []

    # NL-LSTF top 5
    if "top5" in nl:
        for g in nl["top5"]:
            all_results.append({
                "gene":   g["gene"],
                "score":  g["normalized_score"],
                "module": "NL-LSTF"
            })

    # TARSL top 5
    if "top5" in tar:
        for g in tar["top5"]:
            all_results.append({
                "gene":   g["gene"],
                "score":  g["score"],
                "module": "TARSL"
            })

    # SLMGAE top 5
    if "top5" in ae:
        for g in ae["top5"]:
            all_results.append({
                "gene":   g["gene"],
                "score":  g["score"],
                "module": "SLMGAE"
            })

    return {
        "driver_gene": gene,
        "total":       len(all_results),
        "results":     all_results
    }


@app.get("/api/ensemble/{gene}")
def ensemble(gene: str):
    return compute_ensemble(gene.upper())
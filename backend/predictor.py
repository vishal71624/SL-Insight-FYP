import numpy as np
import pickle
from scipy.sparse import load_npz
import os
import torch as _torch
from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

DATA = os.path.join(os.path.dirname(__file__), "data")

# ── Load all artifacts once at startup ──────────────────────
print("[INIT] Loading gene embeddings...")
Z = np.load(os.path.join(DATA, "gene_embeddings.npy"))           # (9856, 128)

print("[INIT] Loading 3D UMAP coordinates...")
Z_3d = np.load(os.path.join(DATA, "gene_embeddings_3d.npy"))     # (9856, 3)

print("[INIT] Loading hub-bias mean scores...")
global_mean = np.load(os.path.join(DATA, "global_mean_scores.npy"))  # (9856,)

print("[INIT] Loading gene mappings...")
with open(os.path.join(DATA, "gene_to_idx.pkl"), "rb") as f:
    gene_to_idx = pickle.load(f)
with open(os.path.join(DATA, "idx_to_gene.pkl"), "rb") as f:
    idx_to_gene = pickle.load(f)

print("[INIT] Loading SL and PPI matrices...")
A_sl = load_npz(os.path.join(DATA, "A_sl_matrix.npz"))
A_ppi = load_npz(os.path.join(DATA, "W_ppi_matrix.npz"))

print(f"[INIT] Ready. {len(gene_to_idx)} genes loaded.\n")


# ── Gene list for autocomplete ───────────────────────────────
def get_gene_list():
    return sorted(gene_to_idx.keys())


# ── 3D galaxy positions ──────────────────────────────────────
def get_galaxy_positions():
    """Return all gene names + 3D coordinates for the galaxy."""
    genes = []
    for idx in range(len(idx_to_gene)):
        name = idx_to_gene[idx]
        x, y, z = Z_3d[idx].tolist()
        genes.append({"name": name, "x": x, "y": y, "z": z, "idx": idx})
    return genes


# ── Network connections for a gene ──────────────────────────
def get_network_data(gene_name: str):
    """Return SL and PPI neighbours of the driver gene."""
    if gene_name not in gene_to_idx:
        return None

    idx = gene_to_idx[gene_name]

    # SL neighbours
    sl_row = A_sl.getrow(idx)
    sl_indices = sl_row.nonzero()[1].tolist()
    sl_neighbours = [
        {"gene": idx_to_gene[i], "weight": float(A_sl[idx, i])}
        for i in sl_indices[:30]   # cap at 30 for display
    ]

    # PPI neighbours
    ppi_row = A_ppi.getrow(idx)
    ppi_indices = ppi_row.nonzero()[1].tolist()
    ppi_neighbours = [
        {"gene": idx_to_gene[i], "weight": float(A_ppi[idx, i])}
        for i in ppi_indices[:30]
    ]

    return {
        "driver": gene_name,
        "sl_neighbours": sl_neighbours,
        "ppi_neighbours": ppi_neighbours,
        "sl_count": len(sl_indices),
        "ppi_count": len(ppi_indices)
    }


# ── Embedding vector for a gene ──────────────────────────────
def get_embedding(gene_name: str):
    """Return the 128-dim embedding of a gene."""
    if gene_name not in gene_to_idx:
        return None
    idx = gene_to_idx[gene_name]
    return {
        "gene": gene_name,
        "idx": idx,
        "embedding": Z[idx].tolist()   # list of 128 floats
    }


# ── Core prediction ──────────────────────────────────────────
def predict_top5(gene_name: str):
    if gene_name not in gene_to_idx:
        return {"error": f"Gene '{gene_name}' not found in dataset."}

    idx = gene_to_idx[gene_name]
    driver_embedding = Z[idx]

    # Raw scores — dot product against all genes
    raw_scores = Z @ driver_embedding

    # Column-wise hub bias subtraction
    normalized_scores = raw_scores - global_mean

    # Remove self
    normalized_scores[idx] = -999.0
    raw_scores[idx] = -999.0

    # Rank
    top5_idx = np.argsort(normalized_scores)[-5:][::-1]

    results = []
    for rank, i in enumerate(top5_idx, 1):
        results.append({
            "rank": rank,
            "gene": idx_to_gene[i],
            "idx": int(i),
            "raw_score": float(raw_scores[i]),
            "normalized_score": float(normalized_scores[i]),
            "hub_bias_correction": float(global_mean[i]),
            "x": float(Z_3d[i][0]),
            "y": float(Z_3d[i][1]),
            "z": float(Z_3d[i][2])
        })

    return {
        "driver_gene": gene_name,
        "driver_idx": idx,
        "driver_embedding_norm": float(np.linalg.norm(driver_embedding)),
        "driver_3d": {
            "x": float(Z_3d[idx][0]),
            "y": float(Z_3d[idx][1]),
            "z": float(Z_3d[idx][2])
        },
        "top5": results
    }
    
def explain_pair(driver: str, candidate: str):
    """Per-dimension contribution between driver and candidate gene."""
    if driver not in gene_to_idx or candidate not in gene_to_idx:
        return {"error": "Gene not found"}

    d_idx = gene_to_idx[driver]
    c_idx = gene_to_idx[candidate]

    z_driver    = Z[d_idx]   # (128,)
    z_candidate = Z[c_idx]   # (128,)

    # Per-dimension dot product contribution
    contributions = (z_driver * z_candidate).tolist()  # (128,)

    # Top contributing dimensions
    contrib_arr = np.array(contributions)
    top_dims    = np.argsort(np.abs(contrib_arr))[-10:][::-1]

    # Network evidence
    d_sl_neighbours  = set(A_sl.getrow(d_idx).nonzero()[1].tolist())
    c_sl_neighbours  = set(A_sl.getrow(c_idx).nonzero()[1].tolist())
    d_ppi_neighbours = set(A_ppi.getrow(d_idx).nonzero()[1].tolist())
    c_ppi_neighbours = set(A_ppi.getrow(c_idx).nonzero()[1].tolist())

    shared_sl  = d_sl_neighbours & c_sl_neighbours
    shared_ppi = d_ppi_neighbours & c_ppi_neighbours

    direct_sl  = c_idx in d_sl_neighbours
    direct_ppi = c_idx in d_ppi_neighbours

    return {
        "driver":        driver,
        "candidate":     candidate,
        "contributions": contributions,
        "top_dims": [
            {
                "dim":          int(d),
                "contribution": float(contrib_arr[d]),
                "driver_val":   float(z_driver[d]),
                "candidate_val":float(z_candidate[d]),
            }
            for d in top_dims
        ],
        "dot_product":   float(np.dot(z_driver, z_candidate)),
        "network": {
            "direct_sl":         direct_sl,
            "direct_ppi":        direct_ppi,
            "shared_sl_count":   len(shared_sl),
            "shared_ppi_count":  len(shared_ppi),
            "shared_sl_genes":   [idx_to_gene[i] for i in list(shared_sl)[:5]],
            "shared_ppi_genes":  [idx_to_gene[i] for i in list(shared_ppi)[:5]],
        }
    }


def get_ranking_shift(gene: str):
    """Show how hub-bias correction shifts the top rankings."""
    if gene not in gene_to_idx:
        return {"error": "Gene not found"}

    idx        = gene_to_idx[gene]
    raw_scores = Z @ Z[idx]

    # Raw top 10
    raw_scores_copy = raw_scores.copy()
    raw_scores_copy[idx] = -999
    raw_top10_idx = np.argsort(raw_scores_copy)[-10:][::-1]

    # Normalized top 10
    norm_scores      = raw_scores - global_mean
    norm_scores_copy = norm_scores.copy()
    norm_scores_copy[idx] = -999
    norm_top10_idx = np.argsort(norm_scores_copy)[-10:][::-1]

    raw_top10  = [{"rank": i+1, "gene": idx_to_gene[j],
                   "score": float(raw_scores[j])} for i, j in enumerate(raw_top10_idx)]
    norm_top10 = [{"rank": i+1, "gene": idx_to_gene[j],
                   "score": float(norm_scores[j])} for i, j in enumerate(norm_top10_idx)]

    return {
        "driver":     gene,
        "raw_top10":  raw_top10,
        "norm_top10": norm_top10,
    }


TARSL_DATA = os.path.join(os.path.dirname(__file__), "data", "tarsl")

print("[INIT] Loading TARSL model...")

with open(os.path.join(TARSL_DATA, "SL_scores_masked.pkl"), "rb") as f:
    tarsl_scores = pickle.load(f)

with open(os.path.join(TARSL_DATA, "gene2idx.pkl"), "rb") as f:
    tarsl_gene2idx = pickle.load(f)

with open(os.path.join(TARSL_DATA, "idx2gene.pkl"), "rb") as f:
    tarsl_idx2gene = pickle.load(f)

with open(os.path.join(TARSL_DATA, "symbol2id.pkl"), "rb") as f:
    tarsl_symbol2id = pickle.load(f)

with open(os.path.join(TARSL_DATA, "id2symbol.pkl"), "rb") as f:
    tarsl_id2symbol = pickle.load(f)

print(f"[INIT] TARSL ready. {len(tarsl_symbol2id)} genes loaded.")


def predict_tarsl(gene_name: str):
    gene_name = gene_name.upper()

    gid = tarsl_symbol2id.get(gene_name)
    if gid is None:
        return {"error": f"Gene '{gene_name}' not found in TARSL dataset"}

    idx = tarsl_gene2idx.get(gid)
    if idx is None:
        return {"error": f"Gene '{gene_name}' not in TARSL model"}

    scores = tarsl_scores[idx]
    top5_idx = np.argsort(scores)[::-1][:5]

    results = []
    for rank, j in enumerate(top5_idx, 1):
        partner_id = tarsl_idx2gene[j]
        partner_name = tarsl_id2symbol.get(partner_id, str(partner_id))
        results.append({
            "rank": rank,
            "gene": partner_name,
            "score": float(scores[j]),
        })

    return {
        "driver_gene": gene_name,
        "module": "TARSL",
        "top5": results
    }


# ── AE / SLMGAE Module ───────────────────────────────────────
AE_DATA = os.path.join(os.path.dirname(__file__), "data", "ae")

print("[INIT] Loading SLMGAE logits...")
ae_logits = np.load(os.path.join(AE_DATA, "slmgae_logits.npy"))

with open(os.path.join(AE_DATA, "ae_gene2idx.pkl"), "rb") as f:
    ae_gene2idx = pickle.load(f)

with open(os.path.join(AE_DATA, "ae_idx2gene.pkl"), "rb") as f:
    ae_idx2gene = pickle.load(f)

print(f"[INIT] SLMGAE ready. {len(ae_gene2idx)} genes loaded.")


def predict_ae(gene_name: str):
    gene_name = gene_name.upper()

    if gene_name not in ae_gene2idx:
        return {"error": f"Gene '{gene_name}' not found in SLMGAE dataset"}

    idx    = ae_gene2idx[gene_name]
    scores = ae_logits[idx].copy()
    scores[idx] = -999.0

    min_s  = scores[scores > -999].min()
    max_s  = scores.max()
    norm   = np.clip((scores - min_s) / (max_s - min_s + 1e-8) * 100, 0, 100)

    top5_idx = np.argsort(norm)[::-1][:5]

    results = []
    for rank, j in enumerate(top5_idx, 1):
        results.append({
            "rank":  rank,
            "gene":  ae_idx2gene[j],
            "score": float(norm[j]),
        })

    return {
        "driver_gene": gene_name,
        "module":      "SLMGAE",
        "top5":        results
    }

# ── GCL Ensemble Module ──────────────────────────────────────
import torch
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

ENSEMBLE_DATA = os.path.join(os.path.dirname(__file__), "data", "ensemble")

print("[INIT] Loading GCL embeddings...")
gcl_embeddings = torch.load(
    os.path.join(ENSEMBLE_DATA, "gene_embeddings.pt"),
    map_location="cpu"
).numpy()

with open(os.path.join(ENSEMBLE_DATA, "gene_to_idx.pkl"), "rb") as f:
    gcl_gene_to_idx = pickle.load(f)

with open(os.path.join(ENSEMBLE_DATA, "idx_to_gene.pkl"), "rb") as f:
    gcl_idx_to_gene = pickle.load(f)

print(f"[INIT] GCL ready. {len(gcl_gene_to_idx)} genes, {gcl_embeddings.shape[1]}-dim embeddings.")


def ensemble_top5(driver_gene: str, candidates: list):
    """
    Given driver gene and list of candidate dicts {gene, score, module},
    re-rank using GCL embedding cosine similarity and return top 5.
    """
    driver_gene = driver_gene.upper()

    if driver_gene not in gcl_gene_to_idx:
        return {"error": f"Driver gene '{driver_gene}' not found in GCL embeddings"}

    driver_idx    = gcl_gene_to_idx[driver_gene]
    driver_emb    = gcl_embeddings[driver_idx].reshape(1, -1)

    scored = []
    for c in candidates:
        gene = c["gene"].upper()

        if gene not in gcl_gene_to_idx:
            gcl_score = 0.0
        else:
            cand_idx  = gcl_gene_to_idx[gene]
            cand_emb  = gcl_embeddings[cand_idx].reshape(1, -1)
            gcl_score = float(cos_sim(driver_emb, cand_emb)[0][0])

        # Normalise original model score to 0-1 range for combining
        orig_score = float(c.get("score", 0))

        # Combined agreement score
        agreement = 0.6 * gcl_score + 0.4 * orig_score

        scored.append({
            "gene":          gene,
            "module":        c.get("module", ""),
            "original_score": orig_score,
            "gcl_similarity": round(gcl_score, 4),
            "agreement_score": round(agreement, 4),
        })

    # Sort by agreement score descending
    scored.sort(key=lambda x: x["agreement_score"], reverse=True)

    return {
        "driver_gene": driver_gene,
        "top5": scored[:5],
        "all_ranked": scored,
    }


ENSEMBLE_DATA = os.path.join(os.path.dirname(__file__), "data", "ensemble")

print("[INIT] Loading GCL embeddings...")
gcl_embeddings = _torch.load(
    os.path.join(ENSEMBLE_DATA, "gene_embeddings.pt"),
    map_location="cpu"
).numpy()

with open(os.path.join(ENSEMBLE_DATA, "gene_to_idx.pkl"), "rb") as f:
    gcl_gene_to_idx = pickle.load(f)

with open(os.path.join(ENSEMBLE_DATA, "idx_to_gene.pkl"), "rb") as f:
    gcl_idx_to_gene = pickle.load(f)

# Load pairs.csv
import pandas as _pd
_pairs_path = os.path.join(ENSEMBLE_DATA, "pairs.csv")
gcl_pairs_df = _pd.read_csv(_pairs_path) if os.path.exists(_pairs_path) else None

print(f"[INIT] GCL ready. {len(gcl_gene_to_idx)} genes.")


def compute_ensemble(driver_gene: str):
    """Full ensemble computation for a driver gene."""
    driver_gene = driver_gene.upper()

    # Get candidates — from pairs.csv if available, else from live models
    if gcl_pairs_df is not None and driver_gene in gcl_pairs_df["geneA"].values:
        sub = gcl_pairs_df[gcl_pairs_df["geneA"] == driver_gene].copy()
    else:
        # Build from live predictions
        nl  = predict_top5(driver_gene)
        tar = predict_tarsl(driver_gene)
        ae  = predict_ae(driver_gene)
        rows = []
        if "top5" in nl:
            for g in nl["top5"]:
                rows.append({"geneA": driver_gene, "geneB": g["gene"],
                             "model": "NL-LSTF", "score": g["normalized_score"]})
        if "top5" in tar:
            for g in tar["top5"]:
                rows.append({"geneA": driver_gene, "geneB": g["gene"],
                             "model": "TARSL", "score": g["score"]})
        if "top5" in ae:
            for g in ae["top5"]:
                rows.append({"geneA": driver_gene, "geneB": g["gene"],
                             "model": "SLMGAE", "score": g["score"]})
        sub = _pd.DataFrame(rows)

    if sub.empty:
        return {"error": f"No candidates found for {driver_gene}"}

    # Get driver embedding index
    driver_idx_gcl = gcl_gene_to_idx.get(driver_gene)

    # Structural scores — cosine similarity
    struct_scores = []
    for _, row in sub.iterrows():
        gene = str(row["geneB"]).upper()
        cand_idx = gcl_gene_to_idx.get(gene)
        if driver_idx_gcl is not None and cand_idx is not None:
            d_emb = gcl_embeddings[driver_idx_gcl].reshape(1, -1)
            c_emb = gcl_embeddings[cand_idx].reshape(1, -1)
            s = float(_cos_sim(d_emb, c_emb)[0][0])
        else:
            s = 0.0
        struct_scores.append(s)
    sub["struct_score"] = struct_scores

    # Model normalisation — rank within model
    sub["model_norm"] = (
        sub.groupby("model")["score"]
        .rank(method="first", ascending=True)
        / sub.groupby("model")["score"].transform("count")
    )

    # Fusion score — cross-multiply ranks
    x = sub["model_norm"].values
    fusion_scores = []
    for i in range(len(x)):
        temp = [x[i] * x[j] for j in range(len(x)) if i != j]
        fusion_scores.append(float(np.mean(temp)) if temp else 0.0)
    fusion_arr = np.array(fusion_scores)
    rng = fusion_arr.max() - fusion_arr.min()
    fusion_norm = (fusion_arr - fusion_arr.min()) / (rng if rng > 0 else 1)
    sub["fusion_score"] = fusion_norm

    # Ensemble score
    sub["ensemble"] = 0.5 * sub["struct_score"] + 0.5 * sub["fusion_score"]

    # Top 5
    ranked = sub.sort_values("ensemble", ascending=False)
    unique = ranked.drop_duplicates(subset=["geneB"])
    top5   = unique.head(5)

    return {
        "driver_gene":  driver_gene,
        "all_pairs":    sub.to_dict(orient="records"),
        "top5":         top5.to_dict(orient="records"),
        "fusion_min":   float(fusion_arr.min()),
        "fusion_max":   float(fusion_arr.max()),
    }
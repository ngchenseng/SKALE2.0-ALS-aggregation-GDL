"""SKALE 2.0 core machine-learning framework and training pipeline.
This script contains the core computational workflow corresponding to Cells 1 to 15:
dependency setup, manifest construction, structural feature loading, ESM2 embedding,
radius-graph construction, Siamese EGNN architecture, FiLM phase conditioning,
sample construction, loss functions, training, checkpointing and QC outputs.
"""
import os
import sys
import subprocess

def section(title: str) -> None:
    print(f"\n=== {title} ===")

def install_dependencies(packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

def mount_google_drive() -> None:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as exc:
        print(f"Google Drive was not mounted automatically: {exc}")
try:
    from IPython.display import display
except Exception:
    display = print

section('Cell 1 — Dependency installation')

install_dependencies(["fair-esm", "biopython", "scikit-learn"])

section('Cell 2 — Imports, reproducibility and device setup')

import os, json, math, hashlib, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Device:", device)

section('Cell 3 — Output directories and input paths')

mount_google_drive()
OUTDIR = os.environ.get("SKALE2_OUTDIR", "/content/drive/MyDrive/SKALE2.0")
os.makedirs(OUTDIR, exist_ok=True)
ESM_CACHE_DIR = os.path.join(OUTDIR, "SKALE2_esm_cache")
os.makedirs(ESM_CACHE_DIR, exist_ok=True)
print("✅ OUTDIR:", OUTDIR)
print("✅ ESM_CACHE_DIR:", ESM_CACHE_DIR)
DIR_SOD1 = os.environ.get("SKALE2_DIR_SOD1", "/content/drive/MyDrive/Structural_analysis/")
DIR_TDP43 = os.environ.get("SKALE2_DIR_TDP43", "/content/drive/MyDrive/Structural_analysis3/")
DIR_MAPT = os.environ.get("SKALE2_DIR_MAPT", "/content/drive/MyDrive/Structural_analysis4/")
DIR_PRNP = os.environ.get("SKALE2_DIR_PRNP", "/content/drive/MyDrive/Structural_analysis6/")
print("✅ DIR_SOD1:", DIR_SOD1)
print("✅ DIR_TDP43:", DIR_TDP43)
print("✅ DIR_MAPT:", DIR_MAPT)
print("✅ DIR_PRNP:", DIR_PRNP)

section('Cell 4 — Manifest construction')

import os, glob
import pandas as pd
import numpy as np

def find_pdb(folder, filename):
    p = os.path.join(folder, filename)
    return p if os.path.exists(p) else None

def find_wt_pdb(folder):
    """Heuristic: pick a pdb containing 'wt' in the filename."""
    if not os.path.isdir(folder): return None
    pdbs = [f for f in os.listdir(folder) if f.lower().endswith(".pdb")]
    wts = [f for f in pdbs if "wt" in f.lower() or "wild" in f.lower()]
    if not wts: return None
    wts = sorted(wts, key=lambda x: (len(x), x))
    return os.path.join(folder, wts[0])

def find_feature_csv(folder, variant_tag, kind):
    """Find CSV matching kind (sasa, hb, nma) and variant tag."""
    if not os.path.isdir(folder): return None
    kind = kind.lower()
    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    strong = [f for f in files if (kind in f.lower()) and (variant_tag.lower() in f.lower())]
    if strong:
        strong = sorted(strong, key=lambda x: (len(x), x))
        return os.path.join(folder, strong[0])
    weak = [f for f in files if kind in f.lower()]
    if weak:
        weak = sorted(weak, key=lambda x: (len(x), x))
        return os.path.join(folder, weak[0])
    return None

def find_kinetics_json(folder, variant_tag):
    """Find kinetics param json."""
    if not os.path.isdir(folder): return None
    files = [f for f in os.listdir(folder) if f.lower().endswith(".json")]
    cand = [f for f in files if ("kin" in f.lower()) and (variant_tag.lower() in f.lower())]
    if cand:
        cand = sorted(cand, key=lambda x: (len(x), x))
        return os.path.join(folder, cand[0])
    cand2 = [f for f in files if "kin" in f.lower()]
    if cand2:
        cand2 = sorted(cand2, key=lambda x: (len(x), x))
        return os.path.join(folder, cand2[0])
    return None
rows = []
sod1_wt = find_wt_pdb(DIR_SOD1)
sod1_g86r = find_pdb(DIR_SOD1, "SOD1_G86R.pdb") or find_pdb(DIR_SOD1, "g86r.pdb") or find_pdb(DIR_SOD1, "g86r0.pdb")
if sod1_wt:
    rows.append(dict(protein="SOD1", variant="WT", pdb_path=sod1_wt,
                     sasa_csv=find_feature_csv(DIR_SOD1, "WT", "sasa"),
                     hb_csv=find_feature_csv(DIR_SOD1, "WT", "hb"),
                     nma_csv=find_feature_csv(DIR_SOD1, "WT", "nma"),
                     kinetics_json=find_kinetics_json(DIR_SOD1, "WT")))
if sod1_g86r:
    rows.append(dict(protein="SOD1", variant="G86R", pdb_path=sod1_g86r,
                     sasa_csv=find_feature_csv(DIR_SOD1, "G86R", "sasa"),
                     hb_csv=find_feature_csv(DIR_SOD1, "G86R", "hb"),
                     nma_csv=find_feature_csv(DIR_SOD1, "G86R", "nma"),
                     kinetics_json=find_kinetics_json(DIR_SOD1, "G86R")))
tdp_wt = find_wt_pdb(DIR_TDP43)
tdp_s332n = find_pdb(DIR_TDP43, "TDP43_S332N.pdb") or find_pdb(DIR_TDP43, "s332n.pdb") or find_pdb(DIR_TDP43, "s332n0.pdb")
if tdp_wt:
    rows.append(dict(protein="TDP43", variant="WT", pdb_path=tdp_wt,
                     sasa_csv=find_feature_csv(DIR_TDP43, "WT", "sasa"),
                     hb_csv=find_feature_csv(DIR_TDP43, "WT", "hb"),
                     nma_csv=find_feature_csv(DIR_TDP43, "WT", "nma"),
                     kinetics_json=find_kinetics_json(DIR_TDP43, "WT")))
if tdp_s332n:
    rows.append(dict(protein="TDP43", variant="S332N", pdb_path=tdp_s332n,
                     sasa_csv=find_feature_csv(DIR_TDP43, "S332N", "sasa"),
                     hb_csv=find_feature_csv(DIR_TDP43, "S332N", "hb"),
                     nma_csv=find_feature_csv(DIR_TDP43, "S332N", "nma"),
                     kinetics_json=find_kinetics_json(DIR_TDP43, "S332N")))
mapt_wt = find_wt_pdb(DIR_MAPT)
mapt_p301l = find_pdb(DIR_MAPT, "p301l2.pdb")
if mapt_wt:
    rows.append(dict(protein="MAPT", variant="WT", pdb_path=mapt_wt,
                     sasa_csv=find_feature_csv(DIR_MAPT, "WT", "sasa"),
                     hb_csv=find_feature_csv(DIR_MAPT, "WT", "hb"),
                     nma_csv=find_feature_csv(DIR_MAPT, "WT", "nma"),
                     kinetics_json=None))
if mapt_p301l:
    rows.append(dict(protein="MAPT", variant="P301L", pdb_path=mapt_p301l,
                     sasa_csv=find_feature_csv(DIR_MAPT, "P301L", "sasa"),
                     hb_csv=find_feature_csv(DIR_MAPT, "P301L", "hb"),
                     nma_csv=find_feature_csv(DIR_MAPT, "P301L", "nma"),
                     kinetics_json=None))
prnp_wt = find_wt_pdb(DIR_PRNP)
prnp_e200k = find_pdb(DIR_PRNP, "e200k0.pdb")
prnp_g127v = find_pdb(DIR_PRNP, "g127v0.pdb")
if prnp_wt:
    rows.append(dict(protein="PRNP", variant="WT", pdb_path=prnp_wt,
                     sasa_csv=find_feature_csv(DIR_PRNP, "WT", "sasa"),
                     hb_csv=find_feature_csv(DIR_PRNP, "WT", "hb"),
                     nma_csv=find_feature_csv(DIR_PRNP, "WT", "nma"),
                     kinetics_json=None))
if prnp_e200k:
    rows.append(dict(protein="PRNP", variant="E200K", pdb_path=prnp_e200k,
                     sasa_csv=find_feature_csv(DIR_PRNP, "E200K", "sasa"),
                     hb_csv=find_feature_csv(DIR_PRNP, "E200K", "hb"),
                     nma_csv=find_feature_csv(DIR_PRNP, "E200K", "nma"),
                     kinetics_json=None))
if prnp_g127v:
    rows.append(dict(protein="PRNP", variant="G127V", pdb_path=prnp_g127v,
                     sasa_csv=find_feature_csv(DIR_PRNP, "G127V", "sasa"),
                     hb_csv=find_feature_csv(DIR_PRNP, "G127V", "hb"),
                     nma_csv=find_feature_csv(DIR_PRNP, "G127V", "nma"),
                     kinetics_json=None))
manifest = pd.DataFrame(rows)
manifest["nma_fluct_csv"] = None
manifest["kinetics_path"] = None
folder_map = {"SOD1": DIR_SOD1, "TDP43": DIR_TDP43, "MAPT": DIR_MAPT, "PRNP": DIR_PRNP}
for i, row in manifest.iterrows():
    prot = row["protein"]
    folder = folder_map.get(prot, None)
    if folder:
        cand = sorted(glob.glob(os.path.join(folder, "*fluctuation*.csv")))
        if cand:
            manifest.at[i, "nma_fluct_csv"] = cand[0]
SOD1_WT_KIN = os.path.join(DIR_SOD1, "k_sf12+13.csv")
if os.path.exists(SOD1_WT_KIN):
    manifest.loc[(manifest.protein=="SOD1") & (manifest.variant=="WT"), "kinetics_path"] = SOD1_WT_KIN
TDP43_WT_KIN = os.path.join(DIR_TDP43, "tdp43_kine_data.csv")
if not os.path.exists(TDP43_WT_KIN):
    cands = sorted(glob.glob(os.path.join(DIR_TDP43, "*kine*.csv")))
    if cands: TDP43_WT_KIN = cands[0]
if os.path.exists(TDP43_WT_KIN):
    manifest.loc[(manifest.protein=="TDP43") & (manifest.variant=="WT"), "kinetics_path"] = TDP43_WT_KIN
tdp43_fluct = os.path.join(DIR_TDP43, "fluctuation_tdp43_s332n.csv")
if os.path.exists(tdp43_fluct):
    mask_tdp = manifest["protein"].eq("TDP43") & manifest["variant"].isin(["WT", "S332N"])
    manifest.loc[mask_tdp, "nma_fluct_csv"] = tdp43_fluct
g127v_fluct = os.path.join(DIR_PRNP, "fluctuation_prnp_g127v.csv")
if os.path.exists(g127v_fluct):
    manifest.loc[(manifest.protein=="PRNP") & (manifest.variant=="G127V"), "nma_fluct_csv"] = g127v_fluct
print("✅ Manifest created with", len(manifest), "rows.")
print("✅ Patched kinetics and fluctuation files.")
display(manifest[["protein", "variant", "nma_fluct_csv", "kinetics_path"]])

section('Cell 5 — Manifest file checks')

def check_path(p):
    return (p is not None) and isinstance(p, str) and (len(p) > 0) and os.path.exists(p)

def quick_check_manifest(df):
    missing = []
    for i, row in df.iterrows():
        for k in ["pdb_path", "sasa_csv", "hb_csv", "nma_csv"]:
            p = row.get(k, None)
            if p is not None and isinstance(p, str) and len(p) > 0:
                if not os.path.exists(p):
                    missing.append((i, row["protein"], row["variant"], k, p))
    if missing:
        print("❌ Missing files:")
        for m in missing[:20]:
            print(m)
    else:
        print("✅ All referenced files exist (for non-empty fields).")
quick_check_manifest(manifest)
for col in ["sasa_csv", "hb_csv", "nma_csv"]:
    p = manifest.iloc[0].get(col, None)
    if check_path(p):
        df = pd.read_csv(p)
        print("\n---", col, "columns ---")
        print(df.columns.tolist()[:30])

section('Cell 6 — PDB and structural feature loaders')

import os
import re
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}

def load_ca_coords_and_sequence(pdb_path: str):
    """
    Returns:
      coords: (L,3) float32 CA coordinates
      seq:    length L string
      res_ids:(L,) residue numbers from PDB (res.id[1])
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    coords, seq, res_ids = [], [], []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != " ":
                    continue
                if "CA" not in res:
                    continue
                aa = AA3_TO_1.get(res.get_resname().upper(), "X")
                coords.append(res["CA"].coord.astype(np.float32))
                seq.append(aa)
                res_ids.append(int(res.id[1]))
        break
    coords = np.stack(coords, axis=0)
    return coords, "".join(seq), np.array(res_ids, dtype=np.int64)

def _find_residue_col(df: pd.DataFrame) -> str:
    candidates = ["Residue_Index", "Residue_Index ", "Residue", "Residue Number",
                  "residue", "res_id", "i", "pos"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find residue index column in {df.columns.tolist()}")

def load_per_residue_scalar(csv_path: str, res_ids: np.ndarray, value_cols=None, verbose=True) -> np.ndarray:
    """
    NaN-safe loader for per-residue files like SASA/NMA.
    - Coerces features to numeric (non-numeric -> NaN)
    - Fills NaN per column using median; if column all NaN -> 0
    - Aligns rows to res_ids; missing residues filled with same per-column fill value
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    rcol = _find_residue_col(df)
    df[rcol] = pd.to_numeric(df[rcol], errors="coerce")
    if value_cols is None:
        value_cols = [c for c in df.columns if c != rcol]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    fill_vals = {}
    for c in value_cols:
        if df[c].isna().all():
            fill_vals[c] = 0.0
        else:
            fill_vals[c] = float(df[c].median(skipna=True))
    na_before = int(df[value_cols].isna().sum().sum())
    df[value_cols] = df[value_cols].fillna(fill_vals)
    na_after = int(df[value_cols].isna().sum().sum())
    df = df.dropna(subset=[rcol]).copy()
    df[rcol] = df[rcol].astype(int)
    sub = df[[rcol] + value_cols].groupby(rcol).mean().reset_index()
    feat = np.zeros((len(res_ids), len(value_cols)), dtype=np.float32)
    default_vec = np.array([fill_vals[c] for c in value_cols], dtype=np.float32)
    m = {int(r): sub[sub[rcol] == int(r)][value_cols].values[0].astype(np.float32)
         for r in sub[rcol].values}
    for i, r in enumerate(res_ids):
        feat[i] = m.get(int(r), default_vec)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if verbose:
        print(f"✅ Loaded {os.path.basename(csv_path)} | cols={len(value_cols)} | NaN before={na_before} after={na_after}")
    return feat

def _parse_resnum_from_aa3num(x):
    """
    Parses strings like 'GLU3', 'SER20' -> 3, 20
    (keeps first integer found)
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    m = re.search(r"(-?\d+)", s)
    return int(m.group(1)) if m else None

def load_hbond_features(hb_csv_path: str, res_ids: np.ndarray, verbose=True) -> np.ndarray:
    """
    HB edge list columns (your files):
      ['Structure','Donor_Residue','Acceptor_Residue','Donor_Atom','Acceptor_Atom','Distance']
    Aggregates into per-residue features aligned to res_ids:
      [donor_count, acceptor_count, total_involvement, mean_dist, min_dist, frac_short]
    """
    df = pd.read_csv(hb_csv_path)
    df.columns = [c.strip() for c in df.columns]
    required = {"Donor_Residue", "Acceptor_Residue", "Distance"}
    if not required.issubset(df.columns):
        raise ValueError(f"HB file missing required columns {required}. Found: {df.columns.tolist()}")
    df["donor_id"] = df["Donor_Residue"].apply(_parse_resnum_from_aa3num)
    df["acceptor_id"] = df["Acceptor_Residue"].apply(_parse_resnum_from_aa3num)
    df["dist"] = pd.to_numeric(df["Distance"], errors="coerce")
    if df["dist"].isna().all():
        df["dist"] = 0.0
    else:
        df["dist"] = df["dist"].fillna(df["dist"].median(skipna=True))
    df = df.dropna(subset=["donor_id", "acceptor_id"], how="all").copy()
    df["donor_id"] = df["donor_id"].astype("Int64")
    df["acceptor_id"] = df["acceptor_id"].astype("Int64")
    med = float(df["dist"].median()) if len(df) else 0.0
    short_thr = 0.32 if med < 1.0 else 3.2
    donor_count = df.groupby("donor_id").size()
    acceptor_count = df.groupby("acceptor_id").size()
    long1 = df[["donor_id", "dist"]].rename(columns={"donor_id": "res_id"})
    long2 = df[["acceptor_id", "dist"]].rename(columns={"acceptor_id": "res_id"})
    long = pd.concat([long1, long2], ignore_index=True).dropna(subset=["res_id"]).copy()
    long["res_id"] = long["res_id"].astype(int)
    total_count = long.groupby("res_id").size()
    mean_dist = long.groupby("res_id")["dist"].mean()
    min_dist = long.groupby("res_id")["dist"].min()
    frac_short = long.groupby("res_id")["dist"].apply(lambda x: float((x <= short_thr).mean()))
    L = len(res_ids)
    feat = np.zeros((L, 6), dtype=np.float32)
    for i, r in enumerate(res_ids.astype(int)):
        feat[i, 0] = float(donor_count.get(r, 0))
        feat[i, 1] = float(acceptor_count.get(r, 0))
        feat[i, 2] = float(total_count.get(r, 0))
        feat[i, 3] = float(mean_dist.get(r, 0.0))
        feat[i, 4] = float(min_dist.get(r, 0.0))
        feat[i, 5] = float(frac_short.get(r, 0.0))
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if verbose:
        print(f"✅ Loaded {os.path.basename(hb_csv_path)} | HB features=6 | edges={len(df)} | short_thr={short_thr}")
    return feat

def load_hb_diff_features(hb_diff_csv_path: str, res_ids: np.ndarray, verbose=True) -> np.ndarray:
    """
    Input columns: Pair, Delta_Occupancy, Delta_Distance
    Pair like 'ALA230→VAL195' (two residues).
    Output per-residue features aligned to res_ids:
      [sum_abs_delta_occ, mean_abs_delta_occ, mean_delta_dist, mean_abs_delta_dist]
    """
    df = pd.read_csv(hb_diff_csv_path)
    df.columns = [c.strip() for c in df.columns]
    required = {"Pair", "Delta_Occupancy", "Delta_Distance"}
    if not required.issubset(df.columns):
        raise ValueError(f"HB-diff file missing {required}. Found: {df.columns.tolist()}")
    def _pair_to_two_resnums(p):
        nums = re.findall(r"(-?\d+)", str(p))
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
        return None, None
    r1, r2 = zip(*df["Pair"].apply(_pair_to_two_resnums))
    df["r1"] = pd.to_numeric(pd.Series(r1), errors="coerce")
    df["r2"] = pd.to_numeric(pd.Series(r2), errors="coerce")
    df["docc"] = pd.to_numeric(df["Delta_Occupancy"], errors="coerce").fillna(0.0)
    df["ddist"] = pd.to_numeric(df["Delta_Distance"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["r1", "r2"], how="any").copy()
    df["r1"] = df["r1"].astype(int)
    df["r2"] = df["r2"].astype(int)
    long1 = df[["r1", "docc", "ddist"]].rename(columns={"r1": "res"})
    long2 = df[["r2", "docc", "ddist"]].rename(columns={"r2": "res"})
    long = pd.concat([long1, long2], ignore_index=True)
    g = long.groupby("res")
    sum_abs_docc = g["docc"].apply(lambda x: float(np.sum(np.abs(x))))
    mean_abs_docc = g["docc"].apply(lambda x: float(np.mean(np.abs(x))) if len(x) else 0.0)
    mean_ddist = g["ddist"].mean()
    mean_abs_ddist = g["ddist"].apply(lambda x: float(np.mean(np.abs(x))) if len(x) else 0.0)
    feat = np.zeros((len(res_ids), 4), dtype=np.float32)
    for i, r in enumerate(res_ids.astype(int)):
        feat[i, 0] = float(sum_abs_docc.get(r, 0.0))
        feat[i, 1] = float(mean_abs_docc.get(r, 0.0))
        feat[i, 2] = float(mean_ddist.get(r, 0.0))
        feat[i, 3] = float(mean_abs_ddist.get(r, 0.0))
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if verbose:
        print(f"✅ Loaded {os.path.basename(hb_diff_csv_path)} | HB-diff features=4 | pairs={len(df)}")
    return feat

section('Cell 7 — Frozen ESM2 residue embeddings')

USE_ESM = True
if USE_ESM:
    import esm
    class FrozenESM2:
        def __init__(self, model_name="esm2_t6_8M_UR50D", cache_dir=ESM_CACHE_DIR):
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
            self.model = self.model.to(device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self.batch_converter = self.alphabet.get_batch_converter()
            self.embed_dim = self.model.embed_dim
            self.cache_dir = cache_dir
            os.makedirs(self.cache_dir, exist_ok=True)
        @torch.no_grad()
        def embed(self, seq: str, key: str) -> torch.Tensor:
            h = hashlib.md5(key.encode("utf-8")).hexdigest()
            cache_path = os.path.join(self.cache_dir, f"{h}.pt")
            if os.path.exists(cache_path):
                return torch.load(cache_path, map_location="cpu")
            data = [("x", seq)]
            _, _, toks = self.batch_converter(data)
            toks = toks.to(device)
            out = self.model(toks, repr_layers=[self.model.num_layers], return_contacts=False)
            reps = out["representations"][self.model.num_layers][0]
            reps = reps[1:1+len(seq)].detach().cpu()
            torch.save(reps, cache_path)
            return reps
    esm2 = FrozenESM2()
    ESM_DIM = esm2.embed_dim
    print("✅ ESM enabled. ESM_DIM =", ESM_DIM)
else:
    esm2 = None
    ESM_DIM = 0
    print("✅ ESM disabled.")

section('Cell 8 — Radius graph construction')

def radius_graph(coords: torch.Tensor, r: float = 10.0, max_neighbors: int = 32) -> torch.Tensor:
    L = coords.size(0)
    d = torch.cdist(coords, coords, p=2)
    d.fill_diagonal_(1e9)
    edges = []
    for i in range(L):
        idx = torch.nonzero(d[i] <= r, as_tuple=False).squeeze(-1)
        if idx.numel() > max_neighbors:
            _, order = torch.topk(d[i][idx], k=max_neighbors, largest=False)
            idx = idx[order]
        if idx.numel() > 0:
            src = torch.full((idx.numel(),), i, device=coords.device, dtype=torch.long)
            edges.append(torch.stack([src, idx], dim=0))
    if len(edges) == 0:
        return torch.empty((2, 0), device=coords.device, dtype=torch.long)
    return torch.cat(edges, dim=1)

section('Cell 9 — SKALE 2.0 Siamese EGNN, FiLM gating and output heads')

class EGNNLayer(nn.Module):
    def __init__(self, h_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*h_dim + 1, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 1),
        )
    def forward(self, h, x, edge_index):
        if edge_index.numel() == 0:
            return h, x
        src, dst = edge_index[0], edge_index[1]
        xi, xj = x[src], x[dst]
        dij = torch.norm(xi - xj, dim=-1, keepdim=True)
        m = self.edge_mlp(torch.cat([h[src], h[dst], dij], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)
        h = self.node_mlp(torch.cat([h, agg], dim=-1))
        phi = self.coord_mlp(m)
        dx = torch.zeros_like(x)
        dx.index_add_(0, src, (xi - xj) * phi)
        x = x + dx
        return h, x

class PhaseFiLM(nn.Module):
    def __init__(self, h_dim: int, n_phases: int = 2):
        super().__init__()
        self.emb = nn.Embedding(n_phases, h_dim)
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(h_dim, 2*h_dim),
            nn.SiLU(),
            nn.Linear(2*h_dim, 2*h_dim),
        )
        self.h_dim = h_dim
    def forward(self, h, phase_id: torch.Tensor):
        p = self.emb(phase_id)
        gb = self.to_gamma_beta(p)
        gamma, beta = gb[:self.h_dim], gb[self.h_dim:]
        return h * (1.0 + gamma) + beta

class SKALE2(nn.Module):
    def __init__(self, in_dim: int, esm_dim: int, h_dim: int = 256, n_layers: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.esm_dim = esm_dim
        self.proj = nn.Linear(in_dim + esm_dim, h_dim)
        self.egnn = nn.ModuleList([EGNNLayer(h_dim) for _ in range(n_layers)])
        self.phase_gate = PhaseFiLM(h_dim, n_phases=2)
        self.risk_global = nn.Sequential(nn.Linear(h_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, 1))
        self.kin_head = nn.Sequential(nn.Linear(h_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, 3))
        self.aux_head = nn.Sequential(nn.Linear(h_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, in_dim))
    def forward_single(self, node_feat, coords, edge_index, phase_id):
        h = self.proj(node_feat)
        x = coords
        for layer in self.egnn:
            h, x = layer(h, x, edge_index)
        h = self.phase_gate(h, phase_id)
        pooled = h.mean(dim=0)
        risk_logit = self.risk_global(pooled).squeeze(-1)
        kin_params = self.kin_head(pooled)
        aux_pred = self.aux_head(h)
        return dict(h=h, risk_logit=risk_logit, kin_params=kin_params, aux_pred=aux_pred)
    def forward(self, wt_batch, mut_batch, phase_id):
        out_wt = self.forward_single(**wt_batch, phase_id=phase_id)
        out_mut = self.forward_single(**mut_batch, phase_id=phase_id)
        dz = out_mut["h"].mean(dim=0) - out_wt["h"].mean(dim=0)
        return out_wt, out_mut, dz

section('Cell 10 — Kinetics JSON loader')

KIN_KEYS = ["k_n", "k_plus", "amax"]

def load_kinetics_params(json_path: str) -> torch.Tensor:
    with open(json_path, "r") as f:
        d = json.load(f)
    vals = []
    for k in KIN_KEYS:
        if k not in d:
            raise KeyError(f"Missing key '{k}' in {json_path}. Available keys: {list(d.keys())}")
        vals.append(float(d[k]))
    return torch.tensor(vals, dtype=torch.float32)

section('Cell 11 — Sample building, feature assembly and Siamese pairing')

import os
import numpy as np
import pandas as pd
import torch
_FLUCT_SCALER_CACHE = {}

def load_fluct_feature_zscore(csv_path: str, res_ids, value_col: str, verbose=False):
    """
    Load ONE fluctuation column aligned to res_ids, then Z-score it using
    (mu, sd) computed from ALL eligible fluct columns in that SAME csv file.
    This ensures WT and mutant from the same file are scaled identically.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    eligible = [c for c in df.columns
                if ("fluct" in c.lower()) and ("delta" not in c.lower()) and ("diff" not in c.lower())]
    key = os.path.abspath(csv_path)
    if key not in _FLUCT_SCALER_CACHE:
        vals = []
        for c in eligible:
            v = pd.to_numeric(df[c], errors="coerce").values
            vals.append(v)
        vals = np.concatenate(vals, axis=0)
        vals = vals[np.isfinite(vals)].astype(np.float32)
        mu = float(np.mean(vals)) if len(vals) else 0.0
        sd = float(np.std(vals)) if len(vals) else 1.0
        sd = max(sd, 1e-8)
        _FLUCT_SCALER_CACHE[key] = (mu, sd, eligible)
        if verbose:
            print(f"ℹ️ Fluct scaler cached for {os.path.basename(csv_path)} | mu={mu:.4g} sd={sd:.4g} | cols={eligible}")
    mu, sd, eligible = _FLUCT_SCALER_CACHE[key]
    x = load_per_residue_scalar(csv_path, res_ids, value_cols=[value_col], verbose=verbose).astype(np.float32)
    x = (x - mu) / sd
    return x

class Sample:
    def __init__(self, protein, variant, coords, node_in, edge_index, risk_label, kin_params):
        self.protein = protein
        self.variant = variant
        self.coords = coords
        self.node_in = node_in
        self.edge_index = edge_index
        self.risk_label = risk_label
        self.kin_params = kin_params
        self.has_kin = kin_params is not None

def pick_fluct_col(df_cols, variant):
    """
    SAFE fluctuation column chooser:
      - must contain 'fluct'
      - must NOT contain 'diff' or 'delta'
      - WT prefers 'wt' columns
      - mutant prefers columns containing the variant token (e.g. s332n, g86r)
      - fallback: first eligible column
    """
    cols = [c.strip() for c in df_cols]
    cols_l = [c.lower() for c in cols]
    v = str(variant).lower().strip()
    good = [cols[i] for i, cl in enumerate(cols_l)
            if ("fluct" in cl) and ("diff" not in cl) and ("delta" not in cl)]
    if not good:
        return None
    if v == "wt":
        wt_like = [c for c in good if "wt" in c.lower()]
        return wt_like[0] if wt_like else good[0]
    hit = [c for c in good if v in c.lower()]
    return hit[0] if hit else good[0]

def pick_kin_signal_col(df: pd.DataFrame, prefer_keywords=None):
    """
    Choose kinetics signal column from a wide-format ThT table.
    For your TDP43 file: time_(min) + 0.10%/0.25%/0.50%/1.00% + stdev_*
    We should:
      - ignore stdev_* columns
      - prefer keywords like ["0.50%","1.00%","0.25%","0.10%"]
      - fallback: first non-stdev non-time numeric column
    """
    cols = [c.strip() for c in df.columns]
    cols_l = [c.lower() for c in cols]
    time_col = None
    for c in cols:
        if "time" in c.lower():
            time_col = c
            break
    sig_cols = []
    for c in cols:
        cl = c.lower()
        if c == time_col:
            continue
        if "stdev" in cl or "std" in cl:
            continue
        sig_cols.append(c)
    if len(sig_cols) == 0:
        return None
    if prefer_keywords:
        for kw in prefer_keywords:
            hits = [c for c in sig_cols if kw.lower() in c.lower()]
            if hits:
                return hits[0]
    return sig_cols[0]

def load_kinetics_params_csv(path: str, prefer_signal_keywords=None) -> torch.Tensor:
    """
    Convert ThT CSV curve into [tlag, kplus, amax].
    Supports:
      - long format: Time (min), Concentration (uM), Replicate, Signal  (handled in your Cell earlier if you kept it)
      - wide format: time + signal columns (TDP43 file)
    This function handles the wide format robustly.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if any("volume" in c.lower() for c in df.columns):
        raise ValueError(f"SEC-like file detected (Volume column). Not ThT kinetics: {path}")
    time_col = None
    for c in df.columns:
        if "time" in c.lower():
            time_col = c
            break
    if time_col is None:
        raise ValueError(f"No time column detected in kinetics file: {path}")
    cols_l = [c.lower() for c in df.columns]
    if ("signal" in cols_l) and any("replicate" in c for c in cols_l):
        sig_col = df.columns[cols_l.index("signal")]
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df[sig_col] = pd.to_numeric(df[sig_col], errors="coerce")
        df = df.dropna(subset=[time_col, sig_col])
        g = df.groupby(time_col)[sig_col].mean().reset_index()
        t = g[time_col].values.astype(np.float32)
        y = g[sig_col].values.astype(np.float32)
    else:
        sig_col = pick_kin_signal_col(df, prefer_keywords=prefer_signal_keywords)
        if sig_col is None:
            raise ValueError(f"Could not choose kinetics signal column from: {df.columns.tolist()}")
        t = pd.to_numeric(df[time_col], errors="coerce").values
        y = pd.to_numeric(df[sig_col], errors="coerce").values
        m = np.isfinite(t) & np.isfinite(y)
        t = t[m].astype(np.float32)
        y = y[m].astype(np.float32)
    if len(t) < 5:
        return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    y0 = float(np.min(y))
    y1 = float(np.max(y))
    amp = max(1e-8, (y1 - y0))
    yn = (y - y0) / amp
    if np.any(yn >= 0.10):
        idx10 = int(np.argmax(yn >= 0.10))
        tlag = float(t[idx10])
    else:
        tlag = float(t[-1])
    dydt = np.gradient(yn, t)
    kplus = float(np.max(dydt))
    amax = float(np.clip(yn[-1], 0.0, 1.0))
    return torch.tensor([tlag, kplus, amax], dtype=torch.float32)

def build_samples(manifest_df: pd.DataFrame):
    samples = []
    in_dim = None
    for _, row in manifest_df.iterrows():
        coords_np, seq, res_ids = load_ca_coords_and_sequence(row["pdb_path"])
        feats = []
        if row.get("sasa_csv") and isinstance(row["sasa_csv"], str) and os.path.exists(row["sasa_csv"]):
            feats.append(load_per_residue_scalar(row["sasa_csv"], res_ids, verbose=True))
        if row.get("hb_csv") and isinstance(row["hb_csv"], str) and os.path.exists(row["hb_csv"]):
            feats.append(load_hbond_features(row["hb_csv"], res_ids, verbose=True))
        if row.get("nma_csv") and isinstance(row["nma_csv"], str) and os.path.exists(row["nma_csv"]):
            feats.append(load_per_residue_scalar(row["nma_csv"], res_ids, verbose=True))
        if row.get("nma_fluct_csv") and isinstance(row["nma_fluct_csv"], str) and os.path.exists(row["nma_fluct_csv"]):
            df_tmp = pd.read_csv(row["nma_fluct_csv"])
            df_tmp.columns = [c.strip() for c in df_tmp.columns]
            fcol = pick_fluct_col(df_tmp.columns.tolist(), row["variant"])
            if fcol is None:
                print(f"⚠️ No usable fluctuation column (non-delta) in {os.path.basename(row['nma_fluct_csv'])}; skipping.")
            else:
                feat_fluct = load_fluct_feature_zscore(row["nma_fluct_csv"], res_ids, fcol, verbose=True)
                feats.append(feat_fluct)
        if len(feats) == 0:
            feats_np = np.zeros((len(seq), 1), dtype=np.float32)
        else:
            feats_np = np.concatenate(feats, axis=1).astype(np.float32)
        if in_dim is None:
            in_dim = feats_np.shape[1]
        if USE_ESM:
            key = f"{row['protein']}|{row['variant']}|{seq}"
            esm_emb = esm2.embed(seq, key).numpy().astype(np.float32)
            node_np = np.concatenate([feats_np, esm_emb], axis=1)
        else:
            node_np = feats_np
        coords = torch.tensor(coords_np, dtype=torch.float32)
        node_in = torch.tensor(node_np, dtype=torch.float32)
        edge_index = radius_graph(coords, r=10.0, max_neighbors=32)
        risk_label = row.get("risk_label", None)
        if risk_label is not None and not pd.isna(risk_label):
            risk_label = float(risk_label)
        else:
            risk_label = None
        kin_params = None
        kin_path = row.get("kinetics_path", None)
        if kin_path is not None and isinstance(kin_path, str) and len(kin_path) > 0 and os.path.exists(kin_path):
            kin_params = load_kinetics_params_csv(
                kin_path,
                prefer_signal_keywords=["0.50%", "1.00%", "0.25%", "0.10%"]
            )
        samples.append(Sample(
            protein=row["protein"], variant=row["variant"],
            coords=coords, node_in=node_in, edge_index=edge_index,
            risk_label=risk_label, kin_params=kin_params
        ))
    return samples, in_dim

def make_siamese_pairs(samples):
    by_prot = {}
    for i, s in enumerate(samples):
        by_prot.setdefault(s.protein, []).append((i, s.variant))
    pairs = []
    for prot, items in by_prot.items():
        wt = [i for i, v in items if str(v).upper() == "WT"]
        if not wt:
            print(f"⚠️ No WT found for {prot}. Skipping Siamese pairing.")
            continue
        wt_idx = wt[0]
        for i, v in items:
            if i != wt_idx:
                pairs.append((wt_idx, i))
    return pairs
samples, IN_DIM = build_samples(manifest)
print("✅ Built samples:", len(samples), "| IN_DIM =", IN_DIM, "| ESM_DIM =", ESM_DIM)
pairs = make_siamese_pairs(samples)
print("✅ Siamese WT→mutant pairs:", pairs)
print("✅ Total pairs:", len(pairs))
kin_samples = [i for i, s in enumerate(samples) if s.has_kin]
print("✅ Kinetics-labeled SAMPLES found:", len(kin_samples))
for i in kin_samples:
    s = samples[i]
    print(s.protein, s.variant, "| kinetics_path =", manifest.loc[i, "kinetics_path"] if "kinetics_path" in manifest.columns else None,
          "| kin_params =", s.kin_params.cpu().numpy())

section('Cell 11.1 — Kinetics parameter quality control')

print("=== Kinetics params QC ===")
kin_samples = [i for i,s in enumerate(samples) if s.has_kin]
for i in kin_samples:
    s = samples[i]
    print(i, s.protein, s.variant, "kin_params =", s.kin_params.cpu().numpy())

section('Cell 11.2 — Kinetics target normalization')

kin_ids = [i for i,s in enumerate(samples) if s.has_kin]
K = torch.stack([samples[i].kin_params for i in kin_ids], dim=0)
mu = K.mean(dim=0)
sd = K.std(dim=0, unbiased=False).clamp(min=1e-6)
print("✅ kin_mu:", mu.cpu().numpy())
print("✅ kin_sd:", sd.cpu().numpy())
for i in kin_ids:
    samples[i].kin_params = (samples[i].kin_params - mu) / sd
print("✅ Normalized kin_params for kinetics-labeled samples.")

section('Cell 12 — Masked losses and optimizer setup')

import torch
import torch.nn.functional as F
import numpy as np
model = SKALE2(in_dim=IN_DIM, esm_dim=ESM_DIM, h_dim=256, n_layers=4).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)

def batch_from_sample(s: Sample):
    return dict(
        node_feat=s.node_in.to(device),
        coords=s.coords.to(device),
        edge_index=s.edge_index.to(device),
    )

def masked_losses(out_wt, out_mut, sw: Sample, sm: Sample):
    """
    Revised masked_losses:
    - Removes tanh clamping so model can predict negative/large Z-scores.
    """
    losses = {}
    if sm.risk_label is not None:
        y = torch.tensor(sm.risk_label, device=device)
        losses["risk_bce"] = F.binary_cross_entropy_with_logits(out_mut["risk_logit"], y)
    kin_src = None
    if getattr(sm, "has_kin", False) and (sm.kin_params is not None):
        kin_src = ("mut", sm, out_mut)
    elif getattr(sw, "has_kin", False) and (sw.kin_params is not None):
        kin_src = ("wt", sw, out_wt)
    if kin_src is not None:
        _, s_kin, out_kin = kin_src
        yk = s_kin.kin_params.to(device)
        kin_pred = out_kin["kin_params"]
        losses["kin_mse"] = F.mse_loss(kin_pred, yk)
    x_true = sm.node_in[:, :IN_DIM].to(device)
    x_pred = out_mut["aux_pred"]
    losses["aux_mse"] = F.mse_loss(x_pred, x_true)
    w = dict(risk_bce=1.0, kin_mse=1.0, aux_mse=0.2)
    total = 0.0
    for k, v in losses.items():
        total = total + w.get(k, 1.0) * v
    return total, losses

section('Cell 13 — Single-step smoke test')

model.train()
phase_id = torch.tensor(0, device=device)
iw, im = pairs[0]
sw, sm = samples[iw], samples[im]
out_wt, out_mut, dz = model(batch_from_sample(sw), batch_from_sample(sm), phase_id)
total, parts = masked_losses(out_wt, out_mut, sw, sm)
opt.zero_grad()
total.backward()
opt.step()
print("✅ Smoke test loss:", float(total.detach().cpu()))
print({k: float(v.detach().cpu()) for k, v in parts.items()})
print("Mut has kinetics?", sm.has_kin, "| predicted kin params:", out_mut["kin_params"].detach().cpu().numpy())

section('Cell 14 — Train/evaluation utilities')

import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score

def split_pairs(pairs, val_frac=0.25, seed=42):
    rng = random.Random(seed)
    pairs2 = pairs.copy()
    rng.shuffle(pairs2)
    n_val = max(1, int(len(pairs2) * val_frac))
    return pairs2[n_val:], pairs2[:n_val]
train_pairs, val_pairs = split_pairs(pairs, val_frac=0.25, seed=SEED)
print("✅ train_pairs:", len(train_pairs), "val_pairs:", len(val_pairs))

def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item()**2
    return total**0.5

def _kin_loss_from_single(model, s: Sample, phase: torch.Tensor):
    """
    Compute kinetics loss for a SINGLE sample by feeding it as (WT, WT).
    This avoids needing mutant kinetics and still trains the kinetics head.
    """
    out_wt, out_mut, dz = model(batch_from_sample(s), batch_from_sample(s), phase)
    total, parts = masked_losses(out_wt, out_mut, s, s)
    kin_mse = parts.get("kin_mse", None)
    return kin_mse, total
@torch.no_grad()
def eval_epoch(model, pairs_list, phase_id=0, kin_ids=None):
    model.eval()
    phase = torch.tensor(phase_id, device=device)
    losses = []
    risk_logits, risk_labels = [], []
    kin_mses = []
    for iw, im in pairs_list:
        sw, sm = samples[iw], samples[im]
        out_wt, out_mut, dz = model(batch_from_sample(sw), batch_from_sample(sm), phase)
        total, parts = masked_losses(out_wt, out_mut, sw, sm)
        losses.append(float(total.detach().cpu()))
        if sm.risk_label is not None:
            risk_logits.append(float(out_mut["risk_logit"].detach().cpu()))
            risk_labels.append(int(sm.risk_label))
        if sm.has_kin and ("kin_mse" in parts):
            kin_mses.append(float(parts["kin_mse"].detach().cpu()))
    if kin_ids is not None and len(kin_ids) > 0:
        for idx in kin_ids:
            s = samples[idx]
            if not s.has_kin:
                continue
            kin_mse, _ = _kin_loss_from_single(model, s, phase)
            if kin_mse is not None:
                kin_mses.append(float(kin_mse.detach().cpu()))
    metrics = {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "risk_acc": np.nan,
        "risk_auc": np.nan,
        "kin_mse": float(np.mean(kin_mses)) if kin_mses else np.nan,
    }
    if len(risk_labels) >= 2 and len(set(risk_labels)) > 1:
        probs = 1 / (1 + np.exp(-np.array(risk_logits)))
        metrics["risk_auc"] = roc_auc_score(risk_labels, probs)
        preds = (probs >= 0.5).astype(int)
        metrics["risk_acc"] = accuracy_score(risk_labels, preds)
    elif len(risk_labels) >= 1:
        probs = 1 / (1 + np.exp(-np.array(risk_logits)))
        preds = (probs >= 0.5).astype(int)
        metrics["risk_acc"] = accuracy_score(risk_labels, preds)
    return metrics

def train_epoch(model, pairs_list, opt, phase_id=0, clip=1.0, kin_ids=None):
    model.train()
    phase = torch.tensor(phase_id, device=device)
    losses = []
    for iw, im in pairs_list:
        sw, sm = samples[iw], samples[im]
        out_wt, out_mut, dz = model(batch_from_sample(sw), batch_from_sample(sm), phase)
        total, parts = masked_losses(out_wt, out_mut, sw, sm)
        if torch.isnan(total) or torch.isinf(total):
            raise RuntimeError("❌ Loss became NaN/Inf. Check your CSV values and scaling.")
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        losses.append(float(total.detach().cpu()))
    if kin_ids is not None and len(kin_ids) > 0:
        for idx in kin_ids:
            s = samples[idx]
            if not s.has_kin:
                continue
            kin_mse, total2 = _kin_loss_from_single(model, s, phase)
            if kin_mse is None:
                continue
            if torch.isnan(kin_mse) or torch.isinf(kin_mse):
                raise RuntimeError("❌ Kinetics loss became NaN/Inf. Check kinetics parsing/scaling.")
            opt.zero_grad()
            (0.1 * kin_mse).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
    return float(np.mean(losses)) if losses else np.nan

section('Cell 15 — Main training loop, checkpointing and QC plots')

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
CKPT_DIR = os.path.join(OUTDIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
BEST_PATH = os.path.join(CKPT_DIR, "best.pt")

def _safe_train_epoch(model, pairs_list, opt, phase_id=0, clip=1.0, kin_ids=None):
    """Calls your train_epoch whether or not it supports kin_ids."""
    try:
        return train_epoch(model, pairs_list, opt, phase_id=phase_id, clip=clip, kin_ids=kin_ids)
    except TypeError:
        return train_epoch(model, pairs_list, opt, phase_id=phase_id, clip=clip)
@torch.no_grad()
def eval_pair_components(model, pairs_list, phase_id=0):
    """
    Evaluate Siamese WT->mutant pair objective and extract components from masked_losses:
      - total (pair total)
      - aux_mse (manual-channel reconstruction)
      - risk_bce (if present; likely None in your case)
      - kin_mse_in_pairs (only if your masked_losses applies kin on mutants; usually None here)
    """
    model.eval()
    phase = torch.tensor(int(phase_id), device=device)
    totals, auxes, risks, kins = [], [], [], []
    for iw, im in pairs_list:
        sw, sm = samples[iw], samples[im]
        out_wt, out_mut, dz = model(batch_from_sample(sw), batch_from_sample(sm), phase)
        total, parts = masked_losses(out_wt, out_mut, sw, sm)
        totals.append(float(total.detach().cpu()))
        if "aux_mse" in parts:
            auxes.append(float(parts["aux_mse"].detach().cpu()))
        if "risk_bce" in parts:
            risks.append(float(parts["risk_bce"].detach().cpu()))
        if "kin_mse" in parts:
            kins.append(float(parts["kin_mse"].detach().cpu()))
    return {
        "pair_total": float(np.mean(totals)) if totals else np.nan,
        "aux_mse": float(np.mean(auxes)) if auxes else np.nan,
        "risk_bce": float(np.mean(risks)) if risks else np.nan,
        "kin_mse_in_pairs": float(np.mean(kins)) if kins else np.nan,
    }
@torch.no_grad()
def eval_kin_mse_on_ids(model, kin_ids, phase_id=0):
    """
    WT-only kinetics supervision QC:
    We compute kin MSE by SELF-PAIRING each WT kinetics sample (s,s),
    then comparing out_mut['kin_params'] to s.kin_params.
    """
    if kin_ids is None or len(kin_ids) == 0:
        return np.nan
    model.eval()
    phase = torch.tensor(int(phase_id), device=device)
    mses = []
    for idx in kin_ids:
        s = samples[idx]
        if not getattr(s, "has_kin", False) or (s.kin_params is None):
            continue
        b = batch_from_sample(s)
        out_wt, out_mut, dz = model(b, b, phase)
        y = s.kin_params.to(device)
        mses.append(float(F.mse_loss(out_mut["kin_params"], y).detach().cpu()))
    return float(np.mean(mses)) if mses else np.nan

def _savefig(name):
    p = os.path.join(OUTDIR, name)
    plt.savefig(p, dpi=220, bbox_inches="tight")
    print("✅ Saved:", p)
kin_samples = [i for i, s in enumerate(samples) if getattr(s, "has_kin", False)]
print("✅ Kinetics-labeled samples:", [(i, samples[i].protein, samples[i].variant) for i in kin_samples])
if len(kin_samples) >= 2:
    train_kin_ids = [kin_samples[0]]
    val_kin_ids   = [kin_samples[1]]
elif len(kin_samples) == 1:
    train_kin_ids = [kin_samples[0]]
    val_kin_ids   = []
else:
    train_kin_ids, val_kin_ids = [], []
print("✅ train_kin_ids:", [(i, samples[i].protein, samples[i].variant) for i in train_kin_ids])
print("✅ val_kin_ids:",   [(i, samples[i].protein, samples[i].variant) for i in val_kin_ids])
history = {
    "train_pair_total": [],
    "val_pair_total": [],
    "train_aux_mse": [],
    "val_aux_mse": [],
    "train_kin_mse": [],
    "val_kin_mse": [],
    "grad_norm": [],
}
best_val = float("inf")
patience = 30
bad = 0
EPOCHS = 300
for epoch in range(1, EPOCHS + 1):
    tr = _safe_train_epoch(model, train_pairs, opt, phase_id=0, clip=1.0, kin_ids=train_kin_ids)
    gn = grad_norm(model)
    tr_comp = eval_pair_components(model, train_pairs, phase_id=0)
    va_comp = eval_pair_components(model, val_pairs, phase_id=0)
    tr_kin = eval_kin_mse_on_ids(model, train_kin_ids, phase_id=0)
    va_kin = eval_kin_mse_on_ids(model, val_kin_ids, phase_id=0)
    history["train_pair_total"].append(float(tr_comp["pair_total"]))
    history["val_pair_total"].append(float(va_comp["pair_total"]))
    history["train_aux_mse"].append(float(tr_comp["aux_mse"]))
    history["val_aux_mse"].append(float(va_comp["aux_mse"]))
    history["train_kin_mse"].append(float(tr_kin) if np.isfinite(tr_kin) else np.nan)
    history["val_kin_mse"].append(float(va_kin) if np.isfinite(va_kin) else np.nan)
    history["grad_norm"].append(float(gn))
    print(
        f"Epoch {epoch:03d} | "
        f"pair_train={tr_comp['pair_total']:.4f} | pair_val={va_comp['pair_total']:.4f} | "
        f"aux_train={tr_comp['aux_mse']:.4f} | aux_val={va_comp['aux_mse']:.4f} | "
        f"kin_train={tr_kin if np.isfinite(tr_kin) else np.nan:.4f} | "
        f"kin_val={va_kin if np.isfinite(va_kin) else np.nan:.4f} | "
        f"grad={gn:.3f}"
    )
    val_score = float(va_comp["pair_total"]) if np.isfinite(va_comp["pair_total"]) else np.inf
    if val_score < best_val:
        best_val = val_score
        bad = 0
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "history": history,
            "manifest": manifest.to_dict("records")
        }, BEST_PATH)
    else:
        bad += 1
        if bad >= patience:
            print("✅ Early stopping triggered.")
            break
print("✅ Best checkpoint:", BEST_PATH)
plt.figure()
plt.plot(history["train_pair_total"], label="train_pair_total")
plt.plot(history["val_pair_total"], label="val_pair_total")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
plt.yscale("symlog")
plt.title("SKALE2 QC: Pair Total Loss (symlog)")
_savefig("QC_pair_total_symlog.png")
plt.show()
plt.figure()
plt.plot(history["train_aux_mse"], label="train_aux_mse")
plt.plot(history["val_aux_mse"], label="val_aux_mse")
plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend()
plt.yscale("symlog")
plt.title("SKALE2 QC: Aux Reconstruction MSE (symlog)")
_savefig("QC_aux_mse_symlog.png")
plt.show()
plt.figure()
plt.plot(history["train_kin_mse"], label="train_kin_mse (WT ids)")
plt.plot(history["val_kin_mse"], label="val_kin_mse (WT ids)")
plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend()
plt.title("SKALE2 QC: Kinetics MSE (WT-only supervision)")
_savefig("QC_kin_mse.png")
plt.show()
plt.figure()
plt.plot(history["grad_norm"], label="grad_norm")
plt.xlabel("epoch"); plt.ylabel("L2 norm"); plt.legend()
plt.title("SKALE2 QC: Gradient Norm")
_savefig("QC_grad_norm.png")
plt.show()
print("ℹ️ Risk metrics remain NaN because risk_label is missing in manifest (expected).")

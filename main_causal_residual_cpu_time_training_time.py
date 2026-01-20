# ============================================================
# GTCN – Building env only, feature_mode="both", model_type="both"
# CAUSAL VERSION with training time + CPU inference time
# ============================================================

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ============================================================
# Imports
# ============================================================
import math, re, gc, random, time
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import weight_norm   # WeightNorm only

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
RESULTS_ROOT = Path("results_CPU_TIME_building_both")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# Repro & device
# ============================================================
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Deterministic backends
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=False)
torch.set_num_threads(1)
if hasattr(torch.backends, "cuda"):
    torch.backends.cuda.matmul.allow_tf32 = False
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.allow_tf32 = False

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Data & environments (only building)
# ============================================================
DATA_DIR = "data"
ENV_FILES = {
    "building": ("database_building_train.csv", "database_building_test.csv"),
}
GRID_M_MAP = {"building": 0.6}

# === AP coordinates ===
AP_COORDS_MAP = {
    "building": {
        1:(132.0,1.0),2:(126.2,14.6),3:(115.0,6.0),4:(88.2,11.2),5:(83.0,2.3),
        6:(72.8,19.0),7:(67.6,17.5),8:(53.4,12.0),9:(42.2,1.0),10:(31.0,9.3),
        11:(13.0,14.0),12:(3.3,-1.2),13:(-6.0,12.0),
    },
}

# ============================================================
# Training config (constant defaults; passed explicitly to run_env)
# ============================================================
BATCH        = 64
BASE_LR      = 1e-3
WEIGHT_DECAY = 1e-4
USE_AMP      = False  # strict determinism -> keep False

# Loss weights
HUBER_BETA       = 1.0
W_HUBER, W_EUCL, W_A_L1 = 1.0, 0.5, 1e-3
LAMBDA_SPARS, LAMBDA_SMOOTH = 1e-3, 1e-3

# Fixed model hyperparameters
FIXED_CFG = {
    "hidden": 64,
    "gcn_layers": 3,
    "tcn_blocks": 3,
    "tcn_kernel": 3,
    "dilations": (1,2,4,8),
    "gate_hidden": 32,
    "gate_min_w": 0.05,
    "gate_temp": 1.0,
    "dropout": 0.05,
    "wd": WEIGHT_DECAY,
}

# ============================================================
# AMP shim
# ============================================================
try:
    from torch.amp import autocast as _autocast
    def autocast(enabled=True):
        return _autocast(device_type="cuda", enabled=(enabled and torch.cuda.is_available()))
    from torch.amp import GradScaler
except Exception:
    from torch.cuda.amp import autocast as _autocast, GradScaler
    def autocast(enabled=True):
        return _autocast(enabled=(enabled and torch.cuda.is_available()))

# ============================================================
# Data utils
# ============================================================
RSS_SENTINEL = -200.0
RTT_SENTINEL_MM = 100000.0
AP_RTT_RE = re.compile(r"^AP(\d+)\s+RTT\(mm\)$")
AP_RSS_RE = re.compile(r"^AP(\d+)\s+RSS\(dBm\)$")

def load_df(p: Path):
    df = pd.read_csv(p, dtype={"LOS APs": "string"}, low_memory=False)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "LOS APs" in df.columns:
        df["LOS APs"] = df["LOS APs"].fillna("")
    return df

def infer_ap_indices(df: pd.DataFrame):
    rss_idxs, rtt_idxs = set(), set()
    for c in df.columns:
        m = AP_RSS_RE.match(c)
        if m: rss_idxs.add(int(m.group(1)))
        m = AP_RTT_RE.match(c)
        if m: rtt_idxs.add(int(m.group(1)))
    idxs = sorted(rss_idxs) if rss_idxs else sorted(rss_idxs | rtt_idxs)
    return idxs

def build_adjacency_full_inv_dist(coords_xy):
    P = np.array(coords_xy, dtype=float)
    N = P.shape[0]
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    with np.errstate(divide='ignore'):
        A = np.where(np.eye(N, dtype=bool), 0.0, 1.0/(D + 1e-12))
    A_tilde = A + np.eye(N, dtype=float)
    deg = A_tilde.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-12))
    return (D_inv_sqrt @ A_tilde @ D_inv_sqrt).astype(np.float32)

def compute_stats_train(df, ap_idxs):
    rss_mean_ap, rss_std_ap, rtt_mean_ap, rtt_std_ap = {}, {}, {}, {}
    for i in ap_idxs:
        rss = df[f"AP{i} RSS(dBm)"].values.astype(float)
        rss_valid = rss[(rss != RSS_SENTINEL) & np.isfinite(rss)]
        if rss_valid.size == 0:
            rss_mean_ap[i], rss_std_ap[i] = -70.0, 10.0
        elif rss_valid.size == 1:
            rss_mean_ap[i], rss_std_ap[i] = float(rss_valid.mean()), 10.0
        else:
            m = float(rss_valid.mean()); s = float(np.std(rss_valid, ddof=1))
            rss_mean_ap[i], rss_std_ap[i] = m, max(s, 1e-6)

        rtt_mm = df[f"AP{i} RTT(mm)"].values.astype(float)
        rtt_m  = (rtt_mm[(rtt_mm != RTT_SENTINEL_MM) & np.isfinite(rtt_mm)] / 1000.0)
        if rtt_m.size == 0:
            rtt_mean_ap[i], rtt_std_ap[i] = 10.0, 1.0
        elif rtt_m.size == 1:
            rtt_mean_ap[i], rtt_std_ap[i] = float(rtt_m.mean()), 1.0
        else:
            m = float(rtt_m.mean()); s = float(np.std(rtt_m, ddof=1))
            rtt_mean_ap[i], rtt_std_ap[i] = m, max(s, 1e-6)
    return rss_mean_ap, rss_std_ap, rtt_mean_ap, rtt_std_ap

def build_features_keep_sentinels(df, ap_xy_z, stats, ap_idxs, grid_m=0.6, use_ap_geom=True, feature_mode="both"):
    rss_mean_ap, rss_std_ap, rtt_mean_ap, rtt_std_ap = stats
    X_scans, y_scans, rp_keys = [], [], []
    for _, row in df.iterrows():
        nodes = []
        for i_idx, i in enumerate(ap_idxs):
            # --- RSS ---
            raw_rss = float(row[f"AP{i} RSS(dBm)"])
            if (not np.isfinite(raw_rss)) or (raw_rss == RSS_SENTINEL):
                rss_val, rss_mask, rss_z = RSS_SENTINEL, 0.0, 0.0
            else:
                rss_val, rss_mask = raw_rss, 1.0
                rss_z = (rss_val - rss_mean_ap[i]) / rss_std_ap[i]
            # --- RTT ---
            raw_rtt_mm = float(row[f"AP{i} RTT(mm)"])
            if (not np.isfinite(raw_rtt_mm)) or (raw_rtt_mm == RTT_SENTINEL_MM):
                rtt_m, rtt_mask, rtt_z = 100.0, 0.0, 0.0
            else:
                rtt_m, rtt_mask = raw_rtt_mm / 1000.0, 1.0
                rtt_z = (rtt_m - rtt_mean_ap[i]) / rtt_std_ap[i]
            # --- Feature vector ---
            feat = []
            if feature_mode in ["rss", "both"]: feat.extend([rss_val, rss_mask, rss_z])
            if feature_mode in ["rtt", "both"]: feat.extend([rtt_m, rtt_mask, rtt_z])
            if use_ap_geom:
                apx_z, apy_z = ap_xy_z[i_idx].tolist(); feat.extend([apx_z, apy_z])
            nodes.append(np.array(feat, np.float32))
        X_scans.append(np.stack(nodes, 0))
        y_scans.append([row["X"]*grid_m, row["Y"]*grid_m])
        rp_keys.append((row["X"], row["Y"]))
    return np.stack(X_scans).astype(np.float32), np.array(y_scans, np.float32), rp_keys

def pack_sequences_by_rp_relaxed(X_scans, y_scans_m, rp_keys, T_expected: int):
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, key in enumerate(rp_keys): buckets[key].append(i)
    X_seq, y_seq, kept, dropped = [], [], 0, 0
    for k in sorted(buckets.keys()):
        idxs = sorted(buckets[k])
        if len(idxs) < T_expected:
            dropped += 1; continue
        idxs = idxs[:T_expected]
        X_seq.append(X_scans[idxs]); y_seq.append(y_scans_m[idxs]); kept += 1
    if kept == 0:
        raise RuntimeError(f"No RP buckets matched T_expected={T_expected}.")
    if dropped > 0:
        print(f"[pack] Dropped {dropped} RP(s) with < {T_expected} scans; kept {kept}.")
    return np.stack(X_seq), np.stack(y_seq)

# ---------- RF-RMSE (per-scan) ----------
def rf_rmse_per_scan(pred_n, gt_n, y_mean, y_std):
    pred_m = pred_n * y_std + y_mean
    gt_m   = gt_n   * y_std + y_mean
    diff   = (pred_m - gt_m).cpu().numpy()
    rf_rmse_scan = float(np.sqrt(np.mean((diff**2).mean(axis=1))))
    return rf_rmse_scan, diff

def infer_expected_T(df_train: pd.DataFrame) -> int:
    counts = df_train.groupby(["X","Y"]).size()
    return int(counts.mode().iloc[0])

# --------------- Rolling statistics ---------------
EPS = 1e-8
MIN_COUNT_STATS = 3

def _nanstd(a, ddof=1):
    m = np.nanmean(a); v = np.nanmean((a - m)**2)
    n = np.sum(~np.isnan(a))
    if n - ddof <= 0: return np.nan
    v = v * (n / max(n - ddof, 1))
    return np.sqrt(max(v, 0.0))

def _skew_kurt(a):
    x = a[np.isfinite(a)]; n = x.size
    if n < 3: return (np.nan, np.nan)
    mu = np.mean(x)
    m2 = np.mean((x - mu)**2); m3 = np.mean((x - mu)**3); m4 = np.mean((x - mu)**4)
    if m2 < EPS: return (0.0, -3.0)
    skew = m3 / ((m2**1.5) + EPS); kurt = m4 / ((m2**2) + EPS) - 3.0
    return (skew, kurt)

def add_rolling_stats_per_ap(X_seq, vars_to_process, windows, min_count=MIN_COUNT_STATS):
    """
    X_seq: (B,T,N,F_base)
    vars_to_process: list of (z_idx, mask_idx)
    windows: list[int]
    """
    B, T, N, F = X_seq.shape
    out_list = [X_seq]

    for z_score_idx, mask_idx in vars_to_process:
        feats_for_var = []
        vals = X_seq[..., z_score_idx]
        mask = X_seq[..., mask_idx] > 0.5
        vals = np.array(vals, dtype=np.float32); mask = np.array(mask, dtype=bool)

        for W in windows:
            mean = np.zeros((B,T,N), np.float32)
            med  = np.zeros((B,T,N), np.float32)
            std  = np.zeros((B,T,N), np.float32)
            skew = np.zeros((B,T,N), np.float32)
            kurt = np.zeros((B,T,N), np.float32)

            for b in range(B):
                for n in range(N):
                    series = vals[b, :, n]; valid = mask[b, :, n]
                    for t in range(T):
                        s = max(0, t - W + 1); e = t + 1
                        w_vals = series[s:e]; w_mask = valid[s:e]
                        if np.sum(w_mask) < min_count: continue
                        xw = np.where(w_mask, w_vals, np.nan)
                        mean[b,t,n] = float(np.nanmean(xw))
                        med[b,t,n]  = float(np.nanmedian(xw))
                        std_val = _nanstd(xw, ddof=1)
                        std[b,t,n]  = 0.0 if (not np.isfinite(std_val)) else float(std_val)
                        sk, ku = _skew_kurt(xw)
                        skew[b,t,n] = 0.0 if (not np.isfinite(sk)) else float(sk)
                        kurt[b,t,n] = 0.0 if (not np.isfinite(ku)) else float(ku)

            featsW = np.stack([mean, med, std, skew, kurt], axis=-1)
            feats_for_var.append(featsW)

        feats_var_allW = np.concatenate(feats_for_var, axis=-1)
        out_list.append(feats_var_allW)

    X_aug = np.concatenate(out_list, axis=-1)
    return X_aug

# ============================================================
# Model - CAUSAL VERSION
# ============================================================
class ResidualAdjacency(nn.Module):
    def __init__(self, A_init=None, N=None, alpha_init=0.1):
        super().__init__()
        if A_init is None:
            assert N is not None
            A0 = torch.eye(N, dtype=torch.float32)
        else:
            A0 = torch.tensor(A_init, dtype=torch.float32)
        self.N = A0.shape[0]
        self.register_buffer("A_base", A0)
        self.S = nn.Parameter(torch.zeros(self.N, self.N))
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(alpha_init)))
    def forward(self):
        R = torch.tanh(0.5*(self.S + self.S.t()))
        alpha = torch.clamp(torch.exp(self.log_alpha), 0.0, 1.0)
        A = torch.clamp(self.A_base + alpha*R, min=0.0)
        A = A + torch.eye(self.N, device=A.device)
        deg = A.sum(1).clamp_min(1e-6)
        D_inv_sqrt = torch.diag(torch.rsqrt(deg))
        return D_inv_sqrt @ A @ D_inv_sqrt
    def l1_penalty(self):
        return torch.tanh(self.S).abs().mean()

class GraphBlock(nn.Module):
    def __init__(self, in_f, out_f, dropout=0.1):
        super().__init__()
        self.lin  = nn.Linear(in_f, out_f, bias=True)
        self.norm = nn.LayerNorm(out_f)
        self.drop = nn.Dropout(dropout)
        self.res  = nn.Linear(in_f, out_f) if in_f != out_f else nn.Identity()
    def forward(self, x, A):
        B, T, N, C = x.shape
        xb = x.reshape(B*T, N, C)
        Ax = torch.einsum("ij,bjf->bif", A, xb)
        h  = self.lin(Ax)
        h  = F.relu(self.norm(h))
        h  = self.drop(h).reshape(B, T, N, -1)
        r  = self.res(x)
        return h + r

class CausalTemporalBlock(nn.Module):
    """
    CAUSAL Temporal Block with WeightNorm (Bai et al., 2018)
    Two Conv1d layers (k=3) with dilation d, left-crop to ensure causality,
    ReLU + Dropout, residual add.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1, use_weightnorm=True):
        super().__init__()
        pad = (kernel_size - 1) * dilation

        conv1 = nn.Conv1d(in_ch,  out_ch, kernel_size, padding=pad, dilation=dilation)
        conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.c1 = weight_norm(conv1) if use_weightnorm else conv1
        self.c2 = weight_norm(conv2) if use_weightnorm else conv2

        self.act  = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        self.pad = pad
        self.use_weightnorm = use_weightnorm

    def forward(self, x):
        y = self.drop(self.act(self.c1(x)))
        y = self.drop(self.act(self.c2(y)))
        res = self.res(x)

        # crop to input length (strict causality)
        if y.size(2) > x.size(2):   y   = y[:, :, :x.size(2)]
        if res.size(2) > x.size(2): res = res[:, :, :x.size(2)]
        return y + res

class NodeGate(nn.Module):
    def __init__(self, feat_dim, hidden=32, min_w=0.05, temperature=1.0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.min_w = min_w
        self.log_temp = nn.Parameter(torch.tensor(math.log(temperature), dtype=torch.float32))
    def forward(self, x):
        t = torch.exp(self.log_temp).clamp_min(1e-3)
        w = torch.sigmoid(self.mlp(x).squeeze(-1) / t)
        if self.min_w > 0: w = self.min_w + (1.0 - self.min_w) * w
        return w

class CausalGCN_TCN_PerScan(nn.Module):
    """
    CAUSAL VERSION: GCN + TCN ("both") or subsets.
    legacy_both: gate BEFORE GCN, no pre-projection (matches original).
    """
    def __init__(self, N, in_feats, hidden=128, gcn_layers=2, tcn_blocks=4,
                 tcn_kernel=3, dilations=(1,2,4,8), dropout=0.1,
                 A_init=None, gate_hidden=32, gate_min_w=0.05, gate_temp=1.0,
                 huber_beta=1.0, use_gcn=True, use_tcn=True):
        super().__init__()
        self.use_gcn = use_gcn
        self.use_tcn = use_tcn
        self.legacy_both = (use_gcn and use_tcn)

        self.resA = ResidualAdjacency(A_init=A_init, N=N)
        self.gate = NodeGate(in_feats, hidden=gate_hidden, min_w=gate_min_w, temperature=gate_temp)

        if self.legacy_both:
            self.in_proj = nn.Identity()
            first_in = in_feats
        else:
            self.in_proj = nn.Linear(in_feats, hidden)
            first_in = hidden

        # GCN stack
        if self.use_gcn:
            gblocks = [GraphBlock(first_in, hidden, dropout)]
            for _ in range(1, gcn_layers):
                gblocks.append(GraphBlock(hidden, hidden, dropout))
            self.gblocks = nn.ModuleList(gblocks)
        else:
            self.gblocks = nn.ModuleList()

        # CAUSAL TCN stack
        if self.use_tcn and tcn_blocks > 0:
            blocks, in_ch = [], hidden
            for d in list(dilations)[:tcn_blocks]:
                blocks.append(CausalTemporalBlock(
                    in_ch, hidden,
                    kernel_size=tcn_kernel,
                    dilation=d,
                    dropout=dropout,
                    use_weightnorm=True
                ))
                in_ch = hidden
            self.tcn = nn.Sequential(*blocks)
        else:
            self.tcn = nn.Identity()

        self.head_scan = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2)
        )
        self.huber_beta = huber_beta

    def forward(self, x, return_weights=False):
        # x: [B,T,N,F]
        A = self.resA() if self.use_gcn else None
        w = self.gate(x)  # [B,T,N]

        if self.legacy_both:
            h = x * w.unsqueeze(-1)
        else:
            h = self.in_proj(x)

        if self.use_gcn:
            for gb in self.gblocks:
                h = gb(h, A)

        # Weighted node pooling -> [B,T,H]
        w_sum = w.sum(dim=2, keepdim=True).clamp_min(1e-6)
        h_pool = (h * w.unsqueeze(-1)).sum(dim=2) / w_sum

        # CAUSAL temporal modeling
        ht = self.tcn(h_pool.permute(0,2,1)).permute(0,2,1) if not isinstance(self.tcn, nn.Identity) else h_pool

        out = self.head_scan(ht)
        return (out, w) if return_weights else out

# ---- ptflops helper ----
def print_complexity_with_ptflops(model, t_len, n_ap, feat_dim, device, out_dir, print_per_layer=False):
    try:
        from ptflops import get_model_complexity_info
    except Exception as e:
        print(f"[ptflops] Not available: {e}")
        return None

    import torch.nn as nn

    class _Wrap(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, x):
            return self.m(x)

    wrap = _Wrap(model).to(device)
    input_res = (t_len, n_ap, feat_dim)  # (T, N, F)

    was_training = model.training
    model.eval()
    macs, params = get_model_complexity_info(
        wrap,
        input_res,
        as_strings=False,
        print_per_layer_stat=print_per_layer,
        verbose=print_per_layer
    )
    if was_training:
        model.train()

    flops = float(macs) * 2.0
    txt = (
        f"[ptflops] Input shape (T,N,F): {list(input_res)}\n"
        f"[ptflops] Params: {int(params):,}\n"
        f"[ptflops] MACs  : {macs/1e6:.2f}M\n"
        f"[ptflops] FLOPs : {flops/1e6:.2f}M  (≈ 2 × MACs)\n"
    )
    print("\n" + txt)

    try:
        out_path = Path(out_dir) / "model_complexity_ptflops.txt"
        with open(out_path, "w") as f:
            f.write(txt)
        print(f"[ptflops] Saved: {out_path}")
    except Exception:
        pass

    return {"params": int(params), "macs": float(macs), "flops": flops}

# ---- CPU inference time helper ----
def measure_cpu_inference_time(model, example_input, n_warmup=10, n_runs=100, n_threads=1):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(n_threads)

    model_cpu = model.to("cpu").eval()
    x_cpu = example_input.to("cpu")

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model_cpu(x_cpu)

        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model_cpu(x_cpu)
        t1 = time.perf_counter()

    avg_ms_per_seq = (t1 - t0) * 1000.0 / n_runs
    T_len = x_cpu.shape[1]
    avg_ms_per_scan = avg_ms_per_seq / float(T_len)

    print(
        f"[CPU-TIME] threads={n_threads} | "
        f"avg per sequence (RP) = {avg_ms_per_seq:.4f} ms, "
        f"per scan = {avg_ms_per_scan:.4f} ms "
        f"(T={T_len}, n_runs={n_runs})"
    )

    torch.set_num_threads(old_threads)
    return avg_ms_per_seq, avg_ms_per_scan

# ============================================================
# Train/Eval for building, feature_mode="both", model_type="both"
# ============================================================
def run_building_both(cfg_base: dict, epochs: int, eval_every: int, roll_windows):
    env_name = "building"
    feature_mode = "both"
    model_type = "both"

    assert model_type in {"gcn","tcn","both"}
    set_seed(SEED)

    TRAIN_CSV, TEST_CSV = ENV_FILES[env_name]
    grid_m = GRID_M_MAP[env_name]
    out_dir = RESULTS_ROOT / f"artifacts_causal_{env_name}_{feature_mode}_{model_type}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_name = str(out_dir)

    print(f"\n=== CAUSAL {env_name.upper()} | mode={feature_mode.upper()} | model={model_type.upper()} "
          f"| epochs={epochs} | eval_every={eval_every} | rolls={list(roll_windows)} | out={out_dir_name} ===")

    # Data
    df_tr = load_df(Path(DATA_DIR)/TRAIN_CSV); df_te = load_df(Path(DATA_DIR)/TEST_CSV)
    T_expected = infer_expected_T(df_tr)
    print(f"[{env_name}] Using T_expected={T_expected} scans/RP")

    ap_idxs_csv = infer_ap_indices(df_tr)
    coords_map = AP_COORDS_MAP.get(env_name)
    if coords_map is None:
        ap_idxs = ap_idxs_csv
        print(f"[{env_name}] Using AP indices from CSV: {ap_idxs}")
    else:
        ap_idxs = sorted(coords_map.keys())
        print(f"[{env_name}] Using AP indices from coords: {ap_idxs}")
    N = len(ap_idxs)

    A0 = None; ap_xy_z = np.zeros((N,2), np.float32)
    if coords_map is not None and len(coords_map) == N:
        coords = [coords_map[i] for i in ap_idxs]
        A0 = build_adjacency_full_inv_dist(coords)
        ap_xy = np.array(coords, np.float32)
        ap_xy_z = (ap_xy - ap_xy.mean(0, keepdims=True)) / (ap_xy.std(0, keepdims=True)+1e-6)

    stats = compute_stats_train(df_tr, ap_idxs)
    Xtr_scans, ytr_scans_m, tr_keys = build_features_keep_sentinels(
        df_tr, ap_xy_z, stats, ap_idxs, grid_m, True, feature_mode=feature_mode)
    Xte_scans, yte_scans_m, te_keys = build_features_keep_sentinels(
        df_te, ap_xy_z, stats, ap_idxs, grid_m, True, feature_mode=feature_mode)

    Xtr_seq_np, ytr_seq_m = pack_sequences_by_rp_relaxed(Xtr_scans, ytr_scans_m, tr_keys, T_expected)
    Xte_seq_np, yte_seq_m = pack_sequences_by_rp_relaxed(Xte_scans, yte_scans_m, te_keys, T_expected)

    # Rolling stats for BOTH
    vars_to_process = [(2, 1), (5, 4)]
    print(f"[{env_name}] Adding rolling stats for mode='{feature_mode}' with windows={list(roll_windows)}")
    Xtr_seq_np = add_rolling_stats_per_ap(Xtr_seq_np, vars_to_process, windows=list(roll_windows))
    Xte_seq_np = add_rolling_stats_per_ap(Xte_seq_np, vars_to_process, windows=list(roll_windows))

    # Targets normalization
    y_mean = torch.tensor(ytr_seq_m.reshape(-1,2).mean(axis=0), dtype=torch.float32, device=device)
    y_std  = torch.tensor(np.clip(ytr_seq_m.reshape(-1,2).std(axis=0), 1e-6, None), dtype=torch.float32, device=device)

    Xtr = torch.tensor(Xtr_seq_np, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr_seq_m,  dtype=torch.float32, device=device)
    Xte = torch.tensor(Xte_seq_np, dtype=torch.float32, device=device)
    yte = torch.tensor(yte_seq_m,  dtype=torch.float32, device=device)
    ytr_n = (ytr - y_mean)/y_std; yte_n = (yte - y_mean)/y_std

    # DataLoaders
    g = torch.Generator(device="cpu").manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(Xtr, ytr_n), batch_size=BATCH, shuffle=True, generator=g, num_workers=0, drop_last=False)
    test_loader  = DataLoader(TensorDataset(Xte, yte_n), batch_size=BATCH, shuffle=False, num_workers=0, drop_last=False)

    # Model flags (both = GCN + TCN)
    use_gcn = True
    use_tcn = True

    cfg = dict(cfg_base)
    model = CausalGCN_TCN_PerScan(
        N=N, in_feats=Xtr.shape[-1], hidden=cfg["hidden"], gcn_layers=cfg["gcn_layers"],
        tcn_blocks=cfg["tcn_blocks"], tcn_kernel=cfg["tcn_kernel"], dilations=cfg["dilations"],
        dropout=cfg["dropout"], A_init=A0, gate_hidden=cfg["gate_hidden"],
        gate_min_w=cfg["gate_min_w"], gate_temp=cfg["gate_temp"], huber_beta=HUBER_BETA,
        use_gcn=use_gcn, use_tcn=use_tcn
    ).to(device)

    # --- ptflops profile ---
    T_len   = int(Xtr.shape[1])
    N_ap    = int(Xtr.shape[2])
    featdim = int(Xtr.shape[3])
    complexity = print_complexity_with_ptflops(
        model,
        t_len=T_len,
        n_ap=N_ap,
        feat_dim=featdim,
        device=device,
        out_dir=out_dir,
        print_per_layer=False
    )
    params = complexity["params"] if complexity is not None else None
    macs   = complexity["macs"]   if complexity is not None else None
    flops  = complexity["flops"]  if complexity is not None else None

    opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=cfg["wd"])
    scaler = GradScaler(enabled=(torch.cuda.is_available() and USE_AMP))
    huber = nn.SmoothL1Loss(beta=HUBER_BETA, reduction='none')

    # Eval helper
    eval_rows = []
    def do_eval(epoch_tag: int):
        model.eval()
        preds_n, gts_n = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb)
                preds_n.append(out.reshape(-1,2).cpu()); gts_n.append(yb.reshape(-1,2).cpu())
        pred_n = torch.cat(preds_n, 0); gt_n = torch.cat(gts_n, 0)
        rf_rmse, diff = rf_rmse_per_scan(pred_n, gt_n, y_mean.cpu(), y_std.cpu())
        print(f"[EVAL] CAUSAL {env_name} | {feature_mode} | {model_type} | epoch={epoch_tag:04d} | RF-RMSE={rf_rmse:.3f} m")
        eval_rows.append({"epoch": int(epoch_tag), "rf_rmse": float(rf_rmse)})
        return rf_rmse, diff

    # ---- Measure training time ----
    train_start = time.perf_counter()

    # ---- Train with periodic eval ----
    for epoch in range(1, epochs+1):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                out, w = model(xb, return_weights=True)

                # Validity masks
                rss_mask_idx, rtt_mask_idx = 1, 4  # for feature_mode="both"
                valid_sum = torch.zeros_like(xb[...,0])
                valid_sum += xb[..., rss_mask_idx]
                valid_sum += xb[..., rtt_mask_idx]
                valid = torch.clamp(valid_sum, 0, 1)

                frac_valid = valid.mean(dim=2, keepdim=True)
                per_el = huber(out, yb).mean(dim=2, keepdim=True)
                loss_huber = (per_el * (frac_valid + 1e-3)).mean()

                out_m = out * y_std + y_mean
                y_m   = yb  * y_std + y_mean
                loss_eucl = torch.sqrt(torch.sum((out_m - y_m)**2, dim=2) + 1e-8).mean()

                reg = 0.0
                if LAMBDA_SPARS  > 0: reg += LAMBDA_SPARS  * w.mean()
                if LAMBDA_SMOOTH > 0 and w.size(1) > 1:
                    reg += LAMBDA_SMOOTH * (w[:,1:,:] - w[:,:-1,:]).abs().mean()

                loss = W_HUBER*loss_huber + W_EUCL*loss_eucl + reg + W_A_L1 * model.resA.l1_penalty()

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        if (eval_every is not None) and (epoch % eval_every == 0):
            do_eval(epoch)

    # ---- Final evaluation (GPU) ----
    final_rf_rmse, diff = do_eval(epochs)

    # Stop training timer
    train_end = time.perf_counter()
    train_seconds = train_end - train_start
    train_minutes = train_seconds / 60.0
    print(f"[TRAIN] {env_name} | {feature_mode} | {model_type} | "
          f"{epochs} epochs took {train_minutes:.2f} min ({train_seconds:.1f} s)")

    # Euclidean error distances for CDF
    err_dists = np.sqrt((diff**2).sum(axis=1))

    # ---- CPU inference timing (after training on GPU) ----
    X_example = Xte[:1].detach()
    avg_ms_seq, avg_ms_scan = measure_cpu_inference_time(
        model,
        X_example,
        n_warmup=10,
        n_runs=100,
        n_threads=1,
    )

    # Save summary CSV (RF-RMSE + complexity + CPU + training time)
    row = {
        "env": env_name,
        "feature_mode": feature_mode,
        "model_type": model_type,
        "rf_rmse": float(final_rf_rmse),
        "N_AP": N,
        "grid_m": grid_m,
        "T_used": int(Xtr.shape[1]),
        "feat_dim": int(Xtr.shape[-1]),
        "causal": True,
        "cpu_ms_per_seq": float(avg_ms_seq),
        "cpu_ms_per_scan": float(avg_ms_scan),
        "params": params,
        "macs": macs,
        "flops": flops,
        "train_seconds": float(train_seconds),
        "train_minutes": float(train_minutes),
        "train_min_per_epoch": float(train_minutes / epochs),
        "epochs": epochs,
        "roll_windows": str(list(roll_windows)),
    }
    pd.DataFrame([row]).to_csv(out_dir/f"results_causal_building_both_both.csv", index=False)

    # Save distances and per-epoch RF-RMSE log
    np.save(out_dir/f"errors_causal_building_both_both.npy", err_dists)
    df_eval = pd.DataFrame(eval_rows)
    df_eval.to_csv(out_dir/f"eval_curve_causal_building_both_both.csv", index=False)

    # Optional RF-RMSE curve
    try:
        import matplotlib.pyplot as plt
        if len(df_eval) > 0:
            plt.figure(figsize=(6.5,4.2))
            plt.plot(df_eval["epoch"], df_eval["rf_rmse"])
            plt.xlabel("Epoch"); plt.ylabel("RF-RMSE (m)")
            plt.title(f"CAUSAL RF-RMSE vs Epoch – {env_name} (both,both)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(out_dir/f"rf_rmse_curve_causal_building_both_both.png", dpi=150)
            plt.close()
    except Exception:
        pass

    torch.cuda.empty_cache(); gc.collect()
    return row, err_dists

# ============================================================
# Main: run ONLY building / both / both
# ============================================================
if __name__ == "__main__":
    from datetime import datetime

    total_epochs = 1500
    rolls = [5, 10, 20]
    eval_every = 200

    print("\n========== CAUSAL RUN: BUILDING | mode=BOTH | model=BOTH ==========\n")
    row, dists = run_building_both(
        cfg_base=FIXED_CFG,
        epochs=total_epochs,
        eval_every=eval_every,
        roll_windows=rolls,
    )
    print("\n=== FINAL SUMMARY (building, both, both) ===")
    print(row)
    print("\nResults saved under:", RESULTS_ROOT)

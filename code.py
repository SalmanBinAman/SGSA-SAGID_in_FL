
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import json

# --------------------------------------------------------------------------- #
#  Config                                                                      #
# --------------------------------------------------------------------------- #
CFG = dict(
    dataset="mnist",            # 'mnist' or 'fmnist'
    n_clients=20,
    mal_frac=0.40,              # 8/20 malicious; keep < 0.5 (SAGID needs honest majority)
    rounds=40,
    k_frac=0.10,                # top-k sparsification budget (10% of parameters)
    local_epochs=1,
    batch=64,
    lr=0.05,
    momentum=0.9,

    # ---- SGSA (attack) knobs ----
    attack_boost=2.5,           # payload norm = boost * honest-top-k norm. >1.5 suppresses/reverses
    attack_concentration=6,     # focus payload on the top-(k/concentration) most-important coords
    attack_noise=0.0,           # optional jitter before norm-matching (0 = pure)

    # ---- SAGID (defense) knobs ----
    ema=0.8,                    # server expected-gradient model EMA factor
    rand_drop=0.02,             # randomized coordinate dropout in robust aggregation
    blacklist_vote=0.5,         # persistent-blacklist threshold (fraction of decision rounds)

    # ---- numerical safeguard (uniform across ALL scenarios) ----
    clip_multiple=6.0,          # per-client update-norm ceiling = clip_multiple * round-0 median norm

    train_subset=12000,         # samples used for federation (speed); raise for full data
    seed=42,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
print("Using device:", DEVICE)


def _finite(x, fill=0.0):
    """Return x with NaN/Inf replaced (works for python float, np array, torch tensor)."""
    if isinstance(x, torch.Tensor):
        return torch.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)
    a = np.asarray(x, dtype=float)
    return np.nan_to_num(a, nan=fill, posinf=fill, neginf=fill)


# --------------------------------------------------------------------------- #
#  Data                                                                        #
# --------------------------------------------------------------------------- #
def load_data(cfg):
    from torchvision import datasets
    norm = (0.1307, 0.3081) if cfg["dataset"] == "mnist" else (0.2860, 0.3530)
    ctor = datasets.MNIST if cfg["dataset"] == "mnist" else datasets.FashionMNIST
    tr = ctor(root="./data", train=True, download=True)
    te = ctor(root="./data", train=False, download=True)

    def to_xy(ds, idx=None):
        X = ds.data.float().div(255.).sub(norm[0]).div(norm[1]).unsqueeze(1)  # (N,1,28,28)
        y = ds.targets.clone()
        if idx is not None:
            X, y = X[idx], y[idx]
        return X.to(DEVICE), y.to(DEVICE)

    g = torch.Generator().manual_seed(cfg["seed"])
    idx = torch.randperm(len(tr), generator=g)[:cfg["train_subset"]]
    Xtr, ytr = to_xy(tr, idx)
    Xte, yte = to_xy(te)
    return Xtr, ytr, Xte, yte


def partition_iid(n_samples, n_clients, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_samples, generator=g)
    return [perm[i::n_clients] for i in range(n_clients)]   # round-robin = IID shards


# --------------------------------------------------------------------------- #
#  Model                                                                       #
# --------------------------------------------------------------------------- #
class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)   # 28 -> 14
        x = F.max_pool2d(F.relu(self.c2(x)), 2)   # 14 -> 7
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


WORKER = SmallCNN().to(DEVICE)   # reusable worker: weights overwritten per client/eval


def get_flat():
    return parameters_to_vector(WORKER.parameters()).detach().clone()


def set_flat(vec):
    vector_to_parameters(vec, WORKER.parameters())


@torch.no_grad()
def evaluate(global_vec, Xte, yte):
    set_flat(_finite(global_vec)); WORKER.eval()
    preds = []
    for i in range(0, Xte.shape[0], 1024):
        logits = _finite(WORKER(Xte[i:i + 1024]))
        preds.append(logits.argmax(1))
    preds = torch.cat(preds).cpu().numpy()
    yt = yte.cpu().numpy()
    return dict(
        acc=accuracy_score(yt, preds),
        prec=precision_score(yt, preds, average="macro", zero_division=0),
        rec=recall_score(yt, preds, average="macro", zero_division=0),
        f1=f1_score(yt, preds, average="macro", zero_division=0),
    )


def local_update(global_vec, Xc, yc, lr, local_epochs, batch, momentum):
    """Train the worker on one client's data; return delta = w_local - w_global."""
    set_flat(_finite(global_vec).clone()); WORKER.train()
    opt = torch.optim.SGD(WORKER.parameters(), lr=lr, momentum=momentum)
    n = Xc.shape[0]
    for _ in range(local_epochs):
        perm = torch.randperm(n, device=Xc.device)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = F.cross_entropy(WORKER(Xc[b]), yc[b])
            loss.backward(); opt.step()
    return _finite(get_flat() - global_vec)


# --------------------------------------------------------------------------- #
#  Sparsification: honest top-k  vs  SGSA                                      #
# --------------------------------------------------------------------------- #
def topk_sparsify(vec, k_frac):
    """Honest compression: keep the top-k coordinates by |magnitude|."""
    k = max(1, int(round(k_frac * vec.numel())))
    out = torch.zeros_like(vec)
    idx = torch.topk(vec.abs(), k).indices
    out[idx] = vec[idx]
    return out


def sgsa_sparsify(vec, k_frac, concentration, boost, noise):
    """
    SELECTIVE GRADIENT SUPPRESSION ATTACK (v2 — strengthened, coordinated).
      1. N = ||honest top-k|| (stealth reference norm)
      2. pick top j = k/concentration coords by |magnitude| (most important; overlap
         across clients under IID -> coordinated)
      3. send SIGN-FLIPPED gradient there (suppression/reversal)
      4. rescale payload norm to N * boost  (boost=1 -> norm-stealthy; boost>1.5 -> reverses)
    """
    k = max(1, int(round(k_frac * vec.numel())))
    a = vec.abs()
    top_idx = torch.topk(a, k).indices
    target_norm = vec[top_idx].norm() + 1e-12

    j = max(1, k // max(1, concentration))
    imp_idx = torch.topk(a, j).indices
    payload = -torch.sign(vec[imp_idx]) * a[imp_idx]
    if noise > 0:
        payload = payload + noise * torch.randn(j, device=vec.device)

    out = torch.zeros_like(vec)
    out[imp_idx] = payload
    out = out / (out.norm() + 1e-12) * target_norm * boost
    return _finite(out)


# --------------------------------------------------------------------------- #
#  SAGID defense                                                               #
# --------------------------------------------------------------------------- #
def sagid_suspicion(U, g_exp, k_frac):
    """Per-client suspicion using the server expected-gradient model g_exp (EMA of
    accepted aggregates). On E = top-k(|g_exp|): 0.5*sign-disagreement + 0.5*(1-cos).
    Cold start (g_exp ~ 0): fall back to median pairwise cosine to bootstrap."""
    C, D = U.shape
    k = max(1, int(round(k_frac * D)))

    if g_exp is None or g_exp.norm() < 1e-8:
        Un = U / (U.norm(dim=1, keepdim=True) + 1e-12)
        S = Un @ Un.T
        S.fill_diagonal_(float("nan"))
        med = torch.nanmedian(S, dim=1).values
        susp = (1.0 - med).clamp(0.0, 2.0) / 2.0
        return _finite(susp.cpu().numpy(), fill=0.5)

    E = torch.topk(g_exp.abs(), k).indices
    ref = g_exp[E]
    refsign = torch.sign(ref)
    UE = U[:, E]
    active = UE != 0

    dis = ((torch.sign(UE) != refsign) & active).float().sum(1) / (active.float().sum(1) + 1e-12)
    cos = (UE @ ref) / (UE.norm(dim=1) * ref.norm() + 1e-12)
    cons = (1.0 - cos).clamp(0.0, 2.0) / 2.0

    susp = 0.5 * dis + 0.5 * cons
    return _finite(susp.cpu().numpy(), fill=0.5)


def _kmeans2_1d(x, n_iter=50):
    x = np.asarray(x, dtype=float)
    c = np.array([x.min(), x.max()], dtype=float)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(n_iter):
        labels = np.abs(x[:, None] - c[None, :]).argmin(axis=1)
        new = c.copy()
        for j in (0, 1):
            if np.any(labels == j):
                new[j] = x[labels == j].mean()
        if np.allclose(new, c):
            break
        c = new
    if c[0] > c[1]:
        c = c[::-1]; labels = 1 - labels
    return labels, c


def flag_clients(suspicion, max_frac=0.49):
    s = np.asarray(suspicion, dtype=float)
    C = len(s)
    if C < 3 or np.ptp(s) < 1e-6:
        return np.zeros(C, dtype=bool)
    labels, centers = _kmeans2_1d(s)
    gap = centers[1] - centers[0]
    if gap < max(0.05, 0.5 * (np.std(s) + 1e-9)):
        return np.zeros(C, dtype=bool)
    flagged = labels.astype(bool)
    cap = int(max_frac * C)
    if flagged.sum() > cap:
        flagged = np.zeros(C, dtype=bool)
        flagged[np.argsort(s)[-cap:]] = True
    return flagged


def robust_aggregate(U, keep_mask, rand_drop):
    """SAGID step 4: norm-clip each survivor to median survivor norm, mean,
    then gentle randomized coordinate dropout."""
    Uk = U[torch.as_tensor(keep_mask, device=U.device)]
    norms = Uk.norm(dim=1)
    med = norms.median()
    scale = torch.clamp(med / (norms + 1e-12), max=1.0)
    Uk = Uk * scale[:, None]
    agg = Uk.mean(dim=0)
    if rand_drop > 0:
        nz = torch.nonzero(agg, as_tuple=True)[0]
        if len(nz) > 0:
            drop = torch.rand(len(nz), device=agg.device) < rand_drop
            agg = agg.clone(); agg[nz[drop]] = 0.0
    return _finite(agg)


# --------------------------------------------------------------------------- #
#  Federated training driver                                                   #
# --------------------------------------------------------------------------- #
def run_federated(scenario, Xtr, ytr, Xte, yte, cfg):
    rng = np.random.default_rng(cfg["seed"])
    clients = partition_iid(Xtr.shape[0], cfg["n_clients"], cfg["seed"])
    n_mal = int(round(cfg["mal_frac"] * cfg["n_clients"])) if scenario != "clean" else 0
    mal_ids = set(rng.choice(cfg["n_clients"], size=n_mal, replace=False).tolist())
    is_mal = np.array([1 if c in mal_ids else 0 for c in range(cfg["n_clients"])])

    torch.manual_seed(cfg["seed"])
    fresh = SmallCNN().to(DEVICE)
    global_w = parameters_to_vector(fresh.parameters()).detach().clone()

    g_exp = torch.zeros_like(global_w)
    ceiling = None   # uniform per-client update-norm safeguard, set on round 0

    hist = dict(acc=[], prec=[], rec=[], f1=[], is_mal=is_mal, det_records=[],
                mal_dev=[], hon_dev=[], mal_sgsa_norm=[], mal_honesteq_norm=[])
    vote_count = np.zeros(cfg["n_clients"]); n_dec = 0

    for r in range(cfg["rounds"]):
        raw, honest_eq_for = [], {}
        for cid in range(cfg["n_clients"]):
            idx = clients[cid]
            delta = local_update(global_w, Xtr[idx], ytr[idx], cfg["lr"],
                                  cfg["local_epochs"], cfg["batch"], cfg["momentum"])
            if cid in mal_ids:
                u = sgsa_sparsify(delta, cfg["k_frac"], cfg["attack_concentration"],
                                  cfg["attack_boost"], cfg["attack_noise"])
                honest_eq_for[cid] = float(topk_sparsify(delta, cfg["k_frac"]).norm())
            else:
                u = topk_sparsify(delta, cfg["k_frac"])
            raw.append(u)

        U = torch.stack(raw)
        U = _finite(U)

        # uniform numerical safeguard: clip every client update to a fixed ceiling
        norms = U.norm(dim=1)
        if ceiling is None:
            fin = norms[torch.isfinite(norms)]
            base = float(fin.median()) if len(fin) > 0 else 1.0
            ceiling = max(1e-6, cfg["clip_multiple"] * base)
        scale = torch.clamp(torch.tensor(ceiling, device=U.device) / (norms + 1e-12), max=1.0)
        U = U * scale[:, None]

        # record transmitted (post-clip) malicious norms + honest reference for stealth plot
        for cid in mal_ids:
            hist["mal_sgsa_norm"].append(float(U[cid].norm()))
            hist["mal_honesteq_norm"].append(honest_eq_for.get(cid, float("nan")))
        for c in range(cfg["n_clients"]):
            (hist["mal_dev"] if is_mal[c] else hist["hon_dev"]).append(float(U[c].norm()))

        if scenario == "defense":
            susp = sagid_suspicion(U, g_exp, cfg["k_frac"])
            flagged = flag_clients(susp)
            if flagged.sum() > 0:
                vote_count += flagged; n_dec += 1
            blacklist = (vote_count >= cfg["blacklist_vote"] * n_dec) if n_dec >= 3 \
                        else np.zeros(cfg["n_clients"], bool)
            keep_mask = ~blacklist
            if keep_mask.sum() < max(2, cfg["n_clients"] // 2):
                keep_mask = susp <= np.quantile(susp, 0.6)
            hist["det_records"].append(dict(susp=susp.copy(), flagged=flagged.copy()))
            agg = robust_aggregate(U, keep_mask, cfg["rand_drop"])
            g_exp = cfg["ema"] * g_exp + (1.0 - cfg["ema"]) * agg
        else:
            agg = _finite(U.mean(dim=0))                # plain FedAvg

        global_w = _finite(global_w + agg)
        m = evaluate(global_w, Xte, yte)
        for k in ("acc", "prec", "rec", "f1"):
            hist[k].append(float(m[k]))
        print(f"  [{scenario:7s}] round {r+1:2d}/{cfg['rounds']}  acc={m['acc']:.4f}", end="\r")
    print()
    hist["global_w"] = global_w
    return hist


# --------------------------------------------------------------------------- #
#  Metrics                                                                     #
# --------------------------------------------------------------------------- #
def summarize(h, last_n=5):
    """All global-model metrics available for one scenario."""
    out = {}
    for k in ("acc", "prec", "rec", "f1"):
        a = _finite(h[k])
        out[k + "_final"] = float(a[-1])
        out[k + "_best"] = float(np.max(a))
        out[k + "_last5_mean"] = float(np.mean(a[-last_n:]))
    a = _finite(h["acc"])
    out["best_acc_round"] = int(np.argmax(a) + 1)
    out["mean_honest_update_norm"] = float(np.mean(_finite(h["hon_dev"]))) if h["hon_dev"] else float("nan")
    out["mean_malicious_update_norm"] = float(np.mean(_finite(h["mal_dev"]))) if h["mal_dev"] else float("nan")
    return out


def detection_metrics(hist):
    is_mal = hist["is_mal"]
    flags = np.stack([rec["flagged"] for rec in hist["det_records"]])
    susp = np.stack([rec["susp"] for rec in hist["det_records"]])
    decision = flags.sum(axis=1) > 0
    n_dec = int(decision.sum())
    if n_dec > 0:
        flag_freq = flags[decision].mean(axis=0)
        mean_susp = susp[decision].mean(axis=0)
    else:
        flag_freq = flags.mean(axis=0); mean_susp = susp.mean(axis=0)
    client_flag = (flag_freq >= 0.5).astype(int)
    cm = confusion_matrix(is_mal, client_flag, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = dict(
        acc=accuracy_score(is_mal, client_flag),
        prec=precision_score(is_mal, client_flag, zero_division=0),
        rec=recall_score(is_mal, client_flag, zero_division=0),
        f1=f1_score(is_mal, client_flag, zero_division=0),
        cm=cm,
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        tpr=float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        fpr=float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        tnr=float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        fnr=float(fn / (fn + tp)) if (fn + tp) else float("nan"),
        flag_freq=flag_freq, mean_susp=mean_susp,
        n_decision_rounds=n_dec, n_total_rounds=int(flags.shape[0]),
    )
    try:
        out["auc"] = roc_auc_score(is_mal, mean_susp)
    except ValueError:
        out["auc"] = float("nan")
    return out


def safe_lim(*arrays, pct=99, pad=1.05, default=1.0):
    """NaN/Inf-safe upper axis limit."""
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays]) if arrays else np.array([])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return default
    lim = float(np.percentile(vals, pct)) * pad
    return lim if (np.isfinite(lim) and lim > 0) else default


# --------------------------------------------------------------------------- #
#  Run the three scenarios                                                     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Loading data ...")
    Xtr, ytr, Xte, yte = load_data(CFG)
    print(f"train {tuple(Xtr.shape)}  test {tuple(Xte.shape)}  "
          f"params={sum(p.numel() for p in WORKER.parameters())}")

    print("\nScenario 1/3: CLEAN");        clean   = run_federated("clean",   Xtr, ytr, Xte, yte, CFG)
    print("Scenario 2/3: SGSA ATTACK");    attack  = run_federated("attack",  Xtr, ytr, Xte, yte, CFG)
    print("Scenario 3/3: SGSA + SAGID");   defense = run_federated("defense", Xtr, ytr, Xte, yte, CFG)
    det = detection_metrics(defense)

    S = {n: summarize(h) for n, h in [("Clean", clean), ("SGSA attack", attack), ("SGSA + SAGID", defense)]}

    # ---------------- global-model metrics: every scenario, every metric -------------- #
    print("\n" + "=" * 86)
    print(" GLOBAL-MODEL METRICS  (test set) — all scenarios, all metrics")
    print("=" * 86)
    hdr = f"{'Scenario':<16}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'BestAcc':>9}{'@round':>7}{'Last5Acc':>10}"
    print(hdr); print("-" * len(hdr))
    for n in S:
        s = S[n]
        print(f"{n:<16}{s['acc_final']:>8.4f}{s['prec_final']:>8.4f}{s['rec_final']:>8.4f}"
              f"{s['f1_final']:>8.4f}{s['acc_best']:>9.4f}{s['best_acc_round']:>7d}{s['acc_last5_mean']:>10.4f}")

    print("\nAvg per-client update norm (||transmitted update||):")
    for n in S:
        s = S[n]
        print(f"  {n:<16} honest={s['mean_honest_update_norm']:.4f}   "
              f"malicious={s['mean_malicious_update_norm'] if not np.isnan(s['mean_malicious_update_norm']) else float('nan'):.4f}")

    # ---------------- attack impact & defense recovery (per metric) -------------------- #
    print("\n" + "=" * 86)
    print(" ATTACK IMPACT  (Clean - Attack)   &   DEFENSE RECOVERY  (Defense - Attack)")
    print("=" * 86)
    print(f"{'Metric':<8}{'Clean':>9}{'Attack':>9}{'Defense':>9}{'Drop(pp)':>10}{'Recover(pp)':>13}{'Recover%':>10}")
    for m in ("acc", "prec", "rec", "f1"):
        c = S["Clean"][m + "_final"]; a = S["SGSA attack"][m + "_final"]; d = S["SGSA + SAGID"][m + "_final"]
        drop = (c - a) * 100; recov = (d - a) * 100
        pct = (100 * (d - a) / (c - a)) if abs(c - a) > 1e-9 else float("nan")
        print(f"{m:<8}{c:>9.4f}{a:>9.4f}{d:>9.4f}{drop:>10.2f}{recov:>13.2f}{pct:>10.1f}")

    # ---------------- detection (defense scenario only) -------------------------------- #
    print("\n" + "=" * 86)
    print(" SAGID DETECTION METRICS")
    print("=" * 86)
    print(f"accuracy={det['acc']:.3f}  precision={det['prec']:.3f}  recall(TPR)={det['rec']:.3f}  "
          f"f1={det['f1']:.3f}  auc={det['auc']:.3f}")
    print(f"TP={det['tp']}  FP={det['fp']}  TN={det['tn']}  FN={det['fn']}   "
          f"TPR={det['tpr']:.3f}  FPR={det['fpr']:.3f}  TNR={det['tnr']:.3f}  FNR={det['fnr']:.3f}")
    print(f"decision rounds (>=1 client flagged): {det['n_decision_rounds']}/{det['n_total_rounds']}")
    print(f"confusion matrix [rows=true 0/1, cols=pred 0/1]:\n{det['cm']}")
    print("per-client flag frequency (R=malicious, B=honest):")
    for c in range(CFG["n_clients"]):
        tag = "MAL " if defense["is_mal"][c] else "hon "
        print(f"   client {c:2d} [{tag}] flag_freq={det['flag_freq'][c]:.2f}  mean_susp={det['mean_susp'][c]:.3f}")

    # ---------------- stealth -------------------------------------------------------- #
    sgsa_n = _finite(attack["mal_sgsa_norm"]); hon_eq = _finite(attack["mal_honesteq_norm"])
    ratio = sgsa_n / (hon_eq + 1e-12)
    ratio = ratio[np.isfinite(ratio)]
    print("\n" + "=" * 86)
    print(" STEALTH")
    print("=" * 86)
    print(f"per-client ||SGSA|| / ||own honest top-k|| = {ratio.mean():.3f} +/- {ratio.std():.3f}  "
          f"(=1.0 => magnitude-stealthy; current boost={CFG['attack_boost']})")

    # --------------------------- figures (NaN/Inf-safe) ------------------------------- #
    R = np.arange(1, CFG["rounds"] + 1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    for name, h, c in [("Clean", clean, "tab:green"), ("SGSA attack", attack, "tab:red"),
                       ("SGSA + SAGID", defense, "tab:blue")]:
        plt.plot(R, _finite(h["acc"]), label=name, lw=2.2, color=c)
    plt.title("Global model accuracy"); plt.xlabel("Round"); plt.ylabel("Test acc"); plt.legend()
    plt.subplot(1, 2, 2)
    for name, h, c in [("Clean", clean, "tab:green"), ("SGSA attack", attack, "tab:red"),
                       ("SGSA + SAGID", defense, "tab:blue")]:
        plt.plot(R, _finite(h["f1"]), label=name, lw=2.2, color=c)
    plt.title("Global model macro-F1"); plt.xlabel("Round"); plt.ylabel("Macro-F1"); plt.legend()
    plt.tight_layout(); plt.savefig("fig1_convergence.png", dpi=130); plt.show()

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1)
    colors = ["tab:red" if m else "tab:blue" for m in defense["is_mal"]]
    plt.bar(np.arange(len(det["flag_freq"])), det["flag_freq"], color=colors)
    plt.axhline(0.5, ls="--", color="black")
    plt.title(f"Flag freq over {det['n_decision_rounds']} decision rounds")
    plt.xlabel("Client"); plt.ylabel("Fraction flagged")
    plt.subplot(1, 3, 2)
    cm = det["cm"]
    plt.imshow(cm, cmap="Blues"); plt.title(f"Confusion (F1={det['f1']:.2f})")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0, 1], ["Honest", "Malicious"]); plt.yticks([0, 1], ["Honest", "Malicious"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.subplot(1, 3, 3)
    fpr, tpr, _ = roc_curve(defense["is_mal"], _finite(det["mean_susp"]))
    plt.plot(fpr, tpr, lw=2.2, label=f"AUC={det['auc']:.3f}"); plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title("Detection ROC"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
    plt.tight_layout(); plt.savefig("fig2_detection.png", dpi=130); plt.show()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    lim = safe_lim(sgsa_n, hon_eq)
    plt.scatter(hon_eq, sgsa_n, s=14, alpha=0.5, color="tab:purple")
    plt.plot([0, lim], [0, lim], "--", color="black", label="magnitude-stealthy (y=x)")
    plt.xlim(0, lim); plt.ylim(0, lim)
    plt.title(f"Per-client SGSA payload norm (boost={CFG['attack_boost']})")
    plt.xlabel("||own honest top-k||"); plt.ylabel("||transmitted SGSA||"); plt.legend()
    plt.subplot(1, 2, 2)
    mets = ["acc", "prec", "rec", "f1"]; x = np.arange(4); w = 0.25
    for i, (name, h, c) in enumerate([("Clean", clean, "tab:green"), ("SGSA", attack, "tab:red"),
                                      ("SGSA+SAGID", defense, "tab:blue")]):
        plt.bar(x + (i - 1) * w, [_finite(h[m])[-1] for m in mets], w, label=name, color=c)
    plt.xticks(x, ["Acc", "Prec", "Rec", "F1"]); plt.ylim(0, 1.05)
    plt.title("Final metrics"); plt.legend()
    plt.tight_layout(); plt.savefig("fig3_stealth_metrics.png", dpi=130); plt.show()

    # --------------------------- save everything ------------------------------------- #
    results = dict(
        config=CFG,
        scenarios={n: S[n] for n in S},
        attack_impact_pp={m: float((S["Clean"][m + "_final"] - S["SGSA attack"][m + "_final"]) * 100)
                          for m in ("acc", "prec", "rec", "f1")},
        defense_recovery_pp={m: float((S["SGSA + SAGID"][m + "_final"] - S["SGSA attack"][m + "_final"]) * 100)
                             for m in ("acc", "prec", "rec", "f1")},
        detection=dict(acc=det["acc"], prec=det["prec"], rec=det["rec"], f1=det["f1"], auc=det["auc"],
                       tp=det["tp"], fp=det["fp"], tn=det["tn"], fn=det["fn"],
                       tpr=det["tpr"], fpr=det["fpr"], tnr=det["tnr"], fnr=det["fnr"],
                       n_decision_rounds=det["n_decision_rounds"], n_total_rounds=det["n_total_rounds"],
                       cm=det["cm"].tolist(), flag_freq=det["flag_freq"].tolist(),
                       mean_susp=_finite(det["mean_susp"]).tolist(),
                       is_mal=defense["is_mal"].tolist()),
        stealth=dict(self_ratio_mean=float(ratio.mean()), self_ratio_std=float(ratio.std())),
        history={n: {k: _finite(h[k]).tolist() for k in ("acc", "prec", "rec", "f1")}
                 for n, h in [("clean", clean), ("attack", attack), ("defense", defense)]},
    )
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: fig1_convergence.png, fig2_detection.png, fig3_stealth_metrics.png, results.json")

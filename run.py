import argparse
from pathlib import Path
import math
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from problems import SBGRProblem
from algorithms import ObjectiveWrapper, ALGORITHMS

# SciPy is recommended for robust statistical tests
try:
    from scipy.stats import friedmanchisquare
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =========================================================
# Helpers
# =========================================================
def make_problem(name: str, dim: int, seed: int, case: str) -> SBGRProblem:
    if name.lower() == "sbgr":
        return SBGRProblem(dim=dim, seed=seed, case=case)
    raise ValueError(f"Unknown problem: {name}")


def dim_for_case(case: str, base_dim: int, highdim_dim: int) -> int:
    case = case.lower()
    if case == "highdim":
        return max(base_dim, highdim_dim)
    return base_dim


def eval_budget_for_case(case: str, base_max_evals: int, highdim_max_evals: Optional[int]) -> int:
    if case == "highdim" and highdim_max_evals is not None:
        return int(highdim_max_evals)
    return int(base_max_evals)


def resample_history(evals: np.ndarray, best: np.ndarray, checkpoints: np.ndarray) -> np.ndarray:
    if len(evals) == 0 or len(best) == 0:
        return np.full_like(checkpoints, np.nan, dtype=float)

    out = np.empty_like(checkpoints, dtype=float)
    j = 0
    current = float(best[0])

    for i, c in enumerate(checkpoints):
        while j + 1 < len(evals) and evals[j + 1] <= c:
            j += 1
            current = float(best[j])
        out[i] = current
    return out


def last_improvement_eval_from_history(history_evals: np.ndarray, history_best: np.ndarray, max_evals: int) -> float:
    if len(history_evals) == 0:
        return np.nan
    he = np.asarray(history_evals, dtype=int)
    hb = np.asarray(history_best, dtype=float)

    # finalize_history غالبًا يضيف نقطة أخيرة عند max_evals بدون تحسين جديد
    if len(he) >= 2 and he[-1] == max_evals and np.isclose(hb[-1], hb[-2], rtol=0, atol=1e-12):
        return float(he[-2])
    return float(he[-1])


def convergence_auc(checkpoints: np.ndarray, curve: np.ndarray) -> float:
    """
    AUC تقريبية لمنحنى best-fitness عبر evaluations.
    lower-is-better => AUC الأقل عادةً أفضل (لكن قارن داخل نفس case فقط)
    """
    x = np.asarray(checkpoints, dtype=float)
    y = np.asarray(curve, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    # تطبيع على طول المجال لتسهيل المقارنة
    width = x[-1] - x[0]
    if width <= 0:
        return float(np.nanmean(y))
    # np.trapz متوافق مع numpy الحالية
    return float(np.trapezoid(y, x) / width)


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> float:
    """
    Wilcoxon signed-rank p-value (تقريب Normal) بدون SciPy.
    مناسب عادةً عندما n >= 10.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    d = x[mask] - y[mask]
    d = d[np.abs(d) > 0]

    n = len(d)
    if n < 10:
        return float("nan")

    order = np.argsort(np.abs(d))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)

    Wpos = np.sum(ranks[d > 0])
    Wneg = np.sum(ranks[d < 0])
    W = min(Wpos, Wneg)

    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    z = (W - mean) / math.sqrt(var + 1e-12)

    p = 2.0 * (0.5 * (1.0 + math.erf(-abs(z) / math.sqrt(2.0))))
    return float(p)


def holm_step_down(p_values: np.ndarray, alpha: float = 0.05):
    """
    Holm-Bonferroni step-down correction
    returns: adjusted_pvals, reject_flags, order_indices
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = np.full(m, np.nan, dtype=float)
    reject = np.zeros(m, dtype=int)

    valid = np.isfinite(p)
    if valid.sum() == 0:
        return adjusted, reject, np.arange(m)

    valid_idx = np.where(valid)[0]
    pv = p[valid]
    order_local = np.argsort(pv)
    ordered_idx = valid_idx[order_local]
    ordered_p = pv[order_local]

    # adjusted p-values (monotonic)
    prev = 0.0
    for rank, pval in enumerate(ordered_p):
        mult = (len(ordered_p) - rank)
        adj = min(1.0, pval * mult)
        adj = max(adj, prev)
        prev = adj
        adjusted[ordered_idx[rank]] = adj

    # reject decisions (step-down)
    stop = False
    for rank, pval in enumerate(ordered_p):
        thr = alpha / (len(ordered_p) - rank)
        idx = ordered_idx[rank]
        if not stop and pval <= thr:
            reject[idx] = 1
        else:
            stop = True
            reject[idx] = 0

    return adjusted, reject, ordered_idx


def safe_nanmean(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def safe_nanmedian(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# =========================================================
# Statistics builders
# =========================================================
VIOL_COLS = [
    "viol_vref",
    "viol_tc",
    "viol_loop_gain",
    "viol_phase_margin",
    "viol_gain_margin",
    "viol_power",
]


def build_case_stats(df_case: pd.DataFrame, algo_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    يرجع:
    1) stats_basic
    2) stats_extended
    3) violations_summary
    4) rank_summary
    """
    if df_case.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    # Basic stats on final fitness
    basic = (
        df_case.groupby("algo")["best_fitness"]
        .agg(["count", "min", "mean", "median", "std", "max"])
        .reindex(algo_names)
        .reset_index()
        .rename(columns={"count": "runs"})
    )

    # Extended stats
    rows_ext = []
    for algo in algo_names:
        sub = df_case[df_case["algo"] == algo].copy()
        if sub.empty:
            rows_ext.append({"algo": algo})
            continue

        feasible_final = sub[sub["is_feasible"] == 1]
        ever_feasible = sub[sub["best_feasible_found"] == 1]

        row = {
            "algo": algo,
            "runs": int(len(sub)),
            "success_rate": float(sub["is_feasible"].mean()),
            "feasible_runs": int(sub["is_feasible"].sum()),
            "ever_feasible_rate": float(sub["best_feasible_found"].mean()),
            "best_fitness_min": float(sub["best_fitness"].min()),
            "best_fitness_mean": float(sub["best_fitness"].mean()),
            "best_fitness_median": float(sub["best_fitness"].median()),
            "best_fitness_std": float(sub["best_fitness"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "PSRR_best_final": float(sub["PSRR_DB"].max()),
            "PSRR_mean_final": float(sub["PSRR_DB"].mean()),
            "PSRR_mean_feasible_final": safe_nanmean(feasible_final["PSRR_DB"].values),
            "best_feasible_PSRR_max": safe_nanmean(np.array([sub["best_feasible_PSRR"].max()])),
            "best_feasible_PSRR_mean": safe_nanmean(ever_feasible["best_feasible_PSRR"].values),
            "best_feasible_fitness_mean": safe_nanmean(ever_feasible["best_feasible_fitness"].values),
            "runtime_sec_mean": float(sub["runtime_sec"].mean()),
            "runtime_sec_std": float(sub["runtime_sec"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "first_feasible_eval_mean": safe_nanmean(sub["first_feasible_eval"].values),
            "first_feasible_eval_median": safe_nanmedian(sub["first_feasible_eval"].values),

            # convergence speed / behavior
            "last_improvement_eval_mean": safe_nanmean(sub["last_improvement_eval"].values),
            "last_improvement_eval_median": safe_nanmedian(sub["last_improvement_eval"].values),
            "convergence_auc_mean": safe_nanmean(sub["convergence_auc"].values),
            "convergence_auc_std": float(sub["convergence_auc"].std(ddof=1)) if len(sub.dropna(subset=["convergence_auc"])) > 1 else 0.0,
        }
        rows_ext.append(row)

    ext = pd.DataFrame(rows_ext)

    # Violations summary (على الحل النهائي الأفضل لكل run)
    vrows = []
    for algo in algo_names:
        sub = df_case[df_case["algo"] == algo].copy()
        if sub.empty:
            vrows.append({"algo": algo})
            continue

        infeas = sub[sub["is_feasible"] == 0]
        row = {
            "algo": algo,
            "runs": int(len(sub)),
            "infeasible_runs": int((sub["is_feasible"] == 0).sum()),
        }
        for c in VIOL_COLS:
            row[f"{c}_count_all"] = int(sub[c].sum())
            row[f"{c}_rate_all"] = float(sub[c].mean())
            row[f"{c}_count_infeasible"] = int(infeas[c].sum()) if len(infeas) else 0
            row[f"{c}_rate_infeasible"] = float(infeas[c].mean()) if len(infeas) else float("nan")
        row["avg_penalty"] = float(sub["penalty"].mean())
        row["max_penalty"] = float(sub["penalty"].max())
        vrows.append(row)

    viol = pd.DataFrame(vrows)

    # Rank summary (per-run ranking داخل نفس case)
    pivot = df_case.pivot(index="run", columns="algo", values="best_fitness").reindex(columns=algo_names)
    rank_rows = []
    if not pivot.empty:
        ranks = pivot.rank(axis=1, method="average", ascending=True)
        for algo in algo_names:
            col = ranks[algo].values if algo in ranks.columns else np.array([])
            rank_rows.append({
                "algo": algo,
                "avg_rank": safe_nanmean(col),
                "median_rank": safe_nanmedian(col),
            })
    rank_df = pd.DataFrame(rank_rows)

    return basic, ext, viol, rank_df


def build_case_wilcoxon(df_case: pd.DataFrame, algo_names: List[str], base_algo: str = "ABC") -> pd.DataFrame:
    rows = []
    if base_algo not in algo_names:
        return pd.DataFrame(columns=["comparison", "p_value", "n_pairs"])

    base = df_case[df_case["algo"] == base_algo].sort_values("run")["best_fitness"].values
    for algo in algo_names:
        if algo == base_algo:
            continue
        other = df_case[df_case["algo"] == algo].sort_values("run")["best_fitness"].values
        n_pairs = int(min(len(base), len(other)))
        p = wilcoxon_signed_rank(base[:n_pairs], other[:n_pairs]) if n_pairs > 0 else float("nan")
        rows.append({
            "comparison": f"{base_algo} vs {algo}",
            "p_value": p,
            "n_pairs": n_pairs,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        adj, rej, _ = holm_step_down(out["p_value"].values, alpha=0.05)
        out["holm_adjusted_p"] = adj
        out["holm_reject_0_05"] = rej
    return out


def build_overall_wilcoxon_blocks(df_all: pd.DataFrame, algo_names: List[str], base_algo: str = "ABC") -> pd.DataFrame:
    """
    مقارنة paired عبر كل blocks = (case, run)
    """
    if df_all.empty or base_algo not in algo_names:
        return pd.DataFrame(columns=["comparison", "p_value", "n_pairs"])

    pivot = df_all.pivot(index=["case", "run"], columns="algo", values="best_fitness").reindex(columns=algo_names)
    rows = []
    if base_algo not in pivot.columns:
        return pd.DataFrame(rows)

    base = pivot[base_algo].values
    for algo in algo_names:
        if algo == base_algo:
            continue
        other = pivot[algo].values
        p = wilcoxon_signed_rank(base, other)
        n_pairs = int(np.sum(np.isfinite(base) & np.isfinite(other)))
        rows.append({
            "comparison": f"{base_algo} vs {algo}",
            "p_value": p,
            "n_pairs": n_pairs,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        adj, rej, _ = holm_step_down(out["p_value"].values, alpha=0.05)
        out["holm_adjusted_p"] = adj
        out["holm_reject_0_05"] = rej
    return out


def friedman_from_pivot(pivot: pd.DataFrame, algo_names: List[str], scope_label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    pivot: rows=blocks, cols=algorithms, values=fitness (lower better)
    returns:
    - friedman result dataframe
    - avg ranks dataframe
    """
    if pivot.empty:
        return pd.DataFrame(), pd.DataFrame()

    pivot = pivot.reindex(columns=algo_names)
    complete = pivot.dropna(axis=0, how="any").copy()

    if complete.empty or complete.shape[0] < 2 or complete.shape[1] < 3:
        fr = pd.DataFrame([{
            "scope": scope_label,
            "n_blocks": int(complete.shape[0]),
            "k_algorithms": int(complete.shape[1]),
            "friedman_chi2": np.nan,
            "friedman_p": np.nan,
            "scipy_available": int(SCIPY_AVAILABLE),
            "note": "Insufficient complete blocks or algorithms for Friedman"
        }])
        return fr, pd.DataFrame()

    ranks = complete.rank(axis=1, method="average", ascending=True)
    avg_ranks = ranks.mean(axis=0).reset_index()
    avg_ranks.columns = ["algo", "avg_rank"]
    avg_ranks["scope"] = scope_label
    avg_ranks = avg_ranks[["scope", "algo", "avg_rank"]].sort_values("avg_rank")

    k = complete.shape[1]
    n = complete.shape[0]
    rbar = ranks.mean(axis=0).values
    chi2_manual = (12 * n / (k * (k + 1))) * np.sum(rbar**2) - 3 * n * (k + 1)

    p_val = np.nan
    chi2_val = float(chi2_manual)
    note = "manual_chi2_only (SciPy not available)"
    if SCIPY_AVAILABLE:
        stat, p = friedmanchisquare(*[complete[c].values for c in complete.columns])
        chi2_val = float(stat)
        p_val = float(p)
        note = "scipy.friedmanchisquare"

    fr = pd.DataFrame([{
        "scope": scope_label,
        "n_blocks": int(n),
        "k_algorithms": int(k),
        "friedman_chi2": chi2_val,
        "friedman_p": p_val,
        "scipy_available": int(SCIPY_AVAILABLE),
        "note": note,
    }])

    return fr, avg_ranks


# =========================================================
# Plotters (per case)
# =========================================================
def plot_convergence(case_dir: Path, case_name: str, algo_names: List[str], checkpoints: np.ndarray, curves: Dict[str, List[np.ndarray]]):
    plt.figure(figsize=(9, 5))
    for algo in algo_names:
        if algo not in curves or len(curves[algo]) == 0:
            continue
        arr = np.vstack(curves[algo])
        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        plt.plot(checkpoints, mu, label=algo)
        plt.fill_between(checkpoints, mu - sd, mu + sd, alpha=0.15)

    plt.xlabel("Function evaluations")
    plt.ylabel("Best fitness (lower is better)")
    plt.title(f"Convergence (mean ± std) - {case_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(case_dir / f"convergence_{case_name.lower()}.png", dpi=220)
    plt.close()


def plot_boxplot_fitness(case_dir: Path, case_name: str, df_case: pd.DataFrame, algo_names: List[str]):
    plt.figure(figsize=(9, 5))
    data = [df_case[df_case["algo"] == a]["best_fitness"].values for a in algo_names]
    plt.boxplot(data, tick_labels=algo_names, showmeans=True)
    plt.ylabel("Final best fitness")
    plt.title(f"Final fitness distribution - {case_name}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(case_dir / f"boxplot_fitness_{case_name.lower()}.png", dpi=220)
    plt.close()


def plot_boxplot_feasible_psrr(case_dir: Path, case_name: str, df_case: pd.DataFrame, algo_names: List[str]):
    plt.figure(figsize=(9, 5))
    data = []
    for a in algo_names:
        vals = df_case[(df_case["algo"] == a) & (df_case["is_feasible"] == 1)]["PSRR_DB"].values
        if len(vals) == 0:
            vals = np.array([np.nan])
        data.append(vals)

    plt.boxplot(data, tick_labels=algo_names, showmeans=True)
    plt.ylabel("PSRR (dB) - feasible runs only")
    plt.title(f"Feasible-only PSRR distribution - {case_name}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(case_dir / f"boxplot_feasible_psrr_{case_name.lower()}.png", dpi=220)
    plt.close()


def plot_success_rate_bar(case_dir: Path, case_name: str, ext: pd.DataFrame, algo_names: List[str]):
    plt.figure(figsize=(9, 5))
    sub = ext.set_index("algo").reindex(algo_names)
    y = sub["success_rate"].values.astype(float)
    x = np.arange(len(algo_names))

    plt.bar(x, y)
    plt.xticks(x, algo_names)
    plt.ylim(0, 1.05)
    plt.ylabel("Success rate (feasible ratio)")
    plt.title(f"Success rate by algorithm - {case_name}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(case_dir / f"success_rate_bar_{case_name.lower()}.png", dpi=220)
    plt.close()


def plot_violations_bar(case_dir: Path, case_name: str, viol: pd.DataFrame, algo_names: List[str]):
    if viol.empty:
        return

    show_cols = [f"{c}_rate_all" for c in VIOL_COLS]
    sub = viol.set_index("algo").reindex(algo_names)

    x = np.arange(len(algo_names))
    width = 0.12

    plt.figure(figsize=(12, 5))
    for i, col in enumerate(show_cols):
        y = sub[col].values.astype(float)
        plt.bar(x + (i - (len(show_cols)-1)/2) * width, y, width=width, label=col.replace("_rate_all", ""))

    plt.xticks(x, algo_names)
    plt.ylim(0, 1.05)
    plt.ylabel("Violation frequency across runs")
    plt.title(f"Constraint violation frequencies - {case_name}")
    plt.legend(ncol=3, fontsize=8)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(case_dir / f"violations_bar_{case_name.lower()}.png", dpi=220)
    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    # Problem setup
    ap.add_argument("--problem", default="sbgr", choices=["sbgr"])
    ap.add_argument("--cases", nargs="*", default=["base"], choices=["base", "hard", "highdim"],
                    help="حالات SBGR للتجارب")
    ap.add_argument("--dim", type=int, default=12, help="عدد الأبعاد للحالتين base/hard")
    ap.add_argument("--highdim-dim", type=int, default=30, help="عدد الأبعاد لحالة highdim (افتراضيًا 30)")

    # Experiment protocol
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--max-evals", type=int, default=30000, help="ميزانية التقييمات للحالات base/hard")
    ap.add_argument("--highdim-max-evals", type=int, default=None,
                    help="ميزانية خاصة لحالة highdim (اختياري)")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)

    # Algorithms / output
    ap.add_argument("--algos", nargs="*", default=["ABC", "GWO", "FA", "PSO", "GA", "ACO"])
    ap.add_argument("--outdir", default="results_experiments")
    ap.add_argument("--progress-step", type=int, default=2000, help="اطبع التقدم كل كم evaluation")
    ap.add_argument("--checkpoint-count", type=int, default=300, help="عدد نقاط إعادة أخذ convergence curve")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    algo_names = [a for a in args.algos if a in ALGORITHMS]
    if len(algo_names) == 0:
        raise ValueError("No valid algorithms selected.")

    cases = [c.lower() for c in args.cases]

    all_rows = []
    all_case_stats = []
    all_case_ext = []
    all_case_viol = []
    all_case_wil = []
    all_case_ranks = []
    all_case_friedman = []
    all_case_avg_ranks = []

    total_jobs = len(cases) * args.runs * len(algo_names)
    job_counter = 0

    print("\n==============================")
    print("Experiment Run Started")
    print("==============================")
    print(f"Cases       : {cases}")
    print(f"Algorithms  : {algo_names}")
    print(f"Runs/case   : {args.runs}")
    print(f"Population  : {args.pop}")
    print(f"Max evals   : base/hard={args.max_evals}, highdim={args.highdim_max_evals if args.highdim_max_evals else args.max_evals}")
    print(f"Highdim dim : {args.highdim_dim}")
    print(f"SciPy stat  : {'YES' if SCIPY_AVAILABLE else 'NO (Friedman p-value will be NaN)'}")
    print(f"Output dir  : {outdir.resolve()}")
    print("==============================\n")

    for case_idx, case in enumerate(cases, start=1):
        case_dim = dim_for_case(case, args.dim, args.highdim_dim)
        case_max_evals = eval_budget_for_case(case, args.max_evals, args.highdim_max_evals)
        checkpoints = np.linspace(1, case_max_evals, args.checkpoint_count, dtype=int)

        case_name = f"SBGR-{case.upper()}"
        case_dir = outdir / f"case_{case}"
        ensure_dir(case_dir)

        print(f"\n############################################################")
        print(f"CASE {case_idx}/{len(cases)}: {case_name} | dim={case_dim} | max_evals={case_max_evals}")
        print(f"############################################################")

        curves = {a: [] for a in algo_names}

        for run in range(args.runs):
            # seed خاص بكل case/run للمشكلة
            case_offset = {"base": 0, "hard": 100000, "highdim": 200000}[case]
            run_seed = args.seed + case_offset + run * 1000

            # نفس المشكلة لكل الخوارزميات داخل نفس run (نزاهة)
            problem = make_problem(args.problem, case_dim, seed=run_seed, case=case)

            print(f"\n========== {case_name} | RUN {run+1}/{args.runs} | seed={run_seed} ==========")

            for algo in algo_names:
                job_counter += 1
                print(
                    f"\n[{job_counter}/{total_jobs}] بدء: case={case}, algo={algo}, "
                    f"pop={args.pop}, dim={case_dim}, max_evals={case_max_evals}"
                )

                def progress_cb(evals, max_evals, best_seen, _algo=algo, _case=case, _run=run+1):
                    pct = 100.0 * evals / max_evals
                    print(
                        f"\r    ... {_case}/run{_run}/{_algo}: {pct:6.2f}% | "
                        f"evals={evals}/{max_evals} | best_seen={best_seen:.6g}",
                        end="",
                        flush=True,
                    )

                rng = np.random.default_rng(run_seed)  # نفس seed داخل نفس run لكل الخوارزمية
                obj = ObjectiveWrapper(
                    func=None,
                    eval_with_metrics=problem.evaluate_with_metrics,
                    max_evals=case_max_evals,
                    progress_cb=progress_cb,
                    progress_step=args.progress_step,
                )

                opt = ALGORITHMS[algo]

                t0 = time.perf_counter()
                res = opt(obj=obj, rng=rng, lb=problem.lb, ub=problem.ub, pop=args.pop, dim=case_dim)
                runtime_sec = time.perf_counter() - t0

                print("")  # newline بعد progress

                # metrics للحل النهائي الأفضل
                m_final = problem.metrics(res.best_x)

                # metrics لأفضل feasible (إذا وجد)
                if obj.best_feasible_x is not None and obj.best_feasible_metrics is not None:
                    m_feas = dict(obj.best_feasible_metrics)
                    best_feasible_f = float(obj.best_feasible_f)
                    best_feasible_found = 1
                else:
                    m_feas = {}
                    best_feasible_f = float("nan")
                    best_feasible_found = 0

                # convergence speed metrics
                y_curve = resample_history(res.history_evals, res.history_best, checkpoints)
                auc_val = convergence_auc(checkpoints, y_curve)
                last_imp_eval = last_improvement_eval_from_history(res.history_evals, res.history_best, case_max_evals)

                print(
                    f"[انتهى] {algo} | best_fitness={res.best_f:.6g} | "
                    f"PSRR={m_final.get('PSRR_DB', float('nan')):.3f} dB | "
                    f"penalty={m_final.get('penalty', 'NA')} | feasible={int(m_final.get('is_feasible', 0))} | "
                    f"first_feasible_eval={obj.first_feasible_eval if obj.first_feasible_eval is not None else 'NA'} | "
                    f"last_impr_eval={int(last_imp_eval) if np.isfinite(last_imp_eval) else 'NA'} | "
                    f"runtime={runtime_sec:.2f}s"
                )

                row = {
                    "case": case,
                    "case_name": case_name,
                    "dim": case_dim,
                    "run": run,
                    "seed": run_seed,
                    "algo": algo,

                    # final best (overall)
                    "best_fitness": float(res.best_f),
                    "PSRR_DB": m_final.get("PSRR_DB", np.nan),
                    "penalty": m_final.get("penalty", np.nan),
                    "is_feasible": int(m_final.get("is_feasible", 0)),
                    "VREF": m_final.get("VREF", np.nan),
                    "TC": m_final.get("TC", np.nan),
                    "LOOP_GAIN_DB": m_final.get("LOOP_GAIN_DB", np.nan),
                    "PHASE_MARGIN_DEG": m_final.get("PHASE_MARGIN_DEG", np.nan),
                    "GAIN_MARGIN_DB": m_final.get("GAIN_MARGIN_DB", np.nan),
                    "POWER_UW": m_final.get("POWER_UW", np.nan),

                    # violations (final best)
                    "viol_vref": int(m_final.get("viol_vref", 0)),
                    "viol_tc": int(m_final.get("viol_tc", 0)),
                    "viol_loop_gain": int(m_final.get("viol_loop_gain", 0)),
                    "viol_phase_margin": int(m_final.get("viol_phase_margin", 0)),
                    "viol_gain_margin": int(m_final.get("viol_gain_margin", 0)),
                    "viol_power": int(m_final.get("viol_power", 0)),

                    # feasibility tracking (ever found)
                    "first_feasible_eval": float(obj.first_feasible_eval) if obj.first_feasible_eval is not None else np.nan,
                    "best_feasible_found": int(best_feasible_found),
                    "best_feasible_fitness": best_feasible_f,
                    "best_feasible_PSRR": m_feas.get("PSRR_DB", np.nan),
                    "best_feasible_VREF": m_feas.get("VREF", np.nan),
                    "best_feasible_TC": m_feas.get("TC", np.nan),
                    "best_feasible_LOOP_GAIN_DB": m_feas.get("LOOP_GAIN_DB", np.nan),
                    "best_feasible_PHASE_MARGIN_DEG": m_feas.get("PHASE_MARGIN_DEG", np.nan),
                    "best_feasible_GAIN_MARGIN_DB": m_feas.get("GAIN_MARGIN_DB", np.nan),
                    "best_feasible_POWER_UW": m_feas.get("POWER_UW", np.nan),

                    # runtime / budget
                    "runtime_sec": float(runtime_sec),
                    "eval_budget": int(case_max_evals),
                    "actual_evals": int(obj.evals),

                    # convergence speed / behavior
                    "last_improvement_eval": float(last_imp_eval),
                    "convergence_auc": float(auc_val),
                }
                all_rows.append(row)

                curves[algo].append(y_curve)

        # ==========================
        # Per-case outputs
        # ==========================
        df_case = pd.DataFrame([r for r in all_rows if r["case"] == case])
        df_case.to_csv(case_dir / f"summary_{case}.csv", index=False)

        basic, ext, viol, rank_df = build_case_stats(df_case, algo_names)
        wil_df = build_case_wilcoxon(df_case, algo_names, base_algo="ABC")

        # Friedman per-case (blocks = runs)
        pivot_case = df_case.pivot(index="run", columns="algo", values="best_fitness")
        friedman_case_df, avg_ranks_case_df = friedman_from_pivot(pivot_case, algo_names, scope_label=f"case:{case}")

        # أضف اسم الحالة للجداول
        for _df in [basic, ext, viol, rank_df, wil_df]:
            if not _df.empty:
                _df.insert(0, "case", case)

        basic.to_csv(case_dir / f"stats_{case}.csv", index=False)
        ext.to_csv(case_dir / f"stats_extended_{case}.csv", index=False)
        viol.to_csv(case_dir / f"violations_summary_{case}.csv", index=False)
        rank_df.to_csv(case_dir / f"rank_summary_{case}.csv", index=False)
        wil_df.to_csv(case_dir / f"wilcoxon_{case}.csv", index=False)
        friedman_case_df.to_csv(case_dir / f"friedman_{case}.csv", index=False)
        avg_ranks_case_df.to_csv(case_dir / f"avg_ranks_{case}.csv", index=False)

        # plots
        plot_convergence(case_dir, case_name, algo_names, checkpoints, curves)
        plot_boxplot_fitness(case_dir, case_name, df_case, algo_names)
        plot_boxplot_feasible_psrr(case_dir, case_name, df_case, algo_names)
        plot_success_rate_bar(case_dir, case_name, ext.drop(columns=["case"]) if "case" in ext.columns else ext, algo_names)
        plot_violations_bar(case_dir, case_name, viol.drop(columns=["case"]) if "case" in viol.columns else viol, algo_names)

        # try latex export (optional, requires Jinja2)
        try:
            with open(case_dir / f"tables_{case}.tex", "w", encoding="utf-8") as f:
                f.write(f"% Case: {case_name}\n\n")
                f.write("% Basic stats\n")
                f.write(basic.to_latex(index=False, float_format="%.6g"))
                f.write("\n\n% Extended stats\n")
                f.write(ext.to_latex(index=False, float_format="%.6g"))
                f.write("\n\n% Violations summary\n")
                f.write(viol.to_latex(index=False, float_format="%.6g"))
                f.write("\n\n% Wilcoxon ABC vs others + Holm\n")
                f.write(wil_df.to_latex(index=False, float_format="%.6g"))
                f.write("\n\n% Friedman per-case\n")
                f.write(friedman_case_df.to_latex(index=False, float_format="%.6g"))
                f.write("\n\n% Avg ranks per-case\n")
                f.write(avg_ranks_case_df.to_latex(index=False, float_format="%.6g"))
                f.write("\n\n% Rank summary\n")
                f.write(rank_df.to_latex(index=False, float_format="%.6g"))
        except Exception as e:
            with open(case_dir / f"tables_{case}_EXPORT_ERROR.txt", "w", encoding="utf-8") as f:
                f.write("LaTeX export skipped or failed.\n")
                f.write(str(e))

        print(f"\n✅ انتهت حالة {case_name}. المخرجات داخل: {case_dir.resolve()}")

        all_case_stats.append(basic)
        all_case_ext.append(ext)
        all_case_viol.append(viol)
        all_case_wil.append(wil_df)
        all_case_ranks.append(rank_df)
        all_case_friedman.append(friedman_case_df)
        all_case_avg_ranks.append(avg_ranks_case_df)

    # ======================================================
    # Combined outputs (all cases)
    # ======================================================
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(outdir / "summary.csv", index=False)

    stats_all = pd.concat(all_case_stats, ignore_index=True) if len(all_case_stats) else pd.DataFrame()
    ext_all = pd.concat(all_case_ext, ignore_index=True) if len(all_case_ext) else pd.DataFrame()
    viol_all = pd.concat(all_case_viol, ignore_index=True) if len(all_case_viol) else pd.DataFrame()
    wil_all = pd.concat(all_case_wil, ignore_index=True) if len(all_case_wil) else pd.DataFrame()
    ranks_all = pd.concat(all_case_ranks, ignore_index=True) if len(all_case_ranks) else pd.DataFrame()
    friedman_all_cases = pd.concat(all_case_friedman, ignore_index=True) if len(all_case_friedman) else pd.DataFrame()
    avg_ranks_all_cases = pd.concat(all_case_avg_ranks, ignore_index=True) if len(all_case_avg_ranks) else pd.DataFrame()

    if not stats_all.empty:
        stats_all.to_csv(outdir / "stats.csv", index=False)
    if not ext_all.empty:
        ext_all.to_csv(outdir / "stats_extended.csv", index=False)
    if not viol_all.empty:
        viol_all.to_csv(outdir / "violations_summary.csv", index=False)
    if not wil_all.empty:
        wil_all.to_csv(outdir / "wilcoxon.csv", index=False)
    if not ranks_all.empty:
        ranks_all.to_csv(outdir / "rank_summary.csv", index=False)
    if not friedman_all_cases.empty:
        friedman_all_cases.to_csv(outdir / "friedman_per_case.csv", index=False)
    if not avg_ranks_all_cases.empty:
        avg_ranks_all_cases.to_csv(outdir / "avg_ranks_per_case.csv", index=False)

    # Overall rank across cases (mean of avg_rank from rank_summary)
    if not ranks_all.empty and "avg_rank" in ranks_all.columns:
        overall_rank = (
            ranks_all.groupby("algo")["avg_rank"]
            .mean()
            .reset_index()
            .sort_values("avg_rank", ascending=True)
        )
        overall_rank.to_csv(outdir / "overall_rank_across_cases.csv", index=False)
    else:
        overall_rank = pd.DataFrame()

    # Overall Friedman across all blocks = (case, run)
    pivot_all_blocks = df_all.pivot(index=["case", "run"], columns="algo", values="best_fitness") if not df_all.empty else pd.DataFrame()
    friedman_overall_df, avg_ranks_overall_df = friedman_from_pivot(pivot_all_blocks, algo_names, scope_label="overall_blocks(case,run)")
    if not friedman_overall_df.empty:
        friedman_overall_df.to_csv(outdir / "friedman_overall_blocks.csv", index=False)
    if not avg_ranks_overall_df.empty:
        avg_ranks_overall_df.to_csv(outdir / "avg_ranks_overall_blocks.csv", index=False)

    # Overall Wilcoxon (ABC vs others) across all blocks + Holm
    wil_overall_blocks = build_overall_wilcoxon_blocks(df_all, algo_names, base_algo="ABC")
    if not wil_overall_blocks.empty:
        wil_overall_blocks.to_csv(outdir / "wilcoxon_overall_blocks.csv", index=False)

    # Combined latex (optional)
    try:
        with open(outdir / "tables_all_cases.tex", "w", encoding="utf-8") as f:
            f.write("% Combined tables across all cases\n\n")
            for title, dfx in [
                ("stats.csv", stats_all),
                ("stats_extended.csv", ext_all),
                ("violations_summary.csv", viol_all),
                ("wilcoxon.csv", wil_all),
                ("rank_summary.csv", ranks_all),
                ("friedman_per_case.csv", friedman_all_cases),
                ("avg_ranks_per_case.csv", avg_ranks_all_cases),
                ("overall_rank_across_cases.csv", overall_rank),
                ("friedman_overall_blocks.csv", friedman_overall_df),
                ("avg_ranks_overall_blocks.csv", avg_ranks_overall_df),
                ("wilcoxon_overall_blocks.csv", wil_overall_blocks),
            ]:
                if dfx is not None and not dfx.empty:
                    f.write(f"% {title}\n")
                    f.write(dfx.to_latex(index=False, float_format='%.6g'))
                    f.write("\n\n")
    except Exception as e:
        with open(outdir / "tables_all_cases_EXPORT_ERROR.txt", "w", encoding="utf-8") as f:
            f.write("Combined LaTeX export skipped or failed.\n")
            f.write(str(e))

    # Optional note if SciPy missing
    if not SCIPY_AVAILABLE:
        with open(outdir / "SCIPY_MISSING_NOTE.txt", "w", encoding="utf-8") as f:
            f.write("SciPy is not installed. Friedman p-values were not computed.\n")
            f.write("Install SciPy to enable scipy.stats.friedmanchisquare.\n")

    print("\n============================================================")
    print("✅ All experiments finished successfully.")
    print("Outputs saved to:", outdir.resolve())
    print("Main files:")
    print(" - summary.csv")
    print(" - stats.csv")
    print(" - stats_extended.csv")
    print(" - violations_summary.csv")
    print(" - wilcoxon.csv (with Holm columns)")
    print(" - friedman_per_case.csv")
    print(" - friedman_overall_blocks.csv")
    print(" - avg_ranks_overall_blocks.csv")
    print(" - rank_summary.csv")
    print(" - case_<name>/ plots + per-case tables")
    print("============================================================")


if __name__ == "__main__":
    main()
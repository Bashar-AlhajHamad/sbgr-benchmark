import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Any

Array = np.ndarray


@dataclass
class RunResult:
    best_x: Array
    best_f: float
    history_evals: Array
    history_best: Array


class ObjectiveWrapper:
    """
    Wrapper لحساب عدد evaluations ومنع تجاوز الميزانية
    + دعم progress callback
    + تتبع أول feasible evaluation وأفضل feasible solution
    """

    def __init__(
        self,
        func: Optional[Callable[[Array], float]],
        max_evals: int,
        progress_cb: Optional[Callable[[int, int, float], None]] = None,
        progress_step: int = 2000,
        eval_with_metrics: Optional[Callable[[Array], Tuple[float, Dict[str, Any]]]] = None,
    ):
        if func is None and eval_with_metrics is None:
            raise ValueError("Either func or eval_with_metrics must be provided.")

        self.func = func
        self.eval_with_metrics = eval_with_metrics

        self.max_evals = int(max_evals)
        self.evals = 0

        self.progress_cb = progress_cb
        self.progress_step = int(progress_step)

        self.best_seen = float("inf")

        # tracking feasibility
        self.first_feasible_eval: Optional[int] = None
        self.best_feasible_f: float = float("inf")
        self.best_feasible_x: Optional[Array] = None
        self.best_feasible_metrics: Optional[Dict[str, Any]] = None

        # latest call info (useful for debugging)
        self.last_f: Optional[float] = None
        self.last_metrics: Optional[Dict[str, Any]] = None

    def __call__(self, x: Array) -> float:
        if self.evals >= self.max_evals:
            return float("inf")

        self.evals += 1
        x_arr = np.asarray(x, dtype=float)

        if self.eval_with_metrics is not None:
            val, metrics = self.eval_with_metrics(x_arr)
            val = float(val)
            metrics = dict(metrics) if metrics is not None else {}
        else:
            val = float(self.func(x_arr))  # type: ignore[arg-type]
            metrics = {}

        self.last_f = val
        self.last_metrics = metrics if metrics else None

        if val < self.best_seen:
            self.best_seen = val

        # feasibility tracking (إذا كانت metrics متوفرة)
        if metrics:
            is_feasible = int(metrics.get("is_feasible", 0))
            if is_feasible == 1:
                if self.first_feasible_eval is None:
                    self.first_feasible_eval = self.evals
                if val < self.best_feasible_f:
                    self.best_feasible_f = val
                    self.best_feasible_x = x_arr.copy()
                    self.best_feasible_metrics = dict(metrics)

        # progress callback
        if self.progress_cb:
            if (self.evals % self.progress_step == 0) or (self.evals == self.max_evals):
                self.progress_cb(self.evals, self.max_evals, self.best_seen)

        return val


def clip(x: Array, lb: Array, ub: Array) -> Array:
    return np.minimum(np.maximum(x, lb), ub)


def init_population(rng: np.random.Generator, pop: int, dim: int, lb: Array, ub: Array) -> Array:
    return rng.uniform(lb, ub, size=(pop, dim))


def record(history: List[Tuple[int, float]], evals: int, best: float):
    if not history or best < history[-1][1] - 1e-12:
        history.append((int(evals), float(best)))


def finalize_history(history: List[Tuple[int, float]], max_evals: int) -> Tuple[Array, Array]:
    if not history:
        return np.array([0, max_evals], dtype=int), np.array([np.inf, np.inf], dtype=float)
    if history[-1][0] < max_evals:
        history.append((max_evals, history[-1][1]))
    he = np.array([h[0] for h in history], dtype=int)
    hb = np.array([h[1] for h in history], dtype=float)
    return he, hb


# =========================================================
# ABC (Artificial Bee Colony)
# =========================================================
def abc_optimize(
    obj: ObjectiveWrapper,
    rng: np.random.Generator,
    lb: Array,
    ub: Array,
    pop: int,
    dim: int,
    limit: Optional[int] = None,
) -> RunResult:
    if limit is None:
        limit = int(0.6 * pop * dim)

    X = init_population(rng, pop, dim, lb, ub)
    fit = np.array([obj(x) for x in X], dtype=float)
    trials = np.zeros(pop, dtype=int)

    best_idx = int(np.argmin(fit))
    best_x = X[best_idx].copy()
    best_f = float(fit[best_idx])

    history: List[Tuple[int, float]] = []
    record(history, obj.evals, best_f)

    def fitness_to_prob(f: Array) -> Array:
        # تحويل fitness لاحتمالات selection (كلما كانت القيمة أصغر كان أفضل)
        q = np.where(f >= 0, 1.0 / (1.0 + f), 1.0 + np.abs(f))
        p = q / (np.sum(q) + 1e-12)
        return p

    while obj.evals < obj.max_evals:
        # Employed bees
        for i in range(pop):
            if obj.evals >= obj.max_evals:
                break

            k = int(rng.integers(0, pop))
            while k == i:
                k = int(rng.integers(0, pop))

            j = int(rng.integers(0, dim))
            phi = rng.uniform(-1.0, 1.0)

            v = X[i].copy()
            v[j] = v[j] + phi * (v[j] - X[k][j])
            v = clip(v, lb, ub)

            fv = obj(v)
            if fv < fit[i]:
                X[i] = v
                fit[i] = fv
                trials[i] = 0
            else:
                trials[i] += 1

        # Onlooker bees
        p = fitness_to_prob(fit)
        t = 0
        i = 0
        while t < pop and obj.evals < obj.max_evals:
            if rng.random() < p[i]:
                t += 1

                k = int(rng.integers(0, pop))
                while k == i:
                    k = int(rng.integers(0, pop))

                j = int(rng.integers(0, dim))
                phi = rng.uniform(-1.0, 1.0)

                v = X[i].copy()
                v[j] = v[j] + phi * (v[j] - X[k][j])
                v = clip(v, lb, ub)

                fv = obj(v)
                if fv < fit[i]:
                    X[i] = v
                    fit[i] = fv
                    trials[i] = 0
                else:
                    trials[i] += 1

            i = (i + 1) % pop

        # Scout bees
        for i in range(pop):
            if obj.evals >= obj.max_evals:
                break
            if trials[i] >= limit:
                X[i] = rng.uniform(lb, ub, size=(dim,))
                fit[i] = obj(X[i])
                trials[i] = 0

        bi = int(np.argmin(fit))
        if fit[bi] < best_f:
            best_f = float(fit[bi])
            best_x = X[bi].copy()
            record(history, obj.evals, best_f)

    he, hb = finalize_history(history, obj.max_evals)
    return RunResult(best_x=best_x, best_f=best_f, history_evals=he, history_best=hb)


# =========================================================
# GWO (Grey Wolf Optimizer)
# =========================================================
def gwo_optimize(
    obj: ObjectiveWrapper,
    rng: np.random.Generator,
    lb: Array,
    ub: Array,
    pop: int,
    dim: int,
) -> RunResult:
    X = init_population(rng, pop, dim, lb, ub)
    fit = np.array([obj(x) for x in X], dtype=float)

    order = np.argsort(fit)
    alpha = X[order[0]].copy()
    beta = X[order[1]].copy()
    delta = X[order[2]].copy()

    best_x = alpha.copy()
    best_f = float(fit[order[0]])

    history: List[Tuple[int, float]] = []
    record(history, obj.evals, best_f)

    max_iters = max(1, (obj.max_evals - obj.evals) // pop + 1)
    for it in range(max_iters):
        if obj.evals >= obj.max_evals:
            break

        a = 2.0 - 2.0 * (it / max_iters)
        X_new = np.zeros_like(X)

        for i in range(pop):
            r1 = rng.random((3, dim))
            r2 = rng.random((3, dim))
            A = 2 * a * r1 - a
            C = 2 * r2

            D_alpha = np.abs(C[0] * alpha - X[i])
            D_beta = np.abs(C[1] * beta - X[i])
            D_delta = np.abs(C[2] * delta - X[i])

            X1 = alpha - A[0] * D_alpha
            X2 = beta - A[1] * D_beta
            X3 = delta - A[2] * D_delta

            X_new[i] = (X1 + X2 + X3) / 3.0

        X = clip(X_new, lb, ub)

        for i in range(pop):
            if obj.evals >= obj.max_evals:
                break
            fit[i] = obj(X[i])

        order = np.argsort(fit)
        alpha = X[order[0]].copy()
        beta = X[order[1]].copy()
        delta = X[order[2]].copy()

        if fit[order[0]] < best_f:
            best_f = float(fit[order[0]])
            best_x = alpha.copy()
            record(history, obj.evals, best_f)

    he, hb = finalize_history(history, obj.max_evals)
    return RunResult(best_x=best_x, best_f=best_f, history_evals=he, history_best=hb)


# =========================================================
# PSO (Particle Swarm Optimization)
# =========================================================
def pso_optimize(
    obj: ObjectiveWrapper,
    rng: np.random.Generator,
    lb: Array,
    ub: Array,
    pop: int,
    dim: int,
    w: float = 0.72,
    c1: float = 1.49,
    c2: float = 1.49,
) -> RunResult:
    X = init_population(rng, pop, dim, lb, ub)
    V = np.zeros_like(X)

    fit = np.array([obj(x) for x in X], dtype=float)
    pbest = X.copy()
    pbest_fit = fit.copy()

    g_idx = int(np.argmin(fit))
    gbest = X[g_idx].copy()
    gbest_fit = float(fit[g_idx])

    v_max = 0.2 * (ub - lb)

    history: List[Tuple[int, float]] = []
    record(history, obj.evals, gbest_fit)

    while obj.evals < obj.max_evals:
        r1 = rng.random((pop, dim))
        r2 = rng.random((pop, dim))

        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -v_max, v_max)

        X = clip(X + V, lb, ub)

        for i in range(pop):
            if obj.evals >= obj.max_evals:
                break

            fi = obj(X[i])
            fit[i] = fi

            if fi < pbest_fit[i]:
                pbest_fit[i] = fi
                pbest[i] = X[i].copy()

                if fi < gbest_fit:
                    gbest_fit = float(fi)
                    gbest = X[i].copy()
                    record(history, obj.evals, gbest_fit)

    he, hb = finalize_history(history, obj.max_evals)
    return RunResult(best_x=gbest, best_f=gbest_fit, history_evals=he, history_best=hb)


# =========================================================
# FA (Firefly Algorithm)
# =========================================================
def fa_optimize(
    obj: ObjectiveWrapper,
    rng: np.random.Generator,
    lb: Array,
    ub: Array,
    pop: int,
    dim: int,
    alpha0: float = 0.25,
    beta0: float = 1.0,
    gamma: float = 1.0,
) -> RunResult:
    X = init_population(rng, pop, dim, lb, ub)
    fit = np.array([obj(x) for x in X], dtype=float)

    best_idx = int(np.argmin(fit))
    best_x = X[best_idx].copy()
    best_f = float(fit[best_idx])

    history: List[Tuple[int, float]] = []
    record(history, obj.evals, best_f)

    step = (ub - lb)
    max_iters = max(1, (obj.max_evals - obj.evals) // pop + 1)

    for it in range(max_iters):
        if obj.evals >= obj.max_evals:
            break

        alpha = alpha0 * (0.97 ** it)

        for i in range(pop):
            for j in range(pop):
                if fit[j] < fit[i]:
                    rij = np.linalg.norm((X[i] - X[j]) / (step + 1e-12))
                    beta = beta0 * np.exp(-gamma * rij * rij)
                    X[i] = X[i] + beta * (X[j] - X[i]) + alpha * (rng.random(dim) - 0.5) * step
                    X[i] = clip(X[i], lb, ub)

            if obj.evals >= obj.max_evals:
                break

            fit[i] = obj(X[i])
            if fit[i] < best_f:
                best_f = float(fit[i])
                best_x = X[i].copy()
                record(history, obj.evals, best_f)

    he, hb = finalize_history(history, obj.max_evals)
    return RunResult(best_x=best_x, best_f=best_f, history_evals=he, history_best=hb)


# =========================================================
# GA (Real-coded): SBX + polynomial mutation
# =========================================================
def _tournament(rng: np.random.Generator, fit: Array, k: int = 3) -> int:
    idx = rng.integers(0, len(fit), size=(k,))
    best = idx[np.argmin(fit[idx])]
    return int(best)


def _sbx_crossover(
    rng: np.random.Generator,
    p1: Array,
    p2: Array,
    lb: Array,
    ub: Array,
    eta: float = 15.0,
    prob: float = 0.9,
) -> Tuple[Array, Array]:
    if rng.random() > prob:
        return p1.copy(), p2.copy()

    dim = p1.shape[0]
    c1 = p1.copy()
    c2 = p2.copy()

    for i in range(dim):
        if rng.random() <= 0.5 and abs(p1[i] - p2[i]) > 1e-14:
            x1 = min(p1[i], p2[i])
            x2 = max(p1[i], p2[i])
            rand = rng.random()

            beta = 1.0 + (2.0 * (x1 - lb[i]) / (x2 - x1 + 1e-12))
            alpha = 2.0 - beta ** (-(eta + 1.0))
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
            child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

            beta = 1.0 + (2.0 * (ub[i] - x2) / (x2 - x1 + 1e-12))
            alpha = 2.0 - beta ** (-(eta + 1.0))
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
            child2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

            c1[i] = np.clip(child1, lb[i], ub[i])
            c2[i] = np.clip(child2, lb[i], ub[i])

    return c1, c2


def _poly_mutation(
    rng: np.random.Generator,
    x: Array,
    lb: Array,
    ub: Array,
    eta: float = 20.0,
    pm: Optional[float] = None,
) -> Array:
    dim = x.shape[0]
    if pm is None:
        pm = 1.0 / dim

    y = x.copy()

    for i in range(dim):
        if rng.random() < pm:
            delta1 = (y[i] - lb[i]) / (ub[i] - lb[i] + 1e-12)
            delta2 = (ub[i] - y[i]) / (ub[i] - lb[i] + 1e-12)
            rand = rng.random()
            mut_pow = 1.0 / (eta + 1.0)

            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                deltaq = 1.0 - val ** mut_pow

            y[i] = np.clip(y[i] + deltaq * (ub[i] - lb[i]), lb[i], ub[i])

    return y


def ga_optimize(
    obj: ObjectiveWrapper,
    rng: np.random.Generator,
    lb: Array,
    ub: Array,
    pop: int,
    dim: int,
    pc: float = 0.9,
) -> RunResult:
    X = init_population(rng, pop, dim, lb, ub)
    fit = np.array([obj(x) for x in X], dtype=float)

    best_idx = int(np.argmin(fit))
    best_x = X[best_idx].copy()
    best_f = float(fit[best_idx])

    history: List[Tuple[int, float]] = []
    record(history, obj.evals, best_f)

    while obj.evals < obj.max_evals:
        elite_idx = int(np.argmin(fit))
        elite = X[elite_idx].copy()
        elite_fit = float(fit[elite_idx])

        offspring = []
        while len(offspring) < pop and obj.evals < obj.max_evals:
            p1 = X[_tournament(rng, fit)]
            p2 = X[_tournament(rng, fit)]

            c1, c2 = _sbx_crossover(rng, p1, p2, lb, ub, prob=pc)
            c1 = _poly_mutation(rng, c1, lb, ub)
            c2 = _poly_mutation(rng, c2, lb, ub)

            offspring.append(c1)
            if len(offspring) < pop:
                offspring.append(c2)

        Y = np.array(offspring[:pop], dtype=float)
        yfit = np.zeros(pop, dtype=float)

        for i in range(pop):
            if obj.evals >= obj.max_evals:
                yfit[i] = float("inf")
            else:
                yfit[i] = obj(Y[i])

        # (mu + lambda) replacement + elitism
        Xc = np.vstack([X, Y, elite.reshape(1, -1)])
        fc = np.concatenate([fit, yfit, np.array([elite_fit])])

        order = np.argsort(fc)
        X = Xc[order[:pop]].copy()
        fit = fc[order[:pop]].copy()

        if fit[0] < best_f:
            best_f = float(fit[0])
            best_x = X[0].copy()
            record(history, obj.evals, best_f)

    he, hb = finalize_history(history, obj.max_evals)
    return RunResult(best_x=best_x, best_f=best_f, history_evals=he, history_best=hb)


# =========================================================
# ACO for continuous optimization: ACOR
# =========================================================
def acor_optimize(
    obj: ObjectiveWrapper,
    rng: np.random.Generator,
    lb: Array,
    ub: Array,
    pop: int,
    dim: int,
    q: float = 0.5,
    zeta: float = 1.0,
) -> RunResult:
    k = pop  # archive size
    m = pop  # samples per iteration

    X = init_population(rng, k, dim, lb, ub)
    fit = np.array([obj(x) for x in X], dtype=float)

    order = np.argsort(fit)
    X = X[order].copy()
    fit = fit[order].copy()

    best_x = X[0].copy()
    best_f = float(fit[0])

    history: List[Tuple[int, float]] = []
    record(history, obj.evals, best_f)

    ranks = np.arange(k)
    denom = q * k
    w = (1.0 / (np.sqrt(2 * np.pi) * denom)) * np.exp(-(ranks ** 2) / (2 * (denom ** 2)))
    w = w / (np.sum(w) + 1e-12)
    cdf = np.cumsum(w)

    while obj.evals < obj.max_evals:
        sigmas = np.zeros((k, dim), dtype=float)

        for i in range(k):
            sigmas[i] = zeta * np.sum(np.abs(X[i] - X), axis=0) / max(1, k - 1)
            sigmas[i] = np.maximum(sigmas[i], 1e-12)

        Y = np.zeros((m, dim), dtype=float)
        for s in range(m):
            r = rng.random()
            i = int(np.searchsorted(cdf, r, side="right"))
            i = min(max(i, 0), k - 1)

            mu = X[i]
            sigma = sigmas[i]

            y = rng.normal(mu, sigma)
            y = clip(y, lb, ub)
            Y[s] = y

        yfit = np.zeros(m, dtype=float)
        for i in range(m):
            if obj.evals >= obj.max_evals:
                yfit[i] = float("inf")
            else:
                yfit[i] = obj(Y[i])

        Xc = np.vstack([X, Y])
        fc = np.concatenate([fit, yfit])

        order = np.argsort(fc)
        X = Xc[order[:k]].copy()
        fit = fc[order[:k]].copy()

        if fit[0] < best_f:
            best_f = float(fit[0])
            best_x = X[0].copy()
            record(history, obj.evals, best_f)

    he, hb = finalize_history(history, obj.max_evals)
    return RunResult(best_x=best_x, best_f=best_f, history_evals=he, history_best=hb)


# =========================================================
# IPA (Immune Plasma Algorithm)
# =========================================================
def ipa_optimize(
    obj: ObjectiveWrapper,
    rng: np.random.Generator,
    lb: Array,
    ub: Array,
    pop: int,
    dim: int,
    donors: int = 1,
    receivers: int = 1,
) -> RunResult:
    X = init_population(rng, pop, dim, lb, ub)
    fit = np.array([obj(x) for x in X], dtype=float)

    best_idx = int(np.argmin(fit))
    best_x = X[best_idx].copy()
    best_f = float(fit[best_idx])

    history: List[Tuple[int, float]] = []
    record(history, obj.evals, best_f)

    def donors_receivers_idx(fit_arr: Array) -> Tuple[List[int], List[int]]:
        order = np.argsort(fit_arr)
        donors_idx = [int(order[i]) for i in range(min(donors, pop))]
        receivers_idx = [int(order[-1 - i]) for i in range(min(receivers, pop))]
        return donors_idx, receivers_idx

    def infection(xk: Array, xm: Array) -> Array:
        j = int(rng.integers(0, dim))
        y = xk.copy()
        y[j] = y[j] + rng.uniform(-1.0, 1.0) * (y[j] - xm[j])
        return clip(y, lb, ub)

    def plasma_transfer(receiver: Array, donor: Array) -> Array:
        y = receiver.copy()
        for j in range(dim):
            y[j] = y[j] + rng.uniform(-1.0, 1.0) * (y[j] - donor[j])
        return clip(y, lb, ub)

    def update_donor(donor: Array) -> Array:
        y = donor.copy()
        for j in range(dim):
            y[j] = y[j] + rng.uniform(-1.0, 1.0) * y[j]
        return clip(y, lb, ub)

    while obj.evals < obj.max_evals:
        # Infection phase
        for i in range(pop):
            if obj.evals >= obj.max_evals:
                break

            m = int(rng.integers(0, pop))
            while m == i:
                m = int(rng.integers(0, pop))

            y = infection(X[i], X[m])
            fy = obj(y)

            if fy < fit[i]:
                X[i] = y
                fit[i] = fy

                if fy < best_f:
                    best_f = float(fy)
                    best_x = y.copy()
                    record(history, obj.evals, best_f)

        if obj.evals >= obj.max_evals:
            break

        # Plasma transfer phase
        donors_idx, receivers_idx = donors_receivers_idx(fit)

        dose_control = np.ones(len(receivers_idx), dtype=int)
        treatment_control = np.ones(len(receivers_idx), dtype=int)

        for ri in range(len(receivers_idx)):
            if obj.evals >= obj.max_evals:
                break

            receiver_index = receivers_idx[ri]
            donor_index = donors_idx[int(rng.integers(0, len(donors_idx)))]

            receiver = X[receiver_index].copy()
            donor = X[donor_index].copy()

            while treatment_control[ri] == 1 and obj.evals < obj.max_evals:
                y = plasma_transfer(receiver, donor)
                fy = obj(y)

                if dose_control[ri] == 1:
                    if fy < fit[donor_index]:
                        dose_control[ri] += 1
                        receiver = y
                        X[receiver_index] = y
                        fit[receiver_index] = fy
                    else:
                        receiver = donor.copy()
                        X[receiver_index] = receiver
                        fit[receiver_index] = fit[donor_index]
                        treatment_control[ri] = 0
                else:
                    if fy < fit[receiver_index]:
                        receiver = y
                        X[receiver_index] = y
                        fit[receiver_index] = fy
                    else:
                        treatment_control[ri] = 0

                if fit[receiver_index] < best_f:
                    best_f = float(fit[receiver_index])
                    best_x = X[receiver_index].copy()
                    record(history, obj.evals, best_f)

        if obj.evals >= obj.max_evals:
            break

        # Donors update phase
        donors_idx, _ = donors_receivers_idx(fit)
        for donor_index in donors_idx:
            if obj.evals >= obj.max_evals:
                break

            if (obj.evals / obj.max_evals) > rng.random():
                y = update_donor(X[donor_index])
            else:
                y = rng.uniform(lb, ub, size=(dim,))

            fy = obj(y)
            X[donor_index] = y
            fit[donor_index] = fy

            if fy < best_f:
                best_f = float(fy)
                best_x = y.copy()
                record(history, obj.evals, best_f)

    he, hb = finalize_history(history, obj.max_evals)
    return RunResult(best_x=best_x, best_f=best_f, history_evals=he, history_best=hb)


ALGORITHMS = {
    "ABC": abc_optimize,
    "GWO": gwo_optimize,
    "IPA": ipa_optimize,
    "FA": fa_optimize,
    "PSO": pso_optimize,
    "GA": ga_optimize,
    "ACO": acor_optimize,  # ACOR for continuous domains
}
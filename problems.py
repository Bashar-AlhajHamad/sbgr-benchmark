import numpy as np
from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class Problem:
    name: str
    dim: int
    lb: np.ndarray
    ub: np.ndarray

    def evaluate(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def metrics(self, x: np.ndarray) -> Dict[str, Any]:
        return {}


def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


class SBGRProblem(Problem):
    """
    SBGR = Surrogate BGR-like constrained optimization problem (بدون SPICE)

    الهدف:
        maximize PSRR
    لكننا نحلها كتصغير:
        fitness = -PSRR + 1000 * penalty

    الحالات المدعومة:
    - base    : الحالة الأساسية
    - hard    : قيود أشد (أصعب)
    - highdim : نفس القيود تقريبًا ولكن بعدد أبعاد أكبر لاختبار scalability
    """

    def __init__(self, dim: int = 12, seed: int = 0, case: str = "base"):
        case = case.lower().strip()
        if case not in {"base", "hard", "highdim"}:
            raise ValueError(f"Unsupported SBGR case: {case}")

        # في حالة highdim نضمن أن الأبعاد أكبر من 12 عادةً (لكن لا نجبر المستخدم)
        self.case = case
        self._constraints = self._build_constraints(case)

        lb, ub = self._build_bounds(dim)
        super().__init__(name=f"SBGR-{case.upper()}", dim=dim, lb=lb, ub=ub)

        rng = np.random.default_rng(seed)
        self._x_star = rng.uniform(self.lb, self.ub)

    # -----------------------------
    # Bounds / constraints
    # -----------------------------
    @staticmethod
    def _build_bounds(dim: int):
        # أول 12 متغيرًا (مشابهة للنسخة الحالية)
        base_lb = np.array(
            [0.5, 0.5, 0.5, 0.5, 0.18, 0.18, 1.0, 1.0, 0.5, 1.4, 0.1, 0.5],
            dtype=float
        )
        base_ub = np.array(
            [50.0, 50.0, 50.0, 50.0, 5.0, 5.0, 200.0, 200.0, 50.0, 3.3, 10.0, 2.0],
            dtype=float
        )

        if dim <= 12:
            return base_lb[:dim], base_ub[:dim]

        # أبعاد إضافية لاختبار scalability (تؤثر عبر d2)
        extra = dim - 12
        extra_lb = np.zeros(extra, dtype=float)
        extra_ub = np.ones(extra, dtype=float)

        lb = np.concatenate([base_lb, extra_lb])
        ub = np.concatenate([base_ub, extra_ub])
        return lb, ub

    @staticmethod
    def _build_constraints(case: str):
        # القيود في النسخة الحالية (مناسبة لمسألة SBGR surrogate)
        base = {
            "VREF_min": 0.95,
            "VREF_max": 1.05,
            "TC_max": 30.0,
            "LoopGain_min": 80.0,
            "PhaseMargin_min": 60.0,
            "GainMargin_min": 10.0,
            "Power_max": 20.0,
        }

        if case == "base":
            return base

        if case == "hard":
            # نجعل القيود أشد قليلًا لاختبار robustness
            return {
                "VREF_min": 0.975,
                "VREF_max": 1.025,
                "TC_max": 24.0,
                "LoopGain_min": 85.0,
                "PhaseMargin_min": 65.0,
                "GainMargin_min": 12.0,
                "Power_max": 17.0,
            }

        if case == "highdim":
            # يمكن إبقاء القيود نفسها أو شدها قليلًا؛ هنا نُبقيها مثل base
            # حتى يكون الاختلاف الأساسي هو زيادة الأبعاد.
            return base.copy()

        raise ValueError(case)

    # -----------------------------
    # Core model
    # -----------------------------
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x - self.lb) / (self.ub - self.lb + 1e-12)

    def _simulate_metrics(self, x: np.ndarray) -> Dict[str, float]:
        x = np.asarray(x, dtype=float)
        s = self._normalize(x)
        center = self._normalize(self._x_star)

        # d2 يشمل كل الأبعاد -> الأبعاد الإضافية تؤثر على المسألة (مهم في highdim)
        d2 = float(np.sum((s - center) ** 2))

        # نستخدم أول 12 متغيرًا في الصيغ التفصيلية، وإذا dim أقل نؤمن fallback
        def sv(i: int, default: float = 0.5) -> float:
            return float(s[i]) if i < len(s) else float(default)

        # مؤشرات surrogate
        loop_gain = 65 + 35*np.exp(-3.2*d2) + 10*np.cos(2*np.pi*(sv(0) + sv(4))) - 6*np.abs(np.sin(3*np.pi*sv(1)))
        loop_gain = float(loop_gain)

        phase_margin = 45 + 30*(1 - np.abs(sv(10, 0.35) - 0.35)) + 10*np.exp(-5*d2) - 8*np.abs(np.sin(2*np.pi*sv(3)))
        phase_margin = float(phase_margin)

        gain_margin = 6 + 14*np.exp(-2.8*d2) + 6*_sigmoid(4*(sv(5) - 0.4)) - 4*np.abs(np.cos(2*np.pi*sv(2)))
        gain_margin = float(gain_margin)

        vref = 1.0 + 0.06*(sv(6) - 0.5) - 0.04*(sv(7) - 0.5) + 0.03*(sv(0) - sv(1)) + 0.02*np.sin(2*np.pi*sv(9))
        vref = float(vref)

        tc = 12 + 38*(np.abs(sv(2) - 0.55) + np.abs(sv(8) - 0.25)) + 18*np.abs(np.sin(2*np.pi*(sv(6) - sv(7))))
        tc = float(tc)

        # القدرة (تقريبية)
        ibias_uA = float(x[8]) if self.dim > 8 else float(0.5 + 49.5*sv(0))
        vdd = float(x[9]) if self.dim > 9 else float(1.4 + 1.9*sv(1))
        power_uW = 2.0 + 0.24*ibias_uA*vdd + 3.0*(sv(0) + sv(1)) + 2.0*np.abs(np.sin(2*np.pi*sv(4)))
        power_uW = float(power_uW)

        psrr = (
            50
            + 0.42*loop_gain
            + 0.18*phase_margin
            + 0.22*gain_margin
            + 8*np.exp(-2.4*d2)
            - 0.55*power_uW
            - 6*np.abs(np.sin(3*np.pi*(sv(6) + sv(7))))
        )
        psrr = float(psrr)

        c = self._constraints

        # تفصيل الانتهاكات
        viol_vref = int(vref < c["VREF_min"] or vref > c["VREF_max"])
        viol_tc = int(tc > c["TC_max"])
        viol_loop_gain = int(loop_gain < c["LoopGain_min"])
        viol_phase_margin = int(phase_margin < c["PhaseMargin_min"])
        viol_gain_margin = int(gain_margin < c["GainMargin_min"])
        viol_power = int(power_uW > c["Power_max"])

        penalty = (
            viol_vref
            + viol_tc
            + viol_loop_gain
            + viol_phase_margin
            + viol_gain_margin
            + viol_power
        )
        is_feasible = int(penalty == 0)

        return {
            # objective-related
            "PSRR_DB": psrr,
            "penalty": int(penalty),
            "is_feasible": is_feasible,

            # specs
            "VREF": vref,
            "TC": tc,
            "LOOP_GAIN_DB": loop_gain,
            "PHASE_MARGIN_DEG": phase_margin,
            "GAIN_MARGIN_DB": gain_margin,
            "POWER_UW": power_uW,

            # violation breakdown
            "viol_vref": viol_vref,
            "viol_tc": viol_tc,
            "viol_loop_gain": viol_loop_gain,
            "viol_phase_margin": viol_phase_margin,
            "viol_gain_margin": viol_gain_margin,
            "viol_power": viol_power,

            # constraints snapshot (مفيد للتوثيق/الdebug أحيانًا)
            "c_VREF_min": c["VREF_min"],
            "c_VREF_max": c["VREF_max"],
            "c_TC_max": c["TC_max"],
            "c_LoopGain_min": c["LoopGain_min"],
            "c_PhaseMargin_min": c["PhaseMargin_min"],
            "c_GainMargin_min": c["GainMargin_min"],
            "c_Power_max": c["Power_max"],
        }

    def fitness_from_metrics(self, m: Dict[str, float]) -> float:
        return float(-m["PSRR_DB"] + 1000.0 * m["penalty"])

    def evaluate(self, x: np.ndarray) -> float:
        m = self._simulate_metrics(x)
        return self.fitness_from_metrics(m)

    def evaluate_with_metrics(self, x: np.ndarray):
        """
        مفيد إذا أردت لاحقًا تقييمًا واحدًا وإرجاع fitness + metrics معًا.
        """
        m = self._simulate_metrics(x)
        f = self.fitness_from_metrics(m)
        return f, m

    def metrics(self, x: np.ndarray) -> Dict[str, Any]:
        return self._simulate_metrics(x)

    @property
    def constraints(self) -> Dict[str, float]:
        return dict(self._constraints)
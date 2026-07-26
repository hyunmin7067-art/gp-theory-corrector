from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    Matern,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

KERNEL_DESCRIPTIONS = {
    "RBF": "잔차가 매우 부드럽고 한 가지 대표 길이 척도로 변한다고 가정",
    "Matérn": "잔차가 연속적이지만 RBF보다 다소 거칠고 불규칙하게 변할 수 있다고 가정",
    "Rational Quadratic": "서로 다른 여러 길이 척도의 변화가 섞여 있다고 가정",
    "Linear": "잔차가 입력값에 따라 대체로 선형적으로 증가하거나 감소한다고 가정",
}

_COMPLEXITY_ORDER = {"Linear": 0, "RBF": 1, "Matérn": 2, "Rational Quadratic": 3}


@dataclass
class FittedResidualGP:
    name: str
    model: GaussianProcessRegressor
    x_scaler: StandardScaler
    y_scaler: StandardScaler

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=float).reshape(-1, 1)
        x_scaled = self.x_scaler.transform(x_arr)
        mean_scaled, std_scaled = self.model.predict(x_scaled, return_std=True)
        mean = self.y_scaler.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
        std = np.maximum(std_scaled * float(self.y_scaler.scale_[0]), 1e-12)
        return mean, std


@dataclass
class AnalysisResult:
    metrics: pd.DataFrame
    baseline_metrics: Dict[str, float]
    cv_predictions: Dict[str, pd.DataFrame]
    fitted_models: Dict[str, FittedResidualGP]
    best_kernel_name: str
    selection_reason: str


def build_candidate_kernels() -> Dict[str, object]:
    def signal():
        return ConstantKernel(1.0, (1e-3, 1e3))
    def noise():
        return WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-8, 1e1))
    return {
        "RBF": signal() * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + noise(),
        "Matérn": signal() * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + noise(),
        "Rational Quadratic": signal() * RationalQuadratic(
            length_scale=1.0,
            alpha=1.0,
            length_scale_bounds=(1e-2, 1e2),
            alpha_bounds=(1e-2, 1e2),
        ) + noise(),
        "Linear": signal() * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e2)) + noise(),
    }


def _fit_one_model(name, kernel, x_train, residual_train, restarts, random_state):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(np.asarray(x_train).reshape(-1, 1))
    y_scaled = y_scaler.fit_transform(np.asarray(residual_train).reshape(-1, 1)).ravel()

    model = GaussianProcessRegressor(
        kernel=clone(kernel),
        alpha=1e-10,
        normalize_y=False,
        n_restarts_optimizer=max(0, int(restarts)),
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x_scaled, y_scaled)
    return FittedResidualGP(name, model, x_scaler, y_scaler)


def _make_splitter(x):
    x_arr = np.asarray(x, dtype=float)
    if len(np.unique(x_arr)) == len(x_arr):
        return LeaveOneOut(), None, "Leave-One-Out"
    groups = pd.factorize(x_arr)[0]
    return LeaveOneGroupOut(), groups, "Leave-One-X-Group-Out"


def _safe_r2(y_true, y_pred):
    if len(y_true) < 2 or np.allclose(np.std(y_true), 0):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def _gaussian_nlpd(y_true, mean, std):
    sigma = np.maximum(np.asarray(std, dtype=float), 1e-9)
    error = np.asarray(y_true, dtype=float) - np.asarray(mean, dtype=float)
    return float(np.mean(0.5 * np.log(2.0 * np.pi * sigma**2) + 0.5 * (error / sigma) ** 2))


def _choose_best(metrics):
    valid = metrics.replace([np.inf, -np.inf], np.nan).dropna(subset=["CV_RMSE", "NLPD"])
    if valid.empty:
        raise ValueError("유효한 커널 성능 결과가 없습니다.")
    min_rmse = float(valid["CV_RMSE"].min())
    tolerance = max(0.03 * min_rmse, 1e-12)
    candidates = valid[valid["CV_RMSE"] <= min_rmse + tolerance].copy()
    candidates["Coverage_gap"] = (candidates["Coverage_95"] - 0.95).abs()
    candidates["Complexity"] = candidates["Kernel"].map(_COMPLEXITY_ORDER).fillna(99)
    candidates = candidates.sort_values(["NLPD", "Coverage_gap", "Complexity", "CV_RMSE"])
    best = str(candidates.iloc[0]["Kernel"])
    if len(candidates) == 1:
        reason = f"{best} 커널이 교차검증 RMSE가 가장 작아 선택되었습니다."
    else:
        reason = (
            f"교차검증 RMSE가 최저값의 3% 이내인 커널이 {len(candidates)}개여서, "
            f"그중 로그 예측밀도(NLPD)가 가장 우수한 {best} 커널을 선택했습니다."
        )
    return best, reason


def analyze_kernels(x, y_exp, y_theory, restarts=1, random_state=42):
    x = np.asarray(x, dtype=float).ravel()
    y_exp = np.asarray(y_exp, dtype=float).ravel()
    y_theory = np.asarray(y_theory, dtype=float).ravel()
    residual = y_exp - y_theory
    kernels = build_candidate_kernels()
    splitter, groups, validation_name = _make_splitter(x)
    splits = list(splitter.split(x.reshape(-1, 1), residual, groups=groups)) if groups is not None else list(splitter.split(x.reshape(-1, 1), residual))

    baseline_rmse = float(np.sqrt(mean_squared_error(y_exp, y_theory)))
    baseline_metrics = {
        "RMSE": baseline_rmse,
        "MAE": float(mean_absolute_error(y_exp, y_theory)),
        "R2": _safe_r2(y_exp, y_theory),
        "Validation": validation_name,
    }

    metrics_rows = []
    cv_predictions = {}
    fitted_models = {}

    for kernel_index, (name, kernel) in enumerate(kernels.items()):
        pred_residual = np.zeros_like(residual)
        pred_std = np.zeros_like(residual)

        for fold_index, (train_idx, test_idx) in enumerate(splits):
            fitted = _fit_one_model(
                name, kernel, x[train_idx], residual[train_idx], restarts,
                random_state + kernel_index * 1000 + fold_index,
            )
            mean, std = fitted.predict(x[test_idx])
            pred_residual[test_idx] = mean
            pred_std[test_idx] = std

        corrected = y_theory + pred_residual
        lower = corrected - 1.96 * pred_std
        upper = corrected + 1.96 * pred_std
        rmse = float(np.sqrt(mean_squared_error(y_exp, corrected)))
        improvement = 100.0 * (baseline_rmse - rmse) / baseline_rmse if baseline_rmse > 1e-12 else float("nan")

        final_fitted = _fit_one_model(
            name, kernel, x, residual, max(restarts, 2), random_state + kernel_index
        )
        fitted_models[name] = final_fitted
        cv_predictions[name] = pd.DataFrame({
            "x": x,
            "y_exp": y_exp,
            "y_theory": y_theory,
            "residual_observed": residual,
            "residual_predicted_cv": pred_residual,
            "residual_std_cv": pred_std,
            "y_corrected_cv": corrected,
            "lower_95_cv": lower,
            "upper_95_cv": upper,
        }).sort_values("x").reset_index(drop=True)

        metrics_rows.append({
            "Kernel": name,
            "CV_RMSE": rmse,
            "CV_MAE": float(mean_absolute_error(y_exp, corrected)),
            "CV_R2": _safe_r2(y_exp, corrected),
            "NLPD": _gaussian_nlpd(y_exp, corrected, pred_std),
            "Coverage_95": float(np.mean((y_exp >= lower) & (y_exp <= upper))),
            "RMSE_Improvement_%": improvement,
            "Optimized_kernel": str(final_fitted.model.kernel_),
            "Log_marginal_likelihood": float(final_fitted.model.log_marginal_likelihood_value_),
            "Validation": validation_name,
        })

    metrics = pd.DataFrame(metrics_rows).sort_values("CV_RMSE").reset_index(drop=True)
    best_name, reason = _choose_best(metrics)
    return AnalysisResult(metrics, baseline_metrics, cv_predictions, fitted_models, best_name, reason)

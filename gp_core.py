from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    Kernel,
    Matern,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut, LeaveOneOut
from sklearn.preprocessing import StandardScaler

KERNEL_DESCRIPTIONS = {
    "RBF": "모든 입력 방향에서 같은 길이 척도로 매우 부드럽게 변하는 잔차를 가정",
    "Matérn": "모든 입력 방향에서 같은 길이 척도를 사용하되 RBF보다 다소 거친 잔차를 허용",
    "Rational Quadratic": "서로 다른 여러 길이 척도의 변화가 섞인 잔차를 가정",
    "Linear": "잔차가 입력변수들의 선형 결합으로 대체로 증가하거나 감소한다고 가정",
    "ARD-RBF": "입력변수마다 별도의 길이 척도를 학습하는 다변수 RBF 커널",
}

_COMPLEXITY_ORDER = {
    "Linear": 0,
    "RBF": 1,
    "Matérn": 2,
    "ARD-RBF": 3,
    "Rational Quadratic": 4,
}


def _as_2d(values: np.ndarray, n_features: Optional[int] = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        if n_features is None or n_features == 1:
            array = array.reshape(-1, 1)
        elif array.size == n_features:
            array = array.reshape(1, -1)
        else:
            raise ValueError(
                f"입력값의 크기({array.size})가 독립변수 수({n_features})와 맞지 않습니다."
            )
    elif array.ndim != 2:
        raise ValueError("입력 행렬은 1차원 또는 2차원이어야 합니다.")

    if n_features is not None and array.shape[1] != n_features:
        raise ValueError(
            f"입력 행렬의 열 수({array.shape[1]})가 독립변수 수({n_features})와 맞지 않습니다."
        )
    return array


@dataclass
class FittedResidualGP:
    name: str
    model: GaussianProcessRegressor
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    feature_names: List[str]

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_array = _as_2d(x, len(self.feature_names))
        x_scaled = self.x_scaler.transform(x_array)
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
    feature_names: List[str]


def build_candidate_kernels(n_features: int) -> Dict[str, Kernel]:
    if n_features < 1:
        raise ValueError("독립변수가 하나 이상 필요합니다.")

    def signal() -> ConstantKernel:
        return ConstantKernel(1.0, (1e-3, 1e3))

    def noise() -> WhiteKernel:
        return WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-8, 1e1))

    kernels: Dict[str, Kernel] = {
        "RBF": signal() * RBF(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 1e2),
        ) + noise(),
        "Matérn": signal() * Matern(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 1e2),
            nu=1.5,
        ) + noise(),
        "Rational Quadratic": signal() * RationalQuadratic(
            length_scale=1.0,
            alpha=1.0,
            length_scale_bounds=(1e-2, 1e2),
            alpha_bounds=(1e-2, 1e2),
        ) + noise(),
        "Linear": signal() * DotProduct(
            sigma_0=1.0,
            sigma_0_bounds=(1e-3, 1e2),
        ) + noise(),
    }

    if n_features >= 2:
        kernels["ARD-RBF"] = signal() * RBF(
            length_scale=np.ones(n_features, dtype=float),
            length_scale_bounds=(1e-2, 1e2),
        ) + noise()
    return kernels


def _fit_one_model(
    name: str,
    kernel: Kernel,
    x_train: np.ndarray,
    residual_train: np.ndarray,
    feature_names: Sequence[str],
    restarts: int,
    random_state: int,
) -> FittedResidualGP:
    x_array = _as_2d(x_train, len(feature_names))
    residual_array = np.asarray(residual_train, dtype=float).ravel()

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x_array)
    y_scaled = y_scaler.fit_transform(residual_array.reshape(-1, 1)).ravel()

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

    return FittedResidualGP(
        name=name,
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_names=list(feature_names),
    )


def _condition_groups(x: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(x)
    return pd.factorize(pd.MultiIndex.from_frame(frame), sort=False)[0]


def _make_splitter(x: np.ndarray, random_state: int):
    groups = _condition_groups(x)
    n_samples = len(x)
    n_groups = len(np.unique(groups))
    has_repeats = n_groups < n_samples

    if has_repeats:
        if n_groups <= 60:
            return LeaveOneGroupOut(), groups, "Leave-One-Condition-Group-Out"
        n_splits = min(5, n_groups)
        return GroupKFold(n_splits=n_splits), groups, f"{n_splits}-Fold Group CV"

    if n_samples <= 60:
        return LeaveOneOut(), None, "Leave-One-Out"
    return KFold(n_splits=5, shuffle=True, random_state=random_state), None, "5-Fold CV"


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.allclose(np.std(y_true), 0):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def _gaussian_nlpd(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    sigma = np.maximum(np.asarray(std, dtype=float), 1e-9)
    error = np.asarray(y_true, dtype=float) - np.asarray(mean, dtype=float)
    return float(
        np.mean(0.5 * np.log(2.0 * np.pi * sigma**2) + 0.5 * (error / sigma) ** 2)
    )


def _choose_best(metrics: pd.DataFrame) -> Tuple[str, str]:
    valid = metrics.replace([np.inf, -np.inf], np.nan).dropna(subset=["CV_RMSE", "NLPD"])
    if valid.empty:
        raise ValueError("유효한 커널 성능 결과가 없습니다.")

    min_rmse = float(valid["CV_RMSE"].min())
    tolerance = max(0.03 * min_rmse, 1e-12)
    candidates = valid[valid["CV_RMSE"] <= min_rmse + tolerance].copy()
    candidates["Coverage_gap"] = (candidates["Coverage_95"] - 0.95).abs()
    candidates["Complexity"] = candidates["Kernel"].map(_COMPLEXITY_ORDER).fillna(99)
    candidates = candidates.sort_values(
        ["NLPD", "Coverage_gap", "Complexity", "CV_RMSE"]
    )

    best = str(candidates.iloc[0]["Kernel"])
    if len(candidates) == 1:
        reason = f"{best} 커널이 교차검증 RMSE가 가장 작아 선택되었습니다."
    else:
        reason = (
            f"교차검증 RMSE가 최저값의 3% 이내인 커널이 {len(candidates)}개여서, "
            f"그중 로그 예측밀도(NLPD)가 가장 우수한 {best} 커널을 선택했습니다."
        )
    return best, reason


def analyze_kernels(
    x: np.ndarray,
    y_exp: np.ndarray,
    y_theory: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    restarts: int = 1,
    random_state: int = 42,
) -> AnalysisResult:
    x_array = _as_2d(x)
    n_samples, n_features = x_array.shape
    names = list(feature_names) if feature_names is not None else [f"x{i + 1}" for i in range(n_features)]
    if len(names) != n_features:
        raise ValueError("독립변수 이름 수가 입력 행렬의 열 수와 맞지 않습니다.")

    y_exp_array = np.asarray(y_exp, dtype=float).ravel()
    y_theory_array = np.asarray(y_theory, dtype=float).ravel()
    if len(y_exp_array) != n_samples or len(y_theory_array) != n_samples:
        raise ValueError("입력 행 수와 실험값·이론값의 개수가 맞지 않습니다.")

    residual = y_exp_array - y_theory_array
    kernels = build_candidate_kernels(n_features)
    splitter, groups, validation_name = _make_splitter(x_array, random_state)
    if groups is None:
        splits = list(splitter.split(x_array, residual))
    else:
        splits = list(splitter.split(x_array, residual, groups=groups))

    baseline_rmse = float(np.sqrt(mean_squared_error(y_exp_array, y_theory_array)))
    baseline_metrics: Dict[str, float] = {
        "RMSE": baseline_rmse,
        "MAE": float(mean_absolute_error(y_exp_array, y_theory_array)),
        "R2": _safe_r2(y_exp_array, y_theory_array),
        "Validation": validation_name,
    }

    metric_rows = []
    cv_predictions: Dict[str, pd.DataFrame] = {}
    fitted_models: Dict[str, FittedResidualGP] = {}

    for kernel_index, (name, kernel) in enumerate(kernels.items()):
        predicted_residual = np.zeros_like(residual)
        predicted_std = np.zeros_like(residual)

        for fold_index, (train_index, test_index) in enumerate(splits):
            fitted = _fit_one_model(
                name=name,
                kernel=kernel,
                x_train=x_array[train_index],
                residual_train=residual[train_index],
                feature_names=names,
                restarts=restarts,
                random_state=random_state + kernel_index * 1000 + fold_index,
            )
            mean, std = fitted.predict(x_array[test_index])
            predicted_residual[test_index] = mean
            predicted_std[test_index] = std

        corrected = y_theory_array + predicted_residual
        lower = corrected - 1.96 * predicted_std
        upper = corrected + 1.96 * predicted_std
        rmse = float(np.sqrt(mean_squared_error(y_exp_array, corrected)))
        improvement = (
            100.0 * (baseline_rmse - rmse) / baseline_rmse
            if baseline_rmse > 1e-12
            else float("nan")
        )

        final_fitted = _fit_one_model(
            name=name,
            kernel=kernel,
            x_train=x_array,
            residual_train=residual,
            feature_names=names,
            restarts=max(restarts, 1),
            random_state=random_state + kernel_index,
        )
        fitted_models[name] = final_fitted

        prediction_frame = pd.DataFrame(x_array, columns=names)
        prediction_frame["y_exp"] = y_exp_array
        prediction_frame["y_theory"] = y_theory_array
        prediction_frame["residual_observed"] = residual
        prediction_frame["residual_predicted_cv"] = predicted_residual
        prediction_frame["residual_std_cv"] = predicted_std
        prediction_frame["y_corrected_cv"] = corrected
        prediction_frame["lower_95_cv"] = lower
        prediction_frame["upper_95_cv"] = upper
        cv_predictions[name] = prediction_frame

        metric_rows.append({
            "Kernel": name,
            "CV_RMSE": rmse,
            "CV_MAE": float(mean_absolute_error(y_exp_array, corrected)),
            "CV_R2": _safe_r2(y_exp_array, corrected),
            "NLPD": _gaussian_nlpd(y_exp_array, corrected, predicted_std),
            "Coverage_95": float(np.mean((y_exp_array >= lower) & (y_exp_array <= upper))),
            "RMSE_Improvement_%": improvement,
            "Optimized_kernel": str(final_fitted.model.kernel_),
            "Log_marginal_likelihood": float(final_fitted.model.log_marginal_likelihood_value_),
            "Validation": validation_name,
        })

    metrics = pd.DataFrame(metric_rows).sort_values("CV_RMSE").reset_index(drop=True)
    best_name, reason = _choose_best(metrics)
    return AnalysisResult(
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        cv_predictions=cv_predictions,
        fitted_models=fitted_models,
        best_kernel_name=best_name,
        selection_reason=reason,
        feature_names=names,
    )


def extract_ard_length_scales(fitted: FittedResidualGP) -> Optional[np.ndarray]:
    """최적화된 합성 커널 안에서 다차원 RBF 길이 척도를 찾는다."""

    def walk(kernel: Kernel) -> Optional[np.ndarray]:
        if isinstance(kernel, RBF):
            scales = np.asarray(kernel.length_scale, dtype=float).ravel()
            if scales.size == len(fitted.feature_names):
                return scales
        for attribute in ("k1", "k2"):
            child = getattr(kernel, attribute, None)
            if child is not None:
                found = walk(child)
                if found is not None:
                    return found
        return None

    return walk(fitted.model.kernel_)

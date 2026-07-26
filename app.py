from __future__ import annotations

import io
import json
from typing import Dict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import numpy as np
import pandas as pd
import streamlit as st

from formula_utils import FormulaError, discover_parameter_names, parse_formula
from gp_core import KERNEL_DESCRIPTIONS, AnalysisResult, analyze_kernels
from sample_data import make_rotation_sample

# Windows 한글 그래프 글꼴 설정
font_path = r"C:\Windows\Fonts\malgun.ttf"
if Path(font_path).exists():
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rcParams["font.family"] = font_name
rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="GPR 이론식 보정기", page_icon="📈", layout="wide")
st.title("가우시안 프로세스 회귀 기반 실험 이론식 보정기")
st.caption(
    "기존 이론식을 없애는 것이 아니라, 이론값과 실험값의 잔차를 네 가지 GP 커널로 학습해 "
    "검증 성능이 가장 좋은 보정 모델을 선택합니다."
)

with st.expander("모델의 수학적 구조", expanded=False):
    st.latex(r"r_i=y_{\mathrm{exp},i}-f_{\mathrm{theory}}(x_i)")
    st.latex(r"y_{\mathrm{corrected}}(x)=f_{\mathrm{theory}}(x)+\mu_{\mathrm{GP}}(x)")
    st.write(
        "GP는 닫힌 형태의 새 다항식을 만드는 것이 아니라, "
        "입력값마다 계산되는 잔차 예측함수와 불확실성을 제공합니다."
    )

st.sidebar.header("분석 설정")
input_mode = st.sidebar.radio("데이터 입력 방법", ["예제 데이터", "CSV 업로드", "표에 직접 입력"])
precision_mode = st.sidebar.selectbox("초매개변수 탐색", ["빠른 분석", "표준 분석", "정밀 분석"], index=1)
restarts_map = {"빠른 분석": 0, "표준 분석": 1, "정밀 분석": 4}
optimizer_restarts = restarts_map[precision_mode]
x_unit = st.sidebar.text_input("x축 단위", value="kg·m²")
y_unit = st.sidebar.text_input("y축 단위", value="s")
grid_points = st.sidebar.slider("예측 곡선 점 개수", 100, 1000, 400, 50)

st.header("1. 실험 데이터 입력")
sample_df = make_rotation_sample()

if input_mode == "예제 데이터":
    data = sample_df.copy()
    st.info("회전진동 예제입니다. 이론식은 2*pi*sqrt(x/kappa), kappa=0.08입니다.")
    st.dataframe(data, use_container_width=True)
elif input_mode == "CSV 업로드":
    uploaded = st.file_uploader("x, y_exp 열이 포함된 CSV 파일", type=["csv"])
    if uploaded is None:
        st.warning("CSV 파일을 업로드하세요.")
        data = pd.DataFrame(columns=["x", "y_exp"])
    else:
        try:
            data = pd.read_csv(uploaded)
            st.dataframe(data, use_container_width=True)
        except Exception as exc:
            st.error(f"CSV 파일을 읽지 못했습니다: {exc}")
            data = pd.DataFrame(columns=["x", "y_exp"])
else:
    data = st.data_editor(
        sample_df.head(8).copy(),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "x": st.column_config.NumberColumn("x", format="%.8g"),
            "y_exp": st.column_config.NumberColumn("실험값 y_exp", format="%.8g"),
        },
    )

st.header("2. 이론식 입력")
default_formula = "2*pi*sqrt(x/kappa)" if input_mode == "예제 데이터" else "x"
formula_text = st.text_input(
    "독립변수는 x로 입력하세요.",
    value=default_formula,
    help="허용 함수: sin, cos, tan, exp, log, sqrt, abs / 상수: pi, e",
)

parameter_names = discover_parameter_names(formula_text)
parameter_values: Dict[str, float] = {}
if parameter_names:
    st.write("이론식 매개변수")
    defaults = [{"parameter": name, "value": 0.08 if name == "kappa" else 1.0} for name in parameter_names]
    param_df = st.data_editor(
        pd.DataFrame(defaults),
        hide_index=True,
        use_container_width=True,
        disabled=["parameter"],
        column_config={
            "parameter": st.column_config.TextColumn("매개변수"),
            "value": st.column_config.NumberColumn("값", format="%.10g"),
        },
    )
    parameter_values = {str(row["parameter"]): float(row["value"]) for _, row in param_df.iterrows()}
else:
    st.caption("추가 매개변수가 없는 이론식입니다.")

st.header("3. 분석 실행")
run_analysis = st.button("네 가지 커널 비교 시작", type="primary", use_container_width=True)


def clean_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"x", "y_exp"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"필수 열이 없습니다: {', '.join(sorted(missing))}")
    cleaned = frame.loc[:, ["x", "y_exp"]].copy()
    cleaned["x"] = pd.to_numeric(cleaned["x"], errors="coerce")
    cleaned["y_exp"] = pd.to_numeric(cleaned["y_exp"], errors="coerce")
    cleaned = cleaned.dropna(subset=["x", "y_exp"])
    cleaned = cleaned[np.isfinite(cleaned["x"]) & np.isfinite(cleaned["y_exp"])]
    cleaned = cleaned.sort_values("x").reset_index(drop=True)
    if len(cleaned) < 5:
        raise ValueError("유효한 데이터가 최소 5개 필요합니다.")
    if cleaned["x"].nunique() < 5:
        raise ValueError("서로 다른 x 값이 최소 5개 필요합니다.")
    if len(cleaned) > 250:
        raise ValueError("현재 버전은 계산 시간을 위해 최대 250개 데이터까지 지원합니다.")
    if np.isclose(cleaned["x"].std(), 0):
        raise ValueError("x 값에 변화가 없습니다.")
    return cleaned


if run_analysis:
    try:
        cleaned = clean_input_frame(data)
        parsed = parse_formula(formula_text, parameter_names)
        theory_function = parsed.build_function(parameter_values)
        x = cleaned["x"].to_numpy(dtype=float)
        y_exp = cleaned["y_exp"].to_numpy(dtype=float)
        y_theory = theory_function(x)

        if cleaned["x"].nunique() < len(cleaned):
            st.info(
                "같은 x에서 반복 측정값이 발견되었습니다. 검증할 때 같은 x의 측정값 전체를 "
                "한 묶음으로 제외해 데이터 누출을 방지합니다."
            )
        if cleaned["x"].nunique() < 8:
            st.warning("서로 다른 x가 8개 미만이므로 커널 순위가 데이터 변화에 민감할 수 있습니다.")

        with st.spinner("각 커널을 교차검증하고 최종 모델을 학습하는 중입니다..."):
            analysis = analyze_kernels(
                x=x,
                y_exp=y_exp,
                y_theory=y_theory,
                restarts=optimizer_restarts,
                random_state=42,
            )

        st.session_state["analysis_payload"] = {
            "cleaned": cleaned,
            "formula_text": formula_text,
            "parameter_names": parameter_names,
            "parameter_values": parameter_values,
            "analysis": analysis,
            "x_unit": x_unit,
            "y_unit": y_unit,
            "grid_points": grid_points,
        }
        st.success("분석이 완료되었습니다.")
    except (ValueError, FormulaError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.exception(exc)


def figure_to_png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


if "analysis_payload" in st.session_state:
    payload = st.session_state["analysis_payload"]
    cleaned = payload["cleaned"]
    formula_text = payload["formula_text"]
    parameter_names = payload["parameter_names"]
    parameter_values = payload["parameter_values"]
    analysis: AnalysisResult = payload["analysis"]
    x_unit = payload["x_unit"]
    y_unit = payload["y_unit"]
    grid_points = payload["grid_points"]

    parsed = parse_formula(formula_text, parameter_names)
    theory_function = parsed.build_function(parameter_values)
    x = cleaned["x"].to_numpy(dtype=float)
    y_exp = cleaned["y_exp"].to_numpy(dtype=float)
    y_theory = theory_function(x)
    residual = y_exp - y_theory

    x_min, x_max = float(np.min(x)), float(np.max(x))
    padding = max((x_max - x_min) * 0.05, 1e-9)
    grid = np.linspace(x_min - padding, x_max + padding, int(grid_points))
    theory_grid = theory_function(grid)

    st.divider()
    st.header("4. 커널 성능 비교")
    baseline = analysis.baseline_metrics
    best_row = analysis.metrics.set_index("Kernel").loc[analysis.best_kernel_name]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("기존 이론식 RMSE", f"{baseline['RMSE']:.6g}")
    col2.metric("선택된 커널", analysis.best_kernel_name)
    col3.metric("최적 모델 CV RMSE", f"{best_row['CV_RMSE']:.6g}")
    col4.metric("RMSE 개선율", f"{best_row['RMSE_Improvement_%']:.2f}%")
    st.write(analysis.selection_reason)

    if best_row["RMSE_Improvement_%"] <= 0:
        st.warning(
            "선택된 GP 모델도 기존 이론식보다 교차검증 RMSE를 줄이지 못했습니다. "
            "현재 데이터에서는 보정이 유효하다고 결론 내리기 어렵습니다."
        )

    display_metrics = analysis.metrics[[
        "Kernel", "CV_RMSE", "CV_MAE", "CV_R2", "NLPD",
        "Coverage_95", "RMSE_Improvement_%", "Validation",
    ]].copy()
    st.dataframe(
        display_metrics.style.format({
            "CV_RMSE": "{:.6g}",
            "CV_MAE": "{:.6g}",
            "CV_R2": "{:.4f}",
            "NLPD": "{:.4f}",
            "Coverage_95": "{:.1%}",
            "RMSE_Improvement_%": "{:.2f}%",
        }),
        use_container_width=True,
    )

    with st.expander("커널별 가정"):
        for name, description in KERNEL_DESCRIPTIONS.items():
            st.markdown(f"**{name}** — {description}")

    fig_perf, ax_perf = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(analysis.metrics))
    ax_perf.bar(positions, analysis.metrics["CV_RMSE"])
    ax_perf.axhline(baseline["RMSE"], linestyle="--", label="기존 이론식 RMSE")
    ax_perf.set_xticks(positions)
    ax_perf.set_xticklabels(analysis.metrics["Kernel"], rotation=15)
    ax_perf.set_ylabel(f"교차검증 RMSE ({y_unit})" if y_unit else "교차검증 RMSE")
    ax_perf.set_title("커널별 교차검증 성능")
    ax_perf.legend()
    ax_perf.grid(axis="y", alpha=0.25)
    st.pyplot(fig_perf)

    st.header("5. 이론식과 최적 보정 모델")
    best_model = analysis.fitted_models[analysis.best_kernel_name]
    residual_mean, residual_std = best_model.predict(grid)
    corrected_mean = theory_grid + residual_mean
    lower = corrected_mean - 1.96 * residual_std
    upper = corrected_mean + 1.96 * residual_std

    fig_best, ax_best = plt.subplots(figsize=(10, 5.5))
    ax_best.scatter(x, y_exp, label="실험값", zorder=4)
    ax_best.plot(grid, theory_grid, linestyle="--", label="기존 이론식")
    ax_best.plot(grid, corrected_mean, label=f"보정 모델 ({analysis.best_kernel_name})")
    ax_best.fill_between(grid, lower, upper, alpha=0.2, label="95% 예측구간")
    ax_best.set_xlabel(f"x ({x_unit})" if x_unit else "x")
    ax_best.set_ylabel(f"y ({y_unit})" if y_unit else "y")
    ax_best.set_title("기존 이론식과 GP 잔차 보정 결과")
    ax_best.legend()
    ax_best.grid(alpha=0.25)
    st.pyplot(fig_best)

    st.latex(r"y_{\mathrm{corrected}}(x)=f_{\mathrm{theory}}(x)+\mu_{\mathrm{GP}}(x)")
    st.code(f"기존 이론식: {formula_text}")
    st.code(f"최적화된 커널: {best_model.model.kernel_}")
    st.caption(
        "예측구간은 GP 잔차 모델과 측정 잡음의 불확실성입니다. "
        "입력한 이론식 매개변수 자체의 불확실성은 포함하지 않습니다."
    )

    st.header("6. 잔차와 네 커널의 학습 결과")
    fig_res, ax_res = plt.subplots(figsize=(10, 5.5))
    ax_res.axhline(0.0, linestyle="--", linewidth=1)
    ax_res.scatter(x, residual, label="관측 잔차", zorder=4)
    for name, model in analysis.fitted_models.items():
        mean, _ = model.predict(grid)
        ax_res.plot(grid, mean, label=name)
    ax_res.set_xlabel(f"x ({x_unit})" if x_unit else "x")
    ax_res.set_ylabel(f"잔차 ({y_unit})" if y_unit else "잔차")
    ax_res.set_title("커널별 잔차 함수")
    ax_res.legend()
    ax_res.grid(alpha=0.25)
    st.pyplot(fig_res)

    st.header("7. 특정 입력값 예측")
    prediction_x = st.number_input("예측할 x", value=float((x_min + x_max) / 2), format="%.10g")
    pred_theory = float(theory_function(np.array([prediction_x]))[0])
    pred_residual, pred_std = best_model.predict(np.array([prediction_x]))
    pred_corrected = pred_theory + float(pred_residual[0])
    pred_std_value = float(pred_std[0])
    pred_low = pred_corrected - 1.96 * pred_std_value
    pred_high = pred_corrected + 1.96 * pred_std_value

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("기존 이론값", f"{pred_theory:.8g}")
    p2.metric("예상 보정량", f"{float(pred_residual[0]):.8g}")
    p3.metric("최종 보정값", f"{pred_corrected:.8g}")
    p4.metric("예측 표준편차", f"{pred_std_value:.8g}")
    st.write(f"95% 예측구간: **[{pred_low:.8g}, {pred_high:.8g}]**")
    if prediction_x < x_min or prediction_x > x_max:
        st.warning(
            f"입력값이 학습 범위 [{x_min:.6g}, {x_max:.6g}] 밖에 있습니다. "
            "외삽 결과는 신뢰도가 낮을 수 있습니다."
        )

    st.header("8. 결과 내려받기")
    best_predictions = pd.DataFrame({
        "x": grid,
        "y_theory": theory_grid,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "y_corrected": corrected_mean,
        "lower_95": lower,
        "upper_95": upper,
        "is_extrapolation": (grid < x_min) | (grid > x_max),
    })

    summary = {
        "formula": formula_text,
        "parameters": parameter_values,
        "x_range": [x_min, x_max],
        "n_observations": int(len(cleaned)),
        "n_unique_x": int(cleaned["x"].nunique()),
        "validation": str(baseline["Validation"]),
        "baseline_metrics": baseline,
        "best_kernel": analysis.best_kernel_name,
        "optimized_kernel": str(best_model.model.kernel_),
        "selection_reason": analysis.selection_reason,
        "best_metrics": {
            key: (None if pd.isna(value) else float(value))
            for key, value in best_row[[
                "CV_RMSE", "CV_MAE", "CV_R2", "NLPD", "Coverage_95", "RMSE_Improvement_%"
            ]].items()
        },
        "limitations": [
            "하나의 독립변수 x만 지원",
            "학습 범위 밖 외삽은 신뢰도가 낮음",
            "이론식 매개변수 불확실성은 예측구간에 포함하지 않음",
            "선택 결과는 데이터 수와 측정 범위에 의존",
        ],
    }

    d1, d2, d3, d4 = st.columns(4)
    d1.download_button(
        "성능표 CSV",
        analysis.metrics.to_csv(index=False).encode("utf-8-sig"),
        file_name="kernel_metrics.csv",
        mime="text/csv",
        use_container_width=True,
    )
    d2.download_button(
        "최적 모델 예측 CSV",
        best_predictions.to_csv(index=False).encode("utf-8-sig"),
        file_name="best_model_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
    d3.download_button(
        "분석 요약 JSON",
        json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="analysis_summary.json",
        mime="application/json",
        use_container_width=True,
    )
    d4.download_button(
        "최적 모델 그래프 PNG",
        figure_to_png(fig_best),
        file_name="best_model_plot.png",
        mime="image/png",
        use_container_width=True,
    )

    st.divider()
    st.subheader("해석 시 주의사항")
    st.markdown(
        """
- 이 프로그램이 보정하는 것은 **특정 장치와 실험 범위에서 관측된 체계적 잔차**입니다.
- 선택된 커널이 현상의 실제 물리 원인을 증명하는 것은 아닙니다.
- GP 보정 후 교차검증 오차가 기존 이론식보다 줄어들어야 보정의 실효성을 주장할 수 있습니다.
- 데이터가 적거나 한 구간에 몰려 있으면 커널 선택이 불안정할 수 있습니다.
- 다른 장치나 크게 다른 조건에 적용하려면 새 데이터로 다시 학습해야 합니다.
        """
    )

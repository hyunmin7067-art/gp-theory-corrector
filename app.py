from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import numpy as np
import pandas as pd
import streamlit as st

from formula_utils import (
    FormulaError,
    discover_parameter_names,
    parse_formula,
    validate_variable_names,
)
from gp_core import (
    KERNEL_DESCRIPTIONS,
    AnalysisResult,
    analyze_kernels,
    build_candidate_kernels,
    extract_ard_length_scales,
)
from sample_data import make_multivariable_sample, make_rotation_sample


# Windows와 Streamlit Cloud에서 한글 글꼴 설정
font_candidates = [
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    Path(r"C:\Windows\Fonts\malgun.ttf"),
]
for font_path in font_candidates:
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
        break
rcParams["axes.unicode_minus"] = False


st.set_page_config(page_title="다변수 GPR 이론식 보정기", page_icon="📈", layout="wide")
st.title("가우시안 프로세스 회귀 기반 다변수 실험 이론식 보정기")
st.caption(
    "실험의 여러 입력조건을 함께 사용하고, 기존 이론식과 실험값의 잔차를 GP로 학습해 "
    "최적 보정 모델과 예측 불확실성을 구합니다. 입력변수가 2개 이상이면 ARD-RBF도 비교합니다."
)

with st.expander("모델의 수학적 구조", expanded=False):
    st.latex(r"\mathbf{x}_i=(x_{i1},x_{i2},\ldots,x_{id})")
    st.latex(r"r_i=y_{\mathrm{exp},i}-f_{\mathrm{theory}}(\mathbf{x}_i)")
    st.latex(r"y_{\mathrm{corrected}}(\mathbf{x})=f_{\mathrm{theory}}(\mathbf{x})+\mu_{\mathrm{GP}}(\mathbf{x})")
    st.write(
        "ARD-RBF는 입력변수마다 별도의 길이 척도를 학습합니다. "
        "따라서 시간, 전력, 온도처럼 서로 다른 조건이 잔차 변화에 미치는 정도를 구분할 수 있습니다."
    )


st.sidebar.header("분석 설정")
input_mode = st.sidebar.radio(
    "데이터 입력 방법",
    ["예제 데이터", "CSV 업로드", "표에 직접 입력"],
)
precision_mode = st.sidebar.selectbox(
    "초매개변수 탐색",
    ["빠른 분석", "표준 분석", "정밀 분석"],
    index=1,
)
restarts_map = {"빠른 분석": 0, "표준 분석": 1, "정밀 분석": 4}
optimizer_restarts = restarts_map[precision_mode]
y_unit = st.sidebar.text_input("결과값 y 단위", value="")
grid_points = st.sidebar.slider("조건 슬라이스 점 개수", 100, 800, 300, 50)


st.header("1. 실험 데이터 입력")
example_name = None
if input_mode == "예제 데이터":
    example_name = st.selectbox(
        "예제 선택",
        ["회전진동 예제(입력 1개)", "시간·전력 제거율 예제(입력 2개, 합성 데이터)"],
    )
    if example_name.startswith("회전진동"):
        data = make_rotation_sample()
        st.info("이론식 예시: 2*pi*sqrt(x/kappa), kappa=0.08")
    else:
        data = make_multivariable_sample()
        st.info(
            "다변수 동작 확인을 위한 합성 데이터입니다. "
            "이론식 예시: 100*(1-exp(-exp(beta0+beta1*power)*time))"
        )
    st.dataframe(data, use_container_width=True)

elif input_mode == "CSV 업로드":
    uploaded = st.file_uploader(
        "독립변수 열과 실험 결과 열이 포함된 CSV 파일",
        type=["csv"],
    )
    if uploaded is None:
        st.warning("CSV 파일을 업로드하세요. 열 이름은 자유롭게 정할 수 있습니다.")
        data = pd.DataFrame()
    else:
        try:
            data = pd.read_csv(uploaded)
            st.dataframe(data, use_container_width=True)
        except Exception as exc:
            st.error(f"CSV 파일을 읽지 못했습니다: {exc}")
            data = pd.DataFrame()

else:
    direct_feature_count = st.selectbox("독립변수 개수", [1, 2, 3, 4, 5, 6], index=1)

    st.write("열 이름 설정")
    name_columns = st.columns(min(3, direct_feature_count))
    direct_feature_names = []
    for index in range(direct_feature_count):
        with name_columns[index % len(name_columns)]:
            direct_feature_names.append(
                st.text_input(
                    f"독립변수 {index + 1} 이름",
                    value=f"x{index + 1}",
                    key=f"direct_feature_name_{index}_{direct_feature_count}",
                ).strip()
            )

    direct_target_name = st.text_input(
        "실험 결과 열 이름",
        value="y_exp",
        key=f"direct_target_name_{direct_feature_count}",
    ).strip()

    proposed_names = [*direct_feature_names, direct_target_name]
    valid_direct_names = (
        all(proposed_names)
        and len(set(proposed_names)) == len(proposed_names)
    )

    if not valid_direct_names:
        st.error("열 이름은 비워 둘 수 없고, 서로 중복될 수 없습니다.")
        data = pd.DataFrame()
    else:
        direct_template = pd.DataFrame(
            {
                column: np.linspace(0.0, 1.0, 8) * (index + 1)
                for index, column in enumerate(direct_feature_names)
            }
        )
        direct_template[direct_target_name] = np.linspace(0.0, 1.0, 8)
        data = st.data_editor(
            direct_template,
            num_rows="dynamic",
            use_container_width=True,
            key="direct_editor_" + "_".join(proposed_names),
        )


available_columns = [str(column) for column in data.columns]

if not available_columns:
    target_column = None
    candidate_features: List[str] = []
else:
    default_target_index = (
        available_columns.index("y_exp")
        if "y_exp" in available_columns
        else len(available_columns) - 1
    )
    target_column = st.selectbox(
        "실험 결과(종속변수) 열",
        options=available_columns,
        index=default_target_index,
        help="예측하거나 보정하려는 결과값 열을 선택하세요. 열 이름은 y_exp일 필요가 없습니다.",
    )
    candidate_features = [
        column for column in available_columns
        if column != target_column
    ]

numeric_defaults = [
    column
    for column in candidate_features
    if column in data.columns and pd.api.types.is_numeric_dtype(data[column])
]
feature_names = st.multiselect(
    "GP와 이론식에 사용할 독립변수 열",
    options=candidate_features,
    default=numeric_defaults,
    help="식별번호나 메모 열은 선택하지 마세요. 선택한 열 전체가 GP 입력이 됩니다.",
)

feature_name_error = None
try:
    if feature_names:
        feature_names = validate_variable_names(feature_names)
except FormulaError as exc:
    feature_name_error = str(exc)
    st.error(feature_name_error)

unit_values: Dict[str, str] = {}
if feature_names and feature_name_error is None:
    with st.expander("독립변수 단위 설정", expanded=False):
        unit_df = st.data_editor(
            pd.DataFrame({"variable": feature_names, "unit": [""] * len(feature_names)}),
            hide_index=True,
            disabled=["variable"],
            use_container_width=True,
            key="feature_units_" + "_".join(feature_names),
            column_config={
                "variable": st.column_config.TextColumn("독립변수"),
                "unit": st.column_config.TextColumn("단위"),
            },
        )
        unit_values = {
            str(row["variable"]): str(row["unit"]).strip()
            for _, row in unit_df.iterrows()
        }


st.header("2. 이론식 입력")
if example_name and example_name.startswith("회전진동"):
    suggested_formula = "2*pi*sqrt(x/kappa)"
elif example_name and example_name.startswith("시간·전력"):
    suggested_formula = "100*(1-exp(-exp(beta0+beta1*power)*time))"
elif {"time", "power"}.issubset(set(feature_names)):
    suggested_formula = "100*(1-exp(-exp(beta0+beta1*power)*time))"
elif feature_names:
    suggested_formula = feature_names[0]
else:
    suggested_formula = "x"

formula_key = "formula_" + "_".join(feature_names or ["none"])
formula_text = st.text_input(
    "선택한 독립변수의 열 이름을 그대로 사용하세요.",
    value=suggested_formula,
    key=formula_key,
    help="허용 함수: sin, cos, tan, exp, log, sqrt, abs / 상수: pi, e",
)

parameter_values: Dict[str, float] = {}
parameter_names: List[str] = []
if feature_names and feature_name_error is None:
    try:
        parameter_names = discover_parameter_names(formula_text, feature_names)
    except FormulaError as exc:
        st.error(str(exc))

if parameter_names:
    default_parameter_values = {
        "kappa": 0.08,
        "beta0": -6.015,
        "beta1": 0.0085709,
    }
    defaults = [
        {
            "parameter": name,
            "value": default_parameter_values.get(name, 1.0),
        }
        for name in parameter_names
    ]
    st.write("이론식 매개변수")
    parameter_df = st.data_editor(
        pd.DataFrame(defaults),
        hide_index=True,
        disabled=["parameter"],
        use_container_width=True,
        key="parameters_" + "_".join(parameter_names),
        column_config={
            "parameter": st.column_config.TextColumn("매개변수"),
            "value": st.column_config.NumberColumn("값", format="%.10g"),
        },
    )
    parameter_values = {
        str(row["parameter"]): float(row["value"])
        for _, row in parameter_df.iterrows()
    }
else:
    st.caption("추가 매개변수가 없는 이론식입니다.")

if target_column is not None:
    st.caption(f"현재 결과 열: {target_column}")


st.header("3. 분석 실행")
kernel_count = len(build_candidate_kernels(max(1, len(feature_names))))
run_analysis = st.button(
    f"{kernel_count}개 커널 비교 시작",
    type="primary",
    use_container_width=True,
    disabled=(not feature_names) or (target_column is None) or feature_name_error is not None,
)


def clean_input_frame(
    frame: pd.DataFrame,
    selected_features: List[str],
    target_name: str,
) -> pd.DataFrame:
    required = set(selected_features) | {target_name}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"필수 열이 없습니다: {', '.join(sorted(missing))}")

    cleaned = frame.loc[:, [*selected_features, target_name]].copy()
    for column in [*selected_features, target_name]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=[*selected_features, target_name])

    finite_mask = np.all(
        np.isfinite(cleaned[[*selected_features, target_name]].to_numpy(dtype=float)),
        axis=1,
    )
    cleaned = cleaned.loc[finite_mask].reset_index(drop=True)

    if len(cleaned) < 5:
        raise ValueError("유효한 데이터가 최소 5개 필요합니다.")
    unique_conditions = cleaned[selected_features].drop_duplicates()
    if len(unique_conditions) < 5:
        raise ValueError("서로 다른 입력조건 조합이 최소 5개 필요합니다.")
    if len(cleaned) > 250:
        raise ValueError("현재 버전은 계산 시간을 위해 최대 250개 데이터까지 지원합니다.")

    constant_features = [
        name for name in selected_features
        if cleaned[name].nunique() < 2 or np.isclose(cleaned[name].std(), 0.0)
    ]
    if constant_features:
        raise ValueError(
            "값이 변하지 않는 독립변수는 GP 입력에서 제외하세요: "
            + ", ".join(constant_features)
        )

    return cleaned.sort_values(selected_features).reset_index(drop=True)


if run_analysis:
    try:
        cleaned = clean_input_frame(data, feature_names, target_column)
        parsed = parse_formula(
            formula_text,
            parameter_names,
            variable_names=feature_names,
        )
        theory_function = parsed.build_function(parameter_values)
        x_matrix = cleaned[feature_names].to_numpy(dtype=float)
        y_exp = cleaned[target_column].to_numpy(dtype=float)
        y_theory = theory_function(cleaned[feature_names])

        n_unique_conditions = len(cleaned[feature_names].drop_duplicates())
        if n_unique_conditions < len(cleaned):
            st.info(
                "같은 입력조건 조합의 반복 측정값이 발견되었습니다. 검증할 때 같은 조건의 측정값을 "
                "한 묶음으로 제외하여 데이터 누출을 방지합니다."
            )
        if n_unique_conditions < 8:
            st.warning("서로 다른 입력조건 조합이 8개 미만이므로 커널 순위가 민감할 수 있습니다.")
        if len(feature_names) >= 3 and len(cleaned) < 10 * len(feature_names):
            st.warning(
                "입력변수 수에 비해 데이터가 적습니다. 변수 수가 늘어날수록 더 다양한 조건의 데이터가 필요합니다."
            )

        with st.spinner("각 커널을 교차검증하고 최종 모델을 학습하는 중입니다..."):
            analysis = analyze_kernels(
                x=x_matrix,
                y_exp=y_exp,
                y_theory=y_theory,
                feature_names=feature_names,
                restarts=optimizer_restarts,
                random_state=42,
            )

        st.session_state["analysis_payload"] = {
            "cleaned": cleaned,
            "feature_names": feature_names,
            "target_column": target_column,
            "formula_text": formula_text,
            "parameter_names": parameter_names,
            "parameter_values": parameter_values,
            "analysis": analysis,
            "feature_units": unit_values,
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


def axis_label(name: str, units: Dict[str, str]) -> str:
    unit = units.get(name, "")
    return f"{name} ({unit})" if unit else name


def nearest_slice_mask(
    cleaned_frame: pd.DataFrame,
    all_features: List[str],
    varying_feature: str,
    fixed_values: Dict[str, float],
) -> np.ndarray:
    fixed_features = [name for name in all_features if name != varying_feature]
    if not fixed_features:
        return np.ones(len(cleaned_frame), dtype=bool)

    values = cleaned_frame[fixed_features].to_numpy(dtype=float)
    target = np.array([fixed_values[name] for name in fixed_features], dtype=float)
    scale = np.std(values, axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    distance = np.sqrt(np.sum(((values - target) / scale) ** 2, axis=1))
    minimum = float(np.min(distance))
    return np.isclose(distance, minimum, rtol=1e-7, atol=1e-10)


if "analysis_payload" in st.session_state:
    payload = st.session_state["analysis_payload"]
    cleaned = payload["cleaned"]
    feature_names = payload["feature_names"]
    target_column = payload["target_column"]
    formula_text = payload["formula_text"]
    parameter_names = payload["parameter_names"]
    parameter_values = payload["parameter_values"]
    analysis: AnalysisResult = payload["analysis"]
    feature_units = payload["feature_units"]
    y_unit = payload["y_unit"]
    grid_points = payload["grid_points"]

    parsed = parse_formula(
        formula_text,
        parameter_names,
        variable_names=feature_names,
    )
    theory_function = parsed.build_function(parameter_values)
    x_matrix = cleaned[feature_names].to_numpy(dtype=float)
    y_exp = cleaned[target_column].to_numpy(dtype=float)
    y_theory = theory_function(cleaned[feature_names])
    residual = y_exp - y_theory

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
        "Kernel",
        "CV_RMSE",
        "CV_MAE",
        "CV_R2",
        "NLPD",
        "Coverage_95",
        "RMSE_Improvement_%",
        "Validation",
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
        for name in analysis.fitted_models:
            st.markdown(f"**{name}** — {KERNEL_DESCRIPTIONS[name]}")

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

    if "ARD-RBF" in analysis.fitted_models:
        with st.expander("ARD 변수별 길이 척도 해석", expanded=False):
            ard_model = analysis.fitted_models["ARD-RBF"]
            length_scales = extract_ard_length_scales(ard_model)
            if length_scales is not None:
                inverse = 1.0 / np.maximum(length_scales, 1e-12)
                relative = inverse / inverse.sum()
                ard_table = pd.DataFrame({
                    "독립변수": feature_names,
                    "표준화 공간 길이 척도": length_scales,
                    "상대 민감도 지표": relative,
                })
                st.dataframe(
                    ard_table.style.format({
                        "표준화 공간 길이 척도": "{:.5g}",
                        "상대 민감도 지표": "{:.1%}",
                    }),
                    use_container_width=True,
                )
                st.caption(
                    "길이 척도가 작을수록 그 변수 방향에서 잔차가 빠르게 변했다는 뜻입니다. "
                    "이는 인과관계를 증명하는 중요도 지표가 아니라 모델 내부의 상대적 변화 척도입니다."
                )

    st.header("5. 교차검증 예측과 실험값 비교")
    best_cv = analysis.cv_predictions[analysis.best_kernel_name]
    parity_min = float(np.min([best_cv["y_exp"].min(), best_cv["y_theory"].min(), best_cv["y_corrected_cv"].min()]))
    parity_max = float(np.max([best_cv["y_exp"].max(), best_cv["y_theory"].max(), best_cv["y_corrected_cv"].max()]))
    parity_padding = max((parity_max - parity_min) * 0.05, 1e-9)

    fig_parity, ax_parity = plt.subplots(figsize=(7, 6))
    ax_parity.scatter(best_cv["y_exp"], best_cv["y_theory"], label="기존 이론식")
    ax_parity.scatter(best_cv["y_exp"], best_cv["y_corrected_cv"], label="GP 보정값(CV)")
    ax_parity.plot(
        [parity_min - parity_padding, parity_max + parity_padding],
        [parity_min - parity_padding, parity_max + parity_padding],
        linestyle="--",
        label="1:1 기준선",
    )
    ax_parity.set_xlabel(f"실험값 ({y_unit})" if y_unit else "실험값")
    ax_parity.set_ylabel(f"예측값 ({y_unit})" if y_unit else "예측값")
    ax_parity.set_title("교차검증 예측값과 실험값")
    ax_parity.legend()
    ax_parity.grid(alpha=0.25)
    st.pyplot(fig_parity)

    st.header("6. 한 변수에 따른 조건 슬라이스")
    st.caption("여러 입력변수 중 하나만 변화시키고 나머지 변수는 고정하여 보정 곡선을 확인합니다.")
    plot_feature = st.selectbox("그래프에서 변화시킬 독립변수", feature_names)
    fixed_features = [name for name in feature_names if name != plot_feature]
    fixed_values: Dict[str, float] = {}

    if fixed_features:
        fixed_columns = st.columns(min(3, len(fixed_features)))
        for index, name in enumerate(fixed_features):
            unique_values = np.sort(cleaned[name].unique().astype(float))
            median_value = float(np.median(unique_values))
            column = fixed_columns[index % len(fixed_columns)]
            with column:
                if len(unique_values) <= 30:
                    default_index = int(np.argmin(np.abs(unique_values - median_value)))
                    fixed_values[name] = float(st.selectbox(
                        f"{name} 고정값",
                        options=unique_values.tolist(),
                        index=default_index,
                        key=f"fixed_{name}",
                    ))
                else:
                    fixed_values[name] = float(st.number_input(
                        f"{name} 고정값",
                        value=median_value,
                        format="%.10g",
                        key=f"fixed_{name}",
                    ))

    plot_min = float(cleaned[plot_feature].min())
    plot_max = float(cleaned[plot_feature].max())
    plot_grid = np.linspace(plot_min, plot_max, int(grid_points))
    slice_frame = pd.DataFrame(index=np.arange(len(plot_grid)))
    for name in feature_names:
        slice_frame[name] = plot_grid if name == plot_feature else fixed_values[name]

    best_model = analysis.fitted_models[analysis.best_kernel_name]
    theory_slice = theory_function(slice_frame[feature_names])
    residual_mean, residual_std = best_model.predict(slice_frame[feature_names].to_numpy(dtype=float))
    corrected_mean = theory_slice + residual_mean
    lower = corrected_mean - 1.96 * residual_std
    upper = corrected_mean + 1.96 * residual_std

    slice_mask = nearest_slice_mask(cleaned, feature_names, plot_feature, fixed_values)
    observed_slice = cleaned.loc[slice_mask].sort_values(plot_feature)

    fig_slice, ax_slice = plt.subplots(figsize=(10, 5.5))
    ax_slice.plot(plot_grid, theory_slice, linestyle="--", label="기존 이론식")
    ax_slice.plot(plot_grid, corrected_mean, label=f"보정 모델 ({analysis.best_kernel_name})")
    ax_slice.fill_between(plot_grid, lower, upper, alpha=0.2, label="95% 예측구간")
    ax_slice.scatter(
        observed_slice[plot_feature],
        observed_slice[target_column],
        label="선택 조건에 가장 가까운 실험값",
        zorder=4,
    )
    ax_slice.set_xlabel(axis_label(plot_feature, feature_units))
    ax_slice.set_ylabel(f"y ({y_unit})" if y_unit else "y")
    ax_slice.set_title("기존 이론식과 GP 잔차 보정 결과")
    ax_slice.legend()
    ax_slice.grid(alpha=0.25)
    st.pyplot(fig_slice)

    if fixed_features:
        nearest_conditions = observed_slice[fixed_features].drop_duplicates()
        st.caption(
            "그래프에 표시한 실험점의 실제 고정 조건: "
            + "; ".join(
                ", ".join(f"{name}={row[name]:.6g}" for name in fixed_features)
                for _, row in nearest_conditions.head(5).iterrows()
            )
        )

    st.latex(r"y_{\mathrm{corrected}}(\mathbf{x})=f_{\mathrm{theory}}(\mathbf{x})+\mu_{\mathrm{GP}}(\mathbf{x})")
    st.code(f"기존 이론식: {formula_text}")
    st.code(f"최적화된 커널: {best_model.model.kernel_}")
    st.caption(
        "예측구간은 GP 잔차 모델과 측정 잡음의 불확실성입니다. "
        "입력한 이론식 매개변수 자체의 불확실성은 포함하지 않습니다."
    )

    st.subheader("커널별 잔차 함수 슬라이스")
    fig_residual, ax_residual = plt.subplots(figsize=(10, 5.5))
    ax_residual.axhline(0.0, linestyle="--", linewidth=1)
    for name, model in analysis.fitted_models.items():
        mean, _ = model.predict(slice_frame[feature_names].to_numpy(dtype=float))
        ax_residual.plot(plot_grid, mean, label=name)
    observed_residual = (
        observed_slice[target_column].to_numpy(dtype=float)
        - theory_function(observed_slice[feature_names])
    )
    ax_residual.scatter(
        observed_slice[plot_feature],
        observed_residual,
        label="관측 잔차",
        zorder=4,
    )
    ax_residual.set_xlabel(axis_label(plot_feature, feature_units))
    ax_residual.set_ylabel(f"잔차 ({y_unit})" if y_unit else "잔차")
    ax_residual.set_title("커널별 잔차 함수")
    ax_residual.legend()
    ax_residual.grid(alpha=0.25)
    st.pyplot(fig_residual)

    st.header("7. 특정 입력조건 예측")
    prediction_columns = st.columns(min(3, len(feature_names)))
    prediction_values: Dict[str, float] = {}
    for index, name in enumerate(feature_names):
        with prediction_columns[index % len(prediction_columns)]:
            prediction_values[name] = float(st.number_input(
                f"예측할 {name}",
                value=float(cleaned[name].median()),
                format="%.10g",
                key=f"prediction_{name}",
            ))

    prediction_frame = pd.DataFrame([prediction_values], columns=feature_names)
    prediction_matrix = prediction_frame.to_numpy(dtype=float)
    predicted_theory = float(theory_function(prediction_frame)[0])
    predicted_residual, predicted_std = best_model.predict(prediction_matrix)
    predicted_corrected = predicted_theory + float(predicted_residual[0])
    predicted_std_value = float(predicted_std[0])
    predicted_low = predicted_corrected - 1.96 * predicted_std_value
    predicted_high = predicted_corrected + 1.96 * predicted_std_value

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("기존 이론값", f"{predicted_theory:.8g}")
    p2.metric("예상 보정량", f"{float(predicted_residual[0]):.8g}")
    p3.metric("최종 보정값", f"{predicted_corrected:.8g}")
    p4.metric("예측 표준편차", f"{predicted_std_value:.8g}")
    st.write(f"95% 예측구간: **[{predicted_low:.8g}, {predicted_high:.8g}]**")

    extrapolated_features = [
        name for name in feature_names
        if prediction_values[name] < float(cleaned[name].min())
        or prediction_values[name] > float(cleaned[name].max())
    ]
    if extrapolated_features:
        st.warning(
            "다음 입력값이 학습 범위를 벗어났습니다: "
            + ", ".join(extrapolated_features)
            + ". 외삽 결과는 신뢰도가 낮을 수 있습니다."
        )

    st.header("8. 결과 내려받기")
    slice_predictions = slice_frame.copy()
    slice_predictions["y_theory"] = theory_slice
    slice_predictions["residual_mean"] = residual_mean
    slice_predictions["residual_std"] = residual_std
    slice_predictions["y_corrected"] = corrected_mean
    slice_predictions["lower_95"] = lower
    slice_predictions["upper_95"] = upper

    summary = {
        "formula": formula_text,
        "parameters": parameter_values,
        "target_column": target_column,
        "input_features": feature_names,
        "input_ranges": {
            name: [float(cleaned[name].min()), float(cleaned[name].max())]
            for name in feature_names
        },
        "n_observations": int(len(cleaned)),
        "n_unique_conditions": int(len(cleaned[feature_names].drop_duplicates())),
        "validation": str(baseline["Validation"]),
        "baseline_metrics": {
            key: (float(value) if isinstance(value, (int, float, np.number)) else str(value))
            for key, value in baseline.items()
        },
        "best_kernel": analysis.best_kernel_name,
        "optimized_kernel": str(best_model.model.kernel_),
        "selection_reason": analysis.selection_reason,
        "best_metrics": {
            key: (None if pd.isna(value) else float(value))
            for key, value in best_row[[
                "CV_RMSE",
                "CV_MAE",
                "CV_R2",
                "NLPD",
                "Coverage_95",
                "RMSE_Improvement_%",
            ]].items()
        },
        "slice": {
            "varying_feature": plot_feature,
            "fixed_values": fixed_values,
        },
        "limitations": [
            "최대 250개 데이터 지원",
            "입력변수가 많을수록 더 다양한 조건의 데이터가 필요함",
            "학습 범위 밖 외삽은 신뢰도가 낮음",
            "이론식 매개변수 불확실성은 예측구간에 포함하지 않음",
            "한 변수 그래프는 나머지 변수를 고정한 조건 슬라이스임",
            "ARD 길이 척도는 인과적 변수 중요도를 뜻하지 않음",
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
        "조건 슬라이스 예측 CSV",
        slice_predictions.to_csv(index=False).encode("utf-8-sig"),
        file_name="slice_predictions.csv",
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
        "보정 그래프 PNG",
        figure_to_png(fig_slice),
        file_name="corrected_slice_plot.png",
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
- 입력변수가 많아질수록 변수 조합을 충분히 포함한 데이터가 필요합니다.
- ARD 길이 척도는 모델이 감지한 변화 속도이며, 인과적 중요도나 물리 법칙 자체를 뜻하지 않습니다.
- 다른 장치나 크게 다른 조건에 적용하려면 새 데이터로 다시 학습해야 합니다.
        """
    )

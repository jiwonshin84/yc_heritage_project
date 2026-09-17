from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. 페이지 설정
# ============================================================

st.set_page_config(
    page_title="모델 성능 및 최적화",
    page_icon="📈",
    layout="wide",
)

st.title("📈 환경 취약도 분류 모델 성능 및 최적화")
st.caption(
    "학습 페이지에서 생성한 모델 평가 결과를 불러와 "
    "후보 모델의 시간순 검증 성능, 2025년 Final Test, "
    "위험 클래스 탐지 성능, 주요 변수 및 최적화 방향을 분석합니다."
)

st.info(
    "📌 이 페이지는 모델을 새로 학습하는 페이지가 아닙니다.\n\n"
    "먼저 **환경 취약도 분류 모델 학습** 페이지에서 학습을 완료한 뒤 "
    "저장된 결과 파일을 기반으로 성능을 비교·해석합니다."
)


# ============================================================
# 2. 파일 경로
# ============================================================

MODEL_DIR = Path("models")
DATA_DIR = Path("data/processed")

MODEL_META_PATH = MODEL_DIR / "model_metadata.json"
MODEL_SUMMARY_PATH = MODEL_DIR / "model_summary.csv"
FOLD_RESULTS_PATH = MODEL_DIR / "fold_results.csv"
FINAL_TEST_PATH = MODEL_DIR / "final_test_result.csv"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"
TARGET_DATA_PATH = DATA_DIR / "[2019_2025] heritage_target_dataset.csv"


# ============================================================
# 3. 한글 이름
# ============================================================

FEATURE_NAME_KO = {
    "temp_avg": "평균 기온",
    "temp_max": "최고 기온",
    "temp_min": "최저 기온",
    "humidity": "평균 습도",
    "rainfall": "일 강수량",
    "wind_speed": "평균 풍속",
    "sunshine_hours": "일조시간",
    "ground_temp": "지면 온도",
    "pm10": "미세먼지(PM10)",
    "pm25": "초미세먼지(PM2.5)",
    "o3": "오존(O₃)",
    "no2": "이산화질소(NO₂)",
    "co": "일산화탄소(CO)",
    "so2": "아황산가스(SO₂)",
    "temp_range": "일교차",
    "temp_change": "평균기온 변화량",
    "humidity_change": "습도 변화량",
    "humidity_std3": "3일 습도 변동성",
    "rainfall_7d": "최근 7일 누적 강수량",
    "rh60_days_7": "7일 RH>60% 일수",
    "rh60_days_28": "28일 RH>60% 일수",
    "rh75_days_7": "7일 RH≥75% 일수",
    "rh75_days_28": "28일 RH≥75% 일수",
    "rh95_days_7": "7일 RH≥95% 일수",
    "rh95_days_28": "28일 RH≥95% 일수",
    "rh75_consecutive_days": "RH≥75% 연속 지속일수",
    "rh95_consecutive_days": "RH≥95% 연속 지속일수",
    "wood_mold_days_7": "7일 목조 곰팡이 조건 일수",
    "wood_mold_days_28": "28일 목조 곰팡이 조건 일수",
    "rh70_days_7": "7일 RH≥70% 일수",
    "rh70_days_28": "28일 RH≥70% 일수",
    "rh70_consecutive_days": "RH≥70% 연속 지속일수",
    "metal_so2_humidity": "금속 고습·SO₂ 복합노출",
    "pm_total": "PM10+PM2.5",
    "pm_load_3d": "3일 미세먼지 누적부하",
    "pm_load_7d": "7일 미세먼지 누적부하",
    "so2_ma7": "SO₂ 7일 이동평균",
    "no2_ma7": "NO₂ 7일 이동평균",
    "o3_ma7": "O₃ 7일 이동평균",
    "month": "월",
    "temp_over_20": "20℃ 초과 여부",
    "rh_over_60": "RH>60% 여부",
    "rh75": "RH≥75% 여부",
    "rh95": "RH≥95% 여부",
    "wood_mold_condition": "목조 곰팡이 조건 여부",
    "rh70_metal": "금속 RH≥70% 여부",
}


METRIC_NAME_KO = {
    "Mean_Accuracy": "평균 Accuracy",
    "Mean_Macro_F1": "평균 Macro F1",
    "Mean_Weighted_F1": "평균 Weighted F1",
    "Mean_Balanced_Accuracy": "평균 Balanced Accuracy",
    "Mean_Danger_Precision": "평균 위험 Precision",
    "Mean_Danger_Recall": "평균 위험 Recall",
    "Mean_Danger_F1": "평균 위험 F1",
    "Accuracy": "Accuracy",
    "Macro_F1": "Macro F1",
    "Weighted_F1": "Weighted F1",
    "Balanced_Accuracy": "Balanced Accuracy",
    "Danger_Precision": "위험 Precision",
    "Danger_Recall": "위험 Recall",
    "Danger_F1": "위험 F1",
}


# ============================================================
# 4. 유틸리티
# ============================================================

def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            path,
            encoding="utf-8",
        )
    except Exception:
        return None


def safe_float(value, default=np.nan):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return default


def metric_text(value) -> str:
    value = safe_float(value)

    if pd.isna(value):
        return "-"

    return f"{value:.4f}"


def pct_text(value) -> str:
    value = safe_float(value)

    if pd.isna(value):
        return "-"

    return f"{value * 100:.1f}%"


def load_metadata() -> dict:
    if not MODEL_META_PATH.exists():
        return {}

    try:
        return json.loads(
            MODEL_META_PATH.read_text(
                encoding="utf-8",
            )
        )
    except Exception:
        return {}


def get_best_model_name(
    metadata: dict,
    model_summary_df: pd.DataFrame | None,
) -> str:
    name = metadata.get(
        "model_name",
        "",
    )

    if name:
        return str(name)

    if (
        model_summary_df is not None
        and not model_summary_df.empty
        and "Model" in model_summary_df.columns
    ):
        return str(
            model_summary_df.iloc[0]["Model"]
        )

    return "-"


def sort_model_summary(
    model_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    학습 페이지와 동일한 모델 선택 우선순위:
    1) Macro F1
    2) 위험 F1
    3) 위험 Recall
    4) Balanced Accuracy
    """
    sort_cols = [
        "Mean_Macro_F1",
        "Mean_Danger_F1",
        "Mean_Danger_Recall",
        "Mean_Balanced_Accuracy",
    ]

    existing = [
        col
        for col in sort_cols
        if col in model_summary_df.columns
    ]

    if not existing:
        return model_summary_df.copy()

    return (
        model_summary_df
        .sort_values(
            existing,
            ascending=[False] * len(existing),
            na_position="last",
        )
        .reset_index(drop=True)
    )


def make_optimization_advice(
    final_row: pd.Series,
    model_summary_df: pd.DataFrame,
    target_df: pd.DataFrame | None,
) -> list[dict]:
    """
    저장된 실제 성능값을 바탕으로 다음 최적화 우선순위를 제안한다.
    """
    advice = []

    macro_f1 = safe_float(
        final_row.get(
            "Macro_F1",
            np.nan,
        )
    )
    bal_acc = safe_float(
        final_row.get(
            "Balanced_Accuracy",
            np.nan,
        )
    )
    danger_recall = safe_float(
        final_row.get(
            "Danger_Recall",
            np.nan,
        )
    )
    danger_f1 = safe_float(
        final_row.get(
            "Danger_F1",
            np.nan,
        )
    )
    danger_count = safe_float(
        final_row.get(
            "Test_Danger_Count",
            np.nan,
        )
    )

    if (
        not pd.isna(danger_count)
        and danger_count < 30
    ):
        advice.append(
            {
                "우선순위": "1",
                "항목": "위험 클래스 표본 보강",
                "이유": (
                    f"2025 Final Test의 위험 표본이 {int(danger_count)}건으로 적어 "
                    "위험 Recall·F1이 소수 사례에 크게 영향을 받을 수 있습니다."
                ),
                "권장": (
                    "연도 확장, 위험 조건 사례 보강, 재질·노출 조합별 "
                    "위험 사례 확인을 우선합니다."
                ),
            }
        )

    if (
        not pd.isna(danger_recall)
        and danger_recall < 0.5
    ):
        advice.append(
            {
                "우선순위": str(len(advice) + 1),
                "항목": "위험 Recall 개선",
                "이유": (
                    f"현재 위험 Recall은 {danger_recall:.4f}로 "
                    "실제 위험 사례를 놓치는 비율이 높을 수 있습니다."
                ),
                "권장": (
                    "class_weight/sample_weight 강화, 위험 클래스 중심 "
                    "threshold 조정, 위험 사례 분석을 권장합니다."
                ),
            }
        )

    if (
        not pd.isna(macro_f1)
        and not pd.isna(bal_acc)
        and macro_f1 - bal_acc > 0.15
    ):
        advice.append(
            {
                "우선순위": str(len(advice) + 1),
                "항목": "클래스 불균형 점검",
                "이유": (
                    "Macro F1과 Balanced Accuracy 사이 차이가 커 "
                    "일부 클래스 성능 편차가 존재할 가능성이 있습니다."
                ),
                "권장": (
                    "클래스별 confusion matrix와 연도별 Fold 성능을 함께 확인합니다."
                ),
            }
        )

    if (
        not pd.isna(danger_f1)
        and danger_f1 < 0.5
    ):
        advice.append(
            {
                "우선순위": str(len(advice) + 1),
                "항목": "위험 클래스 F1 개선",
                "이유": (
                    f"위험 F1이 {danger_f1:.4f}로 낮아 "
                    "위험 탐지의 Precision과 Recall 균형이 충분하지 않습니다."
                ),
                "권장": (
                    "위험 사례의 오분류 원인을 Feature Importance와 함께 분석하고 "
                    "후보 모델의 깊이·leaf 크기·추정기 수를 조정합니다."
                ),
            }
        )

    if len(advice) == 0:
        advice.append(
            {
                "우선순위": "1",
                "항목": "현재 성능 유지 및 일반화 확인",
                "이유": (
                    "주요 지표에서 뚜렷한 경고 신호가 확인되지 않았습니다."
                ),
                "권장": (
                    "새로운 연도 데이터가 확보되면 동일한 시간순 검증 구조로 "
                    "성능 재확인을 권장합니다."
                ),
            }
        )

    return advice


def recommended_hyperparameters(
    best_model_name: str,
    danger_recall: float,
) -> pd.DataFrame:
    """
    현재 모델 계열에 맞춘 다음 실험 후보.
    실제 최적값이 아니라 '다음 탐색 범위'로 제시한다.
    """
    rows = []

    if best_model_name == "Random Forest":
        rows = [
            {
                "파라미터": "n_estimators",
                "현재/기준": "300",
                "다음 탐색 후보": "300, 500, 700",
                "의도": "추정 안정성 확인",
            },
            {
                "파라미터": "max_depth",
                "현재/기준": "15",
                "다음 탐색 후보": "10, 15, 20, None",
                "의도": "과적합·과소적합 균형",
            },
            {
                "파라미터": "min_samples_leaf",
                "현재/기준": "2",
                "다음 탐색 후보": "1, 2, 4, 6",
                "의도": "희소 위험 사례 분할 민감도 조정",
            },
            {
                "파라미터": "max_features",
                "현재/기준": "기본값",
                "다음 탐색 후보": "sqrt, log2, 0.7",
                "의도": "변수 조합 다양성 조정",
            },
        ]

    elif best_model_name == "Extra Trees":
        rows = [
            {
                "파라미터": "n_estimators",
                "현재/기준": "300",
                "다음 탐색 후보": "300, 500, 700",
                "의도": "랜덤 분할 안정성 확인",
            },
            {
                "파라미터": "max_depth",
                "현재/기준": "15",
                "다음 탐색 후보": "10, 15, 20, None",
                "의도": "복잡도 조정",
            },
            {
                "파라미터": "min_samples_leaf",
                "현재/기준": "2",
                "다음 탐색 후보": "1, 2, 4, 6",
                "의도": "소수 위험 사례 반응성 조정",
            },
            {
                "파라미터": "max_features",
                "현재/기준": "기본값",
                "다음 탐색 후보": "sqrt, log2, 0.7, 1.0",
                "의도": "무작위성·일반화 조정",
            },
        ]

    elif best_model_name == "Gradient Boosting":
        rows = [
            {
                "파라미터": "n_estimators",
                "현재/기준": "150",
                "다음 탐색 후보": "100, 150, 250, 350",
                "의도": "부스팅 단계 수 조정",
            },
            {
                "파라미터": "learning_rate",
                "현재/기준": "0.05",
                "다음 탐색 후보": "0.03, 0.05, 0.08, 0.1",
                "의도": "학습 속도와 일반화 균형",
            },
            {
                "파라미터": "max_depth",
                "현재/기준": "3",
                "다음 탐색 후보": "2, 3, 4, 5",
                "의도": "개별 트리 복잡도 조정",
            },
            {
                "파라미터": "min_samples_leaf",
                "현재/기준": "기본값",
                "다음 탐색 후보": "1, 2, 4, 6",
                "의도": "소수 클래스 민감도 조정",
            },
        ]

    else:
        rows = [
            {
                "파라미터": "-",
                "현재/기준": "-",
                "다음 탐색 후보": "-",
                "의도": "학습 결과에서 최종 모델명을 확인해주세요.",
            }
        ]

    df = pd.DataFrame(rows)

    if (
        not pd.isna(danger_recall)
        and danger_recall < 0.5
    ):
        extra = pd.DataFrame(
            [
                {
                    "파라미터": "class/sample weighting",
                    "현재/기준": "balanced sample weight",
                    "다음 탐색 후보": "위험 클래스 가중 강화 실험",
                    "의도": "위험 Recall 개선",
                }
            ]
        )
        df = pd.concat(
            [df, extra],
            ignore_index=True,
        )

    return df


# ============================================================
# 5. 결과 파일 로드
# ============================================================

metadata = load_metadata()
model_summary_df = safe_read_csv(
    MODEL_SUMMARY_PATH
)
fold_results_df = safe_read_csv(
    FOLD_RESULTS_PATH
)
final_test_df = safe_read_csv(
    FINAL_TEST_PATH
)
importance_df = safe_read_csv(
    FEATURE_IMPORTANCE_PATH
)
target_df = safe_read_csv(
    TARGET_DATA_PATH
)


required_ready = all(
    [
        model_summary_df is not None,
        fold_results_df is not None,
        final_test_df is not None,
    ]
)

if not required_ready:
    st.error(
        "모델 평가 결과 파일을 찾을 수 없습니다.\n\n"
        "먼저 **환경 취약도 분류 모델 학습** 페이지에서 "
        "모델 학습을 완료해주세요."
    )

    st.markdown("#### 필요한 파일")

    needed = [
        MODEL_SUMMARY_PATH,
        FOLD_RESULTS_PATH,
        FINAL_TEST_PATH,
        FEATURE_IMPORTANCE_PATH,
        MODEL_META_PATH,
    ]

    for path in needed:
        st.write(
            "✅" if path.exists() else "❌",
            str(path),
        )

    st.stop()


# ============================================================
# 6. 데이터 정리
# ============================================================

model_summary_df = sort_model_summary(
    model_summary_df
)

final_row = final_test_df.iloc[0]

best_model_name = get_best_model_name(
    metadata,
    model_summary_df,
)

final_accuracy = safe_float(
    final_row.get(
        "Accuracy",
        np.nan,
    )
)
final_macro_f1 = safe_float(
    final_row.get(
        "Macro_F1",
        np.nan,
    )
)
final_bal_acc = safe_float(
    final_row.get(
        "Balanced_Accuracy",
        np.nan,
    )
)
final_danger_recall = safe_float(
    final_row.get(
        "Danger_Recall",
        np.nan,
    )
)
final_danger_f1 = safe_float(
    final_row.get(
        "Danger_F1",
        np.nan,
    )
)
final_danger_precision = safe_float(
    final_row.get(
        "Danger_Precision",
        np.nan,
    )
)
final_danger_count = safe_float(
    final_row.get(
        "Test_Danger_Count",
        np.nan,
    )
)


# ============================================================
# 7. 상단 요약
# ============================================================

st.markdown("---")
st.subheader("🏆 최종 모델 성능 요약")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "최종 선택 모델",
    best_model_name,
)

c2.metric(
    "2025 Accuracy",
    metric_text(final_accuracy),
)

c3.metric(
    "2025 Macro F1",
    metric_text(final_macro_f1),
)

c4.metric(
    "Balanced Accuracy",
    metric_text(final_bal_acc),
)

c5.metric(
    "위험 Recall",
    metric_text(final_danger_recall),
)

training_period = metadata.get("training_period", "-").replace("~", " → ")
final_test_period = metadata.get("final_test_period", "-").replace("~", " → ")
validation = metadata.get("validation", "-").replace("~", "–")

st.markdown(
    f"""
    <div style="
        color: #808495;
        font-size: 0.88rem;
        margin-top: 6px;
        margin-bottom: 4px;
    ">
        학습기간: {training_period}
        &nbsp;·&nbsp;
        최종 테스트: {final_test_period}
        &nbsp;·&nbsp;
        검증: {validation}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 8. 모델 순위
# ============================================================

st.markdown("---")
st.subheader("🥇 후보 모델 성능 순위")

st.caption(
    "모델 선택 우선순위: "
    "① 평균 Macro F1 → ② 평균 위험 F1 → "
    "③ 평균 위험 Recall → ④ 평균 Balanced Accuracy"
)

rank_df = model_summary_df.copy()
rank_df.insert(
    0,
    "순위",
    range(
        1,
        len(rank_df) + 1,
    ),
)

rank_display_cols = [
    "순위",
    "Model",
    "Mean_Accuracy",
    "Mean_Macro_F1",
    "Mean_Balanced_Accuracy",
    "Mean_Danger_Precision",
    "Mean_Danger_Recall",
    "Mean_Danger_F1",
    "Danger_Fold_Count",
    "Total_Danger_Samples",
]

rank_display_cols = [
    col
    for col in rank_display_cols
    if col in rank_df.columns
]

rank_view = rank_df[
    rank_display_cols
].copy()

rank_view = rank_view.rename(
    columns={
        "Model": "모델",
        "Mean_Accuracy": "평균 Accuracy",
        "Mean_Macro_F1": "평균 Macro F1",
        "Mean_Balanced_Accuracy": "평균 Balanced Accuracy",
        "Mean_Danger_Precision": "평균 위험 Precision",
        "Mean_Danger_Recall": "평균 위험 Recall",
        "Mean_Danger_F1": "평균 위험 F1",
        "Danger_Fold_Count": "위험 포함 Fold 수",
        "Total_Danger_Samples": "위험 표본 합계",
    }
)

numeric_cols = rank_view.select_dtypes(
    include="number"
).columns

for col in numeric_cols:
    if col not in [
        "순위",
        "위험 포함 Fold 수",
        "위험 표본 합계",
    ]:
        rank_view[col] = rank_view[col].round(4)

st.dataframe(
    rank_view,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 9. 평균 성능 비교 차트
# ============================================================

st.markdown("---")
st.subheader("📊 후보 모델 평균 성능 비교")

compare_metrics = [
    "Mean_Macro_F1",
    "Mean_Balanced_Accuracy",
    "Mean_Danger_Recall",
    "Mean_Danger_F1",
]

compare_metrics = [
    col
    for col in compare_metrics
    if col in model_summary_df.columns
]

compare_df = model_summary_df[
    ["Model"] + compare_metrics
].copy()

compare_long = compare_df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score",
)

compare_long["Metric_KO"] = (
    compare_long["Metric"]
    .map(METRIC_NAME_KO)
    .fillna(compare_long["Metric"])
)

fig_compare = px.bar(
    compare_long,
    x="Model",
    y="Score",
    color="Metric_KO",
    barmode="group",
    text_auto=".3f",
    title="Expanding-Window 평균 검증 성능",
)

fig_compare.update_layout(
    height=470,
    xaxis_title="",
    yaxis_title="점수",
    yaxis_range=[0, 1],
    legend_title="평가지표",
)

st.plotly_chart(
    fig_compare,
    use_container_width=True,
)


# ============================================================
# 10. Fold별 성능 변화
# ============================================================

st.markdown("---")
st.subheader("📈 연도별 Fold 성능 변화")

fold_metric_options = {
    "Macro F1": "Macro_F1",
    "Balanced Accuracy": "Balanced_Accuracy",
    "위험 Recall": "Danger_Recall",
    "위험 F1": "Danger_F1",
    "Accuracy": "Accuracy",
}

available_fold_metrics = {
    label: col
    for label, col in fold_metric_options.items()
    if col in fold_results_df.columns
}

selected_fold_metric_label = st.selectbox(
    "확인할 Fold 평가지표",
    list(
        available_fold_metrics.keys()
    ),
)

selected_fold_metric = (
    available_fold_metrics[
        selected_fold_metric_label
    ]
)

fold_plot_df = fold_results_df.copy()

year_col = None
for candidate in [
    "Validation_Year",
    "Val_Year",
    "Year",
]:
    if candidate in fold_plot_df.columns:
        year_col = candidate
        break

if year_col is not None:
    fig_fold = px.line(
        fold_plot_df,
        x=year_col,
        y=selected_fold_metric,
        color="Model",
        markers=True,
        title=(
            f"검증연도별 {selected_fold_metric_label} 변화"
        ),
    )

    fig_fold.update_layout(
        height=440,
        xaxis_title="검증연도",
        yaxis_title=selected_fold_metric_label,
        yaxis_range=[0, 1],
    )

    st.plotly_chart(
        fig_fold,
        use_container_width=True,
    )

else:
    st.warning(
        "Fold 결과 파일에서 검증연도 컬럼을 찾지 못했습니다."
    )

with st.expander(
    "📋 Fold별 상세 데이터 보기",
    expanded=False,
):
    st.dataframe(
        fold_results_df.round(4),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 11. 2025 Final Test
# ============================================================

st.markdown("---")
st.subheader("🧪 2025 Final Test 집중 분석")

f1, f2, f3, f4 = st.columns(4)

f1.metric(
    "실제 위험 표본",
    (
        f"{int(final_danger_count)}건"
        if not pd.isna(final_danger_count)
        else "-"
    ),
)

f2.metric(
    "위험 Precision",
    metric_text(
        final_danger_precision
    ),
)

f3.metric(
    "위험 Recall",
    metric_text(
        final_danger_recall
    ),
)

f4.metric(
    "위험 F1",
    metric_text(
        final_danger_f1
    ),
)

metric_rows = []

for col in [
    "Accuracy",
    "Macro_F1",
    "Weighted_F1",
    "Balanced_Accuracy",
    "Danger_Precision",
    "Danger_Recall",
    "Danger_F1",
]:
    if col in final_test_df.columns:
        metric_rows.append(
            {
                "평가지표": METRIC_NAME_KO.get(
                    col,
                    col,
                ),
                "점수": safe_float(
                    final_row.get(
                        col,
                        np.nan,
                    )
                ),
            }
        )

final_metric_plot = pd.DataFrame(
    metric_rows
).dropna(
    subset=["점수"]
)

if not final_metric_plot.empty:
    fig_final = px.bar(
        final_metric_plot,
        x="평가지표",
        y="점수",
        text_auto=".3f",
        title="2025 Final Test 주요 지표",
    )

    fig_final.update_layout(
        height=430,
        xaxis_title="",
        yaxis_title="점수",
        yaxis_range=[0, 1],
    )

    st.plotly_chart(
        fig_final,
        use_container_width=True,
    )

if (
    not pd.isna(final_danger_count)
    and final_danger_count < 30
):
    st.warning(
        f"⚠️ 2025 Final Test의 위험 표본은 "
        f"{int(final_danger_count)}건입니다. "
        "위험 Recall·F1은 소수 사례의 영향을 크게 받을 수 있으므로 "
        "Accuracy만으로 모델 성능을 판단하지 않는 것이 좋습니다."
    )


# ============================================================
# 12. 클래스 분포
# ============================================================

if (
    target_df is not None
    and not target_df.empty
    and "target" in target_df.columns
):

    st.markdown("---")
    st.subheader("🎯 Target 클래스 분포")

    class_counts = (
        target_df["target"]
        .value_counts()
        .reindex(
            ["안전", "주의", "위험"],
            fill_value=0,
        )
        .rename_axis("등급")
        .reset_index(name="건수")
    )

    total_target = int(
        class_counts["건수"].sum()
    )

    class_counts["비율"] = np.where(
        total_target > 0,
        class_counts["건수"]
        / total_target,
        0,
    )

    cc1, cc2 = st.columns(
        [1, 1]
    )

    with cc1:
        fig_class = px.pie(
            class_counts,
            names="등급",
            values="건수",
            color="등급",
            color_discrete_map={
                "안전": "#2ECC71",
                "주의": "#F39C12",
                "위험": "#E74C3C",
            },
            hole=0.4,
            title="전체 Target 분포",
        )

        fig_class.update_traces(
            textinfo="percent+label",
        )

        st.plotly_chart(
            fig_class,
            use_container_width=True,
        )

    with cc2:
        class_view = class_counts.copy()
        class_view["비율"] = (
            class_view["비율"] * 100
        ).round(2).astype(str) + "%"

        st.dataframe(
            class_view,
            use_container_width=True,
            hide_index=True,
        )

        danger_total = int(
            class_counts.loc[
                class_counts["등급"] == "위험",
                "건수",
            ].sum()
        )

        if danger_total > 0:
            danger_ratio = (
                danger_total
                / max(total_target, 1)
                * 100
            )

            st.caption(
                f"전체 데이터 중 위험 등급은 "
                f"{danger_total:,}건 ({danger_ratio:.3f}%)입니다."
            )


# ============================================================
# 13. Feature Importance
# ============================================================

if (
    importance_df is not None
    and not importance_df.empty
    and "Feature" in importance_df.columns
):

    st.markdown("---")
    st.subheader("🔎 주요 변수 중요도")

    value_col = None

    for candidate in [
        "Importance",
        "importance",
        "Permutation_Importance",
    ]:
        if candidate in importance_df.columns:
            value_col = candidate
            break

    if value_col is not None:
        only_environment = st.checkbox(
            "재질·노출환경·계절 더미변수 제외",
            value=True,
        )

        imp = importance_df.copy()

        if only_environment:
            imp = imp[
                ~imp["Feature"]
                .astype(str)
                .str.startswith(
                    (
                        "material_",
                        "exposure_",
                        "season_",
                    )
                )
            ]

        imp["Feature_KO"] = (
            imp["Feature"]
            .map(
                lambda x:
                FEATURE_NAME_KO.get(
                    str(x),
                    str(x),
                )
            )
        )

        top_n = st.slider(
            "표시할 중요 변수 수",
            min_value=5,
            max_value=min(
                30,
                max(
                    len(imp),
                    5,
                ),
            ),
            value=min(
                15,
                max(
                    len(imp),
                    5,
                ),
            ),
        )

        top_imp = (
            imp
            .sort_values(
                value_col,
                ascending=False,
            )
            .head(top_n)
            .sort_values(
                value_col,
                ascending=True,
            )
        )

        fig_imp = px.bar(
            top_imp,
            x=value_col,
            y="Feature_KO",
            orientation="h",
            title=(
                f"{best_model_name} 주요 변수 중요도 TOP {top_n}"
            ),
        )

        fig_imp.update_layout(
            height=max(
                420,
                top_n * 30,
            ),
            xaxis_title="중요도",
            yaxis_title="",
        )

        st.plotly_chart(
            fig_imp,
            use_container_width=True,
        )

        st.caption(
            "※ 변수 중요도는 모델이 분류에 활용한 상대적 기여도를 보여주며, "
            "해당 변수가 실제 문화재 훼손의 직접 원인이라는 뜻은 아닙니다."
        )


# ============================================================
# 14. 자동 진단 및 최적화 우선순위
# ============================================================

st.markdown("---")
st.subheader("🛠️ 모델 최적화 우선순위")

advice_df = pd.DataFrame(
    make_optimization_advice(
        final_row,
        model_summary_df,
        target_df,
    )
)

st.dataframe(
    advice_df,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "※ 위 내용은 저장된 평가 지표를 바탕으로 다음 실험 방향을 정리한 것이며, "
    "자동으로 모델을 변경하거나 재학습하지 않습니다."
)


# ============================================================
# 15. 하이퍼파라미터 다음 탐색 후보
# ============================================================

st.markdown("---")
st.subheader("⚙️ 다음 하이퍼파라미터 탐색 후보")

param_df = recommended_hyperparameters(
    best_model_name,
    final_danger_recall,
)

st.dataframe(
    param_df,
    use_container_width=True,
    hide_index=True,
)

st.info(
    "💡 최적화 실험을 진행할 때도 Random Split보다는 현재 연구와 동일하게 "
    "**Expanding-Window 시간순 검증**을 유지하는 것이 좋습니다. "
    "2025년 자료는 하이퍼파라미터 선택에 사용하지 않고 최종 평가용으로 "
    "남겨두는 구조를 유지하세요."
)


# ============================================================
# 16. 해석 요약
# ============================================================

st.markdown("---")
st.subheader("📝 연구 결과 해석")

summary_parts = []

if not pd.isna(final_accuracy):
    summary_parts.append(
        f"2025년 Final Test Accuracy는 **{final_accuracy:.4f}**입니다."
    )

if not pd.isna(final_macro_f1):
    summary_parts.append(
        f"Macro F1은 **{final_macro_f1:.4f}**로 "
        "클래스별 균형 성능을 함께 확인할 필요가 있습니다."
    )

if not pd.isna(final_bal_acc):
    summary_parts.append(
        f"Balanced Accuracy는 **{final_bal_acc:.4f}**입니다."
    )

if not pd.isna(final_danger_recall):
    summary_parts.append(
        f"위험 클래스 Recall은 **{final_danger_recall:.4f}**입니다."
    )

if (
    not pd.isna(final_danger_count)
    and final_danger_count < 30
):
    summary_parts.append(
        "특히 위험 표본 수가 적기 때문에 위험 클래스 지표는 "
        "소수 사례에 따라 크게 변할 수 있습니다."
    )

summary_parts.append(
    "따라서 본 연구에서는 전체 Accuracy뿐 아니라 "
    "Macro F1, Balanced Accuracy, 위험 Recall·F1을 함께 사용하여 "
    "모델 성능을 해석하는 것이 적절합니다."
)

st.markdown(
    " ".join(summary_parts)
)

st.warning(
    "※ 본 모델의 '위험'은 실제 문화재가 훼손되었다는 의미가 아니라, "
    "연구에서 정의한 환경 취약도 점수가 70점 이상인 상태를 분류한 것입니다."
)


# ============================================================
# 17. 결과 파일 상태
# ============================================================

with st.expander(
    "💾 분석에 사용한 파일",
    expanded=False,
):
    files = [
        MODEL_META_PATH,
        MODEL_SUMMARY_PATH,
        FOLD_RESULTS_PATH,
        FINAL_TEST_PATH,
        FEATURE_IMPORTANCE_PATH,
        TARGET_DATA_PATH,
    ]

    for path in files:
        st.write(
            "✅" if path.exists() else "❌",
            str(path),
        )


st.caption(
    "선화여고 · 영천 헤리티지 AI 탐구단"
)

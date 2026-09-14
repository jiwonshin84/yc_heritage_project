from __future__ import annotations

from pathlib import Path
import json

from github import Github
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# 1. 페이지 설정
# ============================================================

st.set_page_config(
    page_title="위험 예측 분류 모델 학습",
    page_icon="📊",
    layout="wide",
)

st.title("📊 문화유산 위험 예측 분류 모델 학습")
st.caption(
    "2019~2025 영천 환경데이터를 이용하여 재질·노출환경별 "
    "환경 취약도 Target을 생성하고, 시간순 검증으로 최적 모델을 선택합니다."
)

st.info(
    "📌 검증 구조\n\n"
    "• Fold 1: 2019 → 2020 검증\n"
    "• Fold 2: 2019~2020 → 2021 검증\n"
    "• Fold 3: 2019~2021 → 2022 검증\n"
    "• Fold 4: 2019~2022 → 2023 검증\n"
    "• Fold 5: 2019~2023 → 2024 검증\n"
    "• 2025년 자료는 모델 선택에 사용하지 않고 Final Test로만 사용합니다."
)


# ============================================================
# 2. 파일 경로
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 최신 학습 데이터 수집 페이지에서 저장되는 파일
PRIMARY_DATA_PATH = DATA_DIR / "[2019_2025] yeongcheon_features.csv"

# 혹시 파일명을 다르게 저장한 경우를 위한 후보
DATA_CANDIDATES = [
    PRIMARY_DATA_PATH,
    DATA_DIR / "yeongcheon_features_2019_2025.csv",
    DATA_DIR / "[2019_2025] yeongcheon.csv",
]

MODEL_PATH = MODEL_DIR / "best_model.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.pkl"
TRAIN_MEDIANS_PATH = MODEL_DIR / "train_medians.pkl"
BUNDLE_PATH = MODEL_DIR / "heritage_risk_bundle.pkl"
MODEL_META_PATH = MODEL_DIR / "model_metadata.json"
MODEL_SUMMARY_PATH = MODEL_DIR / "model_summary.csv"
FOLD_RESULTS_PATH = MODEL_DIR / "fold_results.csv"
FINAL_TEST_PATH = MODEL_DIR / "final_test_result.csv"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"
TARGET_DATA_PATH = DATA_DIR / "[2019_2025] heritage_target_dataset.csv"


# ============================================================
# 3. 한글 Feature 이름
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


# ============================================================
# 4. 유틸리티
# ============================================================

def find_training_data_path() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "학습 데이터 파일을 찾을 수 없습니다.\n"
        "먼저 '학습 데이터 수집' 페이지를 실행하세요.\n"
        f"우선 경로: {PRIMARY_DATA_PATH}"
    )


def upload_file_to_github(
    local_path: Path,
    git_path: str,
) -> None:
    """
    모델/결과 파일을 GitHub 저장소에 선택적으로 업로드.
    Secrets가 없으면 로컬 저장만 수행한다.
    """
    token = st.secrets.get("GITHUB_TOKEN", "")
    repo_name = st.secrets.get("GITHUB_REPO", "")

    if not token or not repo_name:
        return

    try:
        repo = Github(token).get_repo(repo_name)
        content = local_path.read_bytes()
        message = f"model: update {local_path.name}"

        try:
            old = repo.get_contents(git_path)
            repo.update_file(
                path=git_path,
                message=message,
                content=content,
                sha=old.sha,
                branch="main",
            )
        except Exception:
            repo.create_file(
                path=git_path,
                message=message,
                content=content,
                branch="main",
            )

        st.toast(f"☁️ GitHub 업로드 완료: {git_path}", icon="🚀")

    except Exception as e:
        st.warning(f"GitHub 업로드 실패({git_path}): {e}")


def train_based_minmax_score(
    series: pd.Series,
    train_mask: pd.Series,
) -> pd.Series:
    """
    2025 Final Test 정보를 사용하지 않도록
    2019~2024 구간만 이용해 Min-Max 기준을 계산한다.
    """
    train_values = pd.to_numeric(
        series.loc[train_mask],
        errors="coerce",
    ).dropna()

    if train_values.empty:
        return pd.Series(0.0, index=series.index)

    min_value = train_values.min()
    max_value = train_values.max()

    if (
        pd.isna(min_value)
        or pd.isna(max_value)
        or max_value <= min_value
    ):
        return pd.Series(0.0, index=series.index)

    return (
        (series - min_value)
        / (max_value - min_value)
        * 100
    ).clip(0, 100)


# ============================================================
# 5. Target 데이터 생성
# ============================================================

def create_target_dataset(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    수정중(5)의 최신 산식을 사용한다.

    Target:
      0 <= material_risk < 40  -> 안전
      40 <= material_risk < 70 -> 주의
      70 <= material_risk      -> 위험
    """

    source = train_df.copy()
    source["date"] = pd.to_datetime(
        source["date"],
        errors="coerce",
    )

    source = (
        source
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 재질 × 노출환경 15개 조합
    # --------------------------------------------------------

    materials = ["석조", "목조", "금속", "회화", "기타"]
    exposures = ["실외", "반실외", "실내"]

    combos = (
        pd.MultiIndex
        .from_product(
            [materials, exposures],
            names=["material", "exposure"],
        )
        .to_frame()
        .reset_index(drop=True)
    )

    dataset = (
        source
        .assign(_join_key=1)
        .merge(
            combos.assign(_join_key=1),
            on="_join_key",
        )
        .drop(columns="_join_key")
    )

    # --------------------------------------------------------
    # 반드시 존재해야 하는 공통 파생변수
    # --------------------------------------------------------

    required_features = [
        "temp_change",
        "humidity_change",
        "humidity_std3",
        "rainfall_7d",
        "pm_load_7d",
        "so2_ma7",
        "no2_ma7",
        "o3_ma7",
        "metal_so2_humidity",
        "rh60_days_28",
        "temp_over_20",
        "rh75_days_28",
        "rh95_days_28",
        "rh75_consecutive_days",
        "wood_mold_days_28",
        "rh70_days_28",
        "rh70_consecutive_days",
    ]

    missing = [
        col
        for col in required_features
        if col not in dataset.columns
    ]

    if missing:
        raise ValueError(
            "학습 데이터에 필요한 파생변수가 없습니다: "
            f"{missing}\n"
            "학습 데이터 수집 페이지에서 최신 feature_engineering.py를 "
            "적용해 다시 데이터를 생성하세요."
        )

    # --------------------------------------------------------
    # 2019~2024 기준 상대점수
    # --------------------------------------------------------

    train_mask_for_scaling = (
        dataset["date"]
        < pd.Timestamp("2025-01-01")
    )

    relative_score_map = {
        "temp_change": "temp_change_score",
        "humidity_change": "humidity_change_score",
        "humidity_std3": "humidity_std3_score",
        "rainfall_7d": "rainfall_7d_score",
        "pm_load_7d": "pm_load_7d_score",
        "so2_ma7": "so2_ma7_score",
        "no2_ma7": "no2_ma7_score",
        "o3_ma7": "o3_ma7_score",
        "metal_so2_humidity": "metal_so2_score",
    }

    for source_col, score_col in relative_score_map.items():
        dataset[score_col] = train_based_minmax_score(
            dataset[source_col],
            train_mask_for_scaling,
        )

    # --------------------------------------------------------
    # 문헌 임계조건 기반 0~100 점수
    # --------------------------------------------------------

    dataset["general_rh_score"] = (
        dataset["rh60_days_28"] / 28 * 100
    ).clip(0, 100)

    dataset["general_temp_score"] = (
        dataset["temp_over_20"] * 100
    ).clip(0, 100)

    # 목조
    dataset["wood_rh75_score"] = (
        dataset["rh75_days_28"] / 28 * 100
    ).clip(0, 100)

    dataset["wood_rh95_score"] = (
        dataset["rh95_days_28"] / 28 * 100
    ).clip(0, 100)

    dataset["wood_rh75_duration_score"] = (
        dataset["rh75_consecutive_days"] / 28 * 100
    ).clip(0, 100)

    dataset["wood_mold_score"] = (
        dataset["wood_mold_days_28"] / 28 * 100
    ).clip(0, 100)

    # 금속
    dataset["metal_rh70_score"] = (
        dataset["rh70_days_28"] / 28 * 100
    ).clip(0, 100)

    dataset["metal_rh70_duration_score"] = (
        dataset["rh70_consecutive_days"] / 28 * 100
    ).clip(0, 100)

    # --------------------------------------------------------
    # 노출환경 계수
    # --------------------------------------------------------

    dataset["direct_weather_factor"] = (
        dataset["exposure"]
        .map({
            "실외": 1.00,
            "반실외": 0.60,
            "실내": 0.15,
        })
        .astype(float)
    )

    dataset["air_pollution_factor"] = (
        dataset["exposure"]
        .map({
            "실외": 1.00,
            "반실외": 0.75,
            "실내": 0.40,
        })
        .astype(float)
    )

    dataset["climate_factor"] = (
        dataset["exposure"]
        .map({
            "실외": 1.00,
            "반실외": 0.85,
            "실내": 0.70,
        })
        .astype(float)
    )

    # --------------------------------------------------------
    # 노출환경을 반영한 상대점수
    # --------------------------------------------------------

    dataset["rainfall_exposed_score"] = (
        dataset["rainfall_7d_score"]
        * dataset["direct_weather_factor"]
    )

    dataset["temp_change_exposed_score"] = (
        dataset["temp_change_score"]
        * dataset["climate_factor"]
    )

    dataset["humidity_change_exposed_score"] = (
        dataset["humidity_change_score"]
        * dataset["climate_factor"]
    )

    dataset["pm_exposed_score"] = (
        dataset["pm_load_7d_score"]
        * dataset["air_pollution_factor"]
    )

    dataset["so2_exposed_score"] = (
        dataset["so2_ma7_score"]
        * dataset["air_pollution_factor"]
    )

    dataset["no2_exposed_score"] = (
        dataset["no2_ma7_score"]
        * dataset["air_pollution_factor"]
    )

    dataset["o3_exposed_score"] = (
        dataset["o3_ma7_score"]
        * dataset["air_pollution_factor"]
    )

    dataset["metal_so2_exposed_score"] = (
        dataset["metal_so2_score"]
        * dataset["air_pollution_factor"]
    )

    # --------------------------------------------------------
    # 재질별 환경 취약도
    # --------------------------------------------------------

    wood_risk = (
        dataset["wood_rh75_score"] * 0.30
        + dataset["wood_rh75_duration_score"] * 0.25
        + dataset["wood_mold_score"] * 0.20
        + dataset["wood_rh95_score"] * 0.15
        + dataset["humidity_change_exposed_score"] * 0.10
    )

    metal_risk = (
        dataset["metal_rh70_score"] * 0.35
        + dataset["metal_rh70_duration_score"] * 0.25
        + dataset["metal_so2_exposed_score"] * 0.25
        + dataset["humidity_change_exposed_score"] * 0.15
    )

    painting_risk = (
        dataset["general_rh_score"] * 0.30
        + dataset["humidity_change_exposed_score"] * 0.25
        + dataset["temp_change_exposed_score"] * 0.20
        + dataset["pm_exposed_score"] * 0.15
        + dataset["o3_exposed_score"] * 0.10
    )

    stone_risk = (
        dataset["rainfall_exposed_score"] * 0.30
        + dataset["temp_change_exposed_score"] * 0.20
        + dataset["humidity_change_exposed_score"] * 0.20
        + dataset["pm_exposed_score"] * 0.15
        + dataset["so2_exposed_score"] * 0.10
        + dataset["no2_exposed_score"] * 0.05
    )

    other_risk = (
        dataset["general_rh_score"] * 0.25
        + dataset["general_temp_score"] * 0.15
        + dataset["humidity_change_exposed_score"] * 0.20
        + dataset["temp_change_exposed_score"] * 0.15
        + dataset["rainfall_exposed_score"] * 0.10
        + dataset["pm_exposed_score"] * 0.10
        + dataset["so2_exposed_score"] * 0.05
    )

    material = dataset["material"]

    dataset["base_material_risk"] = np.select(
        [
            material.eq("석조"),
            material.eq("목조"),
            material.eq("금속"),
            material.eq("회화"),
            material.eq("기타"),
        ],
        [
            stone_risk,
            wood_risk,
            metal_risk,
            painting_risk,
            other_risk,
        ],
        default=np.nan,
    )

    dataset["material_risk"] = (
        dataset["base_material_risk"]
        .clip(0, 100)
    )

    dataset["target"] = np.select(
        [
            dataset["material_risk"] >= 70,
            dataset["material_risk"] >= 40,
        ],
        ["위험", "주의"],
        default="안전",
    )

    return dataset


# ============================================================
# 6. 머신러닝 입력 데이터 생성
# ============================================================

def prepare_ml_dataset(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:

    dataset_ml = dataset.copy()
    dataset_ml["date"] = pd.to_datetime(
        dataset_ml["date"],
        errors="coerce",
    )

    # display_season/year는 학습 데이터 수집 페이지의
    # 화면 표시용 컬럼이므로 학습 Feature로 사용하지 않는다.
    display_only_cols = [
        col
        for col in ["display_season", "year"]
        if col in dataset_ml.columns
    ]

    categorical_cols = [
        col
        for col in ["material", "exposure", "season"]
        if col in dataset_ml.columns
    ]

    dataset_ml = pd.get_dummies(
        dataset_ml,
        columns=categorical_cols,
        dtype=int,
    )

    # --------------------------------------------------------
    # Target을 직접 계산하는 데 사용한 점수/계수는
    # Leakage 방지를 위해 학습 Feature에서 제외
    # --------------------------------------------------------

    leakage_cols = [
        "date",
        "target",
        "material_risk",
        "base_material_risk",

        "temp_change_score",
        "humidity_change_score",
        "humidity_std3_score",
        "rainfall_7d_score",
        "pm_load_7d_score",
        "so2_ma7_score",
        "no2_ma7_score",
        "o3_ma7_score",
        "metal_so2_score",

        "general_rh_score",
        "general_temp_score",

        "wood_rh75_score",
        "wood_rh95_score",
        "wood_rh75_duration_score",
        "wood_mold_score",

        "metal_rh70_score",
        "metal_rh70_duration_score",

        "direct_weather_factor",
        "air_pollution_factor",
        "climate_factor",

        "rainfall_exposed_score",
        "temp_change_exposed_score",
        "humidity_change_exposed_score",
        "pm_exposed_score",
        "so2_exposed_score",
        "no2_exposed_score",
        "o3_exposed_score",
        "metal_so2_exposed_score",
    ] + display_only_cols

    feature_cols = [
        col
        for col in dataset_ml.columns
        if col not in leakage_cols
    ]

    # --------------------------------------------------------
    # 문자열/객체형 Feature가 남으면 즉시 중단
    # --------------------------------------------------------

    non_numeric = [
        col
        for col in feature_cols
        if not pd.api.types.is_numeric_dtype(dataset_ml[col])
    ]

    if non_numeric:
        raise ValueError(
            "숫자형이 아닌 Feature가 남아 있습니다: "
            f"{non_numeric}"
        )

    # 결측치 확인
    nan_count = int(
        dataset_ml[feature_cols]
        .isna()
        .sum()
        .sum()
    )

    if nan_count > 0:
        raise ValueError(
            f"머신러닝 입력 Feature에 결측값 {nan_count}개가 남아 있습니다."
        )

    return dataset_ml, feature_cols


# ============================================================
# 7. 후보 모델
# ============================================================

def create_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }


# ============================================================
# 8. Expanding-Window 학습 및 최종 모델 선택
# ============================================================

def run_training(
    dataset_ml: pd.DataFrame,
    feature_cols: list[str],
):
    target_names = ["안전", "주의", "위험"]
    validation_years = [2020, 2021, 2022, 2023, 2024]

    fold_results = []
    fold_confusions = {}

    progress = st.progress(0)
    total_runs = len(validation_years) * len(create_models())
    run_no = 0

    for fold_no, val_year in enumerate(
        validation_years,
        start=1,
    ):
        train_start = pd.Timestamp("2019-01-01")
        val_start = pd.Timestamp(f"{val_year}-01-01")
        val_end = pd.Timestamp(f"{val_year + 1}-01-01")

        train_mask = (
            (dataset_ml["date"] >= train_start)
            & (dataset_ml["date"] < val_start)
        )

        val_mask = (
            (dataset_ml["date"] >= val_start)
            & (dataset_ml["date"] < val_end)
        )

        X_train = dataset_ml.loc[
            train_mask,
            feature_cols,
        ]
        y_train = dataset_ml.loc[
            train_mask,
            "target",
        ]

        X_val = dataset_ml.loc[
            val_mask,
            feature_cols,
        ]
        y_val = dataset_ml.loc[
            val_mask,
            "target",
        ]

        if X_train.empty:
            raise ValueError(
                f"Fold {fold_no}의 Train 데이터가 없습니다."
            )

        if X_val.empty:
            raise ValueError(
                f"Fold {fold_no}의 Validation 데이터가 없습니다."
            )

        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        for model_name, model in create_models().items():
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
            )

            y_pred = model.predict(X_val)

            danger_count = int(
                (y_val == "위험").sum()
            )

            report = classification_report(
                y_val,
                y_pred,
                labels=target_names,
                output_dict=True,
                zero_division=0,
            )

            fold_results.append({
                "Fold": fold_no,
                "Validation_Year": val_year,
                "Model": model_name,
                "Train_Count": len(X_train),
                "Validation_Count": len(X_val),
                "Validation_Danger_Count": danger_count,

                "Accuracy": accuracy_score(
                    y_val,
                    y_pred,
                ),

                "Macro_F1": f1_score(
                    y_val,
                    y_pred,
                    labels=target_names,
                    average="macro",
                    zero_division=0,
                ),

                "Weighted_F1": f1_score(
                    y_val,
                    y_pred,
                    labels=target_names,
                    average="weighted",
                    zero_division=0,
                ),

                "Balanced_Accuracy": balanced_accuracy_score(
                    y_val,
                    y_pred,
                ),

                "Safe_F1": report.get(
                    "안전",
                    {},
                ).get("f1-score", 0.0),

                "Caution_F1": report.get(
                    "주의",
                    {},
                ).get("f1-score", 0.0),

                "Danger_Precision": precision_score(
                    y_val,
                    y_pred,
                    labels=["위험"],
                    average="macro",
                    zero_division=0,
                ),

                "Danger_Recall": recall_score(
                    y_val,
                    y_pred,
                    labels=["위험"],
                    average="macro",
                    zero_division=0,
                ),

                "Danger_F1": f1_score(
                    y_val,
                    y_pred,
                    labels=["위험"],
                    average="macro",
                    zero_division=0,
                ),
            })

            fold_confusions[(model_name, val_year)] = confusion_matrix(
                y_val,
                y_pred,
                labels=target_names,
            )

            run_no += 1
            progress.progress(
                run_no / total_runs
            )

    fold_results_df = pd.DataFrame(
        fold_results
    )

    # --------------------------------------------------------
    # 모델별 기본 평균
    # --------------------------------------------------------

    model_summary_df = (
        fold_results_df
        .groupby("Model")
        .agg(
            Fold_Count=("Fold", "count"),
            Mean_Accuracy=("Accuracy", "mean"),
            Mean_Macro_F1=("Macro_F1", "mean"),
            Std_Macro_F1=("Macro_F1", "std"),
            Mean_Weighted_F1=("Weighted_F1", "mean"),
            Mean_Balanced_Accuracy=("Balanced_Accuracy", "mean"),
            Mean_Safe_F1=("Safe_F1", "mean"),
            Mean_Caution_F1=("Caution_F1", "mean"),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # 위험 클래스가 실제 존재했던 Fold만 평균
    # --------------------------------------------------------

    danger_fold_df = (
        fold_results_df[
            fold_results_df["Validation_Danger_Count"] > 0
        ]
        .copy()
    )

    if not danger_fold_df.empty:
        danger_summary = (
            danger_fold_df
            .groupby("Model")
            .agg(
                Danger_Fold_Count=("Fold", "count"),
                Total_Danger_Samples=(
                    "Validation_Danger_Count",
                    "sum",
                ),
                Mean_Danger_Precision=(
                    "Danger_Precision",
                    "mean",
                ),
                Mean_Danger_Recall=(
                    "Danger_Recall",
                    "mean",
                ),
                Mean_Danger_F1=(
                    "Danger_F1",
                    "mean",
                ),
            )
            .reset_index()
        )

        model_summary_df = model_summary_df.merge(
            danger_summary,
            on="Model",
            how="left",
        )
    else:
        model_summary_df["Danger_Fold_Count"] = 0
        model_summary_df["Total_Danger_Samples"] = 0
        model_summary_df["Mean_Danger_Precision"] = np.nan
        model_summary_df["Mean_Danger_Recall"] = np.nan
        model_summary_df["Mean_Danger_F1"] = np.nan

    # --------------------------------------------------------
    # 최적 모델 선택 우선순위
    # 1) Macro F1
    # 2) 위험 F1
    # 3) 위험 Recall
    # 4) Balanced Accuracy
    # --------------------------------------------------------

    model_summary_df = (
        model_summary_df
        .sort_values(
            by=[
                "Mean_Macro_F1",
                "Mean_Danger_F1",
                "Mean_Danger_Recall",
                "Mean_Balanced_Accuracy",
            ],
            ascending=[False, False, False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    best_model_name = (
        model_summary_df.loc[
            0,
            "Model",
        ]
    )

    # ========================================================
    # 최종 학습: 2019~2024
    # ========================================================

    final_train_mask = (
        (dataset_ml["date"] >= pd.Timestamp("2019-01-01"))
        & (dataset_ml["date"] < pd.Timestamp("2025-01-01"))
    )

    final_test_mask = (
        (dataset_ml["date"] >= pd.Timestamp("2025-01-01"))
        & (dataset_ml["date"] < pd.Timestamp("2026-01-01"))
    )

    X_final_train = dataset_ml.loc[
        final_train_mask,
        feature_cols,
    ]
    y_final_train = dataset_ml.loc[
        final_train_mask,
        "target",
    ]

    X_test = dataset_ml.loc[
        final_test_mask,
        feature_cols,
    ]
    y_test = dataset_ml.loc[
        final_test_mask,
        "target",
    ]

    if X_test.empty:
        raise ValueError(
            "2025 Final Test 데이터가 없습니다."
        )

    # 실시간 예측 시 API 결측값을 학습 데이터 기준으로 보정하기 위한
    # 2019~2024 최종 학습 구간 Feature 중앙값 저장
    train_medians = (
        X_final_train
        .median(numeric_only=True)
        .to_dict()
    )

    final_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_final_train,
    )

    best_model = create_models()[
        best_model_name
    ]

    best_model.fit(
        X_final_train,
        y_final_train,
        sample_weight=final_weights,
    )

    y_test_pred = best_model.predict(
        X_test
    )

    final_report_dict = classification_report(
        y_test,
        y_test_pred,
        labels=target_names,
        output_dict=True,
        zero_division=0,
    )

    final_report_text = classification_report(
        y_test,
        y_test_pred,
        labels=target_names,
        zero_division=0,
    )

    final_cm = confusion_matrix(
        y_test,
        y_test_pred,
        labels=target_names,
    )

    final_test_result = {
        "Model": best_model_name,
        "Train_Count": len(X_final_train),
        "Test_Count": len(X_test),
        "Test_Danger_Count": int(
            (y_test == "위험").sum()
        ),
        "Accuracy": accuracy_score(
            y_test,
            y_test_pred,
        ),
        "Macro_F1": f1_score(
            y_test,
            y_test_pred,
            labels=target_names,
            average="macro",
            zero_division=0,
        ),
        "Weighted_F1": f1_score(
            y_test,
            y_test_pred,
            labels=target_names,
            average="weighted",
            zero_division=0,
        ),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_test,
            y_test_pred,
        ),
        "Safe_F1": final_report_dict.get(
            "안전",
            {},
        ).get("f1-score", 0.0),
        "Caution_F1": final_report_dict.get(
            "주의",
            {},
        ).get("f1-score", 0.0),
        "Danger_Precision": precision_score(
            y_test,
            y_test_pred,
            labels=["위험"],
            average="macro",
            zero_division=0,
        ),
        "Danger_Recall": recall_score(
            y_test,
            y_test_pred,
            labels=["위험"],
            average="macro",
            zero_division=0,
        ),
        "Danger_F1": f1_score(
            y_test,
            y_test_pred,
            labels=["위험"],
            average="macro",
            zero_division=0,
        ),
    }

    # --------------------------------------------------------
    # Feature Importance
    # --------------------------------------------------------

    if hasattr(best_model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": best_model.feature_importances_,
        })
    else:
        # 현재 후보 모델은 모두 native importance를 지원하지만
        # 향후 모델이 추가될 경우를 위한 fallback
        perm = permutation_importance(
            best_model,
            X_test,
            y_test,
            scoring="f1_macro",
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )

        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": perm.importances_mean,
        })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "fold_results_df": fold_results_df,
        "model_summary_df": model_summary_df,
        "final_test_result": final_test_result,
        "final_report_text": final_report_text,
        "final_cm": final_cm,
        "feature_cols": feature_cols,
        "train_medians": train_medians,
        "importance_df": importance_df,
    }


# ============================================================
# 9. 저장
# ============================================================

def save_training_outputs(
    outputs: dict,
    dataset: pd.DataFrame,
) -> None:

    best_model = outputs["best_model"]
    feature_cols = outputs["feature_cols"]
    train_medians = outputs.get("train_medians", {})
    best_model_name = outputs["best_model_name"]
    fold_results_df = outputs["fold_results_df"]
    model_summary_df = outputs["model_summary_df"]
    final_test_result = outputs["final_test_result"]
    importance_df = outputs["importance_df"]

    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    joblib.dump(
        feature_cols,
        FEATURE_COLS_PATH,
    )

    joblib.dump(
        train_medians,
        TRAIN_MEDIANS_PATH,
    )

    dataset.to_csv(
        TARGET_DATA_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    fold_results_df.to_csv(
        FOLD_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    model_summary_df.to_csv(
        MODEL_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([
        final_test_result
    ]).to_csv(
        FINAL_TEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "model_name": best_model_name,
        "feature_count": len(feature_cols),
        "training_period": "2019-01-01~2024-12-31",
        "final_test_period": "2025-01-01~2025-12-31",
        "target_definition": {
            "안전": "0 <= material_risk < 40",
            "주의": "40 <= material_risk < 70",
            "위험": "70 <= material_risk",
        },
        "validation": "Expanding-Window 2020~2024",
    }

    MODEL_META_PATH.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 모델 + Feature 목록 + 학습 중앙값 + 메타데이터를
    # 하나의 Bundle로 저장
    # --------------------------------------------------------
    model_bundle = {
        "model": best_model,
        "features": feature_cols,
        "train_medians": train_medians,
        "metadata": metadata,
    }

    joblib.dump(
        model_bundle,
        BUNDLE_PATH,
    )


# ============================================================
# 10. Session State
# ============================================================

if "risk_training_outputs" not in st.session_state:
    st.session_state.risk_training_outputs = None

if "risk_target_dataset" not in st.session_state:
    st.session_state.risk_target_dataset = None

if "risk_dataset_ml" not in st.session_state:
    st.session_state.risk_dataset_ml = None


# ============================================================
# 11. 실행 영역
# ============================================================

data_path_text = None

try:
    data_path_text = str(
        find_training_data_path()
    )

    st.success(
        f"✅ 학습 데이터 확인: {data_path_text}"
    )

except Exception as e:
    st.warning(str(e))


col_btn1, col_btn2 = st.columns(
    [1, 3],
    vertical_alignment="center",
)

with col_btn1:
    train_clicked = st.button(
        "🧠 분류 모델 학습 시작",
        type="primary",
        use_container_width=True,
    )

with col_btn2:
    if st.session_state.risk_training_outputs is None:
        st.info(
            "학습 버튼을 누르면 Target 생성 → Expanding-Window 검증 → "
            "최종 모델 저장까지 순서대로 실행합니다."
        )
    else:
        st.success(
            "✅ 모델 학습 결과가 준비되어 있습니다."
        )


if train_clicked:

    try:
        data_path = find_training_data_path()

        status = st.status(
            "위험 예측 분류 모델 학습 준비 중...",
            expanded=True,
        )

        status.update(
            label="📂 학습 Feature 데이터 로드 중...",
            state="running",
        )

        train_df = pd.read_csv(
            data_path,
            encoding="utf-8-sig",
        )

        train_df["date"] = pd.to_datetime(
            train_df["date"],
            errors="coerce",
        )

        train_df = (
            train_df
            .dropna(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # 연도 안전성 검사
        # ----------------------------------------------------

        expected_years = {
            2019,
            2020,
            2021,
            2022,
            2023,
            2024,
            2025,
        }

        actual_years = set(
            train_df["date"]
            .dt.year
            .unique()
        )

        missing_years = (
            expected_years
            - actual_years
        )

        if missing_years:
            raise ValueError(
                "학습 데이터에 누락 연도가 있습니다: "
                f"{sorted(missing_years)}"
            )

        status.update(
            label="🎯 재질·노출환경별 Target 생성 중...",
            state="running",
        )

        dataset = create_target_dataset(
            train_df
        )

        status.update(
            label="🧩 머신러닝 Feature 구성 및 Leakage 검사 중...",
            state="running",
        )

        dataset_ml, feature_cols = prepare_ml_dataset(
            dataset
        )

        status.update(
            label="🤖 Expanding-Window 모델 학습 및 검증 중...",
            state="running",
        )

        outputs = run_training(
            dataset_ml,
            feature_cols,
        )

        status.update(
            label="💾 최종 모델 및 결과 파일 저장 중...",
            state="running",
        )

        save_training_outputs(
            outputs,
            dataset,
        )

        st.session_state.risk_training_outputs = outputs
        st.session_state.risk_target_dataset = dataset
        st.session_state.risk_dataset_ml = dataset_ml

        status.update(
            label=(
                f"✅ 학습 완료: "
                f"{outputs['best_model_name']}"
            ),
            state="complete",
            expanded=False,
        )

        # ----------------------------------------------------
        # GitHub 선택적 업로드
        # ----------------------------------------------------

        upload_targets = [
            (MODEL_PATH, "models/best_model.pkl"),
            (FEATURE_COLS_PATH, "models/feature_cols.pkl"),
            (TRAIN_MEDIANS_PATH, "models/train_medians.pkl"),
            (BUNDLE_PATH, "models/heritage_risk_bundle.pkl"),
            (MODEL_META_PATH, "models/model_metadata.json"),
            (MODEL_SUMMARY_PATH, "models/model_summary.csv"),
            (FOLD_RESULTS_PATH, "models/fold_results.csv"),
            (FINAL_TEST_PATH, "models/final_test_result.csv"),
            (
                FEATURE_IMPORTANCE_PATH,
                "models/feature_importance.csv",
            ),
        ]

        for local_path, git_path in upload_targets:
            upload_file_to_github(
                local_path,
                git_path,
            )

        st.rerun()

    except Exception as e:
        st.error(
            f"❌ 위험 예측 분류 모델 학습 실패: {e}"
        )


# ============================================================
# 12. 결과 시각화
# ============================================================

outputs = st.session_state.risk_training_outputs
dataset = st.session_state.risk_target_dataset


if (
    outputs is not None
    and dataset is not None
):

    best_model_name = (
        outputs["best_model_name"]
    )

    fold_results_df = (
        outputs["fold_results_df"]
    )

    model_summary_df = (
        outputs["model_summary_df"]
    )

    final_test_result = (
        outputs["final_test_result"]
    )

    final_report_text = (
        outputs["final_report_text"]
    )

    final_cm = (
        outputs["final_cm"]
    )

    importance_df = (
        outputs["importance_df"]
    )

    feature_cols = (
        outputs["feature_cols"]
    )

    st.markdown("---")

    # ========================================================
    # 12-1. 상단 요약
    # ========================================================

    st.subheader("🏆 최종 학습 결과")

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "최종 모델",
        best_model_name,
    )

    k2.metric(
        "2025 Accuracy",
        f"{final_test_result['Accuracy']:.4f}",
    )

    k3.metric(
        "2025 Macro F1",
        f"{final_test_result['Macro_F1']:.4f}",
    )

    k4.metric(
        "Balanced Accuracy",
        f"{final_test_result['Balanced_Accuracy']:.4f}",
    )

    k5.metric(
        "위험 Recall",
        f"{final_test_result['Danger_Recall']:.4f}",
    )

    # ========================================================
    # 12-2. 필터
    # ========================================================

    st.markdown("---")
    st.markdown("##### 🔍 Target 데이터 분석 조건")

    fcol1, fcol2, _ = st.columns([1, 1, 2])

    with fcol1:
        selected_material = st.selectbox(
            "🏛️ 재질",
            ["전체"]
            + list(
                dataset["material"]
                .dropna()
                .unique()
            ),
        )

    with fcol2:
        selected_exposure = st.selectbox(
            "🌿 노출환경",
            ["전체"]
            + list(
                dataset["exposure"]
                .dropna()
                .unique()
            ),
        )

    filtered_df = dataset.copy()

    if selected_material != "전체":
        filtered_df = filtered_df[
            filtered_df["material"]
            == selected_material
        ]

    if selected_exposure != "전체":
        filtered_df = filtered_df[
            filtered_df["exposure"]
            == selected_exposure
        ]

    # ========================================================
    # 12-3. Target 분포
    # ========================================================

    st.markdown("---")

    total_count = len(filtered_df)
    danger_count = int(
        (filtered_df["target"] == "위험").sum()
    )
    caution_count = int(
        (filtered_df["target"] == "주의").sum()
    )
    safe_count = int(
        (filtered_df["target"] == "안전").sum()
    )

    t1, t2, t3, t4 = st.columns(4)

    t1.metric(
        "총 데이터",
        f"{total_count:,}건",
    )

    t2.metric(
        "🔴 위험",
        f"{danger_count:,}건",
        (
            f"{danger_count / total_count * 100:.2f}%"
            if total_count
            else "0%"
        ),
    )

    t3.metric(
        "🟡 주의",
        f"{caution_count:,}건",
        (
            f"{caution_count / total_count * 100:.2f}%"
            if total_count
            else "0%"
        ),
    )

    t4.metric(
        "🟢 안전",
        f"{safe_count:,}건",
        (
            f"{safe_count / total_count * 100:.2f}%"
            if total_count
            else "0%"
        ),
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🎯 위험 등급 분포")

        target_counts = (
            filtered_df["target"]
            .value_counts()
            .reindex(
                ["안전", "주의", "위험"],
                fill_value=0,
            )
            .rename_axis("Target")
            .reset_index(name="Count")
        )

        fig_target = px.pie(
            target_counts,
            names="Target",
            values="Count",
            color="Target",
            color_discrete_map={
                "안전": "#2ECC71",
                "주의": "#F39C12",
                "위험": "#E74C3C",
            },
            hole=0.42,
            title="선택 조건의 환경 취약도 등급",
        )

        fig_target.update_traces(
            textinfo="percent+label",
        )

        st.plotly_chart(
            fig_target,
            use_container_width=True,
        )

    with col_right:
        st.subheader(
            f"📌 주요 Feature 중요도 TOP 10 ({best_model_name})"
        )

        # material/exposure/season 더미를 제외한 환경변수 중심
        env_importance = importance_df[
            ~importance_df["Feature"].str.startswith(
                ("material_", "exposure_", "season_")
            )
        ].copy()

        top10 = env_importance.head(10).copy()

        top10["Feature_KO"] = (
            top10["Feature"]
            .map(
                lambda x:
                FEATURE_NAME_KO.get(
                    x,
                    x,
                )
            )
        )

        top10 = top10.sort_values(
            "Importance",
            ascending=True,
        )

        fig_imp = px.bar(
            top10,
            x="Importance",
            y="Feature_KO",
            orientation="h",
            title="환경 취약도 분류에 기여한 주요 변수",
        )

        fig_imp.update_layout(
            xaxis_title="Feature Importance",
            yaxis_title="",
            height=420,
        )

        st.plotly_chart(
            fig_imp,
            use_container_width=True,
        )

    # ========================================================
    # 12-4. 모델 비교
    # ========================================================

    st.markdown("---")
    st.subheader("📈 후보 모델 평균 성능 비교")

    compare_df = model_summary_df[
        [
            "Model",
            "Mean_Macro_F1",
            "Mean_Balanced_Accuracy",
            "Mean_Danger_Recall",
            "Mean_Danger_F1",
        ]
    ].copy()

    compare_long = compare_df.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score",
    )

    fig_models = px.bar(
        compare_long,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Expanding-Window 평균 성능",
    )

    fig_models.update_layout(
        yaxis_range=[0, 1],
        yaxis_title="점수",
        xaxis_title="",
        height=460,
    )

    st.plotly_chart(
        fig_models,
        use_container_width=True,
    )

    st.dataframe(
        model_summary_df.round(4),
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # 12-5. Fold 결과
    # ========================================================

    with st.expander(
        "📋 Fold별 상세 성능",
        expanded=False,
    ):
        st.dataframe(
            fold_results_df.round(4),
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # 12-6. 2025 Final Test
    # ========================================================

    st.markdown("---")
    st.subheader("🧪 2025 Final Test 상세")

    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        "실제 위험 표본",
        f"{final_test_result['Test_Danger_Count']}건",
    )
    q2.metric(
        "위험 Precision",
        f"{final_test_result['Danger_Precision']:.4f}",
    )
    q3.metric(
        "위험 Recall",
        f"{final_test_result['Danger_Recall']:.4f}",
    )
    q4.metric(
        "위험 F1",
        f"{final_test_result['Danger_F1']:.4f}",
    )

    cm_df = pd.DataFrame(
        final_cm,
        index=[
            "실제 안전",
            "실제 주의",
            "실제 위험",
        ],
        columns=[
            "예측 안전",
            "예측 주의",
            "예측 위험",
        ],
    )

    cma, cmb = st.columns([1, 1])

    with cma:
        st.markdown("##### Confusion Matrix")
        st.dataframe(
            cm_df,
            use_container_width=True,
        )

    with cmb:
        st.markdown("##### Classification Report")
        st.code(
            final_report_text,
            language="text",
        )

    if final_test_result["Test_Danger_Count"] < 30:
        st.warning(
            "⚠️ 2025 Final Test의 위험 표본 수가 적습니다. "
            "위험 Recall·F1은 소수 사례의 영향을 크게 받으므로 "
            "전체 Accuracy만으로 모델을 평가하지 않습니다."
        )

    # ========================================================
    # 12-7. 재질별 위험 산식
    # ========================================================

    st.markdown("---")
    st.subheader("🏛️ 재질별 환경 취약도 가중치 구조")

    weight_rows = [
        {
            "재질": "목조",
            "주요 조건": (
                "RH75 빈도 30% + RH75 연속지속 25% + "
                "20~30℃ 고습조건 20% + RH95 빈도 15% + "
                "습도 변화 10%"
            ),
        },
        {
            "재질": "금속",
            "주요 조건": (
                "RH70 빈도 35% + RH70 연속지속 25% + "
                "고습·SO₂ 복합노출 25% + 습도 변화 15%"
            ),
        },
        {
            "재질": "회화",
            "주요 조건": (
                "RH60 지속 30% + 습도 변화 25% + "
                "기온 변화 20% + PM 15% + O₃ 10%"
            ),
        },
        {
            "재질": "석조",
            "주요 조건": (
                "7일 강수 30% + 기온 변화 20% + "
                "습도 변화 20% + PM 15% + SO₂ 10% + NO₂ 5%"
            ),
        },
        {
            "재질": "기타",
            "주요 조건": (
                "RH60 지속 25% + 20℃ 초과 15% + 습도 변화 20% + "
                "기온 변화 15% + 강수 10% + PM 10% + SO₂ 5%"
            ),
        },
    ]

    st.dataframe(
        pd.DataFrame(weight_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "※ 가중치는 실제 손상 확률이 아니라 프로젝트에서 정의한 "
        "상대적 환경 취약도 점수를 구성하기 위한 값입니다."
    )

    # ========================================================
    # 12-8. Target 데이터 미리보기
    # ========================================================

    st.markdown("---")
    st.subheader("📋 선택 조건 Target 데이터")

    display_cols = [
        "date",
        "material",
        "exposure",
        "material_risk",
        "target",
        "temp_avg",
        "humidity",
        "rainfall_7d",
        "rh75_days_28",
        "rh70_days_28",
        "pm10",
        "pm25",
        "so2",
    ]

    display_cols = [
        col
        for col in display_cols
        if col in filtered_df.columns
    ]

    table_df = (
        filtered_df[display_cols]
        .sort_values(
            "date",
            ascending=False,
        )
        .head(500)
        .copy()
    )

    rename_map = {
        "date": "날짜",
        "material": "재질",
        "exposure": "노출환경",
        "material_risk": "환경 취약도 점수",
        "target": "등급",
        "temp_avg": "평균기온",
        "humidity": "습도",
        "rainfall_7d": "7일 누적강수",
        "rh75_days_28": "28일 RH75 이상 일수",
        "rh70_days_28": "28일 RH70 이상 일수",
        "pm10": "PM10",
        "pm25": "PM2.5",
        "so2": "SO₂",
    }

    table_df = table_df.rename(
        columns=rename_map
    )

    st.dataframe(
        table_df,
        use_container_width=True,
        height=420,
        hide_index=True,
    )

    # ========================================================
    # 12-9. 파일 저장 상태
    # ========================================================

    st.markdown("---")

    with st.expander(
        "💾 생성된 모델/결과 파일",
        expanded=False,
    ):
        saved_files = [
            MODEL_PATH,
            FEATURE_COLS_PATH,
            MODEL_META_PATH,
            MODEL_SUMMARY_PATH,
            FOLD_RESULTS_PATH,
            FINAL_TEST_PATH,
            FEATURE_IMPORTANCE_PATH,
            TARGET_DATA_PATH,
        ]

        for path in saved_files:
            st.write(
                "✅" if path.exists() else "❌",
                str(path),
            )

    st.info(
        "※ '위험'은 실제 문화재가 훼손되었다는 의미가 아닙니다. "
        "본 연구의 환경 취약도 점수가 70점 이상인 상태를 뜻합니다."
    )

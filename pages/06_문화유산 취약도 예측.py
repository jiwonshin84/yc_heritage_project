            ],
        )

        a2.metric(
            "노출환경",
            row[
                "exposure"
            ],
        )

        a3.metric(
            "주의·위험 예측확률",
            f"{row['attention_probability']:.1f}%",
        )

        a4.metric(
            "위험 예측확률",
            f"{row['danger_probability']:.1f}%",
        )

        if (
            "address" in row.index
            and pd.notna(
                row[
                    "address"
                ]
            )
        ):
            st.caption(
                f"📍 소재지: {row['address']}"
            )


# ============================================================
# 17. 예측 기준일 환경 Feature 확인
# ============================================================

st.markdown("---")

with st.expander(
    "🧪 이번 예측에 사용된 기준일 환경 Feature",
    expanded=False,
):

    realtime_for_view = st.session_state.get(
        "recent_40_environment"
    )

    if realtime_for_view is None:
        st.info(
            "현재 세션에 최근 40일 환경 Feature가 없습니다. "
            "상단의 자동 수집·예측 버튼을 다시 실행해주세요."
        )

    else:
        target_environment = (
            select_target_environment(
                realtime_for_view,
                prediction_date,
            )
        )

        selected_feature_view = (
            target_environment
            .T
            .reset_index()
        )

        selected_feature_view.columns = [
            "Feature",
            "값",
        ]

        st.dataframe(
            selected_feature_view,
            use_container_width=True,
            hide_index=True,
            height=520,
        )


# ============================================================
# 18. CSV 다운로드
# ============================================================

st.markdown("---")

download_df = (
    result_df
    .rename(
        columns=rename_map
    )
    .copy()
)

csv_bytes = (
    download_df
    .to_csv(
        index=False
    )
    .encode(
        "utf-8-sig"
    )
)

st.download_button(
    "📥 문화유산별 예측 결과 CSV 다운로드",
    data=csv_bytes,
    file_name=(
        f"영천_문화유산_환경취약도_예측_"
        f"{pd.Timestamp(prediction_date):%Y%m%d}.csv"
    ),
    mime="text/csv",
    type="primary",
    use_container_width=True,
)


st.caption(
    "☁️ 예측 실행 시 동일 결과가 "
    "`data/processed/latest_prediction.csv` 파일로 GitHub에 자동 저장되어 "
    "실시간 대시보드의 안전·주의·위험 현황에 사용됩니다."
)


# ============================================================
# 19. 해석 안내
# ============================================================

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
            TRAIN_MEDIANS_PATH,
            BUNDLE_PATH,
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

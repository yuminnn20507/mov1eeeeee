import math

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# 1. 기본 설정
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "it-teacher-go/data-share-/main/kobis_movies.csv"
)
SEED = 42
TYPE_NAMES = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]
FEATURES = [
    "스크린 수 (로그)",
    "누적 관객 (로그)",
    "10위권 일수",
    "롱런 지수",
]


# 2. 데이터 불러오기 및 전처리
@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_URL, encoding="utf-8")

    required = [
        "movieCd",
        "movieNm",
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]
    missing = [col for col in required if col not in raw.columns]

    if missing:
        raise ValueError(
            f"CSV에 필요한 열이 없습니다: {', '.join(missing)}"
        )

    original_count = len(raw)
    df = raw.copy()

    numeric = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df[numeric].notna().all(axis=1)
    valid &= df["first_scrn"] > 0
    valid &= df["first_week_audi"] > 0
    valid &= df["total_audi"] > 0
    valid &= df["days_in_top10"] >= 0

    df = df.loc[valid].copy()

    df["스크린 수"] = df["first_scrn"]
    df["누적 관객"] = df["total_audi"]
    df["10위권 일수"] = df["days_in_top10"]

    df["롱런 지수"] = (
        df["total_audi"] / df["first_week_audi"]
    ).clip(upper=20)

    df["스크린 수 (로그)"] = df["스크린 수"].map(math.log10)
    df["누적 관객 (로그)"] = df["누적 관객"].map(math.log10)
    df["영화 제목"] = df["movieNm"].fillna("(제목 없음)").astype(str)

    return df.reset_index(drop=True), original_count


# 3. K-평균 분석 함수
def fit_kmeans(data, features, k):
    scaler = StandardScaler()
    values = data[features].astype(float)
    scaled_values = scaler.fit_transform(values)

    model = KMeans(
        n_clusters=k,
        random_state=SEED,
        n_init=10,
    )
    labels = model.fit_predict(scaled_values)

    return scaled_values, labels, model


# 4. 앱 제목 및 데이터 로딩
st.title("🎬 영화 유형 나누기")
st.write(
    "영화의 스크린 규모, 관객 수, 10위권 유지 기간을 "
    "바탕으로 비슷한 영화끼리 묶어 봅니다."
)

try:
    df, original_count = load_data()
except Exception as exc:
    st.error("데이터를 불러오거나 처리하는 중 오류가 발생했습니다.")
    st.exception(exc)
    st.stop()

st.caption(
    f"원본 데이터 전체 {original_count:,}편 중 "
    f"분석에 필요한 값이 모두 있는 영화 {len(df):,}편을 분석합니다."
)


# 5. 분석 속성 선택
st.subheader("1. 유형 분류에 사용할 속성")

selected_features = st.multiselect(
    "분류에 사용할 속성을 두 개 이상 선택하세요.",
    options=FEATURES,
    default=FEATURES,
    help=(
        "스크린 수와 누적 관객은 상용로그로 변환합니다. "
        "롱런 지수는 누적 관객 ÷ 첫 주 관객이며 최대 20입니다."
    ),
)

if len(selected_features) < 2:
    st.warning("분석을 위해 속성을 두 개 이상 선택해 주세요.")
    st.stop()

if len(df) < 7:
    st.error("묶음 수 2~7을 비교하려면 분석 가능한 영화가 최소 7편 필요합니다.")
    st.stop()


# 6. 묶음 수 선택
st.subheader("2. 묶음 수 정하기")

cluster_count = st.slider(
    "몇 개의 유형으로 묶을까요? (K)",
    min_value=2,
    max_value=7,
    value=3,
    step=1,
)

st.caption(
    f"현재 선택: {cluster_count}개 유형 "
    f"({', '.join(TYPE_NAMES[:cluster_count])})"
)


# 7. 선택한 K로 군집 분석
X_scaled, labels, model = fit_kmeans(
    df, selected_features, cluster_count
)

result = df.copy()
result["군집 번호"] = labels

# 누적 관객 평균이 높은 군집부터 유형 기호 지정
cluster_order = (
    result.groupby("군집 번호")["누적 관객"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

label_map = {
    cluster_id: TYPE_NAMES[i]
    for i, cluster_id in enumerate(cluster_order)
}

result["영화 유형"] = result["군집 번호"].map(label_map)
result["영화 유형"] = pd.Categorical(
    result["영화 유형"],
    categories=TYPE_NAMES[:cluster_count],
    ordered=True,
)


# 8. 유형별 요약 표
st.subheader("3. 유형별 특징")

summary = (
    result.groupby("영화 유형", observed=False)
    .agg(
        **{
            "영화 수": ("영화 제목", "count"),
            "평균 스크린 수": ("스크린 수", "mean"),
            "평균 누적 관객": ("누적 관객", "mean"),
            "평균 10위권 일수": ("10위권 일수", "mean"),
            "평균 롱런 지수": ("롱런 지수", "mean"),
        }
    )
    .reset_index()
)

summary["평균 스크린 수"] = summary["평균 스크린 수"].round(1)
summary["평균 누적 관객"] = summary["평균 누적 관객"].round(0)
summary["평균 10위권 일수"] = summary["평균 10위권 일수"].round(1)
summary["평균 롱런 지수"] = summary["평균 롱런 지수"].round(2)

st.dataframe(summary, use_container_width=True, hide_index=True)


# 9. 산점도
st.subheader("4. 영화 유형 산점도")

x_feature = selected_features[0]
y_feature = selected_features[1]

fig_scatter = px.scatter(
    result,
    x=x_feature,
    y=y_feature,
    color="영화 유형",
    hover_name="영화 제목",
    hover_data={
        "영화 유형": True,
        "스크린 수": ":,.0f",
        "누적 관객": ":,.0f",
        "10위권 일수": True,
        "롱런 지수": ":.2f",
        "군집 번호": False,
    },
    category_orders={
        "영화 유형": TYPE_NAMES[:cluster_count]
    },
    title=f"{x_feature}과 {y_feature}에 따른 영화 유형",
)

fig_scatter.update_traces(
    marker={"size": 10, "opacity": 0.75}
)
fig_scatter.update_layout(
    height=600,
    legend_title_text="영화 유형",
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.caption(
    "점 하나는 영화 한 편입니다. 가까이 있는 점일수록 "
    "선택한 두 속성의 값이 비슷합니다. 로그 속성은 변환된 값으로 표시됩니다."
)


# 10. 유형별 영화 제목 목록
st.subheader("5. 유형별 영화 제목")

for type_name in TYPE_NAMES[:cluster_count]:
    movies = result.loc[
        result["영화 유형"] == type_name
    ].sort_values("누적 관객", ascending=False)

    with st.expander(f"{type_name} 유형 — {len(movies)}편"):
        if movies.empty:
            st.write("이 유형에 속한 영화가 없습니다.")
        else:
            for _, row in movies.iterrows():
                st.write(
                    f"- {row['영화 제목']} "
                    f"(누적 관객 {row['누적 관객']:,.0f}명)"
                )


# 11. K=1~7 관성 계산
st.subheader("6. 묶음 수에 따른 관성 변화")

st.write(
    "관성은 각 영화와 자신이 속한 묶음 중심 사이 거리의 "
    "제곱을 모두 더한 값입니다. 값이 작을수록 각 영화가 "
    "해당 중심에 가깝습니다."
)

inertia_rows = []

for k in range(1, 8):
    _, _, comparison_model = fit_kmeans(
        df, selected_features, k
    )
    inertia_rows.append(
        {
            "묶음 수": k,
            "관성": float(comparison_model.inertia_),
        }
    )

inertia_df = pd.DataFrame(inertia_rows)

inertia_df["직전 값 대비 감소량"] = (
    inertia_df["관성"].shift(1) - inertia_df["관성"]
)

# K=1은 비교할 이전 값이 없음
inertia_df.loc[
    inertia_df["묶음 수"] == 1,
    "직전 값 대비 감소량",
] = float("nan")


# 12. 관성 꺾은선 그래프
fig_elbow = go.Figure()

fig_elbow.add_trace(
    go.Scatter(
        x=inertia_df["묶음 수"],
        y=inertia_df["관성"],
        mode="lines+markers+text",
        text=[
            f"{value:,.1f}"
            for value in inertia_df["관성"]
        ],
        textposition="top center",
        name="관성",
        line={"width": 3},
        marker={"size": 9},
    )
)

fig_elbow.add_vline(
    x=cluster_count,
    line_dash="dash",
    line_width=2,
    annotation_text=f"현재 K = {cluster_count}",
    annotation_position="top",
)

fig_elbow.update_layout(
    title="묶음 수(K)에 따른 관성 변화",
    xaxis_title="묶음 수 (K)",
    yaxis_title="관성 (거리 제곱의 합)",
    xaxis={
        "tickmode": "linear",
        "dtick": 1,
        "range": [0.8, 7.2],
    },
    height=500,
    showlegend=False,
)

st.plotly_chart(fig_elbow, use_container_width=True)


# 13. 관성값과 감소량 표
st.markdown("#### 묶음 수별 관성값과 감소량")

display_inertia = inertia_df.copy()

display_inertia["관성"] = display_inertia["관성"].map(
    lambda value: f"{value:,.3f}"
)

display_inertia["직전 값 대비 감소량"] = (
    display_inertia["직전 값 대비 감소량"].map(
        lambda value: "" if pd.isna(value) else f"{value:,.3f}"
    )
)

st.dataframe(
    display_inertia,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "감소량은 바로 앞 묶음 수의 관성에서 현재 묶음 수의 관성을 뺀 값입니다. "
    "관성은 K가 커질수록 감소하는 경향이 있습니다. 감소 폭이 작아지는 "
    "지점은 적절한 K를 판단하는 참고 자료가 될 수 있습니다."
)


# 14. 실루엣 점수
st.subheader("7. 현재 묶음의 실루엣 점수")

try:
    score = silhouette_score(X_scaled, labels)

    st.metric(
        label=f"K = {cluster_count}일 때 실루엣 점수",
        value=f"{score:.4f}",
    )

    if score >= 0.7:
        interpretation = "묶음이 비교적 뚜렷하게 구분됩니다."
    elif score >= 0.5:
        interpretation = "묶음이 어느 정도 잘 구분됩니다."
    elif score >= 0.25:
        interpretation = "묶음의 구분이 약한 편입니다."
    elif score >= 0:
        interpretation = "묶음 사이의 구분이 뚜렷하지 않습니다."
    else:
        interpretation = "일부 영화가 다른 묶음에 더 가까울 수 있습니다."

    st.write(
        "실루엣 점수는 -1부터 1 사이이며, 1에 가까울수록 "
        f"묶음이 잘 구분된다는 뜻입니다. {interpretation}"
    )

except Exception as exc:
    st.warning(
        "실루엣 점수를 계산하지 못했습니다. "
        "각 묶음에 충분한 영화가 있는지 확인해 주세요."
    )
    st.caption(str(exc))


# 15. 분석 안내
st.divider()

st.info(
    "K-평균은 선택한 속성을 표준화한 뒤 거리를 기준으로 영화를 묶습니다. "
    "속성을 바꾸면 군집 결과, 산점도, 유형별 요약, 영화 목록, 관성 및 "
    "실루엣 점수가 함께 바뀝니다. 관성만으로 가장 좋은 묶음 수가 "
    "자동으로 결정되는 것은 아닙니다."
)

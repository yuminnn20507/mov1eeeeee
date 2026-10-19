
import math
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
SEED = 42

TYPE_NAMES = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]
FEATURES = [
    "스크린 수 (로그)",
    "누적 관객 (로그)",
    "10위권 일수",
    "롱런 지수"
]


# --------------------------------------------------
# 1. 데이터 불러오기 및 정리
# --------------------------------------------------
@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_URL, encoding="utf-8")

    required = [
        "movieCd",
        "movieNm",
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10"
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
        "days_in_top10"
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


# --------------------------------------------------
# 2. K-평균 군집 분류 함수
# --------------------------------------------------
def fit_kmeans(X_scaled, k):
    model = KMeans(
        n_clusters=k,
        random_state=SEED,
        n_init=10
    )
    labels = model.fit_predict(X_scaled)
    return model, labels


def make_label_map(df, labels):
    temp = df[["누적 관객"]].copy()
    temp["군집 번호"] = labels

    # 누적 관객 평균이 많은 군집부터 ㉮, ㉯, ㉰ 순서로 표시
    cluster_order = (
        temp.groupby("군집 번호")["누적 관객"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    return {
        cluster_id: TYPE_NAMES[i]
        for i, cluster_id in enumerate(cluster_order)
    }


# --------------------------------------------------
# 3. 화면 제목 및 데이터 불러오기
# --------------------------------------------------
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
    f"분석에 필요한 값이 모두 있는 영화 {len(df):,}편을 분류합니다."
)


# --------------------------------------------------
# 4. 사용할 속성 선택
# --------------------------------------------------
st.subheader("1. 유형 분류에 사용할 속성")

selected_features = st.multiselect(
    "분류에 사용할 속성을 두 개 이상 선택해 주세요.",
    options=FEATURES,
    default=FEATURES,
    help=(
        "스크린 수와 누적 관객은 상용로그로 변환합니다. "
        "롱런 지수는 누적 관객 ÷ 첫 주 관객이며 최대 20입니다."
    )
)

if len(selected_features) < 2:
    st.warning("분류에 사용할 속성을 두 개 이상 선택해 주세요.")
    st.stop()

if len(df) < 7:
    st.error("묶음 수 1~7을 비교하려면 분석할 영화가 최소 7편 필요합니다.")
    st.stop()

X = df[selected_features].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# 5. 묶음 수 선택
# --------------------------------------------------
st.subheader("2. 묶음 수 정하기")

cluster_count = st.slider(
    "몇 개의 유형으로 나눌까요?",
    min_value=2,
    max_value=7,
    value=3,
    step=1
)

st.caption(
    f"현재 {cluster_count}개 유형으로 분류합니다. "
    "묶음 수나 위의 속성을 바꾸면 결과가 자동으로 다시 계산됩니다."
)


# --------------------------------------------------
# 6. 현재 선택한 묶음 수로 군집 생성
# --------------------------------------------------
selected_model, selected_labels = fit_kmeans(
    X_scaled, cluster_count
)

label_map = make_label_map(df, selected_labels)

result = df.copy()
result["군집 번호"] = selected_labels
result["영화 유형"] = result["군집 번호"].map(label_map)

type_order = TYPE_NAMES[:cluster_count]
result["영화 유형"] = pd.Categorical(
    result["영화 유형"],
    categories=type_order,
    ordered=True
)


# --------------------------------------------------
# 7. 현재 묶음 수 결과 요약
# --------------------------------------------------
st.subheader("3. 분류 결과 요약")

summary = (
    result.groupby("영화 유형", observed=False)
    .agg(
        영화 수=("영화 제목", "size"),
        평균 누적 관객=("누적 관객", "mean"),
        평균 스크린 수=("스크린 수", "mean"),
        평균 10위권 일수=("10위권 일수", "mean"),
        평균 롱런 지수=("롱런 지수", "mean")
    )
    .reset_index()
)

summary["평균 누적 관객"] = summary["평균 누적 관객"].round(0)
summary["평균 스크린 수"] = summary["평균 스크린 수"].round(1)
summary["평균 10위권 일수"] = summary["평균 10위권 일수"].round(1)
summary["평균 롱런 지수"] = summary["평균 롱런 지수"].round(2)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 8. 산점도
# --------------------------------------------------
st.subheader("4. 영화 유형 산점도")

col_x, col_y = st.columns(2)

with col_x:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0
    )

with col_y:
    y_default = 1 if len(selected_features) > 1 else 0
    y_feature = st.selectbox(
        "세로축 속성",
        options=selected_features,
        index=y_default
    )

if x_feature == y_feature:
    st.info(
        "가로축과 세로축이 같습니다. "
        "서로 다른 속성을 선택하면 관계를 더 쉽게 살펴볼 수 있습니다."
    )

fig_scatter = px.scatter(
    result,
    x=x_feature,
    y=y_feature,
    color="영화 유형",
    hover_name="영화 제목",
    hover_data={
        "스크린 수": ":,.0f",
        "누적 관객": ":,.0f",
        "10위권 일수": ":.0f",
        "롱런 지수": ":.2f",
        "군집 번호": False
    },
    category_orders={"영화 유형": type_order},
    title=f"{cluster_count}개 영화 유형의 분포",
    labels={
        x_feature: x_feature,
        y_feature: y_feature,
        "영화 유형": "영화 유형"
    }
)

fig_scatter.update_layout(
    height=560,
    legend_title_text="영화 유형"
)

st.plotly_chart(fig_scatter, use_container_width=True)


# --------------------------------------------------
# 9. 유형별 영화 제목 목록
# --------------------------------------------------
st.subheader("5. 유형별 영화 제목")

for type_name in type_order:
    group = result[result["영화 유형"] == type_name].copy()

    if group.empty:
        continue

    group = group.sort_values("누적 관객", ascending=False)

    with st.expander(
        f"{type_name} 유형 — {len(group)}편",
        expanded=True
    ):
        st.write(
            "누적 관객이 많은 영화부터 표시합니다."
        )

        title_table = group[
            [
                "영화 제목",
                "스크린 수",
                "누적 관객",
                "10위권 일수",
                "롱런 지수"
            ]
        ].copy()

        title_table["스크린 수"] = title_table["스크린 수"].map(
            lambda x: f"{x:,.0f}"
        )
        title_table["누적 관객"] = title_table["누적 관객"].map(
            lambda x: f"{x:,.0f}"
        )
        title_table["10위권 일수"] = title_table["10위권 일수"].map(
            lambda x: f"{x:,.0f}"
        )
        title_table["롱런 지수"] = title_table["롱런 지수"].map(
            lambda x: f"{x:.2f}"
        )

        st.dataframe(
            title_table,
            use_container_width=True,
            hide_index=True
        )


# --------------------------------------------------
# 10. 묶음 수 1~7의 제곱 거리 합 계산
# --------------------------------------------------
st.subheader("6. 묶음 수에 따른 제곱 거리 합")

st.write(
    "각 영화와 해당 영화가 속한 군집 중심 사이의 거리를 제곱한 뒤 "
    "모두 더한 값입니다. 같은 속성으로 묶음 수 1~7을 비교합니다."
)

inertia_rows = []

for k in range(1, 8):
    model_k, labels_k = fit_kmeans(X_scaled, k)

    inertia_rows.append({
        "묶음 수": k,
        "제곱 거리 합": model_k.inertia_
    })

inertia_df = pd.DataFrame(inertia_rows)

fig_elbow = go.Figure()

fig_elbow.add_trace(
    go.Scatter(
        x=inertia_df["묶음 수"],
        y=inertia_df["제곱 거리 합"],
        mode="lines+markers",
        name="제곱 거리 합",
        line=dict(width=3),
        marker=dict(size=9)
    )
)

# 현재 선택한 묶음 수 위치에 세로선 표시
fig_elbow.add_vline(
    x=cluster_count,
    line_dash="dash",
    line_width=2,
    annotation_text=f"현재 선택: {cluster_count}",
    annotation_position="top"
)

fig_elbow.update_layout(
    title="묶음 수와 제곱 거리 합",
    xaxis_title="묶음 수 (K)",
    yaxis_title="제곱 거리 합",
    xaxis=dict(
        tickmode="linear",
        dtick=1
    ),
    height=480
)

st.plotly_chart(fig_elbow, use_container_width=True)


# --------------------------------------------------
# 11. 제곱 거리 합 및 감소량 표
# --------------------------------------------------
inertia_table = inertia_df.copy()

inertia_table["제곱 거리 합"] = inertia_table["제곱 거리 합"].round(3)

inertia_table["이전 값 대비 감소량"] = (
    inertia_df["제곱 거리 합"].shift(1)
    - inertia_df["제곱 거리 합"]
)

# 첫 번째 묶음 수는 비교할 이전 값이 없으므로 빈칸
inertia_table.loc[0, "이전 값 대비 감소량"] = float("nan")
inertia_table["이전 값 대비 감소량"] = (
    inertia_table["이전 값 대비 감소량"].round(3)
)

st.dataframe(
    inertia_table,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "감소량은 바로 앞 묶음 수의 제곱 거리 합에서 "
    "현재 묶음 수의 제곱 거리 합을 뺀 값입니다."
)


# --------------------------------------------------
# 12. 선택한 묶음 수의 실루엣 점수
# --------------------------------------------------
st.subheader("7. 현재 묶음의 실루엣 점수")

silhouette = silhouette_score(
    X_scaled,
    selected_labels
)

st.metric(
    label=f"{cluster_count}개 유형의 실루엣 점수",
    value=f"{silhouette:.3f}"
)

st.write(
    "실루엣 점수는 -1에서 1 사이입니다. "
    "1에 가까울수록 같은 유형의 영화끼리는 가깝고 "
    "다른 유형의 영화와는 잘 구분된다는 뜻입니다."
)

st.caption(
    "참고: 제곱 거리 합은 묶음 수가 많아질수록 대체로 감소합니다. "
    "실루엣 점수와 꺾은선 그래프를 함께 살펴보고 "
    "적절한 묶음 수를 판단해 보세요."
)

import math
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
SEED = 42
TYPE_NAMES = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]
FEATURES = ["스크린 수 (로그)", "누적 관객 (로그)", "10위권 일수", "롱런 지수"]


@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_URL, encoding="utf-8")
    required = [
        "movieCd", "movieNm", "first_scrn", "first_week_audi",
        "total_audi", "days_in_top10"
    ]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"CSV에 필요한 열이 없습니다: {', '.join(missing)}")

    original_count = len(raw)
    df = raw.copy()
    numeric = ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df[numeric].notna().all(axis=1)
    valid &= df["first_scrn"] > 0
    valid &= df["total_audi"] > 0
    valid &= df["first_week_audi"] > 0
    valid &= df["days_in_top10"] >= 0
    df = df.loc[valid].copy()

    df["스크린 수"] = df["first_scrn"]
    df["누적 관객"] = df["total_audi"]
    df["10위권 일수"] = df["days_in_top10"]
    df["롱런 지수"] = (df["total_audi"] / df["first_week_audi"]).clip(upper=20)
    df["스크린 수 (로그)"] = df["스크린 수"].map(math.log10)
    df["누적 관객 (로그)"] = df["누적 관객"].map(math.log10)
    df["영화 제목"] = df["movieNm"].fillna("(제목 없음)").astype(str)
    return df.reset_index(drop=True), original_count


def fit_kmeans(data, features, k):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[features].astype(float))
    model = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = model.fit_predict(scaled)
    return scaled, model, labels


st.title("🎬 영화 유형 나누기")
st.write("영화의 스크린 규모, 관객 수, 10위권 유지 기간을 바탕으로 비슷한 영화끼리 묶어 봅니다.")

try:
    df, original_count = load_data()
except Exception as exc:
    st.error("데이터를 불러오거나 처리하는 중 오류가 발생했습니다.")
    st.exception(exc)
    st.stop()

st.caption(
    f"원본 데이터 전체 {original_count:,}편 중, 분석에 필요한 값이 모두 있는 영화 "
    f"{len(df):,}편을 분류합니다."
)

st.subheader("1. 유형 분류에 사용할 속성")
selected_features = st.multiselect(
    "분류에 사용할 속성을 두 개 이상 선택해 주세요. 기본값은 네 가지 속성 전체입니다.",
    options=FEATURES,
    default=FEATURES,
    help="스크린 수와 누적 관객은 상용로그로 변환합니다. 롱런 지수는 누적 관객 ÷ 첫 주 관객이며 최대 20입니다."
)
if len(selected_features) < 2:
    st.warning("K-평균 분류를 위해 속성을 두 개 이상 선택해 주세요.")
    st.stop()

if len(df) < 7:
    st.error("묶음 수 2~7을 비교하려면 분석에 사용할 영화가 최소 7편 필요합니다.")
    st.stop()

st.subheader("2. 묶음 수 정하기")
cluster_count = st.slider("묶음 수 (K)", min_value=2, max_value=7, value=3, step=1)

# 선택한 속성으로 표준화하고 K=2~7의 군집 내 제곱합(inertia) 계산
scaled, selected_model, selected_labels = fit_kmeans(df, selected_features, cluster_count)
elbow_rows = []
for k in range(1, 8):
    if k == 1:

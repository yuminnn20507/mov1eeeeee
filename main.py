import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 페이지 설정
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
RANDOM_STATE = 42

FEATURES = {
    "스크린 수 (로그)": "스크린 수 (로그)",
    "누적 관객 (로그)": "누적 관객 (로그)",
    "10위권 일수": "10위권 일수",
    "롱런 지수": "롱런 지수",
}
RAW_FEATURES = ["스크린 수", "누적 관객", "10위권 일수", "롱런 지수"]
FEATURE_COLUMNS = list(FEATURES.keys())


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    required = [
        "movieCd", "movieNm", "first_scrn", "first_week_audi",
        "total_audi", "days_in_top10"
    ]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError("필요한 열이 없습니다: " + ", ".join(missing_columns))

    for col in ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 네 가지 속성에 필요한 값이 없거나 첫 주 관객이 0인 영화 제외
    valid = df[
        ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]
    ].notna().all(axis=1)
    valid &= df["first_scrn"] > 0
    valid &= df["total_audi"] > 0
    valid &= df["first_week_audi"] > 0
    valid &= df["days_in_top10"] >= 0
    df = df.loc[valid].copy()

    df["스크린 수"] = df["first_scrn"]
    df["누적 관객"] = df["total_audi"]
    df["10위권 일수"] = df["days_in_top10"]
    df["롱런 지수"] = (df["total_audi"] / df["first_week_audi"]).clip(upper=20)

    # 값의 크기 차이를 줄이기 위해 상용로그 사용
    df["스크린 수 (로그)"] = df["스크린 수"].apply(lambda x: __import__("math").log10(x))
    df["누적 관객 (로그)"] = df["누적 관객"].apply(lambda x: __import__("math").log10(x))
    df["영화 제목"] = df["movieNm"].fillna("(제목 없음)").astype(str)
    return df.reset_index(drop=True)


st.title("🎬 영화 유형 나누기")
st.markdown("영화의 스크린 규모, 관객 수, 10위권 유지 기간을 바탕으로 비슷한 영화끼리 묶어 봅니다.")

try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결이나 CSV 주소를 확인해 주세요.")
    st.exception(e)
    st.stop()

total_count = len(pd.read_csv(DATA_URL, encoding="utf-8"))
st.caption(f"원본 데이터 전체 {total_count:,}편 중, 분석에 필요한 값이 모두 있는 영화 {len(df):,}편을 분류합니다.")

st.subheader("1. 유형 분류에 사용할 속성")
selected_features = st.multiselect(
    "두 개 이상 선택해 주세요. 기본값은 네 가지 속성 전체입니다.",
    options=FEATURE_COLUMNS,
    default=FEATURE_COLUMNS,
    help="스크린 수와 누적 관객은 상용로그로 변환한 값을 사용합니다. 롱런 지수는 누적 관객을 첫 주 관객으로 나눈 값이며 최대 20입니다."
)

if len(selected_features) < 2:
    st.warning("K-평균 분류를 위해 속성을 두 개 이상 선택해 주세요.")
    st.stop()

# 선택 속성 표준화 후 K-평균 분류
X = df[selected_features].astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
raw_labels = kmeans.fit_predict(X_scaled)
df["원래 묶음 번호"] = raw_labels

# 누적 관객의 원래 단위 평균이 큰 묶음부터 ㉮, ㉯, ㉰ 부여

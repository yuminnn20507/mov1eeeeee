cluster_order = (
    df.groupby("원래 묶음 번호")["누적 관객"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
label_map = {cluster_id: ["㉮", "㉯", "㉰"][rank] for rank, cluster_id in enumerate(cluster_order)}
df["영화 유형"] = df["원래 묶음 번호"].map(label_map)
df["영화 유형"] = pd.Categorical(df["영화 유형"], categories=["㉮", "㉯", "㉰"], ordered=True)

st.subheader("2. 분류 결과")
c1, c2, c3, c4 = st.columns(4)
c1.metric("원본 전체 편수", f"{total_count:,}편")
c2.metric("분석에 사용한 편수", f"{len(df):,}편")
c3.metric("제외한 편수", f"{total_count - len(df):,}편")
c4.metric("분류 유형 수", "3개")

st.caption("분류는 선택한 속성의 표준화 값으로 수행합니다. 유형 번호는 매번 같은 난수 시드를 사용하며, ㉮는 누적 관객 평균이 가장 높은 묶음입니다.")

st.subheader("3. 2차원 산점도")
plot_x = st.selectbox("가로축 속성", options=selected_features, index=0, key="plot_x")
plot_y_default = 1 if len(selected_features) > 1 else 0
plot_y = st.selectbox("세로축 속성", options=selected_features, index=plot_y_default, key="plot_y")

fig2 = px.scatter(
    df,
    x=plot_x,
    y=plot_y,
    color="영화 유형",
    hover_name="영화 제목",
    hover_data={"영화 유형": True, "movieCd": True, plot_x: ":.3f", plot_y: ":.3f"},
    category_orders={"영화 유형": ["㉮", "㉯", "㉰"]},
    title="선택한 속성으로 본 영화 유형"
)
fig2.update_traces(marker={"size": 8, "opacity": 0.78})
fig2.update_layout(legend_title_text="영화 유형", height=560)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("4. 3차원 산점도")
if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 위에서 속성을 세 개 이상 선택해 주세요.")
else:
    col1, col2, col3 = st.columns(3)
    default_z = 2 if len(selected_features) > 2 else 0
    with col1:
        plot_3d_x = st.selectbox("X축 속성", selected_features, index=0, key="plot_3d_x")
    with col2:
        plot_3d_y = st.selectbox("Y축 속성", selected_features, index=1, key="plot_3d_y")
    with col3:
        plot_3d_z = st.selectbox("Z축 속성", selected_features, index=default_z, key="plot_3d_z")

    fig3 = px.scatter_3d(
        df,
        x=plot_3d_x,
        y=plot_3d_y,
        z=plot_3d_z,
        color="영화 유형",
        hover_name="영화 제목",
        hover_data={"영화 유형": True, "movieCd": True},
        category_orders={"영화 유형": ["㉮", "㉯", "㉰"]},
        title="마우스로 돌려 보는 3차원 영화 유형"
    )
    fig3.update_traces(marker={"size": 3, "opacity": 0.8})
    fig3.update_layout(legend_title_text="영화 유형", height=680)
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("5. 유형별 편수와 속성 평균")
summary = df.groupby("영화 유형", observed=False).agg(
    편수=("movieNm", "size"),
    스크린_수_평균=("스크린 수", "mean"),
    누적_관객_평균=("누적 관객", "mean"),
    십위권_일수_평균=("10위권 일수", "mean"),
    롱런_지수_평균=("롱런 지수", "mean"),
).reindex(["㉮", "㉯", "㉰"]).reset_index()

summary = summary.rename(columns={
    "영화 유형": "영화 유형",
    "스크린_수_평균": "스크린 수 평균",
    "누적_관객_평균": "누적 관객 평균",
    "십위권_일수_평균": "10위권 일수 평균",
    "롱런_지수_평균": "롱런 지수 평균",
})
summary["편수"] = summary["편수"].fillna(0).astype(int)
for col in ["스크린 수 평균", "누적 관객 평균", "10위권 일수 평균", "롱런 지수 평균"]:
    summary[col] = summary[col].round(2)
st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("6. 유형별 누적 관객 상위 영화 5편")
for label in ["㉮", "㉯", "㉰"]:
    group = df[df["영화 유형"] == label].sort_values("누적 관객", ascending=False).head(5)
    with st.expander(f"{label} 유형 — 상위 {len(group)}편", expanded=True):
        if group.empty:
            st.write("이 유형에 해당하는 영화가 없습니다.")
        else:
            display = group[["영화 제목", "누적 관객", "스크린 수", "10위권 일수", "롱런 지수"]].copy()
            display = display.rename(columns={
                "영화 제목": "영화 제목",
                "누적 관객": "누적 관객",
                "스크린 수": "스크린 수",
                "10위권 일수": "10위권 일수",
                "롱런 지수": "롱런 지수",
            })
            display["누적 관객"] = display["누적 관객"].map(lambda x: f"{x:,.0f}")
            display["스크린 수"] = display["스크린 수"].map(lambda x: f"{x:,.0f}")
            display["10위권 일수"] = display["10위권 일수"].map(lambda x: f"{x:,.0f}")
            display["롱런 지수"] = display["롱런 지수"].map(lambda x: f"{x:.2f}")
            st.dataframe(display, use_container_width=True, hide_index=True)

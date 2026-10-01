import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

# 웹페이지 기본 설정
st.set_page_config(
    page_title="서울 100년 기온 분석 및 예측기",
    page_icon="🌡️",
    layout="wide"
)

# 데이터 로드 및 전처리 (캐싱 적용)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    
    # 데이터 불러오기 (UTF-8 인코딩)
    df = pd.read_csv(url, encoding='utf-8')
    
    # 열 이름 공백 제거
    df.columns = df.columns.str.strip()
    
    # 날짜 및 기온 데이터 변환
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    for col in ['평균기온', '최저기온', '최고기온']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 1. 2025년 이하 데이터만 대상
    df = df[df['연도'] <= 2025]
    
    # 2. 관측일(유효 평균기온 데이터)이 300일 이상인 해만 추출
    year_counts = df.groupby('연도')['평균기온'].count()
    valid_years = year_counts[year_counts >= 300].index
    
    df_valid = df[df['연도'].isin(valid_years)]
    
    # 연도별 평균 집계
    yearly_df = df_valid.groupby('연도').agg(
        연평균기온=('평균기온', 'mean'),
        연평균최저기온=('최저기온', 'mean'),
        연평균최고기온=('최고기온', 'mean')
    ).reset_index().round(2)
    
    # 누락된 연도를 구분하기 위해 전체 연도 범위 생성 후 좌측 결합
    full_years = pd.DataFrame({'연도': range(int(yearly_df['연도'].min()), int(yearly_df['연도'].max()) + 1)})
    yearly_df = pd.merge(full_years, yearly_df, on='연도', how='left')
    
    # 10년 이동평균 추세선 계산
    yearly_df['10년_이동평균'] = yearly_df['연평균기온'].rolling(window=10, min_periods=5).mean().round(2)
    
    return yearly_df

# 메인 화면 구성
st.title("🌡️ 서울 기온 분석 및 예측기")
st.markdown("지난 서울의 기온 관측 데이터를 바탕으로 **지난 100년간의 추이**를 분석하고 **선형 회귀 모델**을 통해 미래 기온을 예측합니다.")
st.markdown("---")

try:
    yearly_df = load_data()
    
    # 관측 데이터가 유효한 행들만 필터링 (회귀 분석 및 통계 계산용)
    valid_df = yearly_df.dropna(subset=['연평균기온']).copy()
    
    # 회귀 분석용 독립 변수 계산 (1908년부터 지난 연수)
    valid_df['경과연수'] = valid_df['연도'] - 1908
    
    X = valid_df['경과연수']
    Y = valid_df['연평균기온']
    
    # 선형 회귀분석 수행
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    
    # ==========================================
    # [기존 기능 1] 주요 통계 지표 및 100년 추이 그래프
    # ==========================================
    st.subheader("📊 1. 서울 연평균 기온 변동 추이 분석")
    
    min_year_row = valid_df.loc[valid_df['연평균기온'].idxmin()]
    max_year_row = valid_df.loc[valid_df['연평균기온'].idxmax()]
    start_year = int(valid_df['연도'].min())
    end_year = int(valid_df['연도'].max())
    count_years = len(valid_df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("학습 데이터 연도 수", f"{count_years}개 해")
    with col2:
        st.metric("시작 연도", f"{start_year}년")
    with col3:
        st.metric("끝 연도", f"{end_year}년")
    with col4:
        st.metric("상관계수 (r)", f"{r_value:.3f}")
        
    # Plotly 시각화 (connectgaps=False로 설정하여 누락 구간 선 끊기)
    fig_history = go.Figure()
    
    fig_history.add_trace(go.Scatter(
        x=yearly_df['연도'],
        y=yearly_df['연평균기온'],
        mode='lines+markers',
        name='연평균 기온',
        connectgaps=False,
        line=dict(color='#FF5722', width=2),
        marker=dict(size=5),
        hovertemplate='%{x}년: %{y}℃<extra></extra>'
    ))
    
    fig_history.add_trace(go.Scatter(
        x=yearly_df['연도'],
        y=yearly_df['10년_이동평균'],
        mode='lines',
        name='10년 이동평균 (추세)',
        connectgaps=False,
        line=dict(color='#2196F3', width=3, dash='dash'),
        hovertemplate='%{x}년 10년 평균: %{y}℃<extra></extra>'
    ))
    
    fig_history.update_layout(
        title={'text': "서울 연도별 평균 기온 및 10년 이동평균 추이 (누락 구간 분리)", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="연도",
        yaxis_title="기온 (℃)",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_history, use_container_width=True)
    
    # 최고/최저 기온 및 요약 표
    tab1, tab2 = st.columns([1, 1])
    with tab1:
        st.markdown("#### 🔍 최고 / 최저 기온 평균 비교")
        fig_sub = go.Figure()
        fig_sub.add_trace(go.Scatter(x=yearly_df['연도'], y=yearly_df['연평균최고기온'], name='연평균 최고기온', connectgaps=False, line=dict(color='#E53935')))
        fig_sub.add_trace(go.Scatter(x=yearly_df['연도'], y=yearly_df['연평균최저기온'], name='연평균 최저기온', connectgaps=False, line=dict(color='#1E88E5')))
        fig_sub.update_layout(
            title="연도별 최고/최저 기온 평균 비교",
            xaxis_title="연도",
            yaxis_title="기온 (℃)",
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig_sub, use_container_width=True)
        
    with tab2:
        st.markdown("#### 📋 연도별 데이터 요약표")
        st.dataframe(
            yearly_df[['연도', '연평균기온', '연평균최저기온', '연평균최고기온', '10년_이동평균']],
            height=350,
            use_container_width=True
        )

    st.markdown("---")

    # ==========================================
    # [추가 기능 2] 선형 회귀 기온 예측기
    # ==========================================
    st.subheader("🔮 2. 선형 회귀 기반 기온 예측기")
    
    selected_year = st.slider(
        "예측하고 싶은 연도를 선택하세요 (1900년 ~ 2100년)",
        min_value=1900,
        max_value=2100,
        value=2025,
        step=1
    )
    
    # 선택한 연도의 회귀 예측 기온 계산
    selected_x = selected_year - 1908
    predicted_temp = slope * selected_x + intercept
    
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric(
            label=f"🎯 {selected_year}년 예상 연평균 기온",
            value=f"{predicted_temp:.2f} ℃"
        )
    with res_col2:
        st.info(
            f"**회귀 방정식**: $Y = {slope:.4f} \\times (연도 - 1908) + {intercept:.4f}$\n\n"
            f"서울의 연평균 기온은 1908년을 기준(0년)으로 매년 약 **{slope:.3f}℃**씩 상승하는 추세를 보입니다."
        )
        
    # 산점도 + 회귀선 그래프
    future_years = np.arange(start_year, 2101)
    future_x = future_years - 1908
    future_pred = slope * future_x + intercept
    
    fig_pred = go.Figure()
    
    # 관측 데이터 산점도
    fig_pred.add_trace(go.Scatter(
        x=valid_df['연도'],
        y=valid_df['연평균기온'],
        mode='markers',
        name='실제 관측 연평균 기온',
        marker=dict(color='#FF5722', size=7, opacity=0.8),
        hovertemplate='%{x}년 실제: %{y}℃<extra></extra>'
    ))
    
    # 회귀선
    fig_pred.add_trace(go.Scatter(
        x=future_years,
        y=future_pred,
        mode='lines',
        name='선형 회귀선',
        line=dict(color='#1E88E5', width=3),
        hovertemplate='%{x}년 추정: %{y:.2f}℃<extra></extra>'
    ))
    
    # 선택된 연도 강조 표시
    fig_pred.add_trace(go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode='markers+text',
        name=f'선택 연도 ({selected_year}년)',
        marker=dict(color='#4CAF50', size=14, symbol='star'),
        text=[f"  <b>{selected_year}년 ({predicted_temp:.2f}℃)</b>"],
        textposition="top center",
        hovertemplate='%{x}년 선택 예측치: %{y:.2f}℃<extra></extra>'
    ))
    
    fig_pred.update_layout(
        title={'text': "서울 연도별 평균 기온 관측치 및 회귀 예측선 (1900년~2100년)", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="연도",
        yaxis_title="평균 기온 (℃)",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")

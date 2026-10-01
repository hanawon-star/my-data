import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

# 웹페이지 기본 설정
st.set_page_config(
    page_title="서울 기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

# 데이터 로드 및 전처리 (캐싱 적용)
@st.cache_data
def load_and_process_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    
    # 데이터 불러오기 (UTF-8 인코딩)
    df = pd.read_csv(url, encoding='utf-8')
    
    # 열 이름 공백 제거
    df.columns = df.columns.str.strip()
    
    # 날짜 및 연도 열 생성
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    df['평균기온'] = pd.to_numeric(df['평균기온'], errors='coerce')
    
    # 1. 2025년 이전(포함) 데이터만 선택
    df = df[df['연도'] <= 2025]
    
    # 2. 관측일(유효 기온 데이터)이 300일 이상인 해만 추출
    year_counts = df.groupby('연도')['평균기온'].count()
    valid_years = year_counts[year_counts >= 300].index
    
    df_valid = df[df['연도'].isin(valid_years)]
    
    # 연도별 평균 기온 계산
    yearly_df = df_valid.groupby('연도')['평균기온'].mean().reset_index()
    yearly_df['평균기온'] = yearly_df['평균기온'].round(2)
    
    return yearly_df

# 메인 레이아웃
st.title("🌡️ 서울 기온 예측기")
st.markdown("지난 서울의 기온 관측 데이터를 바탕으로 **선형 회귀 모델**을 생성하여 연도별 예상 기온을 예측합니다.")
st.markdown("---")

try:
    yearly_df = load_and_process_data()
    
    # 회귀 직선 계산용 독립 변수 (1908년부터 지난 연수)
    yearly_df['경과연수'] = yearly_df['연도'] - 1908
    
    X = yearly_df['경과연수']
    Y = yearly_df['평균기온']
    
    # 선형 회귀분석 수행
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    
    # 정보 요약 변수
    count_years = len(yearly_df)
    start_year = int(yearly_df['연도'].min())
    end_year = int(yearly_df['연도'].max())
    corr = r_value  # 상관계수
    r_squared = r_value ** 2  # 결정계수
    
    # 1. 회귀 모델 정보 및 통계 표시
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("학습 데이터 연도 수", f"{count_years}개 해")
    with col2:
        st.metric("기준 시작 연도", f"{start_year}년")
    with col3:
        st.metric("기준 끝 연도", f"{end_year}년")
    with col4:
        st.metric("상관계수 (r)", f"{corr:.3f}")
        
    st.markdown("---")
    
    # 2. 연도 선택 슬라이더 및 예상 기온 표시
    st.subheader("🔮 예측하고 싶은 연도를 선택하세요")
    selected_year = st.slider(
        "연도 선택 (1900년 ~ 2100년)",
        min_value=1900,
        max_value=2100,
        value=2025,
        step=1
    )
    
    # 선택 연도 예측 기온 계산
    selected_x = selected_year - 1908
    predicted_temp = slope * selected_x + intercept
    
    # 예측 결과 크게 표시
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric(
            label=f"🎯 {selected_year}년 예상 연평균 기온",
            value=f"{predicted_temp:.2f} ℃"
        )
    with res_col2:
        st.info(
            f"**회귀 방정식**: $Y = {slope:.4f} \\times (연도 - 1908) + {intercept:.4f}$\n\n"
            f"서울의 기온은 1908년을 기준(0년)으로 매년 약 **{slope:.3f}℃**씩 상승하는 추세를 보입니다."
        )
        
    st.markdown("---")
    
    # 3. Plotly 산점도 + 회귀 직선 그래프
    st.subheader("📈 연평균 기온 산점도 및 회귀 직선")
    
    # 1908년~2100년 범위의 회귀선 라인 데이터 생성
    future_years = np.arange(yearly_df['연도'].min(), 2101)
    future_x = future_years - 1908
    future_pred = slope * future_x + intercept
    
    fig = go.Figure()
    
    # 관측 데이터 산점도
    fig.add_trace(go.Scatter(
        x=yearly_df['연도'],
        y=yearly_df['평균기온'],
        mode='markers',
        name='실제 관측 연평균 기온',
        marker=dict(color='#FF5722', size=7, opacity=0.8),
        hovertemplate='%{x}년 실제: %{y}℃<extra></extra>'
    ))
    
    # 회귀 직선
    fig.add_trace(go.Scatter(
        x=future_years,
        y=future_pred,
        mode='lines',
        name='선형 회귀선',
        line=dict(color='#1E88E5', width=3),
        hovertemplate='%{x}년 추정: %{y:.2f}℃<extra></extra>'
    ))
    
    # 사용자가 선택한 연도 강조 표시
    fig.add_trace(go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode='markers+text',
        name=f'선택한 연도 ({selected_year}년)',
        marker=dict(color='#4CAF50', size=14, symbol='star'),
        text=[f"  <b>{selected_year}년 ({predicted_temp:.2f}℃)</b>"],
        textposition="top center",
        hovertemplate='%{x}년 선택 예측치: %{y:.2f}℃<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "서울 연도별 평균 기온 관측치 및 회귀 예측선 (1900년~2100년)",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="연도",
        yaxis_title="평균 기온 (℃)",
        template="plotly_white",
        height=550,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")

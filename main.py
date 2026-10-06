import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 웹페이지 기본 설정
st.set_page_config(
    page_title="서울 100년 기온 분석 및 예측 모델 종합",
    page_icon="🌡️",
    layout="wide"
)

# 데이터 로드 및 전처리 (캐싱 적용)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    
    # 데이터 불러오기 (UTF-8 인코딩)
    df = pd.read_csv(url, encoding='utf-8')
    df.columns = df.columns.str.strip()
    
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
    
    # 연도별 평균 기온 집계 ('연평균기온' 컬럼명 명시)
    yearly_df = df_valid.groupby('연도').agg(
        연평균기온=('평균기온', 'mean'),
        연평균최저기온=('최저기온', 'mean'),
        연평균최고기온=('최고기온', 'mean')
    ).reset_index().round(2)
    
    # 누락 연도 구분을 위한 전체 연도 범위 생성 후 재결합
    full_years = pd.DataFrame({'연도': range(int(yearly_df['연도'].min()), int(yearly_df['연도'].max()) + 1)})
    yearly_df = pd.merge(full_years, yearly_df, on='연도', how='left')
    
    # 10년 이동평균 계산
    yearly_df['10년_이동평균'] = yearly_df['연평균기온'].rolling(window=10, min_periods=5).mean().round(2)
    
    return yearly_df

# 메인 화면 타이틀
st.title("🌡️️ 서울 기온 통합 분석 및 회귀 모델 종합")
st.markdown("서울 기온 데이터의 **100년 추이 분석**, **학습 기간별 회귀선 비교**, **다항 회귀(곡선) 및 2050년 예측 성능 평가**를 제공합니다.")
st.markdown("---")

try:
    yearly_df = load_data()
    valid_df = yearly_df.dropna(subset=['연평균기온']).copy()
    
    # 연도 스케일링 (1908년을 0으로 변환)
    base_year = 1908
    valid_df['경과연수'] = valid_df['연도'] - base_year

    # ==========================================
    # [기능 1] 서울 연평균 기온 변동 추이 분석
    # ==========================================
    st.subheader("📊 1. 서울 연평균 기온 변동 추이 분석")
    
    start_year = int(valid_df['연도'].min())
    end_year = int(valid_df['연도'].max())
    count_years = len(valid_df)
    
    # 전체 데이터 기준 선형회귀
    X_all = valid_df['경과연수']
    Y_all = valid_df['연평균기온']
    slope_all, intercept_all, r_value_all, _, _ = stats.linregress(X_all, Y_all)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("학습 데이터 연도 수", f"{count_years}개 해")
    with col2:
        st.metric("시작 연도", f"{start_year}년")
    with col3:
        st.metric("끝 연도", f"{end_year}년")
    with col4:
        st.metric("상관계수 (r)", f"{r_value_all:.3f}")
        
    fig_history = go.Figure()
    fig_history.add_trace(go.Scatter(
        x=yearly_df['연도'], y=yearly_df['연평균기온'],
        mode='lines+markers', name='연평균 기온', connectgaps=False,
        line=dict(color='#FF5722', width=2), marker=dict(size=5),
        hovertemplate='%{x}년: %{y}℃<extra></extra>'
    ))
    fig_history.add_trace(go.Scatter(
        x=yearly_df['연도'], y=yearly_df['10년_이동평균'],
        mode='lines', name='10년 이동평균', connectgaps=False,
        line=dict(color='#2196F3', width=3, dash='dash'),
        hovertemplate='%{x}년 10년 평균: %{y}℃<extra></extra>'
    ))
    fig_history.update_layout(
        title={'text': "서울 연도별 평균 기온 및 10년 이동평균 추이", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="연도", yaxis_title="기온 (℃)", hovermode="x unified",
        template="plotly_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_history, use_container_width=True)
    
    tab1, tab2 = st.columns([1, 1])
    with tab1:
        st.markdown("#### 🔍 최고 / 최저 기온 평균 비교")
        fig_sub = go.Figure()
        fig_sub.add_trace(go.Scatter(x=yearly_df['연도'], y=yearly_df['연평균최고기온'], name='연평균 최고기온', connectgaps=False, line=dict(color='#E53935')))
        fig_sub.add_trace(go.Scatter(x=yearly_df['연도'], y=yearly_df['연평균최저기온'], name='연평균 최저기온', connectgaps=False, line=dict(color='#1E88E5')))
        fig_sub.update_layout(title="연도별 최고/최저 기온 평균 비교", xaxis_title="연도", yaxis_title="기온 (℃)", template="plotly_white", height=320)
        st.plotly_chart(fig_sub, use_container_width=True)
        
    with tab2:
        st.markdown("#### 📋 연도별 데이터 요약표")
        st.dataframe(yearly_df[['연도', '연평균기온', '연평균최저기온', '연평균최고기온', '10년_이동평균']], height=320, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # [기능 2] 학습 기간별(50년 vs 100년) 회귀선 비교
    # ==========================================
    st.subheader("🧪 2. 과거 훈련 데이터(50년 vs 100년) 회귀 모델 비교")
    
    train_100 = valid_df[(valid_df['연도'] >= 1906) & (valid_df['연도'] <= 2005)]
    train_50 = valid_df[(valid_df['연도'] >= 1956) & (valid_df['연도'] <= 2005)]
    test_recent = valid_df[(valid_df['연도'] >= 2006) & (valid_df['연도'] <= 2025)]
    
    slope_100, intercept_100, _, _, _ = stats.linregress(train_100['경과연수'], train_100['연평균기온'])
    slope_50, intercept_50, _, _, _ = stats.linregress(train_50['경과연수'], train_50['연평균기온'])
    
    mae_100 = mean_absolute_error(test_recent['연평균기온'], slope_100 * test_recent['경과연수'] + intercept_100)
    mae_50 = mean_absolute_error(test_recent['연평균기온'], slope_50 * test_recent['경과연수'] + intercept_50)
    
    st.info(
        f"- **최근 100년 학습 (1906~2005)**: 기울기 = **{slope_100:.4f}℃/년**, 최근 20년 테스트 MAE = **{mae_100:.3f}℃**\n"
        f"- **최근 50년 학습 (1956~2005)**: 기울기 = **{slope_50:.4f}℃/년**, 최근 20년 테스트 MAE = **{mae_50:.3f}℃**\n\n"
        f"👉 최근 50년 모델의 기울기가 더 가파르며, 온난화 가속도를 반영해 최근 20년 데이터 예측 오차가 더 적습니다."
    )

    st.markdown("---")

    # ==========================================
    # [기능 3] 다항 회귀(1차, 3차, 9차 곡선) 평가 및 2050년 예측
    # ==========================================
    st.subheader("📈 3. 다항 회귀(1차, 3차, 9차 곡선) 모델 평가 (2005년 기준 분할)")
    
    train_df = valid_df[valid_df['연도'] < 2005].copy()
    test_df = valid_df[valid_df['연도'] >= 2005].copy()
    
    col_tr, col_te = st.columns(2)
    with col_tr:
        st.metric("훈련용 데이터 연도 수 (< 2005년)", f"{len(train_df)}개 연도", f"{int(train_df['연도'].min())}년 ~ {int(train_df['연도'].max())}년")
    with col_te:
        st.metric("테스트용 데이터 연도 수 (≥ 2005년)", f"{len(test_df)}개 연도", f"{int(test_df['연도'].min())}년 ~ {int(test_df['연도'].max())}년")

    degrees = [1, 3, 9]
    models = {}
    eval_results = []
    
    X_tr = train_df['경과연수'].values
    y_tr = train_df['연평균기온'].values
    X_te = test_df['경과연수'].values
    y_te = test_df['연평균기온'].values
    
    x_2050 = 2050 - base_year
    
    for deg in degrees:
        coefs = np.polyfit(X_tr, y_tr, deg)
        poly_func = np.poly1d(coefs)
        models[deg] = poly_func
        
        pred_test = poly_func(X_te)
        mae = mean_absolute_error(y_te, pred_test)
        mse = mean_squared_error(y_te, pred_test)
        r2 = r2_score(y_te, pred_test)
        pred_2050 = poly_func(x_2050)
        
        eval_results.append({
            '모델 차수': f'{deg}차 곡선' if deg > 1 else '1차 (직선)',
            '테스트 MAE (평균 오차)': f"{mae:.3f} ℃",
            '테스트 MSE': f"{mse:.3f}",
            '테스트 R²': f"{r2:.3f}",
            '2050년 예상 기온': f"{pred_2050:.2f} ℃"
        })
        
    eval_table = pd.DataFrame(eval_results)
    st.dataframe(eval_table, use_container_width=True, hide_index=True)
    
    st.warning(
        f"💡 **오버피팅 관찰**:\n"
        f"- **1차/3차 모델**은 테스트 데이터 오차가 약 **{eval_results[0]['테스트 MAE (평균 오차)']} ~ {eval_results[1]['테스트 MAE (평균 오차)']}**로 안정적입니다.\n"
        f"- **9차 고차 곡선**은 훈련 데이터 노이즈에 과적합되어 2050년 예상 기온이 **{eval_results[2]['2050년 예상 기온']}**로 극단적 폭발 현상을 보입니다."
    )

    # 곡선 시각화 차트
    x_range_years = np.linspace(int(valid_df['연도'].min()), 2050, 400)
    x_range_scaled = x_range_years - base_year
    
    fig_poly = go.Figure()
    fig_poly.add_trace(go.Scatter(x=train_df['연도'], y=train_df['연평균기온'], mode='markers', name=f'훈련 데이터 ({len(train_df)}개)', marker=dict(color='#2196F3', size=6, opacity=0.7)))
    fig_poly.add_trace(go.Scatter(x=test_df['연도'], y=test_df['연평균기온'], mode='markers', name=f'테스트 데이터 ({len(test_df)}개)', marker=dict(color='#E53935', size=8, symbol='diamond')))
    
    colors = {1: '#4CAF50', 3: '#FF9800', 9: '#9C27B0'}
    styles = {1: 'dash', 3: 'solid', 9: 'dot'}
    
    for deg in degrees:
        fig_poly.add_trace(go.Scatter(
            x=x_range_years, y=models[deg](x_range_scaled),
            mode='lines', name=f'{deg}차 회귀선',
            line=dict(color=colors[deg], width=2.5, dash=styles[deg])
        ))
        
    fig_poly.update_layout(
        title={'text': "다항 회귀 모델별 예측 곡선 비교 (1908년 ~ 2050년)", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="연도", yaxis_title="평균 기온 (℃)",
        yaxis=dict(range=[np.min(valid_df['연평균기온']) - 2, np.max(valid_df['연평균기온']) + 5]),
        template="plotly_white", height=500, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_poly, use_container_width=True)

except Exception as e:
    st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")

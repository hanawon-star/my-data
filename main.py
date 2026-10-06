import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 웹페이지 기본 설정
st.set_page_config(
    page_title="서울 100년 기온 분석 및 회귀 모델 비교",
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
    
    # 연도별 평균 집계
    yearly_df = df_valid.groupby('연도').agg(
        연평균기온=('평균기온', 'mean'),
        연평균최저기온=('최저기온', 'mean'),
        연평균최고기온=('최고기온', 'mean')
    ).reset_index().round(2)
    
    # 누락된 연도 구분을 위한 전체 연도 범위 재구성
    full_years = pd.DataFrame({'연도': range(int(yearly_df['연도'].min()), int(yearly_df['연도'].max()) + 1)})
    yearly_df = pd.merge(full_years, yearly_df, on='연도', how='left')
    
    # 10년 이동평균 추세선
    yearly_df['10년_이동평균'] = yearly_df['연평균기온'].rolling(window=10, min_periods=5).mean().round(2)
    
    return yearly_df

# 메인 화면 구성
st.title("🌡️ 서울 기온 분석 및 학습 모델 비교기")
st.markdown("지난 서울 기온 관측 데이터를 바탕으로 **시계열 추이 분석**, **전체 데이터 회귀 모델**, **학습 기간별(50년 vs 100년) 모델 비교 및 테스트 평가**를 제공합니다.")
st.markdown("---")

try:
    yearly_df = load_data()
    valid_df = yearly_df.dropna(subset=['연평균기온']).copy()
    valid_df['경과연수'] = valid_df['연도'] - 1908

    # ==========================================
    # [기존 기능 1] 주요 통계 지표 및 100년 추이 그래프
    # ==========================================
    st.subheader("📊 1. 서울 연평균 기온 변동 추이 분석")
    
    start_year = int(valid_df['연도'].min())
    end_year = int(valid_df['연도'].max())
    count_years = len(valid_df)
    
    # 전체 데이터 기준 선형회귀
    X_all = valid_df['경과연수']
    Y_all = valid_df['연평균기온']
    slope_all, intercept_all, r_value_all, p_value_all, std_err_all = stats.linregress(X_all, Y_all)
    
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
    # [기존 기능 2] 전체 데이터 선형 회귀 예측기
    # ==========================================
    st.subheader("🔮 2. 전체 데이터 기반 선형 회귀 예측기")
    
    selected_year = st.slider("예측하고 싶은 연도를 선택하세요 (1900년 ~ 2100년)", 1900, 2100, 2025, step=1)
    selected_x = selected_year - 1908
    predicted_temp = slope_all * selected_x + intercept_all
    
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric(label=f"🎯 {selected_year}년 예상 연평균 기온", value=f"{predicted_temp:.2f} ℃")
    with res_col2:
        st.info(f"**전체 데이터 회귀식**: $Y = {slope_all:.4f} \\times (연도 - 1908) + {intercept_all:.4f}$\n\n매년 약 **{slope_all:.3f}℃**씩 상승하는 추세입니다.")
        
    future_years = np.arange(start_year, 2101)
    future_x = future_years - 1908
    future_pred = slope_all * future_x + intercept_all
    
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=valid_df['연도'], y=valid_df['연평균기온'], mode='markers', name='실제 관측치', marker=dict(color='#FF5722', size=6, opacity=0.8)))
    fig_pred.add_trace(go.Scatter(x=future_years, y=future_pred, mode='lines', name='전체 데이터 회귀선', line=dict(color='#1E88E5', width=3)))
    fig_pred.add_trace(go.Scatter(x=[selected_year], y=[predicted_temp], mode='markers+text', name=f'선택 연도 ({selected_year}년)', marker=dict(color='#4CAF50', size=13, symbol='star'), text=[f" <b>{selected_year}년 ({predicted_temp:.2f}℃)</b>"], textposition="top center"))
    fig_pred.update_layout(title={'text': "서울 연도별 평균 기온 관측치 및 전체 회귀선", 'x': 0.5, 'xanchor': 'center'}, xaxis_title="연도", yaxis_title="평균 기온 (℃)", template="plotly_white", height=450, hovermode="x unified")
    st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # [신규 기능 3] 훈련/테스트 데이터 분리 및 모델 비교 평가
    # ==========================================
    st.subheader("🧪 3. 과거 훈련 데이터(50년 vs 100년) 회귀 모델 비교 및 테스트 성능 평가")
    st.markdown("과거 연도를 훈련 데이터로 다르게 학습시켰을 때, 최근 20년(2006년~2025년) 공통 테스트 데이터를 얼마나 잘 예측하는지 평가합니다.")
    
    # 데이터 분할
    train_100 = valid_df[(valid_df['연도'] >= 1906) & (valid_df['연도'] <= 2005)]
    train_50 = valid_df[(valid_df['연도'] >= 1956) & (valid_df['연도'] <= 2005)]
    test_df = valid_df[(valid_df['연도'] >= 2006) & (valid_df['연도'] <= 2025)]
    
    # 1. 최근 100년 모델 학습 (1906~2005)
    slope_100, intercept_100, r_100, _, _ = stats.linregress(train_100['경과연수'], train_100['연평균기온'])
    pred_test_100 = slope_100 * test_df['경과연수'] + intercept_100
    mae_100 = mean_absolute_error(test_df['연평균기온'], pred_test_100)
    mse_100 = mean_squared_error(test_df['연평균기온'], pred_test_100)
    r2_100 = r2_score(test_df['연평균기온'], pred_test_100)
    
    # 2. 최근 50년 모델 학습 (1956~2005)
    slope_50, intercept_50, r_50, _, _ = stats.linregress(train_50['경과연수'], train_50['연평균기온'])
    pred_test_50 = slope_50 * test_df['경과연수'] + intercept_50
    mae_50 = mean_absolute_error(test_df['연평균기온'], pred_test_50)
    mse_50 = mean_squared_error(test_df['연평균기온'], pred_test_50)
    r2_50 = r2_score(test_df['연평균기온'], pred_test_50)
    
    # 3. 전체 데이터 모델 테스트 평가 (참고용)
    pred_test_all = slope_all * test_df['경과연수'] + intercept_all
    mae_all = mean_absolute_error(test_df['연평균기온'], pred_test_all)
    mse_all = mean_squared_error(test_df['연평균기온'], pred_test_all)
    r2_all = r2_score(test_df['연평균기온'], pred_test_all)

    # 지표 비교 표 생성
    metrics_summary = pd.DataFrame({
        '구분': ['최근 100년 학습 모델', '최근 50년 학습 모델', '전체 데이터 학습 모델 (참고)'],
        '학습 기간': ['1906년 ~ 2005년', '1956년 ~ 2005년', f'{start_year}년 ~ {end_year}년'],
        '학습 데이터 수': [f"{len(train_100)}개 연도", f"{len(train_50)}개 연도", f"{len(valid_df)}개 연도"],
        '기울기 (℃/년)': [round(slope_100, 4), round(slope_50, 4), round(slope_all, 4)],
        '테스트 MAE (℃)': [round(mae_100, 3), round(mae_50, 3), round(mae_all, 3)],
        '테스트 MSE (℃²)': [round(mse_100, 3), round(mse_50, 3), round(mse_all, 3)],
        '테스트 R²': [round(r2_100, 3), round(r2_50, 3), round(r2_all, 3)]
    })
    
    st.markdown("#### 📊 모델별 예측 성능 및 기울기 비교 표 (테스트 구간: 2006년 ~ 2025년)")
    st.dataframe(metrics_summary, use_container_width=True, hide_index=True)

    # 주요 분석 인사이트 설명
    st.info(
        f"💡 **모델 비교 요약 및 분석 결과**:\n"
        f"- **기울기 비교**: 최근 50년 모델의 기울기(**{slope_50:.4f}℃/년**)가 최근 100년 모델(**{slope_100:.4f}℃/년**)보다 높습니다. 이는 최근 들어 온난화 속도가 빠르게 가속화되었음을 보여줍니다.\n"
        f"- **예측 성능 비교**: 가속화된 최신 상승 추세를 더 잘 반영한 **최근 50년 학습 모델**이 최근 20년(2006~2025) 테스트 데이터에 대해 더 낮은 오차(MAE: {mae_50:.3f}℃ vs {mae_100:.3f}℃)와 높고 우수한 결정계수($R^2$: {r2_50:.3f} vs {r2_100:.3f})를 기록하며 훨씬 뛰어난 예측 성능을 발휘합니다."
    )

    # 4. 회귀선 비교 시각화 그래프
    eval_x_range = np.arange(1906, 2026)
    eval_x_diff = eval_x_range - 1908
    
    pred_line_100 = slope_100 * eval_x_diff + intercept_100
    pred_line_50 = slope_50 * eval_x_diff + intercept_50
    pred_line_all = slope_all * eval_x_diff + intercept_all
    
    fig_comp = go.Figure()
    
    # 실제 데이터 점 표시 (학습 / 테스트 색상 구분)
    fig_comp.add_trace(go.Scatter(
        x=train_100['연도'], y=train_100['연평균기온'],
        mode='markers', name='과거 훈련 데이터 (1906~2005)',
        marker=dict(color='#78909C', size=6, opacity=0.6)
    ))
    fig_comp.add_trace(go.Scatter(
        x=test_df['연도'], y=test_df['연평균기온'],
        mode='markers', name='테스트 데이터 (2006~2025)',
        marker=dict(color='#E53935', size=9, symbol='diamond')
    ))
    
    # 각 모델 회귀선
    fig_comp.add_trace(go.Scatter(
        x=eval_x_range, y=pred_line_100, mode='lines',
        name=f'최근 100년 모델 회귀선 (기울기: {slope_100:.4f})',
        line=dict(color='#FB8C00', width=2.5, dash='dash')
    ))
    fig_comp.add_trace(go.Scatter(
        x=eval_x_range, y=pred_line_50, mode='lines',
        name=f'최근 50년 모델 회귀선 (기울기: {slope_50:.4f})',
        line=dict(color='#43A047', width=3)
    ))
    fig_comp.add_trace(go.Scatter(
        x=eval_x_range, y=pred_line_all, mode='lines',
        name=f'전체 모델 회귀선 (기울기: {slope_all:.4f})',
        line=dict(color='#1E88E5', width=2, dash='dot')
    ))
    
    fig_comp.update_layout(
        title={'text': "학습 기간별 회귀선 비교 및 테스트 데이터 예측 차트", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="연도", yaxis_title="기온 (℃)",
        template="plotly_white", height=520, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)

except Exception as e:
    st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")

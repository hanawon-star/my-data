import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 웹페이지 기본 설정
st.set_page_config(
    page_title="서울 기온 다항 회귀(곡선) 예측 모델",
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
            
    # 2025년 이하 및 관측일 300일 이상 연도 필터링
    df = df[df['연도'] <= 2025]
    year_counts = df.groupby('연도')['평균기온'].count()
    valid_years = year_counts[year_counts >= 300].index
    
    df_valid = df[df['연도'].isin(valid_years)]
    
    # 연도별 평균 기온 집계 (열 이름을 '연평균기온'으로 명확히 지정)
    yearly_df = df_valid.groupby('연도').agg(
        연평균기온=('평균기온', 'mean'),
        연평균최저기온=('최저기온', 'mean'),
        연평균최고기온=('최고기온', 'mean')
    ).reset_index().round(2)
    
    # 전체 연도 범위 정렬
    full_years = pd.DataFrame({'연도': range(int(yearly_df['연도'].min()), int(yearly_df['연도'].max()) + 1)})
    yearly_df = pd.merge(full_years, yearly_df, on='연도', how='left')
    
    # 10년 이동평균
    yearly_df['10년_이동평균'] = yearly_df['연평균기온'].rolling(window=10, min_periods=5).mean().round(2)
    
    return yearly_df

# 메인 화면 구성
st.title("🌡️ 서울 기온 곡선 회귀(다항 회귀) 예측 모델")
st.markdown("1차(직선), 3차(곡선), 9차(고차 곡선) 모델을 **2005년 이전 훈련 데이터**로만 학습시키고, **2005년 이후 테스트 데이터**로 일반화 성능을 평가합니다.")
st.markdown("---")

try:
    yearly_df = load_data()
    valid_df = yearly_df.dropna(subset=['연평균기온']).copy()
    
    # 수치 안정성을 위해 연도를 스케일링 (1908년을 0으로 변환)
    base_year = 1908
    valid_df['X_scaled'] = valid_df['연도'] - base_year

    # ==========================================
    # [데이터 분할] 2005년 미만은 훈련용, 2005년 이상은 테스트용
    # ==========================================
    train_df = valid_df[valid_df['연도'] < 2005].copy()
    test_df = valid_df[valid_df['연도'] >= 2005].copy()
    
    train_count = len(train_df)
    test_count = len(test_df)
    
    # 상단 데이터 분할 정보 명시
    col_tr, col_te, col_base = st.columns(3)
    with col_tr:
        st.metric("훈련용 데이터 연도 수 (2005년 이전)", f"{train_count}개 연도", delta=f"{int(train_df['연도'].min())}년 ~ {int(train_df['연도'].max())}년")
    with col_te:
        st.metric("테스트용 데이터 연도 수 (2005년 이후)", f"{test_count}개 연도", delta=f"{int(test_df['연도'].min())}년 ~ {int(test_df['연도'].max())}년")
    with col_base:
        st.metric("연도 스케일링 기준", f"연도 - {base_year}", delta="고차 곡선 계산 안정화")

    st.markdown("---")

    # ==========================================
    # [다항 회귀 모델 학습 및 테스트 데이터 채점]
    # ==========================================
    degrees = [1, 3, 9]
    models = {}
    eval_results = []
    
    X_train = train_df['X_scaled'].values
    y_train = train_df['연평균기온'].values
    
    X_test = test_df['X_scaled'].values
    y_test = test_df['연평균기온'].values
    
    # 2050년 계산용 스케일 값
    x_2050 = 2050 - base_year
    
    for deg in degrees:
        # numpy.polyfit을 통한 다항 회귀 피팅 (훈련용 데이터로만 학습)
        coefs = np.polyfit(X_train, y_train, deg)
        poly_func = np.poly1d(coefs)
        models[deg] = poly_func
        
        # 순수 테스트 데이터 예측값 산출
        pred_test = poly_func(X_test)
        
        # 테스트 데이터 기반 채점 (MAE: 평균 몇 도나 빗나가는지)
        mae = mean_absolute_error(y_test, pred_test)
        mse = mean_squared_error(y_test, pred_test)
        r2 = r2_score(y_test, pred_test)
        
        # 2050년 예측값 계산
        pred_2050 = poly_func(x_2050)
        
        eval_results.append({
            '모델 차수': f'{deg}차 곡선' if deg > 1 else '1차 (직선)',
            '테스트 MAE (평균 오차)': f"{mae:.3f} ℃",
            '테스트 MSE': f"{mse:.3f}",
            '테스트 R²': f"{r2:.3f}",
            '2050년 예상 기온': f"{pred_2050:.2f} ℃"
        })
        
    eval_table = pd.DataFrame(eval_results)
    
    st.subheader("📊 다항 회귀 모델 채점 결과 (학습에 안 쓴 테스트 데이터 기준)")
    st.dataframe(eval_table, use_container_width=True, hide_index=True)
    
    # 과적합(Overfitting) 인사이트
    st.warning(
        f"💡 **분석 포인트 (오버피팅 관찰)**:\n"
        f"- **1차(직선) / 3차 곡선**: 훈련 데이터의 완만한 상승 추세를 적절히 반영하여 테스트 데이터에서도 평균 오차가 약 **{eval_results[0]['테스트 MAE (평균 오차)']} ~ {eval_results[1]['테스트 MAE (평균 오차)']}** 수준으로 안정적입니다.\n"
        f"- **9차 곡선**: 훈련 데이터의 국소적 노이즈에 과도하게 맞추어져(오버피팅), 학습 영역을 벗어나는 순간 예측치가 **{eval_results[2]['2050년 예상 기온']}**로 비현실적 폭주를 보입니다."
    )

    st.markdown("---")

    # ==========================================
    # [시각화] 훈련 / 테스트 데이터 및 모델별 곡선 비교
    # ==========================================
    st.subheader("📈 다항 회귀선 시각화 (1908년 ~ 2050년)")
    
    # 연속 연도 축 생성 (1908~2050)
    x_range_years = np.linspace(int(valid_df['연도'].min()), 2050, 400)
    x_range_scaled = x_range_years - base_year
    
    fig = go.Figure()
    
    # 1. 훈련 데이터 산점도
    fig.add_trace(go.Scatter(
        x=train_df['연도'], y=train_df['연평균기온'],
        mode='markers', name=f'훈련 데이터 (<2005년, {train_count}개)',
        marker=dict(color='#2196F3', size=6, opacity=0.7)
    ))
    
    # 2. 테스트 데이터 산점도
    fig.add_trace(go.Scatter(
        x=test_df['연도'], y=test_df['연평균기온'],
        mode='markers', name=f'테스트 데이터 (≥2005년, {test_count}개)',
        marker=dict(color='#E53935', size=8, symbol='diamond')
    ))
    
    # 3. 각 모델별 추정 곡선
    colors = {1: '#4CAF50', 3: '#FF9800', 9: '#9C27B0'}
    line_styles = {1: 'dash', 3: 'solid', 9: 'dot'}
    
    for deg in degrees:
        y_curve = models[deg](x_range_scaled)
        
        fig.add_trace(go.Scatter(
            x=x_range_years, y=y_curve,
            mode='lines',
            name=f'{deg}차 회귀선' if deg > 1 else '1차 회귀선 (직선)',
            line=dict(color=colors[deg], width=2.5, dash=line_styles[deg])
        ))
        
    fig.update_layout(
        title={'text': "훈련/테스트 데이터와 차수별(1차, 3차, 9차) 예측 곡선 비교", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="연도",
        yaxis_title="평균 기온 (℃)",
        yaxis=dict(range=[np.min(valid_df['연평균기온']) - 2, np.max(valid_df['연평균기온']) + 5]),
        template="plotly_white",
        height=550,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")

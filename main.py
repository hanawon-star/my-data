import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 페이지 기본 설정
st.set_page_config(
    page_title="서울 100년 기온 변화 분석",
    page_icon="🌡️",
    layout="wide"
)

# 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"
    
    # 인코딩 시도 (cp949 -> utf-8 -> euc-kr)
    for enc in ['cp949', 'utf-8', 'euc-kr']:
        try:
            df = pd.read_csv(url, encoding=enc)
            break
        except Exception:
            continue
            
    # 컬럼 이름 정리 (공백 제거)
    df.columns = df.columns.str.strip()
    
    # 컬럼명 매핑
    date_col = [c for c in df.columns if '날짜' in c][0]
    avg_temp_col = [c for c in df.columns if '평균' in c][0]
    min_temp_col = [c for c in df.columns if '최저' in c][0]
    max_temp_col = [c for c in df.columns if '최고' in c][0]
    
    # 날짜 데이터 처리
    df[date_col] = pd.to_datetime(df[date_col])
    df['연도'] = df[date_col].dt.year
    
    # 기온 데이터 숫자형 변환
    for col in [avg_temp_col, min_temp_col, max_temp_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 날짜와 평균기온이 유효한 행만 필터링
    df = df.dropna(subset=['연도', avg_temp_col])
    
    return df, date_col, avg_temp_col, min_temp_col, max_temp_col

# 메인 화면 구성
st.title("🌡️ 지난 100년간 서울의 기온은 어떻게 변했을까?")
st.markdown("기상청 데이터를 바탕으로 지난 100여 년간 서울의 연평균 기온 변동 추이를 분석합니다.")
st.markdown("---")

try:
    df, date_col, avg_temp_col, min_temp_col, max_temp_col = load_data()
    
    # 연도별 집계
    yearly_df = df.groupby('연도').agg(
        연평균기온=(avg_temp_col, 'mean'),
        연평균최저기온=(min_temp_col, 'mean'),
        연평균최고기온=(max_temp_col, 'mean')
    ).reset_index().round(2)
    
    # 전체 연도 범위를 생성하여 관측되지 않은 연도(누락 연도)를 NaN으로 유도
    full_years = pd.DataFrame({'연도': range(yearly_df['연도'].min(), yearly_df['연도'].max() + 1)})
    yearly_df = pd.merge(full_years, yearly_df, on='연도', how='left')
    
    # 이동평균 계산 (10년 단위 추세선)
    yearly_df['10년_이동평균'] = yearly_df['연평균기온'].rolling(window=10, min_periods=5).mean().round(2)
    
    # 데이터가 존재하는 연도 기반 통계 계산
    valid_df = yearly_df.dropna(subset=['연평균기온'])
    min_year_row = valid_df.loc[valid_df['연평균기온'].idxmin()]
    max_year_row = valid_df.loc[valid_df['연평균기온'].idxmax()]
    first_year_avg = valid_df.iloc[0]['연평균기온']
    last_year_avg = valid_df.iloc[-1]['연평균기온']
    temp_change = round(last_year_avg - first_year_avg, 2)
    
    # 주요 지표 요약 (Metric Cards)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="가장 추웠던 해",
            value=f"{int(min_year_row['연도'])}년",
            delta=f"{min_year_row['연평균기온']} ℃"
        )
    with col2:
        st.metric(
            label="가장 따뜻했던 해",
            value=f"{int(max_year_row['연도'])}년",
            delta=f"{max_year_row['연평균기온']} ℃"
        )
    with col3:
        st.metric(
            label="분석 기간",
            value=f"{int(valid_df['연도'].min())}년 ~ {int(valid_df['연도'].max())}년",
            delta=f"총 {len(valid_df)}개 관측 연도"
        )
    with col4:
        st.metric(
            label="관측 시작 대비 변화",
            value=f"{temp_change:+} ℃",
            delta="온난화 추세" if temp_change > 0 else "냉각 추세"
        )
        
    st.markdown("### 📈 서울 연평균 기온 변동 추이 그래프")
    
    # Plotly 시각화 (connectgaps=False 로 설정하여 데이터 없는 구간은 선이 끊김)
    fig = go.Figure()
    
    # 연평균 기온 선 그래프
    fig.add_trace(go.Scatter(
        x=yearly_df['연도'],
        y=yearly_df['연평균기온'],
        mode='lines+markers',
        name='연평균 기온',
        connectgaps=False,  # 누락된 데이터 구간 선 끊기
        line=dict(color='#FF5722', width=2),
        marker=dict(size=5),
        hovertemplate='%{x}년: %{y}℃<extra></extra>'
    ))
    
    # 10년 이동평균 추세선
    fig.add_trace(go.Scatter(
        x=yearly_df['연도'],
        y=yearly_df['10년_이동평균'],
        mode='lines',
        name='10년 이동평균 (추세)',
        connectgaps=False,  # 누락된 데이터 구간 선 끊기
        line=dict(color='#2196F3', width=3, dash='dash'),
        hovertemplate='%{x}년 10년 평균: %{y}℃<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "서울 연도별 평균 기온 및 10년 이동평균 추이",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="연도",
        yaxis_title="기온 (℃)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 상세 옵션 및 데이터 테이블
    st.markdown("---")
    st.markdown("### 📊 상세 데이터 탐색")
    
    tab1, tab2 = st.columns([1, 1])
    
    with tab1:
        st.markdown("#### 🔍 최고/최저 기온 비교")
        fig_sub = go.Figure()
        fig_sub.add_trace(go.Scatter(
            x=yearly_df['연도'], 
            y=yearly_df['연평균최고기온'], 
            name='연평균 최고기온', 
            connectgaps=False,
            line=dict(color='#E53935')
        ))
        fig_sub.add_trace(go.Scatter(
            x=yearly_df['연도'], 
            y=yearly_df['연평균최저기온'], 
            name='연평균 최저기온', 
            connectgaps=False,
            line=dict(color='#1E88E5')
        ))
        fig_sub.update_layout(
            title="연도별 최고/최저 기온 평균 비교",
            xaxis_title="연도",
            yaxis_title="기온 (℃)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_sub, use_container_width=True)
        
    with tab2:
        st.markdown("#### 📋 연도별 데이터 요약표")
        st.dataframe(
            yearly_df[['연도', '연평균기온', '연평균최저기온', '연평균최고기온', '10년_이동평균']],
            height=380,
            use_container_width=True
        )

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")

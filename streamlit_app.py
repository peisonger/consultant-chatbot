# -*- coding: utf-8 -*-
import os
import re
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai

# -----------------------------
# 기본 세팅
# -----------------------------
st.set_page_config(page_title="비밀상담사 – Q1/Q2/Q3 챗봇", page_icon="💡", layout="wide")
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# -----------------------------
# API 설정
# -----------------------------
def _get_api_key():
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return os.getenv("GOOGLE_API_KEY", "")

API_KEY = _get_api_key()
if API_KEY:
    genai.configure(api_key=API_KEY)
    GEMINI = genai.GenerativeModel("gemini-2.5-flash")
else:
    st.error("⚠️ Google Gemini API 키가 없습니다.")
    GEMINI = None

# -----------------------------
# CSV 로드
# -----------------------------
@st.cache_data
def read_csv_safe(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

df_profiles = read_csv_safe("data/analysis_cafe_profiles.csv")       # Q1
df_revisit  = read_csv_safe("data/under_30per_re_rate.csv")          # Q2
df_diag     = read_csv_safe("data/analysis_problem_diagnosis.csv")   # Q3
df_ts_summary = read_csv_safe("data/analysis_timeseries_summary.csv") # Q4
df_ts_graph   = read_csv_safe("data/analysis_timeseries_graph_data.csv") # Q5
df_info = read_csv_safe("data/big_data_set1_f.csv") #Q5

# -----------------------------
# Q5용 마스터 데이터 생성 (상권명 결합)
# -----------------------------
@st.cache_data
def create_q5_master_df(df1, df2):
    if df1.empty or df2.empty:
        return pd.DataFrame()
    # Q3 상세분석 파일 + Q5 상권정보 파일 결합
    df_merged = pd.merge(
        df1,
        df2[['ENCODED_MCT', 'HPSN_MCT_BZN_CD_NM']],
        on='ENCODED_MCT',
        how='left'
    )
    return df_merged

df_q5_master = create_q5_master_df(df_diag, df_info)

# -----------------------------
# 점수 자동 추출 (수정된 버전)
# -----------------------------
def extract_scores(detail):
    if pd.isna(detail):
        return np.nan, np.nan
    
    # 1. 0~7 사이의 유효한 점수만 필터링
    text = str(detail)
    nums = re.findall(r"(\d\.\d+|\d+)", text)
    nums = [float(x) for x in nums if 0 <= float(x) <= 7] 
    
    # 2. 유효한 점수 중 마지막 2개를 사용
    if len(nums) >= 2:
        return nums[-2], nums[-1] # 마지막 두 개가 유치력/수익성일 확률이 높음
    elif len(nums) == 1:
        return nums[0], np.nan
    else:
        return np.nan, np.nan

if "DIAGNOSIS_DETAILS" in df_diag.columns:
    df_diag[["acq_score", "profit_score"]] = df_diag["DIAGNOSIS_DETAILS"].apply(
        lambda x: pd.Series(extract_scores(x))
    )

# -----------------------------
# 날씨, 계절 데이터
# -----------------------------
def get_current_season():
    """현재 월을 기준으로 계절을 반환 (봄, 여름, 가을, 겨울)"""
    month = datetime.date.today().month
    if month in [3, 4, 5]:
        return "봄"
    elif month in [6, 7, 8]:
        return "여름"
    elif month in [9, 10, 11]:
        return "가을"
    else: # 12, 1, 2
        return "겨울"
def get_weather_summary():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=37.56&longitude=126.97&current=temperature_2m,weathercode&timezone=Asia%2FSeoul"
        r = requests.get(url, timeout=5).json()
        t = r["current"]["temperature_2m"]
        code = r["current"]["weathercode"]
        code_map = {0:"맑음",1:"대체로 맑음",2:"부분 흐림",3:"흐림",61:"비",71:"눈",80:"소나기",95:"뇌우"}
        return f"{code_map.get(code,'기타')}, {t:.1f}℃"
    except Exception:
        return "날씨 정보를 가져올 수 없음"

weather = get_weather_summary()

# -----------------------------
# 공통 LLM 호출
# -----------------------------
def ask_gemini(prompt):
    if not GEMINI:
        return "⚠️ 모델 연결 오류 (API 키 확인 필요)"
    try:
        res = GEMINI.generate_content(prompt)
        if not res or not getattr(res, "text", None):
            return "⚠️ Gemini 응답이 비어 있습니다. 잠시 후 다시 시도해 주세요."
        return res.text.strip()
    except Exception as e:
        return f"Gemini 호출 오류: {e}"

# -----------------------------
# Q1 – 고객 특성 (데이터 근거 명시 + '가을' 계절 + 오류 수정)
# -----------------------------
def handle_q1(user_input):
    current_season = get_current_season()

    # 0. 파일 로드 확인
    if df_profiles is None or df_profiles.empty: 
        return "⚠️ Q1 고객 데이터 파일(analysis_cafe_profiles.csv)이 로드되지 않았습니다."
    
    # 1. 가맹점 매칭 (오류 수정 코드 적용)
    try:
        mct_nm_series = df_profiles["MCT_NM"]
        if mct_nm_series is None:
            return "⚠️ Q1 데이터의 'MCT_NM' 컬럼을 읽는 데 실패했습니다."
            
        # [오류 수정] .apply(str) + regex=False
        match = df_profiles[mct_nm_series.apply(str).str.contains(user_input, na=False, regex=False)] 
        
        if match.empty:
            names = re.findall(r'(\w+)', user_input)
            if names:
                match = df_profiles[mct_nm_series.apply(str).str.contains(names[0], na=False, regex=False)]
        
        if match.empty:
            return "⚠️ 해당 카페의 고객 데이터(Q1)가 없습니다. 가게 이름을 정확히 입력했는지 확인해주세요."
            
    except AttributeError as e:
        return f"Q1 분석 중 오류 발생: {e}. (Python 3.13 호환성 문제일 수 있습니다.)"
    except Exception as e:
        return f"Q1 알 수 없는 오류: {e}"

    # 2. 데이터 추출
    row = match.iloc[0]
    store = row.get("MCT_NM", "(미상)")
    
    age_cols = [c for c in row.index if c.startswith(("M12_MAL", "M12_FME"))]
    visit_cols = [c for c in row.index if c.startswith("RC_M1_SHC_")]
    
    def top_label(cols):
        try:
            s = row[cols].astype(float)
            return s.idxmax(), float(s.max())
        except Exception: return None, None
        
    age_col, age_val = top_label(age_cols)
    visit_col, visit_val = top_label(visit_cols)
    
    col2ko = {
        "M12_MAL_1020_RAT":"남성 10·20대","M12_MAL_30_RAT":"남성 30대","M12_MAL_40_RAT":"남성 40대",
        "M12_MAL_50_RAT":"남성 50대","M12_MAL_60_RAT":"남성 60대+",
        "M12_FME_1020_RAT":"여성 10·20대","M12_FME_30_RAT":"여성 30대","M12_FME_40_RAT":"여성 40대",
        "M12_FME_50_RAT":"여성 50대","M12_FME_60_RAT":"여성 60대+",
        "RC_M1_SHC_RSD_UE_CLN_RAT":"주거 고객","RC_M1_SHC_WP_UE_CLN_RAT":"직장 고객","RC_M1_SHC_FLP_UE_CLN_RAT":"유동 고객"
    }
    age_txt = col2ko.get(age_col, "주요 고객층 미확인")
    visit_txt = col2ko.get(visit_col, "주요 방문 목적 미확인")
    
    # 3. [핵심 1] LLM에게 보낼 프롬프트 생성 
    prompt = f"""
당신은 마케팅 AI '비밀상담사'입니다.
아래 [데이터 분석 근거]와 [날씨 정보]를 바탕으로 맞춤형 마케팅 전략을 제안하세요.

[데이터 분석 근거]
- 가맹점명: {store}
- 주요 고객층: {age_txt} ({age_val if age_val else 'N/A'}%)
- 주요 방문 목적: {visit_txt} ({visit_val if visit_val else 'N/A'}%)

[날씨 정보]
- 현재 계절: {current_season}
- 현재 날씨: {weather}

[요청 사항]
1. 위 근거를 조합해 [현재 계절({current_season})] 및 [현재 날씨({weather})]에 맞는 '{age_txt} {visit_txt}' 고객 대상 프로모션 2가지 제안.
2. 적합한 홍보 채널 1개와 이유 설명.
3. 홍보 문구(이모지, 해시태그 포함, 100자 이내) 제안.
"""
    
    # 4. [핵심 2] 사용자에게 먼저 보여줄 '분석 근거' 요약본 생성
    analysis_summary = f"""
데이터 분석이 완료되었습니다.

**[데이터 분석 근거]**
- **가맹점명:** {store}
- **주요 고객층:** {age_txt} ({age_val if age_val else 'N/A'}%)
- **주요 방문 목적:** {visit_txt} ({visit_val if visit_val else 'N/A'}%)

**[날씨 정보]**
- **현재 계절:** {current_season}
- **현재 날씨:** {weather}

---
위 분석을 바탕으로 맞춤형 마케팅 전략을 제안합니다.
"""
    
    # 5. [핵심 3] LLM 답변을 받아와서 요약본과 합쳐서 반환
    llm_response = ask_gemini(prompt)
    
    return f"{analysis_summary}\n\n{llm_response}"

# -----------------------------
# Q2 – 재방문률 마케팅: 가게 이름을 타겟팅하여 해당 매장의 업종, 평균 재방문률, 공략 대상 고객층 데이터를 추출
# -----------------------------
def handle_q2(user_input):
    match = df_revisit[df_revisit["MCT_NM"].astype(str).apply(lambda x: isinstance(x, str) and x in user_input)]
    if match.empty:
        return "⚠️ 해당 가맹점의 재방문률 데이터가 없습니다."
    r = match.iloc[0]
    store = r.get("MCT_NM", "(미상)")
    industry = r.get("map", "일반소매업")
    avg_rate = r.get("avg_re_rate", np.nan)
    target_seg = r.get("target_per_segment", "(미상)")

    prompt = f"""
당신은 재방문률을 높이는 '날씨 기반 마케팅 전문가'입니다.

[가맹점명] {store}
[업종] {industry}
[평균 재방문률] {avg_rate}%
[타깃 고객층] {target_seg}
[현재 날씨] {weather}

[요청 사항]
1. '{weather}'월의 날씨를 고려해 '{target_seg}' 고객층을 공략하는 전략 3가지.
2. 각 전략에 적합한 홍보 채널과 이유 설명.
3. 100자 이내의 홍보 문구 작성 (이모지, 해시태그 포함).
"""
    return ask_gemini(prompt)

# -----------------------------
# Q3 – 문제 진단 + 그래프: extract_scores 함수가 DIAGNOSIS_DETAILS라는 텍스트 컬럼에서 정규식(Regex)을 사용해 '고객 유치력 점수'(acq_score)와 '수익 창출력 점수'(profit_score)를 숫자로 추출한 뒤, 3.5점을 기준으로 두 점수를 조합하여 '스타 매장', '숨은 맛집형', '박리다매형', '위기 매장형' 4가지 diag_type으로 분류.
# -----------------------------
def handle_q3(user_input):
    match = df_diag[df_diag["MCT_NM"].astype(str).apply(lambda x: isinstance(x, str) and x in user_input)]
    if match.empty:
        return "⚠️ 해당 가맹점을 찾을 수 없습니다."
    r = match.iloc[0]
    store = r.get("MCT_NM", "(미상)")
    industry = r.get("map", "일반소매업")
    acq = r.get("acq_score", np.nan)
    prof = r.get("profit_score", np.nan)

    # 유형 분류
    if pd.notna(acq) and pd.notna(prof):
        if acq >= 3.5 and prof >= 3.5:
            diag_type = "⭐️ 스타 매장"
            diag_msg = "고객 확보와 수익성이 모두 우수합니다."
            color = "#4B64E6"
        elif acq < 3.5 and prof >= 3.5:
            diag_type = "🌿 숨은 맛집형"
            diag_msg = "재방문율이 높으니 신규 유입을 강화해 보세요."
            color = "#2E8B57"
        elif acq >= 3.5 and prof < 3.5:
            diag_type = "💸 박리다매형"
            diag_msg = "수익성을 높일 전략이 필요합니다."
            color = "#E67E22"
        else:
            diag_type = "⚠️ 위기 매장형"
            diag_msg = "고객 유입과 수익 모두 개선이 필요합니다."
            color = "#D35454"
    else:
        diag_type = "❕ 데이터 부족"
        diag_msg = "점수 데이터 부족."
        color = "gray"

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.axvline(3.5, color="gray", linestyle="--")
    ax.axhline(3.5, color="gray", linestyle="--")
    ax.scatter([acq], [prof], s=180, color=color, edgecolor="black", label="내 가게")
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0.5, 6.5)
    ax.set_xlabel("고객 유치력")
    ax.set_ylabel("수익 창출력")
    ax.text(1,6.2,"⭐️ 스타 매장",fontsize=10,color="#4B64E6")
    ax.text(4.1,6.2,"🌿 숨은 맛집",fontsize=10,color="#2E8B57")
    ax.text(1,1.0,"⚠️ 위기 매장",fontsize=10,color="#D35454")
    ax.text(4.1,1.0,"💸 박리다매",fontsize=10,color="#E67E22")
    ax.legend(loc="lower right")
    st.pyplot(fig, use_container_width=True)

    prompt = f"""
당신은 마케팅 컨설턴트입니다.
[{store}] 매장은 업종 [{industry}]이며, 고객 유치력 {acq}, 수익 창출력 {prof}점입니다.
매장 유형은 {diag_type} ({diag_msg})이며, 현재 날씨는 {weather}입니다.

[요청 사항]
1. 수익성과 고객 유입을 동시에 높일 실행 전략 3가지.
2. 각 전략에 적합한 마케팅 채널과 이유 설명.
3. 홍보 문구(이모지, 해시태그 포함, 100자 이내) 작성.
"""
    return ask_gemini(prompt)


# -----------------------------
# Q4 – 특정 매장 시계열 분석: df_ts_summary에서 매장 데이터를 찾아 '성수기 월'(high_months_str)과 '비수기 월'(low_months_str) 텍스트를 가져온 뒤 Matplotlib으로 그래프 시각화
# -----------------------------
def handle_q4(user_input):
    
    # 0. 파일 로드 확인
    if df_ts_summary.empty or df_ts_graph.empty:
        return "⚠️ Q4 '탐라' 분석 파일(analysis_timeseries_*.csv)이 로드되지 않았습니다."

    # 1. 분석 대상 찾기
    target_name = "탐라"
    target_id = "FEDAD7667E"
    
    # (질문에 '탐라' 또는 ID가 포함되어 있는지 확인)
    if target_name not in user_input and target_id not in user_input:
        return "⚠️ '탐라' 또는 'FEDAD7667E' 키워드가 포함된 질문을 해주세요."

    # 2. 요약(summary) 파일에서 데이터 찾기
    match_summary = df_ts_summary[df_ts_summary["MCT_ID"] == target_id]
    
    if match_summary.empty:
        return f"⚠️ {target_name} ({target_id})의 요약 데이터(analysis_timeseries_summary.csv)가 없습니다."

    r = match_summary.iloc[0]
    store = r.get("MCT_NM", target_name)
    store_id = r.get("MCT_ID", target_id)
    high_months_str = r.get("high_months", "알 수 없음") # 1점에 가까운 월
    low_months_str = r.get("low_months", "알 수 없음")  # 6점에 가까운 월

    # 3.  1단계: 시계열 그래프 출력 (st.pyplot)
    st.markdown(f"--- \n### 📊 {store} ({store_id}) 월별 **매출 랭크** 분석")
    
    df_graph = df_ts_graph[df_ts_graph["MCT_ID"] == target_id].sort_values(by="month_str")
    
    if not df_graph.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        x_labels = df_graph["month_str"].astype(str)
        
        # 'sales_rank' -> 'sales_score' (요약 파일 생성 로직 기준)
        y_values = df_graph["sales_score"] 
        
        ax.plot(x_labels, y_values, marker='o', linestyle='-', label="월별 매출 랭크")
        ax.set_title(f"{store} 매출 랭크 시계열", fontsize=14)
        ax.set_xlabel("월 (YYYYMM)")
        
        # Y축 레이블 변경 (1점=성수기)
        ax.set_ylabel("매출 랭크 (1점=성수기, 6점=비수기)") 
        
        # Y축 반전 (1점이 위로, 6점이 아래로)
        ax.invert_yaxis() 
        # Y축 눈금 1~6+ (데이터 최대값에 따라 유동적)
        ax.set_yticks(np.arange(1, max(7, int(y_values.max()) + 2))) 
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # X축 레이블 간격 조정 (약 8개 표시)
        tick_spacing = max(1, len(x_labels) // 8) 
        ax.set_xticks(x_labels[::tick_spacing])
        ax.set_xticklabels(x_labels[::tick_spacing], rotation=45)
        
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True) # 챗봇 응답창에 그래프 출력
    else:
        st.warning(f"⚠️ {store}의 그래프 데이터(analysis_timeseries_graph_data.csv)가 없습니다.")

    # 4. 2단계: AI의 자체 '근거' 제시 (st.markdown)
    # (LLM 호출 전에 챗봇이 스스로 분석한 내용을 먼저 출력)
    diagnosis_text = f"""
네, [{store}]의 2년간 **매출 랭크** 추이를 분석했습니다.

분석 결과, 랭크가 가장 **낮은(매출이 높은)** 달은 **{high_months_str}** 등 명확한 시즌에 집중되어 있습니다.
반면, 랭크가 **높은(매출이 낮은)** 달은 **{low_months_str}** 등 뚜렷한 행사가 없는 **'비시즌'**에는 매출 랭크가 하락하는 경향을 보입니다.
"""
    st.markdown(diagnosis_text)
    
    # 5. 3단계: LLM 프롬프트 생성 (LLM 답변만 반환)
    
    # [AI의 핵심 진단] 부분에 챗봇이 방금 제시한 근거를 그대로 넣어줌
    ai_diagnosis_prompt_part = f"""
"매출 랭크가 낮은(매출이 높은) {high_months_str}은(는) 각각 명절, 연말 등 명확한 시즌 특수로 인한 것입니다.
반면, {low_months_str} 등은 뚜렷한 시즌이 없어 매출 랭크가 높아져(매출이 낮아져) **'비시즌'**이 문제입니다."
"""
    
    prompt = f"""
당신은 마케팅 AI '비밀상담사'입니다. 아래 [AI의 데이터 분석 결과]를 바탕으로 '비시즌' 마케팅 전략을 제안해주세요.

[AI의 데이터 분석 결과]
- 가맹점명: {store} ({store_id})
- 매출 최성수기 Top 3 (랭크 1점에 가까운 월): {high_months_str}
- 매출 최비수기 Top 3 (랭크 6점에 가까운 월): {low_months_str}

[AI의 핵심 진단]
{ai_diagnosis_prompt_part}

[요청 사항]
위 [AI의 핵심 진단]에 따라,
매출이 낮은 '비시즌'(예: {low_months_str} 등)에 고객의 발길을 끌 수 있는
'시즌과 무관한' 또는 '인위적인 시즌(이벤트)'을 만드는
구체적인 마케팅 방안 2가지를 제안해주세요.

각 방안에 대해 홍보 문구(해시태그 포함, 100자 이내)도 함께 제안해주세요.
"""
    return ask_gemini(prompt)

# -----------------------------
# Q5 – 경쟁사 비교 분석 (신규)
# -----------------------------
def handle_q5(user_input):
    if df_q5_master.empty:
        return "⚠️ 경쟁 분석에 필요한 데이터 파일(big_data_set1_f.csv 또는 analysis_problem_diagnosis (1).csv)이 로드되지 않았습니다."

    # 1. 사용자 입력에서 가게 이름 추출 (예: "킴스** 경쟁 분석")
    match = re.search(r"(.+?)(\s*경쟁\s*분석)", user_input)
    if not match:
        return "⚠️ 가게 이름을 포함하여 '**가게** 경쟁 분석' 형식으로 질문해주세요."
    store_name = match.group(1).strip()

    # 2. 내 가게 정보 및 경쟁사 리스트 분석
    target_store_info = df_q5_master[df_q5_master['MCT_NM'] == store_name]
    if target_store_info.empty:
        return f"⚠️ 분석 대상인 '{store_name}' 정보를 데이터에서 찾을 수 없습니다."

    target_bzn = target_store_info['HPSN_MCT_BZN_CD_NM'].iloc[0]
    target_zcd = target_store_info['HPSN_MCT_ZCD_NM'].iloc[0]

    if pd.isna(target_bzn) or pd.isna(target_zcd):
        return f"⚠️ '{store_name}'의 상권 또는 업종 정보가 명확하지 않아 비교 분석을 할 수 없습니다."

    competitor_df = df_q5_master[
        (df_q5_master['HPSN_MCT_BZN_CD_NM'] == target_bzn) &
        (df_q5_master['HPSN_MCT_ZCD_NM'] == target_zcd)
    ].copy()

    df_final_list = competitor_df.sort_values(by='M12_SME_RY_SAA_PCE_RT', ascending=True).copy()
    df_final_list['Rank'] = range(1, len(df_final_list) + 1)

    my_store_rank_info = df_final_list[df_final_list['MCT_NM'] == store_name].squeeze()
    first_place_store_info = df_final_list.iloc[0]

    # 3.1단계: 분석 결과 표 출력
    st.subheader(f"📍 해당 상권 내 '{target_zcd}' 업종 순위")
    cols_to_show = ['Rank', 'MCT_NM', 'M12_SME_RY_SAA_PCE_RT']
    st.dataframe(df_final_list[cols_to_show].rename(columns={
        'Rank': '순위', 'MCT_NM': '가게 이름', 'M12_SME_RY_SAA_PCE_RT': '업종 내 매출 순위(%)'
    }))
    st.caption("※ '업종 내 매출 순위(%)'는 낮을수록 순위가 높습니다.")

    # 4. 2단계: LLM 프롬프트 생성 (핵심 격차 분석)
    customer_cols = [col for col in my_store_rank_info.index if 'RAT' in col and ('MAL' in col or 'FME' in col)]
    my_profile = my_store_rank_info[customer_cols].astype(float)
    first_profile = first_place_store_info[customer_cols].astype(float)
    gap = first_profile - my_profile
    biggest_gap_col = gap.idxmax()
    biggest_gap_value = gap.max()
    
    col_to_korean = {
        "M12_MAL_1020_RAT":"10·20대 남성 고객 비중", "M12_MAL_30_RAT":"30대 남성 고객 비중",
        "M12_MAL_40_RAT":"40대 남성 고객 비중", "M12_MAL_50_RAT":"50대 남성 고객 비중",
        "M12_MAL_60_RAT":"60대 이상 남성 고객 비중", "M12_FME_1020_RAT":"10·20대 여성 고객 비중",
        "M12_FME_30_RAT":"30대 여성 고객 비중", "M12_FME_40_RAT":"40대 여성 고객 비중",
        "M12_FME_50_RAT":"50대 여성 고객 비중", "M12_FME_60_RAT":"60대 이상 여성 고객 비중"
    }
    gap_label = col_to_korean.get(biggest_gap_col, biggest_gap_col)

    prompt = f"""
당신은 대한민국 최고의 마케팅 전략가 '비밀상담사'입니다.
'{my_store_rank_info['MCT_NM']}' 가게 사장님에게 1위 달성을 위한 구체적이고 실행 가능한 전략을 제안해야 합니다.

[데이터 분석 근거]
- 내 가게: {my_store_rank_info['MCT_NM']}
- 상권/업종: {target_bzn} / {target_zcd}
- 현재 상권 내 순위: {my_store_rank_info['Rank']}위 (매출 상위 {my_store_rank_info['M12_SME_RY_SAA_PCE_RT']:.1f}%)

[1위 매장 '{first_place_store_info['MCT_NM']}' 비교 분석]
- 1위 매장 매출: 상위 {first_place_store_info['M12_SME_RY_SAA_PCE_RT']:.1f}%
- **[핵심 격차]**: 1위 매장은 '{gap_label}'이 {first_place_store_info[biggest_gap_col]:.1f}%를 차지하는 반면, 
  내 가게는 {my_store_rank_info[biggest_gap_col]:.1f}%에 불과합니다. 즉, 1위 매장이 '{gap_label}' 고객층을 **{biggest_gap_value:.1f}%p 더 많이** 확보하고 있습니다.

[요청 사항]
1. 위 [핵심 격차]를 극복하기 위해, '{gap_label}' 고객층을 집중 공략할 수 있는 마케팅 전략 2가지를 제안해주세요.
2. 현재 계절인 **'선선한 가을 저녁'**이라는 시즌성을 반드시 고려하여, 현실적이고 창의적인 아이디어를 제시해주세요.
3. 각 전략마다 구체적인 실행 방안과 매력적인 홍보 문구(해시태그 포함)를 함께 제안해주세요.
"""
    # 5. 3단계: LLM 호출 및 결과 반환
    return ask_gemini(prompt)

# -----------------------------
# Streamlit 실행
# -----------------------------
col1, col2 = st.columns([1, 5])

with col1:
    st.image("img/chatbot_icon.png", width=150) 

with col2:
    st.title("비밀상담사 – 날씨별 맞춤형 전략 제공 챗봇") 

st.caption("💬 '고객 특성', '재방문율 마케팅', '문제 진단', '시계열 분석', '경쟁 분석' 키워드를 포함하면 자동 인식됩니다.")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for role, msg in st.session_state["messages"]:
    with st.chat_message(role):
        st.markdown(msg)

user_input = st.chat_input("사장님, 어떤 고민이 있으신가요?")

if user_input:
    st.session_state["messages"].append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("분석 중입니다…"):
            if "고객 특성" in user_input:
                answer = handle_q1(user_input)
            elif "재방문율 마케팅" in user_input:
                answer = handle_q2(user_input)
            elif "문제 진단" in user_input:
                answer = handle_q3(user_input)
            elif "월별 매출을 분석하고" in user_input:
                answer = handle_q4(user_input)
            elif "경쟁 분석" in user_input:
                answer = handle_q5(user_input)
            else:
                answer = "⚠️ 분석 주제를 인식할 수 없습니다.\n적절한 키워드를 포함해 주세요."
        st.markdown(answer)
        st.session_state["messages"].append(("assistant", answer))



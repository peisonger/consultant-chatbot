# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import google.generativeai as genai
import requests

# -----------------------------
# 기본 세팅
# -----------------------------
st.set_page_config(page_title="비밀상담사 – Q1/Q3 챗봇", page_icon="💡", layout="wide")
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# -----------------------------
# API KEY (secrets / env)
# -----------------------------
def _get_api_key():
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return os.getenv("GOOGLE_API_KEY", "")

API_KEY = _get_api_key()
GEMINI = None
if API_KEY:
    # 1. transport="rest" 옵션을 제거하고 라이브러리 기본값(gRPC)을 사용하도록 권장합니다.
    #    (이게 v1beta 충돌의 원인일 수 있습니다.)
    genai.configure(api_key=API_KEY) 
    
    try:
        # 2. 가장 표준적인 'latest' 버전의 모델 이름을 명시적으로 사용합니다.
        GEMINI = genai.GenerativeModel("gemini-2.5-flash")
        
    except Exception as e:
        # 3. 만약 모델 초기화 자체에서 오류가 난다면, streamlit 화면에 에러를 띄웁니다.
        st.error(f"❌ 모델 초기화에 실패했습니다: {e}")
        GEMINI = None
else:
    st.warning("⚠️ Gemini API 키가 없습니다. .streamlit/secrets.toml 또는 환경변수 GOOGLE_API_KEY 를 설정하세요.")

# -----------------------------
# 안전한 CSV 로드
# -----------------------------
@st.cache_data
def read_csv_safe(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

@st.cache_data
def load_all():
    df_profiles = read_csv_safe("data/analysis_cafe_profiles.csv")          # Q1
    df_diag     = read_csv_safe("data/analysis_problem_diagnosis.csv")      # Q3
    try:
        df_map  = read_csv_safe("data/map.csv")                              # 업종 통합용(있으면)
    except Exception:
        df_map  = pd.DataFrame(columns=["HPSN_MCT_ZCD_NM","map"])
    return df_profiles, df_diag, df_map

df_profiles, df_diag, df_map = load_all()

# 점수 자동 추출 (임시)
import re
def extract_scores(detail):
    if isinstance(detail, str):
        nums = re.findall(r"(\d\.\d+|\d+)", detail)
        nums = [float(x) for x in nums if 0 <= float(x) <= 7]
        if len(nums) >= 2:
            return nums[-2], nums[-1]
    return None, None

if "DIAGNOSIS_DETAILS" in df_diag.columns:
    df_diag["acq_score"], df_diag["profit_score"] = zip(*df_diag["DIAGNOSIS_DETAILS"].map(extract_scores))

# -----------------------------
# 업종 통합(map 붙이기)
# -----------------------------
if not df_map.empty and {"HPSN_MCT_ZCD_NM","map"}.issubset(df_map.columns):
    if "HPSN_MCT_ZCD_NM" in df_profiles.columns:
        df_profiles = df_profiles.merge(
            df_map[["HPSN_MCT_ZCD_NM","map"]], on="HPSN_MCT_ZCD_NM", how="left"
        )
    if "HPSN_MCT_ZCD_NM" in df_diag.columns:
        df_diag = df_diag.merge(
            df_map[["HPSN_MCT_ZCD_NM","map"]], on="HPSN_MCT_ZCD_NM", how="left"
        )
    INDUSTRIES = sorted(df_map["map"].dropna().unique().tolist())
else:
    # map.csv가 없으면 원본 업종 컬럼 사용
    if "HPSN_MCT_ZCD_NM" in df_diag.columns:
        INDUSTRIES = sorted(df_diag["HPSN_MCT_ZCD_NM"].dropna().unique().tolist())
    else:
        INDUSTRIES = ["전체"]

# -----------------------------
# 계절/날씨
# -----------------------------
SEASON_TO_MONTHS = {
    "봄":[3,4,5],"여름":[6,7,8],"가을":[9,10,11],"겨울":[12,1,2]
}

def get_weather_summary() -> str:
    """간단 현재 날씨 문장 (실패 시 요약문 반환)"""
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=37.56&longitude=126.97&current=temperature_2m,weathercode"
            "&timezone=Asia%2FSeoul"
        )
        r = requests.get(url, timeout=5).json()
        t = r["current"]["temperature_2m"]
        code = r["current"]["weathercode"]
        code_map = {0:"맑음",1:"대체로 맑음",2:"부분 흐림",3:"흐림",45:"안개",51:"이슬비",
                    61:"비",63:"강한 비",71:"눈",80:"소나기",95:"뇌우"}
        return f"{code_map.get(code,'기타 날씨')}, {t:.1f}℃"
    except Exception:
        return "날씨 정보를 가져올 수 없음"

# -----------------------------
# 사이드바: 모드/Q1/Q3 설정
# -----------------------------
st.title("🧠 비밀상담사 – Q1/Q2/Q3 챗봇")
st.caption("Colab 분석 결과(Q1/Q2/Q3)를 Streamlit 챗봇으로 연결")

with st.sidebar:
    st.header("⚙️ 상담 설정")
    mode = st.radio("질문 선택", ["Q1 – 고객 특성", "Q2 – 재방문률 마케팅", "Q3 – 문제 진단"], horizontal=False)
    sel_season = st.selectbox("계절", list(SEASON_TO_MONTHS.keys()))
    sel_industry = st.selectbox("업종(통합/원본)", INDUSTRIES)

# 필터링(업종)
if sel_industry != "전체":
    df_prof_sel = df_profiles[df_profiles.get("map", df_profiles.get("HPSN_MCT_ZCD_NM","")) == sel_industry]
    df_diag_sel = df_diag[df_diag.get("map", df_diag.get("HPSN_MCT_ZCD_NM","")) == sel_industry]
else:
    df_prof_sel, df_diag_sel = df_profiles.copy(), df_diag.copy()

# Q1: 카페 상호 / Q2, Q3: 가맹점 상호
with st.sidebar:
    if mode.startswith("Q1"):
        cafe_list = sorted(df_prof_sel.get("MCT_NM", pd.Series()).dropna().unique().tolist())
        sel_cafe = st.selectbox("카페 상호 (Q1)", cafe_list) if cafe_list else None
        st.caption("Q1: 고객 특성 + 계절/날씨로 프로모션/채널/문구 제안")

    elif mode.startswith("Q2"):
        shop_list = sorted(df_diag_sel.get("MCT_NM", pd.Series()).dropna().unique().tolist())
        sel_shop = st.selectbox("가맹점 상호 (Q2)", shop_list) if shop_list else None
        st.caption("Q2: 재방문률 + 날씨 기반 마케팅 전략/채널/문구 제안")

    elif mode.startswith("Q3"):
        shop_list = sorted(df_diag_sel.get("MCT_NM", pd.Series()).dropna().unique().tolist())
        sel_shop = st.selectbox("가맹점 상호 (Q3)", shop_list) if shop_list else None
        st.caption("Q3: 진단 결과 + 계절/날씨로 객단가 개선 전략/채널/문구 제안")

# -----------------------------
# 프롬프트(Q1)
# -----------------------------
def prompt_q1(row: pd.Series, season: str, weather: str) -> str:
    # 최고 비중 고객층/방문목적 추출
    age_cols   = [c for c in row.index if c.startswith(("M12_MAL","M12_FME"))]
    visit_cols = [c for c in row.index if c.startswith("RC_M1_SHC_")]

    def top_label(cols):
        try:
            s = row[cols].astype(float)
            return s.idxmax(), float(s.max())
        except Exception:
            return None, None

    age_k, age_v   = top_label(age_cols)
    visit_k, visit_v = top_label(visit_cols)

    col2ko = {
        "M12_MAL_1020_RAT":"남성 10·20대","M12_MAL_30_RAT":"남성 30대","M12_MAL_40_RAT":"남성 40대","M12_MAL_50_RAT":"남성 50대","M12_MAL_60_RAT":"남성 60대+",
        "M12_FME_1020_RAT":"여성 10·20대","M12_FME_30_RAT":"여성 30대","M12_FME_40_RAT":"여성 40대","M12_FME_50_RAT":"여성 50대","M12_FME_60_RAT":"여성 60대+",
        "RC_M1_SHC_RSD_UE_CLN_RAT":"주거 고객","RC_M1_SHC_WP_UE_CLN_RAT":"직장 고객","RC_M1_SHC_FLP_UE_CLN_RAT":"유동 고객"
    }
    age_txt   = col2ko.get(age_k, "주요 고객층 미확인")
    visit_txt = col2ko.get(visit_k,"주요 방문 목적 미확인")

    store = row.get("MCT_NM","(미상)")

    return f"""
당신은 마케팅 AI '비밀상담사'입니다.
아래 [데이터 분석 근거]와 [날씨 정보]를 바탕으로 사장님께 맞춤형 마케팅 전략을 제안하세요.

[데이터 분석 근거]
- 가맹점명: {store}
- 주요 고객층: {age_txt} ({age_v if age_v is not None else 'N/A'}%)
- 주요 방문목적: {visit_txt} ({visit_v if visit_v is not None else 'N/A'}%)

[날씨 정보]
- ({season}), 현재/예상 날씨: {weather}

[요청 사항]
1. 위 근거를 조합하여 '({season}/{weather})' 상황에 '({age_txt} {visit_txt})'의 지갑을 열 수 있는 프로모션 아이디어 2가지.
2. 이 프로모션을 홍보하기에 가장 효과적인 마케팅 채널 1개와 그 이유.
3. 방금 추천한 바로 그 채널에 올릴 홍보 문구(해시태그 포함) 2줄.
"""


# -----------------------------
# (Q2) 재방문률 마케팅 로직
# -----------------------------
if mode.startswith("Q2"):
    @st.cache_data
    def load_revisit():
        return read_csv_safe("data/under_30per_re_rate.csv")

    df_revisit = load_revisit()

    with st.sidebar:
        shop_list = sorted(df_revisit.get("MCT_NM", pd.Series()).dropna().unique().tolist())
        sel_shop = st.selectbox("가맹점 상호 (Q2)", shop_list) if shop_list else None

    # 데이터 조회
    if sel_shop:
        row = df_revisit[df_revisit["MCT_NM"] == sel_shop].head(1)
        if not row.empty:
            r = row.squeeze()
            map_type = r.get("map", "(미상)")
            avg_rate = r.get("avg_re_rate", np.nan)
            low_flag = r.get("is_low_mct_re_rate", False)
            worst_month = int(r.get("worst_month", 0))
            best_month = int(r.get("best_month", 0))
            target_seg = r.get("target_per_segment", "(미상)")
            industry_avg = r.get("industry_avg_re_rate", np.nan)

            # 기본 메시지
            if low_flag:
                st.warning(f"📉 {sel_shop}의 재방문률은 **30% 미만**입니다. 새로운 마케팅 전략이 필요합니다.")
            else:
                st.info(f"📊 재방문률이 가장 낮은 달은 **{worst_month}월**, 공략 대상 고객은 **{target_seg}**입니다.")

            # LLM 프롬프트 생성
            weather = get_weather_summary()
            prompt_q2 = f"""
당신은 재방문률을 높이는 '날씨 기반 마케팅 전문가'입니다.
아래 [데이터 분석 근거]와 [날씨 정보]를 바탕으로 사장님께 맞춤형 전략을 제안하세요.

[데이터 분석 근거]
- 가맹점명: {sel_shop}
- 업종: {map_type}
- 평균 재방문률: {avg_rate}%
- 산업 평균 대비: {industry_avg}%
- 주요 타겟 고객: {target_seg}
- 재방문률이 낮은 달: {worst_month}월
- 재방문률이 높은 달: {best_month}월

[날씨 정보]
- 현재 계절: {sel_season}
- 실시간 기상 요약: {weather}
- {worst_month}월 평균적으로 비율이 낮은 이유를 날씨 경향과 연결해 설명하세요.

[요청 사항]
1. {worst_month}월의 날씨 경향을 활용해 {target_seg} 고객을 공략할 구체적인 마케팅 전략 3가지 제시.
2. 각 전략에 적합한 마케팅 채널 1개씩 추천하고 이유를 설명.
3. 채널별 홍보 문구(해시태그 포함, 50자 이내) 작성.
"""

            st.markdown("### 💬 대화")
            for role, msg in st.session_state[key_history]:
                st.chat_message(role).markdown(msg)

            user_input = st.chat_input("사장님, 재방문률 관련 고민을 입력해주세요 (예: 7월 재방문률이 낮아요).")

            if user_input:
                st.session_state[key_history].append(("user", user_input))
                st.chat_message("user").markdown(user_input)
                with st.chat_message("assistant"):
                    with st.spinner("전략 생성 중입니다…"):
                        try:
                            res = GEMINI.generate_content(prompt_q2)
                            answer = res.text if res else "LLM 응답을 받을 수 없습니다."
                        except Exception as e:
                            answer = f"❌ 오류 발생: {e}"
                    st.markdown(answer)
                    st.session_state[key_history].append(("assistant", answer))
        else:
            st.warning("선택한 가맹점의 데이터가 없습니다.")

# -----------------------------
# 프롬프트(Q3) — 개선 버전
# -----------------------------
def prompt_q3(row: pd.Series, season: str, weather: str) -> str:
    store = row.get("MCT_NM", "(미상)")
    core = row.get("FINAL_DIAGNOSIS", "(미상)")
    detail = str(row.get("DIAGNOSIS_DETAILS", "(세부 없음)"))
    acq = row.get("acq_score", None)
    prof = row.get("profit_score", None)

    return f"""
당신은 마케팅 컨설턴트입니다.

[가게 진단 결과]
- 가맹점명: {store}
- 핵심 문제: {core}
- 세부 내용: {detail}
- 고객 유치력 점수: {acq if pd.notna(acq) else 'N/A'}
- 수익 창출력 점수: {prof if pd.notna(prof) else 'N/A'}

[현재 계절/날씨]
- ({season}) {weather}

[요청 사항]
1. 위 진단 내용을 한 문장으로 요약해주세요.
2. '{weather}' 날씨 조건에서 '객단가'를 높일 수 있는 메뉴/세트 구성 아이디어를 2가지 제안해주세요.
3. 제안한 아이디어를 홍보하기에 가장 효과적인 마케팅 채널 1개를 추천하고, 그 이유를 간단히 설명해주세요.
4. 방금 추천한 채널에 올릴 홍보 문구를 작성해주세요 (이모지/해시태그 포함, 100자 이내).
"""


# -----------------------------
# (Q3) 자동 진단 요약 + 4분면 시각화
# -----------------------------
if mode.startswith("Q3"):
    # 1️⃣ 자동 진단 텍스트 요약 (LLM 없이)
    if sel_shop and not df_diag_sel.empty:
        row = df_diag_sel[df_diag_sel["MCT_NM"] == sel_shop].head(1)
        if not row.empty:
            r = row.squeeze()
            acq = r.get("acq_score", np.nan)
            prof = r.get("profit_score", np.nan)

            # 기본 진단 카테고리
            if pd.notna(acq) and pd.notna(prof):
                if acq >= 3.5 and prof >= 3.5:
                    st.success("⭐️ 현재 매장은 **‘스타 매장’**으로 분류됩니다. 고객 확보와 수익성이 모두 우수합니다.")
                elif acq < 3.5 and prof >= 3.5:
                    st.info("🌿 현재 매장은 **‘숨은 맛집형’**입니다. 재방문율이 높으니 신규 유입을 강화해 보세요.")
                elif acq >= 3.5 and prof < 3.5:
                    st.warning("💸 현재 매장은 **‘박리다매형’**입니다. 수익성을 높일 전략이 필요합니다.")
                else:
                    st.error("⚠️ 현재 매장은 **‘위기 매장형’**입니다. 고객 유입과 수익 모두 개선이 필요합니다.")
            else:
                st.caption("점수 데이터가 없어 자동 진단 결과를 표시할 수 없습니다.")
        else:
            st.caption("선택한 가맹점의 진단 데이터가 없습니다.")

    # 2️⃣ 4분면 시각화
    with st.expander("📊 4분면 보기 (고객 유치력 × 수익 창출력)", expanded=True):
        if not df_diag_sel.empty and "MCT_NM" in df_diag_sel.columns and sel_shop:
            row = df_diag_sel[df_diag_sel["MCT_NM"] == sel_shop].head(1)
            if not row.empty:
                r = row.squeeze()
                acq = float(r.get("acq_score", np.nan))
                prof = float(r.get("profit_score", np.nan))
                if not np.isnan(acq) and not np.isnan(prof):
                    fig, ax = plt.subplots(figsize=(5.5, 5.5))
                    ax.axvline(3.5, color="gray", linestyle="--")
                    ax.axhline(3.5, color="gray", linestyle="--")
                    ax.scatter([acq], [prof], s=180, color="red", edgecolor="black", label="내 가게")
                    ax.set_xlim(0.5, 6.5)
                    ax.set_ylim(0.5, 6.5)
                    ax.set_xlabel("고객 유치력")
                    ax.set_ylabel("수익 창출력")

                    # 사분면 라벨
                    ax.text(1,6.2,"스타 매장",fontsize=10,color="#4B64E6")
                    ax.text(4.1,6.2,"숨은 맛집",fontsize=10,color="#2E8B57")
                    ax.text(1,1.0,"위기 매장",fontsize=10,color="#D35454")
                    ax.text(4.1,1.0,"박리다매",fontsize=10,color="#E67E22")
                    ax.legend(loc="lower right")
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.info("❕ 점수 데이터가 없어 그래프를 표시할 수 없습니다.")
        else:
            st.caption("좌측에서 가맹점을 선택하면 4분면 그래프를 볼 수 있습니다.")



# -----------------------------
# 말풍선 대화 상태
# -----------------------------
key_history = "history_q1" if mode.startswith("Q1") else "history_q3"
if key_history not in st.session_state:
    st.session_state[key_history] = []  # [(role, msg)]

st.markdown("### 💬 대화")
for role, msg in st.session_state[key_history]:
    st.chat_message(role).markdown(msg)

user_input = st.chat_input("사장님, 어떤 고민이 있으신가요? (예: 장마 시작되면 뭘 팔까요?)")

# -----------------------------
# 사용자 입력 처리 및 LLM 호출
# -----------------------------
if user_input:
    st.session_state[key_history].append(("user", user_input))
    st.chat_message("user").markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("분석 중입니다…"):
            weather = get_weather_summary()

            if mode.startswith("Q1"):
                if not sel_cafe:
                    answer = "좌측에서 카페 상호를 먼저 선택해 주세요."
                else:
                    row = df_prof_sel[df_prof_sel["MCT_NM"] == sel_cafe].head(1)
                    if row.empty:
                        answer = "해당 카페의 Q1 데이터가 없습니다."
                    else:
                        prompt = prompt_q1(row.squeeze(), sel_season, weather)
                        res = GEMINI.generate_content(prompt)
                        answer = res.text if res else "LLM 응답을 받을 수 없습니다."
            else:  # Q3
                if not sel_shop:
                    answer = "좌측에서 가맹점 상호를 먼저 선택해 주세요."
                else:
                    row = df_diag_sel[df_diag_sel["MCT_NM"] == sel_shop].head(1)
                    if row.empty:
                        answer = "해당 가맹점의 Q3 진단 데이터가 없습니다."
                    else:
                        prompt = prompt_q3(row.squeeze(), sel_season, weather)
                        res = GEMINI.generate_content(prompt)
                        answer = res.text if res else "LLM 응답을 받을 수 없습니다."

        st.markdown(answer)
        st.session_state[key_history].append(("assistant", answer))
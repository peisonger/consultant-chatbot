import pandas as pd
import numpy as np
import re

# '탐라' 가게의 ID와 이름을 사용자가 제공한 '탐라(FEDAD7667E)'를 기반으로 설정
TARGET_ID = 'FEDAD7667E'
TARGET_NAME = '탐라'
RANK_METRIC_COL = 'RC_M1_SAA' 

print(f"시작: '{TARGET_NAME}' ({TARGET_ID})의 '매출 랭크' 기반 요약 파일 생성 (수정된 로직)")

# 1. 원본 대용량 파일 로드
try:
    df_big = pd.read_csv("data/big_data_set2_f.csv")
except UnicodeDecodeError:
    df_big = pd.read_csv("data/big_data_set2_f.csv", encoding='cp949')
except FileNotFoundError:
    print("오류: 'data/big_data_set2_f.csv' 파일을 찾을 수 없습니다.")
    raise

# 2. 목표 가맹점 데이터 필터링
df_store = df_big[df_big['ENCODED_MCT'] == TARGET_ID].copy()

if df_store.empty:
    print(f"오류: big_data_set2_f.csv에서 ID '{TARGET_ID}'를 찾을 수 없습니다.")
else:
    print(f"'{TARGET_ID}' 데이터 {len(df_store)}행 필터링 완료")

    # 3. '매출 랭크' 데이터 추출 (예: '1_10%...' -> 1.0)
    try:
        # ⭐️ [수정] 정규식을 사용해 숫자만 정확히 추출 (예: '1_10%미만' -> 1)
        df_store['sales_score'] = df_store[RANK_METRIC_COL].str.extract(r'^(\d+)').astype(float)
        print(f"'sales_score' 컬럼 생성 완료. (예: '1_10%...' -> 1.0)")
    except Exception as e:
        print(f"오류: 'sales_score' 변환 중 문제 발생: {e}")
        df_store = pd.DataFrame(columns=['TA_YM', 'sales_score'])

    df_store_clean = df_store[['TA_YM', 'sales_score']].dropna().sort_values(by='TA_YM')
    print(f"클리닝 완료: {len(df_store_clean)}개월치 유효 '랭크' 데이터 확보")

    if not df_store_clean.empty:
        # 4. (파일 1) 그래프용 데이터 파일 저장
        df_graph_data = df_store_clean.copy()
        df_graph_data['MCT_ID'] = TARGET_ID
        df_graph_data.rename(columns={'TA_YM': 'month_str', 'sales_score': 'sales_score'}, inplace=True)
        
        df_graph_data.to_csv("data/analysis_timeseries_graph_data.csv", index=False, encoding='utf-8-sig')
        print(f"✅ 저장 완료 (그래프용): data/analysis_timeseries_graph_data.csv")

        # 5. (파일 2) 요약(진단)용 데이터 파일 생성
        
        # ⭐️ [핵심 수정] 1점이 '성수기' (매출 높음)
        # 1점에 가까울수록(ascending=True) 성수기
        df_sorted_high = df_store_clean.sort_values(by='sales_score', ascending=True)
        high_months_list = df_sorted_high.head(3)['TA_YM'].tolist()

        # ⭐️ [핵심 수정] 6점에 가까울수록(ascending=False) 비수기
        df_sorted_low = df_store_clean.sort_values(by='sales_score', ascending=False)
        low_months_list = df_sorted_low.head(3)['TA_YM'].tolist()

        def format_month(year_month):
            return f"{int(str(year_month)[-2:])}월"

        low_months_str = ", ".join(sorted(list(set([format_month(ym) for ym in low_months_list]))))
        high_months_str = ", ".join(sorted(list(set([format_month(ym) for ym in high_months_list]))))
        
        print(f"성수기 (1점에 가까운 월): {high_months_str}")
        print(f"비수기 (6점에 가까운 월): {low_months_str}")

        # 6. 요약 파일 저장
        df_summary = pd.DataFrame({
            'MCT_ID': [TARGET_ID],
            'MCT_NM': [TARGET_NAME],
            'high_months': [high_months_str],
            'low_months': [low_months_str]
        })
        
        df_summary.to_csv("data/analysis_timeseries_summary.csv", index=False, encoding='utf-8-sig')
        print(f"✅ 저장 완료 (요약용): data/analysis_timeseries_summary.csv")
    else:
        print("오류: 유효한 랭크 데이터가 없어 파일을 생성하지 못했습니다.")

print("\n--- 작업 완료 ---")
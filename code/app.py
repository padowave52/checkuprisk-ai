
import re
import json
from io import BytesIO
from pathlib import Path

import fitz
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CheckupRisk AI", page_icon="🩺", layout="wide")
st.title("🩺 CheckupRisk AI")
st.subheader("건강검진표 PDF 자동해석 테스트")
st.write(
    "건강검진표 PDF를 업로드하면 일반건강검진 수치표 페이지를 자동으로 찾고, "
    "성별·나이를 반영한 건강검진 해석, KNHANES 기반 AI 예측모델, 관리 우선순위를 제공합니다."
)
st.caption("※ 본 결과는 건강관리 참고용이며 의학적 진단이나 처방을 대체하지 않습니다.")

# =========================================================
# 1. PDF 텍스트 추출 및 검진 수치 추출 함수
# =========================================================

def normalize_value_line(text):
    if text is None:
        return None
    return str(text).strip().replace(" ", "").replace(",", "")


def find_pair_in_text(text):
    if text is None:
        return None, None
    t = normalize_value_line(text)
    m = re.search(r"(\d{1,4}(?:\.\d+)?)\/(\d{1,4}(?:\.\d+)?)", t)
    if m:
        return m.group(1), m.group(2)
    return None, None


def extract_first_number_after_keyword(line, keywords):
    if line is None:
        return None
    for keyword in keywords:
        if keyword in line:
            after = line.split(keyword, 1)[-1]
            nums = re.findall(r"\d+(?:\.\d+)?", after)
            if nums:
                return nums[0]
    return None


def is_numeric_line(text):
    if text is None:
        return False
    return bool(re.fullmatch(r"\d{1,4}(?:\.\d+)?", normalize_value_line(text)))


def find_index_contains_any(lines, keywords):
    for i, line in enumerate(lines):
        for keyword in keywords:
            if keyword in line:
                return i
    return None


def find_next_result_after(lines, keywords, max_lookahead=6):
    idx = find_index_contains_any(lines, keywords)
    if idx is None:
        return None

    same_line_value = extract_first_number_after_keyword(lines[idx], keywords)
    if same_line_value is not None:
        return normalize_value_line(same_line_value)

    for j in range(idx + 1, min(idx + 1 + max_lookahead, len(lines))):
        candidate = lines[j]
        if is_numeric_line(candidate):
            return normalize_value_line(candidate)
        left, right = find_pair_in_text(candidate)
        if left is not None and right is not None:
            return normalize_value_line(candidate)

    return None


def extract_egfr(lines):
    idx = find_index_contains_any(lines, ["신사구체여과율", "e-GFR", "eGFR"])
    if idx is None:
        return None

    for j in range(idx, min(idx + 6, len(lines))):
        nums = re.findall(r"\d+(?:\.\d+)?", lines[j])
        for num in nums:
            try:
                value = float(num)
            except Exception:
                continue
            if abs(value - 1.73) < 0.01:
                continue
            if 5 <= value <= 200:
                return str(int(value)) if value.is_integer() else str(value)
    return None


def find_general_checkup_page(page_texts):
    keyword_weights = {
        "키(cm)": 5, "몸무게": 5, "체질량지수": 5, "허리둘레": 4,
        "수축기/이완기": 5, "공복혈당": 5, "총콜레스테롤": 5, "총 콜레스테롤": 5,
        "고밀도 콜레스테롤": 4, "중성지방": 4, "저밀도 콜레스테롤": 4,
        "혈청 크레아티닌": 4, "신사구체여과율": 4, "e-GFR": 4,
        "에이에스티": 3, "AST": 3, "에이엘티": 3, "ALT": 3,
        "감마지티피": 3, "GTP": 3, "요단백": 4, "검사항목": 2, "참고치": 2,
    }
    negative_keywords = [
        "추가검진 결과서", "유방초음파", "갑상선초음파", "상복부초음파",
        "위암 검진", "유방암 검진", "Summary", "Recommendation",
        "초음파 검사", "판독소견", "Mammography", "Ultrasonography",
    ]

    best_page_num, best_text, best_score = None, "", -999

    for page_num, text in page_texts.items():
        score = 0
        for keyword, weight in keyword_weights.items():
            if keyword in text:
                score += weight
        for keyword in negative_keywords:
            if keyword in text:
                score -= 5
        if "공복혈당" in text and ("총콜레스테롤" in text or "총 콜레스테롤" in text):
            score += 5
        if "키(cm)" in text and "체질량지수" in text:
            score += 5
        if "혈청 크레아티닌" in text and ("e-GFR" in text or "신사구체여과율" in text):
            score += 5

        if score > best_score:
            best_page_num, best_text, best_score = page_num, text, score

    return best_page_num, best_text, best_score


def cut_general_checkup_table(lines):
    start_idx, end_idx = None, len(lines)

    for i, line in enumerate(lines):
        if "키(cm)" in line and "몸무게" in line:
            start_idx = i
            break

    if start_idx is None:
        for i, line in enumerate(lines):
            if "체질량지수" in line:
                start_idx = i
                break

    stop_keywords = [
        "간염검사", "B형 간염", "B형간염", "C형 간염", "C형간염",
        "정신건강", "우울증", "조기정신증", "골밀도검사", "폐기능검사",
        "인지기능장애", "노인신체기능검사", "노인기능평가", "예방접종", "배뇨장애",
    ]

    if start_idx is not None:
        for i in range(start_idx + 1, len(lines)):
            if any(keyword in lines[i] for keyword in stop_keywords):
                end_idx = i
                break
        return lines[start_idx:end_idx]

    return lines


def extract_checkup_values_from_pdf(uploaded_file):
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    page_texts = {}
    for page_num, page in enumerate(doc, start=1):
        page_texts[page_num] = page.get_text("text")

    result_page_num, target_text, page_score = find_general_checkup_page(page_texts)
    if not target_text.strip():
        target_text = "\n".join(page_texts.values())

    raw_lines = [line.strip() for line in target_text.splitlines() if line.strip()]
    lines = cut_general_checkup_table(raw_lines)

    result = {}

    # 키 / 몸무게
    height, weight = None, None
    for line in lines:
        if "키(cm)" in line and "몸무게" in line:
            height, weight = find_pair_in_text(line)
            if height is not None and weight is not None:
                break

    if height is None or weight is None:
        idx = find_index_contains_any(lines, ["키(cm)", "몸무게"])
        if idx is not None:
            for j in range(idx + 1, min(idx + 4, len(lines))):
                height, weight = find_pair_in_text(lines[j])
                if height is not None and weight is not None:
                    break

    result["height"] = height
    result["weight"] = weight

    result["BMI"] = find_next_result_after(lines, ["체질량지수", "BMI"])
    result["waist"] = find_next_result_after(lines, ["허리둘레"])

    # 시력
    vision_left, vision_right = None, None
    for line in lines:
        if "시력" in line:
            vision_left, vision_right = find_pair_in_text(line)
            if vision_left is not None and vision_right is not None:
                break

    if vision_left is None or vision_right is None:
        idx = find_index_contains_any(lines, ["시력"])
        if idx is not None:
            for j in range(idx + 1, min(idx + 4, len(lines))):
                vision_left, vision_right = find_pair_in_text(lines[j])
                if vision_left is not None and vision_right is not None:
                    break

    result["vision_left"] = vision_left
    result["vision_right"] = vision_right

    # 청력
    hearing_left, hearing_right = None, None
    for line in lines:
        if "청력" in line:
            hearing_left, hearing_right = find_pair_in_text(line)
            if hearing_left is not None and hearing_right is not None:
                break

            line_no_space = line.replace(" ", "")
            m = re.search(r"(정상|질환의심)\/(정상|질환의심)", line_no_space)
            if m:
                hearing_left, hearing_right = m.group(1), m.group(2)
                break

    result["hearing_left"] = hearing_left
    result["hearing_right"] = hearing_right

    # 혈압
    bp_value = None
    for line in lines:
        if "수축기/이완기" in line:
            left, right = find_pair_in_text(line)
            if left is not None and right is not None:
                bp_value = f"{left}/{right}"
                break

    if bp_value is None:
        idx = find_index_contains_any(lines, ["수축기/이완기", "(수축기/이완기)"])
        if idx is not None:
            for j in range(idx + 1, min(idx + 5, len(lines))):
                left, right = find_pair_in_text(lines[j])
                if left is not None and right is not None:
                    bp_value = f"{left}/{right}"
                    break

    systolic, diastolic = find_pair_in_text(bp_value)
    result["systolic_bp"] = systolic
    result["diastolic_bp"] = diastolic

    result["hemoglobin"] = find_next_result_after(lines, ["혈색소"])
    result["fasting_glucose"] = find_next_result_after(lines, ["공복혈당"])
    result["total_cholesterol"] = find_next_result_after(lines, ["총콜레스테롤", "총 콜레스테롤"])
    result["HDL"] = find_next_result_after(lines, ["고밀도 콜레스테롤", "HDL"])

    triglyceride = None
    for line in lines:
        compact = line.replace(" ", "")
        if compact.startswith("중성지방(") or "중성지방(mg/dL)" in compact:
            nums = re.findall(r"\d+(?:\.\d+)?", line)
            if nums:
                triglyceride = nums[0]
                break
    if triglyceride is None:
        triglyceride = find_next_result_after(lines, ["중성지방"])
    result["triglyceride"] = triglyceride

    result["LDL"] = find_next_result_after(lines, ["저밀도 콜레스테롤", "LDL"])
    result["creatinine"] = find_next_result_after(lines, ["혈청 크레아티닌", "크레아티닌"])
    result["eGFR"] = extract_egfr(lines)
    result["AST"] = find_next_result_after(lines, ["에이에스티", "AST"])
    result["ALT"] = find_next_result_after(lines, ["에이엘티", "ALT"])
    result["GGT"] = find_next_result_after(lines, ["감마지티피", "γGTP", "r-GTP", "GTP"])

    urine_value = None
    for line in lines:
        if "요단백" in line:
            if "☑" in line or "■" in line or "▣" in line:
                if "정상" in line:
                    urine_value = "정상"
                elif "경계" in line:
                    urine_value = "경계"
                elif "단백뇨" in line:
                    urine_value = "단백뇨 의심"
                break

            if "정상" in line:
                urine_value = "정상"
                break
            elif "경계" in line:
                urine_value = "경계"
                break
            elif "단백뇨" in line:
                urine_value = "단백뇨 의심"
                break

    result["urine_protein"] = urine_value

    return result, lines, result_page_num, page_score, raw_lines


# =========================================================
# 2. 이름 / 단위 매핑
# =========================================================

unit_map = {
    "height": "cm",
    "weight": "kg",
    "BMI": "kg/㎡",
    "waist": "cm",
    "vision_left": "-",
    "vision_right": "-",
    "hearing_left": "dB 또는 판정",
    "hearing_right": "dB 또는 판정",
    "systolic_bp": "mmHg",
    "diastolic_bp": "mmHg",
    "hemoglobin": "g/dL",
    "fasting_glucose": "mg/dL",
    "total_cholesterol": "mg/dL",
    "HDL": "mg/dL",
    "triglyceride": "mg/dL",
    "LDL": "mg/dL",
    "creatinine": "mg/dL",
    "eGFR": "mL/min/1.73㎡",
    "AST": "IU/L",
    "ALT": "IU/L",
    "GGT": "IU/L",
    "urine_protein": "-"
}

name_map = {
    "height": "키",
    "weight": "몸무게",
    "BMI": "체질량지수",
    "waist": "허리둘레",
    "vision_left": "좌안 시력",
    "vision_right": "우안 시력",
    "hearing_left": "좌측 청력",
    "hearing_right": "우측 청력",
    "systolic_bp": "수축기혈압",
    "diastolic_bp": "이완기혈압",
    "hemoglobin": "혈색소",
    "fasting_glucose": "공복혈당",
    "total_cholesterol": "총콜레스테롤",
    "HDL": "HDL 콜레스테롤",
    "triglyceride": "중성지방",
    "LDL": "LDL 콜레스테롤",
    "creatinine": "혈청 크레아티닌",
    "eGFR": "e-GFR",
    "AST": "AST",
    "ALT": "ALT",
    "GGT": "γGTP",
    "urine_protein": "요단백"
}

display_order = [
    "height",
    "weight",
    "BMI",
    "waist",
    "vision_left",
    "vision_right",
    "hearing_left",
    "hearing_right",
    "systolic_bp",
    "diastolic_bp",
    "hemoglobin",
    "fasting_glucose",
    "total_cholesterol",
    "HDL",
    "triglyceride",
    "LDL",
    "creatinine",
    "eGFR",
    "AST",
    "ALT",
    "GGT",
    "urine_protein"
]


# =========================================================
# 3. 판정 / 점수 / 연령 보정
# =========================================================

def to_float(value):
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None




# =========================
# KNHANES AI 예측모델 연결
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model"


@st.cache_resource
def load_ai_model_config():
    """
    clean 모델과 threshold 설정 불러오기
    필요 파일:
    - model\target_*_clean.pkl
    - model\clean_model_thresholds.json
    """
    threshold_path = MODEL_DIR / "clean_model_thresholds.json"

    if not threshold_path.exists():
        return None, None

    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold_config = json.load(f)

    models = {}

    for target in threshold_config.keys():
        model_path = MODEL_DIR / f"{target}_clean.pkl"

        if model_path.exists():
            models[target] = joblib.load(model_path)

    return threshold_config, models


def calculate_egfr_for_app(age, sex, creatinine):
    """
    앱에서 추출된 creatinine으로 eGFR 계산.
    CKD-EPI 2021 creatinine equation 형태.
    """
    if age is None or creatinine is None:
        return np.nan

    try:
        age = float(age)
        scr = float(creatinine)
    except Exception:
        return np.nan

    if scr <= 0:
        return np.nan

    if sex == "여":
        kappa = 0.7
        alpha = -0.241
        sex_factor = 1.012
    else:
        kappa = 0.9
        alpha = -0.302
        sex_factor = 1.0

    egfr = (
        142
        * (min(scr / kappa, 1) ** alpha)
        * (max(scr / kappa, 1) ** -1.200)
        * (0.9938 ** age)
        * sex_factor
    )

    return egfr


def urine_protein_to_code(value):
    """
    PDF 추출 요단백 텍스트를 KNHANES 모델 입력용 숫자로 변환.
    정상 = 1, 경계 = 2, 단백뇨 의심 = 3
    """
    if value is None:
        return np.nan

    value = str(value)

    if "정상" in value or "음성" in value:
        return 1
    elif "경계" in value:
        return 2
    elif "단백뇨" in value or "양성" in value:
        return 3
    else:
        return np.nan


def make_ai_input_features(extracted, sex, age):
    """
    PDF에서 추출한 값 + 입력한 성별/나이를 AI 모델 입력변수 형태로 변환.
    03_train_clean_prediction_models.py에서 저장한 feature 이름과 맞춰야 함.
    """
    sex_code = 1 if sex == "남" else 2
    female = 1 if sex == "여" else 0
    male = 1 if sex == "남" else 0

    creatinine = to_float(extracted.get("creatinine"))

    features = {
        "age": age,
        "sex_code": sex_code,
        "female": female,
        "male": male,

        "height": to_float(extracted.get("height")),
        "weight": to_float(extracted.get("weight")),
        "waist": to_float(extracted.get("waist")),
        "BMI": to_float(extracted.get("BMI")),

        "systolic_bp": to_float(extracted.get("systolic_bp")),
        "diastolic_bp": to_float(extracted.get("diastolic_bp")),

        "fasting_glucose": to_float(extracted.get("fasting_glucose")),
        "HbA1c": np.nan,

        "total_cholesterol": to_float(extracted.get("total_cholesterol")),
        "HDL": to_float(extracted.get("HDL")),
        "triglyceride": to_float(extracted.get("triglyceride")),
        "LDL": to_float(extracted.get("LDL")),

        "AST": to_float(extracted.get("AST")),
        "ALT": to_float(extracted.get("ALT")),
        "hemoglobin": to_float(extracted.get("hemoglobin")),

        "creatinine": creatinine,
        "eGFR_calc": calculate_egfr_for_app(age, sex, creatinine),
        "urine_protein": urine_protein_to_code(extracted.get("urine_protein")),
    }

    return features


def get_model_use_type(target):
    """
    모델 성능에 따라 앱에서의 활용 구분.
    - 주요 AI 예측모델: 성능이 비교적 좋은 모델
    - 보조 선별모델: AI 확률과 검진 수치 판정을 함께 보는 모델
    """
    main_models = [
        "target_obesity",
        "target_metabolic_syndrome",
        "target_dyslipidemia",
    ]

    support_models = [
        "target_diabetes",
        "target_hypertension",
        "target_anemia",
        "target_liver_dysfunction",
        "target_ckd",
    ]

    if target in main_models:
        return "주요 AI 예측모델"
    elif target in support_models:
        return "보조 선별모델"
    else:
        return "AI 예측모델"


def ai_grade_from_probability(probability, threshold):
    """
    AI 예측모델 등급.
    threshold 이상이면 주의, threshold보다 충분히 높으면 높음.
    """
    high_cutoff = min(threshold + 0.25, 0.70)

    if probability >= high_cutoff:
        return "높음"
    elif probability >= threshold:
        return "주의"
    else:
        return "낮음"


def ai_prediction_label(probability, threshold):
    if probability >= threshold:
        return "AI 예측 위험군"
    else:
        return "AI 예측 낮음"


def predict_ai_risks(extracted, sex, age):
    """
    clean 모델 + threshold를 이용해 AI 질병 위험도 출력.
    """
    threshold_config, models = load_ai_model_config()

    if threshold_config is None or models is None:
        return pd.DataFrame()

    input_features = make_ai_input_features(extracted, sex, age)

    rows = []

    for target, config in threshold_config.items():
        if target not in models:
            continue

        saved_model = models[target]
        model = saved_model["model"]
        feature_cols = saved_model["features"]

        X = pd.DataFrame([
            {feature: input_features.get(feature, np.nan) for feature in feature_cols}
        ])

        probability = float(model.predict_proba(X)[0, 1])
        threshold = float(config["threshold"])
        ai_grade = ai_grade_from_probability(probability, threshold)

        rows.append({
            "질병/위험군": config.get("target_name_kr", target),
            "AI 예측확률(%)": round(probability * 100, 1),
            "추천 threshold(%)": round(threshold * 100, 1),
            "AI 등급": ai_grade,
            "AI 판정": ai_prediction_label(probability, threshold),
            "모델 활용": get_model_use_type(target),
            "모델 기준": config.get("definition", ""),
            "사용 변수 수": len(feature_cols),
            "제거한 누수 변수": ", ".join(config.get("removed_features", [])),
        })

    ai_df = pd.DataFrame(rows)

    if not ai_df.empty:
        ai_df = ai_df.sort_values("AI 예측확률(%)", ascending=False).reset_index(drop=True)
        ai_df.insert(0, "순위", range(1, len(ai_df) + 1))

    return ai_df

def grade_to_score(grade):
    if grade == "정상":
        return 100
    elif grade == "주의":
        return 70
    elif grade == "위험":
        return 40
    elif grade == "확인필요":
        return 60
    return 60


def score_to_grade(score):
    if score >= 85:
        return "정상"
    elif score >= 65:
        return "주의"
    return "위험"


def get_age_group(age):
    if age >= 70:
        return "70세 이상"
    elif age >= 65:
        return "65~69세"
    elif age >= 50:
        return "50~64세"
    elif age >= 40:
        return "40~49세"
    return "40세 미만"


def age_context(domain, age):
    if age >= 65 and domain in ["혈압", "혈당", "지질", "신장기능", "심혈관"]:
        return " 65세 이상에서는 같은 수치라도 만성질환 및 합병증 위험 관리가 더 중요합니다."
    elif age >= 50 and domain in ["혈압", "혈당", "지질", "심혈관"]:
        return " 50세 이상에서는 혈압·혈당·지질 수치를 정기적으로 추적하는 것이 좋습니다."
    return ""


def judge_bmi(value):
    v = to_float(value)
    if v is None:
        return "확인필요", "BMI 값을 확인할 수 없습니다."
    if v < 18.5:
        return "주의", "저체중 범위입니다. 영양 상태와 체중 변화를 확인하는 것이 좋습니다."
    elif v < 25:
        return "정상", "BMI가 정상 범위입니다."
    elif v < 30:
        return "주의", "과체중 범위입니다. 체중 관리가 필요합니다."
    return "위험", "비만 범위입니다. 고혈압, 당뇨병, 이상지질혈증 위험과 관련될 수 있습니다."


def judge_waist(value, sex):
    v = to_float(value)
    if v is None:
        return "확인필요", "허리둘레 값을 확인할 수 없습니다."
    cutoff = 90 if sex == "남" else 85
    if v >= cutoff:
        return "주의", f"{sex}성 기준 복부비만 기준에 해당합니다. 대사증후군 위험요인으로 볼 수 있습니다."
    return "정상", "허리둘레가 기준 범위 내에 있습니다."


def judge_bp(sbp, dbp, age):
    s = to_float(sbp)
    d = to_float(dbp)
    if s is None or d is None:
        return "확인필요", "혈압 값을 확인할 수 없습니다."
    if s >= 140 or d >= 90:
        return "위험", "고혈압 의심 범위입니다. 혈압 재측정 및 진료 상담이 필요합니다." + age_context("혈압", age)
    elif s >= 120 or d >= 80:
        return "주의", "고혈압 전단계 범위입니다. 생활습관 관리와 정기적인 혈압 확인이 필요합니다." + age_context("혈압", age)
    return "정상", "혈압이 정상 범위입니다." + age_context("혈압", age)


def judge_hemoglobin(value, sex):
    v = to_float(value)
    if v is None:
        return "확인필요", "혈색소 값을 확인할 수 없습니다."
    lower = 13.0 if sex == "남" else 12.0
    upper = 16.5 if sex == "남" else 15.5
    if v < lower:
        return "주의", "혈색소가 낮아 빈혈 가능성을 확인할 필요가 있습니다."
    elif v > upper:
        return "주의", "혈색소가 참고치보다 높습니다. 필요 시 추가 확인이 필요합니다."
    return "정상", "혈색소가 참고치 범위 내에 있습니다."


def judge_glucose(value, age):
    v = to_float(value)
    if v is None:
        return "확인필요", "공복혈당 값을 확인할 수 없습니다."
    if v >= 126:
        return "위험", "당뇨병 의심 범위입니다. 공복혈당 재검사 또는 HbA1c 확인이 필요합니다." + age_context("혈당", age)
    elif v >= 100:
        return "주의", "공복혈당장애 의심 범위입니다. 당뇨병 예방을 위한 식습관과 운동 관리가 필요합니다." + age_context("혈당", age)
    return "정상", "공복혈당이 정상 범위입니다." + age_context("혈당", age)


def judge_total_cholesterol(value, age):
    v = to_float(value)
    if v is None:
        return "확인필요", "총콜레스테롤 값을 확인할 수 없습니다."
    if v >= 240:
        return "위험", "총콜레스테롤이 높아 이상지질혈증 의심 범위입니다." + age_context("지질", age)
    elif v >= 200:
        return "주의", "총콜레스테롤이 참고치보다 높습니다. 지질 관리가 필요합니다." + age_context("지질", age)
    return "정상", "총콜레스테롤이 참고치 범위 내에 있습니다." + age_context("지질", age)


def judge_hdl(value, age):
    v = to_float(value)
    if v is None:
        return "확인필요", "HDL 콜레스테롤 값을 확인할 수 없습니다."
    if v >= 60:
        return "정상", "HDL 콜레스테롤이 양호한 범위입니다." + age_context("지질", age)
    elif v >= 40:
        return "주의", "HDL 콜레스테롤이 충분히 높지 않습니다. 운동과 생활습관 관리가 필요합니다." + age_context("지질", age)
    return "위험", "HDL 콜레스테롤이 낮아 심혈관질환 위험요인으로 작용할 수 있습니다." + age_context("지질", age)


def judge_triglyceride(value, age):
    v = to_float(value)
    if v is None:
        return "확인필요", "중성지방 값을 확인할 수 없습니다."
    if v >= 200:
        return "위험", "중성지방이 높아 고중성지방혈증 의심 범위입니다." + age_context("지질", age)
    elif v >= 150:
        return "주의", "중성지방이 참고치보다 높습니다. 식습관과 체중 관리가 필요합니다." + age_context("지질", age)
    return "정상", "중성지방이 참고치 범위 내에 있습니다." + age_context("지질", age)


def judge_ldl(value, age):
    v = to_float(value)
    if v is None:
        return "확인필요", "LDL 콜레스테롤 값을 확인할 수 없습니다."
    if v >= 160:
        return "위험", "LDL 콜레스테롤이 높아 심혈관질환 위험요인으로 볼 수 있습니다." + age_context("지질", age)
    elif v >= 130:
        return "주의", "LDL 콜레스테롤이 참고치보다 높습니다. 지질 관리가 필요합니다." + age_context("지질", age)
    return "정상", "LDL 콜레스테롤이 참고치 범위 내에 있습니다." + age_context("지질", age)


def judge_creatinine(value):
    v = to_float(value)
    if v is None:
        return "확인필요", "크레아티닌 값을 확인할 수 없습니다."
    if v > 1.5:
        return "위험", "크레아티닌이 참고치보다 높아 신장기능 이상 가능성을 확인해야 합니다."
    return "정상", "크레아티닌이 참고치 범위 내에 있습니다."


def judge_egfr(value, age):
    v = to_float(value)
    if v is None:
        return "확인필요", "e-GFR 값을 확인할 수 없습니다."
    if v < 45:
        return "위험", "e-GFR이 낮아 신장기능 저하 가능성이 있습니다." + age_context("신장기능", age)
    elif v < 60:
        return "주의", "e-GFR이 60 미만으로 신장기능 확인이 필요합니다." + age_context("신장기능", age)
    elif v < 90 and age >= 65:
        return "정상", "e-GFR이 60 이상으로 기준상 정상 범위입니다. 다만 65세 이상에서는 신장기능 변화를 정기적으로 추적하는 것이 좋습니다."
    return "정상", "e-GFR이 60 이상으로 기준상 정상 범위입니다."


def judge_ast(value):
    v = to_float(value)
    if v is None:
        return "확인필요", "AST 값을 확인할 수 없습니다."
    if v > 40:
        return "주의", "AST가 참고치보다 높아 간기능 확인이 필요합니다."
    return "정상", "AST가 참고치 범위 내에 있습니다."


def judge_alt(value):
    v = to_float(value)
    if v is None:
        return "확인필요", "ALT 값을 확인할 수 없습니다."
    if v > 35:
        return "주의", "ALT가 참고치보다 높아 간기능 이상 가능성을 확인해야 합니다."
    return "정상", "ALT가 참고치 범위 내에 있습니다."


def judge_ggt(value, sex):
    v = to_float(value)
    if v is None:
        return "확인필요", "γGTP 값을 확인할 수 없습니다."
    cutoff = 63 if sex == "남" else 35
    if v > cutoff:
        return "주의", "γGTP가 참고치보다 높습니다. 음주, 지방간, 약물, 간기능 상태를 확인할 필요가 있습니다."
    return "정상", "γGTP가 참고치 범위 내에 있습니다."


def judge_urine_protein(value):
    if value is None:
        return "확인필요", "요단백 결과를 확인할 수 없습니다."
    value = str(value)
    if "단백뇨" in value or "양성" in value:
        return "위험", "단백뇨 의심 결과입니다. 신장기능 확인이 필요합니다."
    elif "경계" in value:
        return "주의", "요단백 경계 결과입니다. 추적 확인이 필요합니다."
    elif "정상" in value or "음성" in value:
        return "정상", "요단백 결과가 정상입니다."
    return "확인필요", "요단백 결과 해석이 필요합니다."


def judge_vision(left, right, age):
    l = to_float(left)
    r = to_float(right)
    if l is None or r is None:
        return "확인필요", "시력 값을 확인할 수 없습니다."
    if l < 0.5 or r < 0.5:
        desc = "시력 저하 가능성이 있어 안과 확인을 권장합니다."
        if age >= 50:
            desc += " 50세 이상에서는 백내장, 녹내장, 망막질환 등 정기 안과검진도 중요합니다."
        return "주의", desc
    desc = "시력이 비교적 양호한 범위입니다."
    if age >= 50:
        desc += " 다만 연령 증가에 따라 정기 안과검진을 유지하는 것이 좋습니다."
    return "정상", desc


def judge_hearing(left, right, age):
    if left is None or right is None:
        return "확인필요", "청력 값을 확인할 수 없습니다."

    age_msg = " 65세 이상에서는 노인성 난청 여부를 정기적으로 확인하는 것이 좋습니다." if age >= 65 else ""

    if str(left) == "질환의심" or str(right) == "질환의심":
        return "주의", "청력 질환의심 결과가 포함되어 있어 이비인후과 확인을 권장합니다." + age_msg

    if str(left) == "정상" and str(right) == "정상":
        return "정상", "청력 결과가 정상입니다." + age_msg

    l = to_float(left)
    r = to_float(right)

    if l is not None and r is not None:
        if l >= 40 or r >= 40:
            return "주의", "청력 수치가 40dB 이상으로 질환의심 기준에 해당할 수 있습니다." + age_msg
        return "정상", "청력 수치가 정상 범위입니다." + age_msg

    return "확인필요", "청력 결과 해석이 필요합니다." + age_msg


def age_penalty_for_domain(domain, score, age):
    """
    나이를 건강점수에 실제 반영하는 함수.
    진단 기준을 바꾸는 것이 아니라, 관리 우선순위 산정용 보정.
    이미 만점인 영역은 감점하지 않음.
    """
    if score >= 100:
        return 0

    if age >= 70:
        penalty_map = {
            "혈압": 10,
            "혈당": 8,
            "지질": 8,
            "신장기능": 10,
            "감각기능": 5,
            "비만": 5,
            "간기능": 3,
            "혈액": 3,
        }
    elif age >= 65:
        penalty_map = {
            "혈압": 8,
            "혈당": 6,
            "지질": 6,
            "신장기능": 8,
            "감각기능": 5,
            "비만": 4,
            "간기능": 3,
            "혈액": 3,
        }
    elif age >= 50:
        penalty_map = {
            "혈압": 5,
            "혈당": 4,
            "지질": 4,
            "신장기능": 4,
            "감각기능": 2,
            "비만": 2,
            "간기능": 2,
            "혈액": 2,
        }
    elif age >= 40:
        penalty_map = {
            "혈압": 2,
            "혈당": 2,
            "지질": 2,
            "신장기능": 2,
            "감각기능": 0,
            "비만": 0,
            "간기능": 0,
            "혈액": 0,
        }
    else:
        penalty_map = {}

    return penalty_map.get(domain, 0)


def analyze_checkup_values(extracted, sex, age):
    rows = []

    def add_row(domain, variable, item_name, value, unit, grade, interpretation):
        rows.append({
            "영역": domain,
            "항목": item_name,
            "model_variable": variable,
            "값": value,
            "단위": unit,
            "판정": grade,
            "설명": interpretation,
            "기본점수": grade_to_score(grade)
        })

    grade, desc = judge_bmi(extracted.get("BMI"))
    add_row("비만", "BMI", "체질량지수", extracted.get("BMI"), "kg/㎡", grade, desc)

    grade, desc = judge_waist(extracted.get("waist"), sex)
    add_row("비만", "waist", "허리둘레", extracted.get("waist"), "cm", grade, desc)

    grade, desc = judge_vision(extracted.get("vision_left"), extracted.get("vision_right"), age)
    add_row("감각기능", "vision", "시력", f"{extracted.get('vision_left')} / {extracted.get('vision_right')}", "-", grade, desc)

    grade, desc = judge_hearing(extracted.get("hearing_left"), extracted.get("hearing_right"), age)
    add_row("감각기능", "hearing", "청력", f"{extracted.get('hearing_left')} / {extracted.get('hearing_right')}", "-", grade, desc)

    grade, desc = judge_bp(extracted.get("systolic_bp"), extracted.get("diastolic_bp"), age)
    add_row("혈압", "blood_pressure", "혈압", f"{extracted.get('systolic_bp')} / {extracted.get('diastolic_bp')}", "mmHg", grade, desc)

    grade, desc = judge_hemoglobin(extracted.get("hemoglobin"), sex)
    add_row("혈액", "hemoglobin", "혈색소", extracted.get("hemoglobin"), "g/dL", grade, desc)

    grade, desc = judge_glucose(extracted.get("fasting_glucose"), age)
    add_row("혈당", "fasting_glucose", "공복혈당", extracted.get("fasting_glucose"), "mg/dL", grade, desc)

    grade, desc = judge_total_cholesterol(extracted.get("total_cholesterol"), age)
    add_row("지질", "total_cholesterol", "총콜레스테롤", extracted.get("total_cholesterol"), "mg/dL", grade, desc)

    grade, desc = judge_hdl(extracted.get("HDL"), age)
    add_row("지질", "HDL", "HDL 콜레스테롤", extracted.get("HDL"), "mg/dL", grade, desc)

    grade, desc = judge_triglyceride(extracted.get("triglyceride"), age)
    add_row("지질", "triglyceride", "중성지방", extracted.get("triglyceride"), "mg/dL", grade, desc)

    grade, desc = judge_ldl(extracted.get("LDL"), age)
    add_row("지질", "LDL", "LDL 콜레스테롤", extracted.get("LDL"), "mg/dL", grade, desc)

    grade, desc = judge_creatinine(extracted.get("creatinine"))
    add_row("신장기능", "creatinine", "혈청 크레아티닌", extracted.get("creatinine"), "mg/dL", grade, desc)

    grade, desc = judge_egfr(extracted.get("eGFR"), age)
    add_row("신장기능", "eGFR", "e-GFR", extracted.get("eGFR"), "mL/min/1.73㎡", grade, desc)

    grade, desc = judge_urine_protein(extracted.get("urine_protein"))
    add_row("신장기능", "urine_protein", "요단백", extracted.get("urine_protein"), "-", grade, desc)

    grade, desc = judge_ast(extracted.get("AST"))
    add_row("간기능", "AST", "AST", extracted.get("AST"), "IU/L", grade, desc)

    grade, desc = judge_alt(extracted.get("ALT"))
    add_row("간기능", "ALT", "ALT", extracted.get("ALT"), "IU/L", grade, desc)

    grade, desc = judge_ggt(extracted.get("GGT"), sex)
    add_row("간기능", "GGT", "γGTP", extracted.get("GGT"), "IU/L", grade, desc)

    analysis_df = pd.DataFrame(rows)

    domain_df = (
        analysis_df
        .groupby("영역", as_index=False)["기본점수"]
        .mean()
    )

    domain_df["기본점수"] = domain_df["기본점수"].round(1)
    domain_df["연령보정"] = domain_df.apply(
        lambda row: age_penalty_for_domain(row["영역"], row["기본점수"], age),
        axis=1
    )
    domain_df["점수"] = (domain_df["기본점수"] - domain_df["연령보정"]).clip(lower=0).round(1)
    domain_df["등급"] = domain_df["점수"].apply(score_to_grade)
    domain_df["연령반영"] = domain_df["연령보정"].apply(
        lambda x: "반영" if x > 0 else "해당 없음"
    )

    weights = {
        "비만": 0.15,
        "혈압": 0.20,
        "혈당": 0.20,
        "지질": 0.20,
        "간기능": 0.10,
        "신장기능": 0.15
    }

    total_score = 0
    total_weight = 0

    for domain, weight in weights.items():
        row = domain_df[domain_df["영역"] == domain]

        if not row.empty:
            total_score += float(row["점수"].iloc[0]) * weight
            total_weight += weight

    if total_weight > 0:
        total_score = round(total_score / total_weight, 1)
    else:
        total_score = 0

    total_grade = score_to_grade(total_score)

    return analysis_df, domain_df, total_score, total_grade


# =========================================================
# 4. 질병 위험도 계산
# =========================================================

def age_risk_points(age):
    if age >= 70:
        return 25
    elif age >= 65:
        return 20
    elif age >= 60:
        return 15
    elif age >= 50:
        return 10
    elif age >= 40:
        return 5
    else:
        return 0


def risk_grade(score):
    if score >= 70:
        return "높음"
    elif score >= 40:
        return "주의"
    return "낮음"


def clamp_score(score):
    return max(0, min(100, round(score, 1)))


def add_risk_row(rows, disease, score, factors, prevention):
    score = clamp_score(score)
    rows.append({
        "질병/위험군": disease,
        "위험도": score,
        "등급": risk_grade(score),
        "주요 관련 요인": ", ".join(factors) if factors else "뚜렷한 고위험 요인 없음",
        "예방 및 관리 방법": prevention
    })


def estimate_disease_risks(extracted, sex, age):
    bmi = to_float(extracted.get("BMI"))
    waist = to_float(extracted.get("waist"))
    sbp = to_float(extracted.get("systolic_bp"))
    dbp = to_float(extracted.get("diastolic_bp"))
    glucose = to_float(extracted.get("fasting_glucose"))
    tc = to_float(extracted.get("total_cholesterol"))
    hdl = to_float(extracted.get("HDL"))
    tg = to_float(extracted.get("triglyceride"))
    ldl = to_float(extracted.get("LDL"))
    creatinine = to_float(extracted.get("creatinine"))
    egfr = to_float(extracted.get("eGFR"))
    ast = to_float(extracted.get("AST"))
    alt = to_float(extracted.get("ALT"))
    ggt = to_float(extracted.get("GGT"))
    urine = str(extracted.get("urine_protein"))

    waist_cutoff = 90 if sex == "남" else 85
    age_point = age_risk_points(age)
    rows = []

    # 1. 당뇨병 위험
    factors = []
    score = age_point
    if glucose is not None:
        if glucose >= 126:
            score += 55
            factors.append(f"공복혈당 {glucose:g} mg/dL")
        elif glucose >= 100:
            score += 30
            factors.append(f"공복혈당 {glucose:g} mg/dL")
    if bmi is not None and bmi >= 25:
        score += 8 if bmi < 30 else 15
        factors.append(f"BMI {bmi:g}")
    if waist is not None and waist >= waist_cutoff:
        score += 10
        factors.append(f"허리둘레 {waist:g} cm")
    if tg is not None and tg >= 150:
        score += 5
        factors.append(f"중성지방 {tg:g} mg/dL")
    add_risk_row(
        rows,
        "당뇨병 위험",
        score,
        factors,
        "단 음료·과자·야식·과도한 탄수화물을 줄이고, 채소·단백질·통곡물 중심의 규칙적 식사를 권장합니다. 식후 걷기와 주 5회 이상 유산소 운동이 도움이 됩니다."
    )

    # 2. 고혈압 위험
    factors = []
    score = age_point
    if sbp is not None and dbp is not None:
        if sbp >= 140 or dbp >= 90:
            score += 60
            factors.append(f"혈압 {sbp:g}/{dbp:g} mmHg")
        elif sbp >= 120 or dbp >= 80:
            score += 30
            factors.append(f"혈압 {sbp:g}/{dbp:g} mmHg")
    if bmi is not None and bmi >= 25:
        score += 8
        factors.append(f"BMI {bmi:g}")
    if waist is not None and waist >= waist_cutoff:
        score += 8
        factors.append(f"허리둘레 {waist:g} cm")
    add_risk_row(
        rows,
        "고혈압 위험",
        score,
        factors,
        "저염식이 중요합니다. 국물, 찌개, 라면, 젓갈, 가공식품 섭취를 줄이고, 가정혈압을 주기적으로 측정하세요. 빠르게 걷기, 자전거 등 유산소 운동과 가벼운 근력운동을 병행하세요."
    )

    # 3. 이상지질혈증 위험
    factors = []
    score = age_point * 0.5
    if tc is not None:
        if tc >= 240:
            score += 30
            factors.append(f"총콜레스테롤 {tc:g} mg/dL")
        elif tc >= 200:
            score += 15
            factors.append(f"총콜레스테롤 {tc:g} mg/dL")
    if ldl is not None:
        if ldl >= 160:
            score += 30
            factors.append(f"LDL {ldl:g} mg/dL")
        elif ldl >= 130:
            score += 20
            factors.append(f"LDL {ldl:g} mg/dL")
    if tg is not None:
        if tg >= 200:
            score += 25
            factors.append(f"중성지방 {tg:g} mg/dL")
        elif tg >= 150:
            score += 15
            factors.append(f"중성지방 {tg:g} mg/dL")
    if hdl is not None and hdl < 40:
        score += 20
        factors.append(f"HDL {hdl:g} mg/dL")
    add_risk_row(
        rows,
        "이상지질혈증 위험",
        score,
        factors,
        "튀김, 가공육, 버터, 패스트푸드, 포화지방 섭취를 줄이고 생선, 채소, 통곡물, 견과류를 적절히 섭취하세요. 유산소 운동과 체중 관리가 도움이 됩니다."
    )

    # 4. 대사증후군 위험
    criteria = 0
    factors = []
    if waist is not None and waist >= waist_cutoff:
        criteria += 1
        factors.append(f"복부비만 기준 해당: 허리둘레 {waist:g} cm")
    if tg is not None and tg >= 150:
        criteria += 1
        factors.append(f"중성지방 {tg:g} mg/dL")
    if hdl is not None and hdl < 40:
        criteria += 1
        factors.append(f"HDL {hdl:g} mg/dL")
    if sbp is not None and dbp is not None and (sbp >= 130 or dbp >= 85):
        criteria += 1
        factors.append(f"혈압 {sbp:g}/{dbp:g} mmHg")
    if glucose is not None and glucose >= 100:
        criteria += 1
        factors.append(f"공복혈당 {glucose:g} mg/dL")
    score = criteria * 20 + (age_point * 0.5)
    add_risk_row(
        rows,
        "대사증후군 위험",
        score,
        factors,
        "복부비만, 혈압, 혈당, 지질 수치를 함께 관리해야 합니다. 식사량 조절, 단순당 줄이기, 주 5회 이상 유산소 운동, 주 2회 이상 근력운동을 권장합니다."
    )

    # 5. 심혈관질환 위험
    factors = []
    score = age_point * 1.4
    if sbp is not None and dbp is not None and (sbp >= 140 or dbp >= 90):
        score += 25
        factors.append(f"혈압 {sbp:g}/{dbp:g} mmHg")
    elif sbp is not None and dbp is not None and (sbp >= 120 or dbp >= 80):
        score += 12
        factors.append(f"혈압 {sbp:g}/{dbp:g} mmHg")
    if ldl is not None and ldl >= 130:
        score += 20
        factors.append(f"LDL {ldl:g} mg/dL")
    if tg is not None and tg >= 150:
        score += 10
        factors.append(f"중성지방 {tg:g} mg/dL")
    if glucose is not None and glucose >= 100:
        score += 10
        factors.append(f"공복혈당 {glucose:g} mg/dL")
    if bmi is not None and bmi >= 25:
        score += 5
        factors.append(f"BMI {bmi:g}")
    add_risk_row(
        rows,
        "심혈관질환 위험",
        score,
        factors,
        "혈압, LDL, 중성지방, 혈당을 함께 관리하세요. 저염식, 포화지방 제한, 규칙적 유산소 운동, 금연, 절주가 중요합니다."
    )

    # 6. 지방간/간기능 이상 위험
    factors = []
    score = age_point * 0.4
    if alt is not None and alt > 35:
        score += 25
        factors.append(f"ALT {alt:g} IU/L")
    if ast is not None and ast > 40:
        score += 20
        factors.append(f"AST {ast:g} IU/L")
    ggt_cutoff = 63 if sex == "남" else 35
    if ggt is not None and ggt > ggt_cutoff:
        score += 25
        factors.append(f"γGTP {ggt:g} IU/L")
    if bmi is not None and bmi >= 25:
        score += 10
        factors.append(f"BMI {bmi:g}")
    if tg is not None and tg >= 150:
        score += 10
        factors.append(f"중성지방 {tg:g} mg/dL")
    add_risk_row(
        rows,
        "지방간/간기능 이상 위험",
        score,
        factors,
        "음주를 줄이거나 중단하고, 야식·과식·단 음료를 줄이세요. 체중 관리와 규칙적인 운동이 도움이 되며, 간수치가 지속적으로 높으면 소화기내과 상담이 필요합니다."
    )

    # 7. 만성신장질환 위험
    factors = []
    score = age_point
    if egfr is not None:
        if egfr < 45:
            score += 55
            factors.append(f"e-GFR {egfr:g}")
        elif egfr < 60:
            score += 35
            factors.append(f"e-GFR {egfr:g}")
        elif egfr < 90 and age >= 65:
            score += 10
            factors.append(f"e-GFR {egfr:g}, 연령 {age}세")
    if creatinine is not None and creatinine > 1.5:
        score += 25
        factors.append(f"크레아티닌 {creatinine:g} mg/dL")
    if "단백뇨" in urine or "양성" in urine:
        score += 35
        factors.append("요단백 이상")
    if sbp is not None and dbp is not None and (sbp >= 140 or dbp >= 90):
        score += 10
        factors.append("혈압 상승")
    if glucose is not None and glucose >= 126:
        score += 10
        factors.append("혈당 상승")
    add_risk_row(
        rows,
        "만성신장질환 위험",
        score,
        factors,
        "혈압과 혈당 관리가 중요합니다. 저염식을 실천하고, e-GFR과 요단백을 정기적으로 확인하세요. 수치 저하나 단백뇨가 지속되면 신장내과 상담을 권장합니다."
    )

    risk_df = pd.DataFrame(rows)
    risk_df = risk_df.sort_values("위험도", ascending=False).reset_index(drop=True)
    risk_df.insert(0, "순위", range(1, len(risk_df) + 1))

    return risk_df


# =========================================================
# 5. 관리 우선순위 및 맞춤 관리 방법
# =========================================================

def get_problem_items(analysis_df, domain):
    sub = analysis_df[analysis_df["영역"] == domain].copy()
    problem = sub[sub["판정"].isin(["위험", "주의", "확인필요"])]

    if problem.empty:
        return "특이 위험 항목 없음"

    items = []
    for _, row in problem.iterrows():
        value_text = "" if pd.isna(row["값"]) else str(row["값"])
        unit_text = "" if pd.isna(row["단위"]) else str(row["단위"])
        items.append(f"{row['항목']} {value_text} {unit_text}({row['판정']})")

    return ", ".join(items)


def get_management_advice(domain, grade, problem_items):
    if domain == "혈압":
        if grade == "위험":
            return "혈압 재측정 후 지속적으로 높으면 내과 상담을 권장합니다. 국물, 찌개, 라면, 젓갈, 가공식품 등 짠 음식 섭취를 줄이고 저염식을 실천하세요. 빠르게 걷기, 자전거, 수영 같은 유산소 운동을 주 5회 이상 실천하고, 주 2회 이상 가벼운 근력운동을 병행하는 것이 좋습니다. 금연과 절주도 함께 필요합니다."
        elif grade == "주의":
            return "혈압이 상승 경향을 보이므로 가정혈압을 주기적으로 확인하세요. 저염식, 체중 관리, 규칙적인 유산소 운동이 필요합니다. 걷기 운동과 가벼운 근력운동을 병행하면 혈압 관리에 도움이 됩니다."
        return "혈압은 현재 양호합니다. 짠 음식 섭취를 과도하게 하지 않고, 규칙적인 운동과 정기적인 혈압 확인을 유지하세요."

    elif domain == "혈당":
        if grade == "위험":
            return "공복혈당이 높게 나온 경우 재검사 또는 HbA1c 확인이 필요합니다. 단 음료, 과자, 흰빵, 흰쌀밥 위주의 식사, 야식 섭취를 줄이세요. 채소, 단백질, 통곡물 중심으로 규칙적으로 식사하고, 식후 10~20분 걷기와 유산소 운동을 꾸준히 실천하는 것이 좋습니다."
        elif grade == "주의":
            return "공복혈당 관리가 필요합니다. 단순당, 탄산음료, 과자, 야식을 줄이고 식사를 규칙적으로 하세요. 식후 걷기, 빠르게 걷기, 자전거 같은 유산소 운동을 꾸준히 실천하는 것이 좋습니다."
        return "혈당은 현재 양호합니다. 규칙적인 식사, 적절한 탄수화물 섭취, 정기적인 운동 습관을 유지하세요."

    elif domain == "지질":
        if grade == "위험":
            return "총콜레스테롤, LDL, 중성지방 수치가 높으면 이상지질혈증과 심혈관질환 위험이 증가할 수 있습니다. 튀김, 가공육, 버터, 패스트푸드, 과도한 육류 지방 섭취를 줄이세요. 생선, 채소, 통곡물, 견과류를 적절히 섭취하고, 유산소 운동과 체중 관리를 병행하는 것이 좋습니다. 필요 시 내과 상담을 권장합니다."
        elif grade == "주의":
            return "지질 관리가 필요합니다. 튀김, 가공식품, 포화지방 섭취를 줄이고 생선, 채소, 통곡물 위주의 식사를 늘리세요. 빠르게 걷기, 자전거, 수영 같은 유산소 운동과 체중 관리가 도움이 됩니다."
        return "지질 수치는 현재 양호합니다. 포화지방 섭취를 과도하게 하지 않고, 채소와 단백질을 균형 있게 섭취하며 운동 습관을 유지하세요."

    elif domain == "비만":
        if grade == "위험":
            return "체중과 허리둘레 관리가 필요합니다. 야식, 단 음료, 과식, 잦은 외식을 줄이고 채소, 단백질, 통곡물 중심으로 식사하세요. 유산소 운동과 근력운동을 병행하면 체지방 감소와 대사 건강 개선에 도움이 됩니다."
        elif grade == "주의":
            return "체중 또는 허리둘레가 관리가 필요한 범위입니다. 식사량을 조절하고 야식과 단 음료를 줄이세요. 유산소 운동과 근력운동을 함께 하면 체중 관리에 효과적입니다."
        return "비만 관련 지표는 현재 양호합니다. 현재 체중과 허리둘레를 유지하도록 규칙적인 식사와 운동을 지속하세요."

    elif domain == "간기능":
        if grade in ["위험", "주의"]:
            return "간기능 수치가 높으면 음주, 지방간, 약물, 건강보조식품 등의 영향을 확인해야 합니다. 음주를 줄이거나 중단하고, 야식과 과식을 줄이며 체중 관리를 실천하세요. 수치가 지속적으로 높으면 소화기내과 상담을 권장합니다."
        return "간기능은 현재 양호합니다. 과음과 불필요한 약물·보충제 복용을 피하고, 규칙적인 식사와 운동을 유지하세요."

    elif domain == "신장기능":
        if grade in ["위험", "주의"]:
            return "신장기능 관리를 위해 혈압과 혈당을 함께 관리하는 것이 중요합니다. 짜게 먹는 습관을 줄이고, 요단백과 e-GFR을 정기적으로 확인하세요. e-GFR 저하나 단백뇨가 지속되면 신장내과 상담을 권장합니다."
        return "신장기능은 현재 양호합니다. 혈압과 혈당을 잘 관리하고, 저염식과 정기검진을 유지하세요."

    elif domain == "감각기능":
        if grade in ["위험", "주의", "확인필요"]:
            return "시력 또는 청력 저하 가능성이 있으면 안과 또는 이비인후과 검사를 권장합니다. 시야 흐림, 난청, 이명, 어지럼 등이 있으면 추가 확인이 필요합니다."
        return "시력과 청력은 현재 큰 이상이 없어 보입니다. 불편감이 생기면 안과 또는 이비인후과에서 확인하세요."

    elif domain == "혈액":
        if grade in ["위험", "주의"]:
            return "혈색소 이상이 있으면 빈혈, 영양 상태, 출혈 여부 등을 확인할 필요가 있습니다. 철분, 단백질, 비타민이 포함된 균형 잡힌 식사를 하고, 지속되면 내과 상담을 권장합니다."
        return "혈색소는 현재 참고치 범위입니다. 균형 잡힌 식사와 정기검진을 유지하세요."

    return "정기적인 건강검진과 생활습관 관리를 유지하세요."


def make_priority_table(domain_df, analysis_df):
    priority_df = domain_df.copy()
    priority_df = priority_df.sort_values("점수", ascending=True).reset_index(drop=True)
    priority_df.insert(0, "순위", range(1, len(priority_df) + 1))

    priority_df["관리 필요 항목"] = priority_df["영역"].apply(
        lambda domain: get_problem_items(analysis_df, domain)
    )

    priority_df["추천 관리 방법"] = priority_df.apply(
        lambda row: get_management_advice(
            row["영역"],
            row["등급"],
            row["관리 필요 항목"]
        ),
        axis=1
    )

    return priority_df


# =========================================================
# 6. Streamlit 화면
# =========================================================

st.subheader("기본 정보 입력")

col_sex, col_age = st.columns(2)

sex = col_sex.selectbox(
    "성별",
    ["여", "남"],
    index=0
)

age = col_age.number_input(
    "나이",
    min_value=0,
    max_value=120,
    value=25,
    step=1
)

uploaded_file = st.file_uploader(
    "건강검진표 PDF를 업로드하세요.",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success("PDF 업로드 완료")

    if st.button("PDF 분석 시작"):
        with st.spinner("PDF 전체 페이지를 탐색하고 일반건강검진 결과표를 찾는 중입니다..."):
            extracted, lines, result_page_num, page_score, raw_lines = extract_checkup_values_from_pdf(uploaded_file)

        st.info(
            f"일반건강검진 수치표로 추정되는 페이지: {result_page_num}페이지 "
            f"/ 탐색 점수: {page_score}"
        )

        df = pd.DataFrame([
            {
                "항목": name_map.get(key, key),
                "model_variable": key,
                "추출값": extracted.get(key),
                "단위": unit_map.get(key, "")
            }
            for key in display_order
        ])

        st.subheader("1. PDF 자동 추출 결과")
        st.dataframe(df, use_container_width=True)

        success_count = df["추출값"].notna().sum()
        total_count = len(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("추출 성공 항목 수", f"{success_count} / {total_count}")
        col2.metric("자동 선택 페이지", f"{result_page_num} 페이지")
        col3.metric("페이지 탐색 점수", page_score)

        st.subheader("2. 추출에 사용한 일반건강검진 수치표 줄")
        with st.expander("수치표 부분 보기"):
            for i, line in enumerate(lines, start=1):
                st.text(f"{i:03d}: {line}")

        st.subheader("3. 선택된 페이지 전체 원문 줄")
        with st.expander("선택 페이지 전체 원문 보기"):
            for i, line in enumerate(raw_lines, start=1):
                st.text(f"{i:03d}: {line}")

        analysis_df, domain_df, total_score, total_grade = analyze_checkup_values(extracted, sex, age)
        risk_df = estimate_disease_risks(extracted, sex, age)
        ai_risk_df = predict_ai_risks(extracted, sex, age)

        st.subheader("4. 건강검진 항목별 판정")

        st.dataframe(
            analysis_df[["영역", "항목", "값", "단위", "판정", "설명", "기본점수"]],
            use_container_width=True
        )

        st.subheader("5. 영역별 건강점수")

        col_score1, col_score2, col_score3 = st.columns(3)

        col_score1.metric("종합 건강점수", f"{total_score}점")
        col_score2.metric("종합 등급", total_grade)
        col_score3.metric("연령대", get_age_group(age))

        st.caption("※ 기본점수는 검사 수치 기준 점수이고, 점수는 나이를 반영한 관리 우선순위용 보정 점수입니다.")
        st.dataframe(domain_df, use_container_width=True)

        st.subheader("6. 검진 수치 기반 질병 위험도")
        st.caption("※ 검진 수치를 바탕으로 한 규칙 기반 위험도입니다. AI 예측모델 결과와 함께 해석합니다.")
        st.dataframe(
            risk_df[["순위", "질병/위험군", "위험도", "등급", "주요 관련 요인", "예방 및 관리 방법"]],
            use_container_width=True
        )

        st.subheader("7. KNHANES 기반 AI 질병 위험 예측")

        if ai_risk_df.empty:
            st.warning(
                "AI 예측모델을 불러오지 못했습니다. "
                "model 폴더에 target_*_clean.pkl 파일과 clean_model_thresholds.json 파일이 있는지 확인하세요."
            )
        else:
            st.caption(
                "※ 본 결과는 국민건강영양조사 기반 머신러닝 예측모델 결과입니다. "
                "정답 기준에 직접 사용된 변수는 입력에서 제외한 누수 제거 모델을 사용했습니다."
            )

            main_ai_df = ai_risk_df[ai_risk_df["모델 활용"] == "주요 AI 예측모델"].copy()
            support_ai_df = ai_risk_df[ai_risk_df["모델 활용"] == "보조 선별모델"].copy()

            simple_cols = [
                "질병/위험군",
                "AI 예측확률(%)",
            ]

            def show_probability_table_left(table_df):
                """
                예측확률 표를 화면 왼쪽에 좁게 배치.
                표가 전체 화면으로 길게 퍼지지 않게 해서 예측확률 값이 왼쪽에 가깝게 보이도록 함.
                """
                show_df = table_df[simple_cols].reset_index(drop=True)

                left_area, right_blank = st.columns([0.46, 0.54])

                with left_area:
                    st.dataframe(
                        show_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "질병/위험군": st.column_config.TextColumn(
                                "질병/위험군",
                                width="medium"
                            ),
                            "AI 예측확률(%)": st.column_config.NumberColumn(
                                "AI 예측확률(%)",
                                format="%.1f",
                                width="small"
                            ),
                        }
                    )

            st.markdown("### 7-1. 주요 AI 예측모델 결과")
            if main_ai_df.empty:
                st.info("주요 AI 예측모델 결과가 없습니다.")
            else:
                show_probability_table_left(main_ai_df)

            st.markdown("### 7-2. 보조 선별모델 결과")
            if support_ai_df.empty:
                st.info("보조 선별모델 결과가 없습니다.")
            else:
                show_probability_table_left(support_ai_df)

            st.markdown("### AI 예측확률 TOP 3")
            top3_ai = ai_risk_df.head(3)

            for i, (_, row) in enumerate(top3_ai.iterrows(), start=1):
                left_top3_area, right_top3_blank = st.columns([0.46, 0.54])

                with left_top3_area:
                    st.markdown(
                        f"""
**{i}순위. {row['질병/위험군']}**

- AI 예측확률: **{row['AI 예측확률(%)']}%**
"""
                    )

        st.subheader("8. 관리 우선순위 및 맞춤 관리 방법")

        priority_df = make_priority_table(domain_df, analysis_df)

        st.dataframe(
            priority_df[["순위", "영역", "기본점수", "연령보정", "점수", "등급", "관리 필요 항목", "추천 관리 방법"]],
            use_container_width=True
        )

        st.markdown("### 관리 우선순위 요약")

        for _, row in priority_df.iterrows():
            st.markdown(
                f"""
**{int(row['순위'])}순위. {row['영역']} 관리**

- 기본점수: **{row['기본점수']}점**
- 연령보정: **-{row['연령보정']}점**
- 최종점수: **{row['점수']}점**
- 등급: **{row['등급']}**
- 관리 필요 항목: {row['관리 필요 항목']}
- 추천 관리 방법: {row['추천 관리 방법']}
"""
            )

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="extracted_values")
            analysis_df.to_excel(writer, index=False, sheet_name="analysis")
            domain_df.to_excel(writer, index=False, sheet_name="domain_scores")
            risk_df.to_excel(writer, index=False, sheet_name="rule_based_risk")
            if not ai_risk_df.empty:
                ai_risk_df.to_excel(writer, index=False, sheet_name="ai_model_risk")
            priority_df.to_excel(writer, index=False, sheet_name="priority_advice")

        st.download_button(
            label="추출 및 판정 결과 엑셀 다운로드",
            data=output.getvalue(),
            file_name="checkup_analysis_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.warning(
            "현재 단계는 PDF 자동추출, 검진 수치 기반 해석, KNHANES 기반 AI 예측모델 연결 단계입니다. "
            "AI 예측결과는 질병별 예측확률 중심으로 제공합니다."
        )

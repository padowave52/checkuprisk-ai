from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


warnings.filterwarnings("ignore")


# =========================
# 경로 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "knhanes"
OUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "model"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("DATA_DIR:", DATA_DIR)
print("OUT_DIR:", OUT_DIR)
print("MODEL_DIR:", MODEL_DIR)


# =========================
# 데이터 파일 자동 찾기 / 불러오기
# =========================

def find_data_file(data_dir):
    candidate_files = []

    # 국민건강영양조사 원시자료는 sas7bdat를 우선 사용
    for ext in ["*.sas7bdat", "*.csv", "*.xlsx", "*.xls"]:
        candidate_files.extend(data_dir.glob(ext))

    if len(candidate_files) == 0:
        raise FileNotFoundError(
            "data\\knhanes 폴더 안에 국민건강영양조사 원시자료 파일이 없습니다."
        )

    return candidate_files[0]


def load_knhanes_data(data_path):
    suffix = data_path.suffix.lower()

    if suffix == ".sas7bdat":
        try:
            df = pd.read_sas(data_path, format="sas7bdat", encoding="cp949")
        except UnicodeDecodeError:
            df = pd.read_sas(data_path, format="sas7bdat", encoding="utf-8")

    elif suffix == ".csv":
        try:
            df = pd.read_csv(data_path, encoding="cp949")
        except UnicodeDecodeError:
            df = pd.read_csv(data_path, encoding="utf-8")

    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(data_path)

    else:
        raise ValueError("지원하지 않는 파일 형식입니다.")

    wrong_columns = {"pdf_id", "item_name", "value", "unit", "model_variable"}

    if wrong_columns.issubset(set(df.columns)):
        raise ValueError(
            "현재 읽은 파일은 국민건강영양조사 원시자료가 아니라 "
            "건강검진표 PDF 추출 결과 파일입니다. "
            "data\\knhanes 폴더에는 hn24_all.sas7bdat 같은 원시자료만 넣어주세요."
        )

    return df


DATA_PATH = find_data_file(DATA_DIR)

print("\n선택된 데이터 파일:", DATA_PATH.name)
print("전체 경로:", DATA_PATH)

df = load_knhanes_data(DATA_PATH)

print("\n원본 데이터 크기:", df.shape)
print("원본 변수 수:", len(df.columns))


# =========================
# KNHANES 변수명 설정
# =========================

COLS = {
    "age": "age",
    "sex": "sex",
    "height": "HE_ht",
    "weight": "HE_wt",
    "waist": "HE_wc",
    "BMI": "HE_BMI",
    "systolic_bp": "HE_sbp",
    "diastolic_bp": "HE_dbp",
    "fasting_glucose": "HE_glu",
    "HbA1c": "HE_HbA1c",
    "total_cholesterol": "HE_chol",
    "HDL": "HE_HDL_st2",
    "triglyceride": "HE_TG",
    "LDL": "HE_LDL_drct",
    "AST": "HE_ast",
    "ALT": "HE_alt",
    "hemoglobin": "HE_HB",
    "creatinine": "HE_crea",
    "urine_protein": "HE_Upro",
}


def exists(col):
    return col in df.columns


available_cols = {k: v for k, v in COLS.items() if exists(v)}
missing_cols = {k: v for k, v in COLS.items() if not exists(v)}

print("\n[사용 가능한 변수]")
for k, v in available_cols.items():
    print(f"{k}: {v}")

print("\n[없는 변수]")
for k, v in missing_cols.items():
    print(f"{k}: {v}")


# =========================
# 전처리 데이터 생성
# =========================

work = pd.DataFrame(index=df.index)

for model_name, col_name in available_cols.items():
    work[model_name] = df[col_name]


def to_numeric_series(s):
    return pd.to_numeric(s, errors="coerce")


for col in work.columns:
    work[col] = to_numeric_series(work[col])


# =========================
# 성별 처리
# =========================
# KNHANES에서 보통 sex: 1=남자, 2=여자

if "sex" not in work.columns:
    raise ValueError("sex 변수가 없습니다. 성별 변수명을 다시 확인하세요.")

work["sex_code"] = work["sex"]
work["female"] = np.where(work["sex_code"] == 2, 1, 0)
work["male"] = np.where(work["sex_code"] == 1, 1, 0)


# =========================
# eGFR 계산
# =========================
# CKD-EPI 2021 creatinine equation 형태
# creatinine 단위가 mg/dL인 경우를 가정

def calculate_egfr_2021(row):
    age = row.get("age", np.nan)
    scr = row.get("creatinine", np.nan)
    female = row.get("female", np.nan)

    if pd.isna(age) or pd.isna(scr) or pd.isna(female):
        return np.nan

    if scr <= 0:
        return np.nan

    if female == 1:
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


if "creatinine" in work.columns and "age" in work.columns:
    work["eGFR_calc"] = work.apply(calculate_egfr_2021, axis=1)
else:
    work["eGFR_calc"] = np.nan


# =========================
# 라벨 생성
# =========================

label_info = {}

# 1. 당뇨병 위험
# 공복혈당 >=126 또는 HbA1c >=6.5
diabetes_known = pd.Series(False, index=work.index)
diabetes_positive = pd.Series(False, index=work.index)

if "fasting_glucose" in work.columns:
    diabetes_known = diabetes_known | work["fasting_glucose"].notna()
    diabetes_positive = diabetes_positive | (work["fasting_glucose"] >= 126)

if "HbA1c" in work.columns:
    diabetes_known = diabetes_known | work["HbA1c"].notna()
    diabetes_positive = diabetes_positive | (work["HbA1c"] >= 6.5)

work["target_diabetes"] = np.where(diabetes_known, diabetes_positive.astype(int), np.nan)
label_info["target_diabetes"] = "공복혈당 >=126 또는 HbA1c >=6.5"


# 2. 고혈압 위험
# 수축기 >=140 또는 이완기 >=90
if "systolic_bp" in work.columns and "diastolic_bp" in work.columns:
    bp_known = work["systolic_bp"].notna() & work["diastolic_bp"].notna()
    bp_positive = (work["systolic_bp"] >= 140) | (work["diastolic_bp"] >= 90)
    work["target_hypertension"] = np.where(bp_known, bp_positive.astype(int), np.nan)
else:
    work["target_hypertension"] = np.nan

label_info["target_hypertension"] = "수축기혈압 >=140 또는 이완기혈압 >=90"


# 3. 이상지질혈증 위험
# 총콜레스테롤 >=240 또는 LDL >=160 또는 TG >=200 또는 HDL <40
lipid_known = pd.Series(False, index=work.index)
lipid_positive = pd.Series(False, index=work.index)

if "total_cholesterol" in work.columns:
    lipid_known = lipid_known | work["total_cholesterol"].notna()
    lipid_positive = lipid_positive | (work["total_cholesterol"] >= 240)

if "LDL" in work.columns:
    lipid_known = lipid_known | work["LDL"].notna()
    lipid_positive = lipid_positive | (work["LDL"] >= 160)

if "triglyceride" in work.columns:
    lipid_known = lipid_known | work["triglyceride"].notna()
    lipid_positive = lipid_positive | (work["triglyceride"] >= 200)

if "HDL" in work.columns:
    lipid_known = lipid_known | work["HDL"].notna()
    lipid_positive = lipid_positive | (work["HDL"] < 40)

work["target_dyslipidemia"] = np.where(lipid_known, lipid_positive.astype(int), np.nan)
label_info["target_dyslipidemia"] = "총콜레스테롤 >=240 또는 LDL >=160 또는 TG >=200 또는 HDL <40"


# 4. 대사증후군 위험
# 허리둘레, TG, HDL, 혈압, 공복혈당 5개 중 3개 이상
metabolic_required = [
    "waist",
    "triglyceride",
    "HDL",
    "systolic_bp",
    "diastolic_bp",
    "fasting_glucose",
    "sex_code",
]

if all(col in work.columns for col in metabolic_required):
    metabolic_known = work[metabolic_required].notna().all(axis=1)

    waist_positive = (
        ((work["sex_code"] == 1) & (work["waist"] >= 90))
        | ((work["sex_code"] == 2) & (work["waist"] >= 85))
    )

    tg_positive = work["triglyceride"] >= 150

    hdl_positive = (
        ((work["sex_code"] == 1) & (work["HDL"] < 40))
        | ((work["sex_code"] == 2) & (work["HDL"] < 50))
    )

    bp_positive = (work["systolic_bp"] >= 130) | (work["diastolic_bp"] >= 85)
    glucose_positive = work["fasting_glucose"] >= 100

    metabolic_count = (
        waist_positive.astype(int)
        + tg_positive.astype(int)
        + hdl_positive.astype(int)
        + bp_positive.astype(int)
        + glucose_positive.astype(int)
    )

    work["target_metabolic_syndrome"] = np.where(
        metabolic_known,
        (metabolic_count >= 3).astype(int),
        np.nan
    )
else:
    work["target_metabolic_syndrome"] = np.nan

label_info["target_metabolic_syndrome"] = "대사증후군 구성요소 5개 중 3개 이상"


# 5. 지방간/간기능 이상 위험
# AST >40 또는 ALT >35
liver_known = pd.Series(False, index=work.index)
liver_positive = pd.Series(False, index=work.index)

if "AST" in work.columns:
    liver_known = liver_known | work["AST"].notna()
    liver_positive = liver_positive | (work["AST"] > 40)

if "ALT" in work.columns:
    liver_known = liver_known | work["ALT"].notna()
    liver_positive = liver_positive | (work["ALT"] > 35)

work["target_liver_dysfunction"] = np.where(liver_known, liver_positive.astype(int), np.nan)
label_info["target_liver_dysfunction"] = "AST >40 또는 ALT >35"


# 6. 만성신장질환 위험
# 계산 eGFR <60 또는 요단백 이상
ckd_known = pd.Series(False, index=work.index)
ckd_positive = pd.Series(False, index=work.index)

if "eGFR_calc" in work.columns:
    ckd_known = ckd_known | work["eGFR_calc"].notna()
    ckd_positive = ckd_positive | (work["eGFR_calc"] < 60)

if "urine_protein" in work.columns:
    ckd_known = ckd_known | work["urine_protein"].notna()
    # HE_Upro 코딩은 자료에 따라 다를 수 있어 2 이상을 양성 의심으로 처리
    ckd_positive = ckd_positive | (work["urine_protein"] >= 2)

work["target_ckd"] = np.where(ckd_known, ckd_positive.astype(int), np.nan)
label_info["target_ckd"] = "계산 eGFR <60 또는 요단백 코드 >=2"


# 7. 빈혈 위험
# 남자 Hb <13, 여자 Hb <12
if "hemoglobin" in work.columns:
    anemia_known = work["hemoglobin"].notna() & work["sex_code"].notna()

    anemia_positive = (
        ((work["sex_code"] == 1) & (work["hemoglobin"] < 13))
        | ((work["sex_code"] == 2) & (work["hemoglobin"] < 12))
    )

    work["target_anemia"] = np.where(anemia_known, anemia_positive.astype(int), np.nan)
else:
    work["target_anemia"] = np.nan

label_info["target_anemia"] = "남자 혈색소 <13 또는 여자 혈색소 <12"


# 8. 비만 위험
# BMI >=25
if "BMI" in work.columns:
    obesity_known = work["BMI"].notna()
    obesity_positive = work["BMI"] >= 25
    work["target_obesity"] = np.where(obesity_known, obesity_positive.astype(int), np.nan)
else:
    work["target_obesity"] = np.nan

label_info["target_obesity"] = "BMI >=25"


# =========================
# 모델별 입력 변수와 누수 제거 변수 설정
# =========================

base_feature_candidates = [
    "age",
    "sex_code",
    "female",
    "male",
    "height",
    "weight",
    "waist",
    "BMI",
    "systolic_bp",
    "diastolic_bp",
    "fasting_glucose",
    "HbA1c",
    "total_cholesterol",
    "HDL",
    "triglyceride",
    "LDL",
    "AST",
    "ALT",
    "hemoglobin",
    "creatinine",
    "eGFR_calc",
    "urine_protein",
]

base_feature_cols = [col for col in base_feature_candidates if col in work.columns]

# 핵심: target을 만드는 데 직접 사용한 변수는 해당 모델 입력에서 제거
leakage_exclusion = {
    "target_diabetes": [
        "fasting_glucose",
        "HbA1c",
    ],
    "target_hypertension": [
        "systolic_bp",
        "diastolic_bp",
    ],
    "target_dyslipidemia": [
        "total_cholesterol",
        "HDL",
        "triglyceride",
        "LDL",
    ],
    "target_metabolic_syndrome": [
        "waist",
        "triglyceride",
        "HDL",
        "systolic_bp",
        "diastolic_bp",
        "fasting_glucose",
    ],
    "target_liver_dysfunction": [
        "AST",
        "ALT",
    ],
    "target_ckd": [
        "creatinine",
        "eGFR_calc",
        "urine_protein",
    ],
    "target_anemia": [
        "hemoglobin",
    ],
    "target_obesity": [
        "BMI",
        "weight",
    ],
}


target_cols = [
    "target_diabetes",
    "target_hypertension",
    "target_dyslipidemia",
    "target_metabolic_syndrome",
    "target_liver_dysfunction",
    "target_ckd",
    "target_anemia",
    "target_obesity",
]


target_name_kr = {
    "target_diabetes": "당뇨병 위험",
    "target_hypertension": "고혈압 위험",
    "target_dyslipidemia": "이상지질혈증 위험",
    "target_metabolic_syndrome": "대사증후군 위험",
    "target_liver_dysfunction": "지방간/간기능 이상 위험",
    "target_ckd": "만성신장질환 위험",
    "target_anemia": "빈혈 위험",
    "target_obesity": "비만 위험",
}


def get_clean_features(target_col):
    exclude_cols = set(leakage_exclusion.get(target_col, []))
    clean_features = [col for col in base_feature_cols if col not in exclude_cols]

    # target 자체는 당연히 제외
    clean_features = [col for col in clean_features if not col.startswith("target_")]

    return clean_features


print("\n[기본 입력 후보 변수]")
print(base_feature_cols)

print("\n[모델별 누수 제거 후 입력 변수]")
for target_col in target_cols:
    print("\n", target_col, "/", target_name_kr.get(target_col, target_col))
    print("제외 변수:", leakage_exclusion.get(target_col, []))
    print("사용 변수:", get_clean_features(target_col))


# =========================
# 모델 학습 함수
# =========================

def train_one_clean_model(work_df, target_col):
    feature_cols = get_clean_features(target_col)

    if len(feature_cols) < 3:
        return None, None, None, feature_cols, "too_few_features"

    model_df = work_df[feature_cols + [target_col]].copy()

    # target 없는 행 제거
    model_df = model_df.dropna(subset=[target_col])

    # 입력변수가 전부 결측인 행 제거
    model_df = model_df.dropna(how="all", subset=feature_cols)

    if model_df.empty:
        return None, None, None, feature_cols, "empty"

    X = model_df[feature_cols]
    y = model_df[target_col].astype(int)

    if y.nunique() < 2:
        return None, None, None, feature_cols, "one_class"

    positive_count = int(y.sum())
    negative_count = int((y == 0).sum())

    if positive_count < 20 or negative_count < 20:
        return None, None, None, feature_cols, "too_few_cases"

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "target": target_col,
        "target_name_kr": target_name_kr.get(target_col, target_col),
        "definition": label_info.get(target_col, ""),
        "leakage_removed_features": ", ".join(leakage_exclusion.get(target_col, [])),
        "used_features": ", ".join(feature_cols),
        "n_features": len(feature_cols),
        "n_total": len(model_df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": round(float(y.mean()), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "auroc": round(roc_auc_score(y_test, y_prob), 4),
        "auprc": round(average_precision_score(y_test, y_prob), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "status": "trained_clean",
    }

    importances = pipe.named_steps["model"].feature_importances_

    fi_df = pd.DataFrame({
        "target": target_col,
        "target_name_kr": target_name_kr.get(target_col, target_col),
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return pipe, metrics, fi_df, feature_cols, "trained_clean"


# =========================
# 전체 모델 학습
# =========================

performance_rows = []
feature_importance_list = []
model_schema_rows = []
leakage_report_rows = []

for target_col in target_cols:
    print("\n==============================")
    print("학습 대상:", target_col, "/", target_name_kr.get(target_col, target_col))
    print("정의:", label_info.get(target_col, ""))
    print("제거한 누수 변수:", leakage_exclusion.get(target_col, []))

    model, metrics, fi_df, feature_cols, status = train_one_clean_model(work, target_col)

    leakage_report_rows.append({
        "target": target_col,
        "target_name_kr": target_name_kr.get(target_col, target_col),
        "definition": label_info.get(target_col, ""),
        "removed_features": ", ".join(leakage_exclusion.get(target_col, [])),
        "used_features": ", ".join(feature_cols),
        "n_used_features": len(feature_cols),
        "status": status,
    })

    if status != "trained_clean":
        print("학습 생략:", status)

        performance_rows.append({
            "target": target_col,
            "target_name_kr": target_name_kr.get(target_col, target_col),
            "definition": label_info.get(target_col, ""),
            "leakage_removed_features": ", ".join(leakage_exclusion.get(target_col, [])),
            "used_features": ", ".join(feature_cols),
            "status": status
        })

        continue

    print("누수 제거 모델 학습 완료")
    print("AUROC:", metrics["auroc"])
    print("AUPRC:", metrics["auprc"])
    print("F1:", metrics["f1"])

    performance_rows.append(metrics)
    feature_importance_list.append(fi_df)

    model_file = MODEL_DIR / f"{target_col}_clean.pkl"

    save_obj = {
        "target": target_col,
        "target_name_kr": target_name_kr.get(target_col, target_col),
        "definition": label_info.get(target_col, ""),
        "features": feature_cols,
        "leakage_removed_features": leakage_exclusion.get(target_col, []),
        "model": model,
    }

    joblib.dump(save_obj, model_file)

    model_schema_rows.append({
        "target": target_col,
        "target_name_kr": target_name_kr.get(target_col, target_col),
        "model_file": str(model_file),
        "features": ", ".join(feature_cols),
        "removed_features": ", ".join(leakage_exclusion.get(target_col, [])),
        "definition": label_info.get(target_col, "")
    })

    print("모델 저장:", model_file)


# =========================
# 결과 저장
# =========================

performance_df = pd.DataFrame(performance_rows)

if feature_importance_list:
    feature_importance_df = pd.concat(feature_importance_list, ignore_index=True)
else:
    feature_importance_df = pd.DataFrame()

model_schema_df = pd.DataFrame(model_schema_rows)
leakage_report_df = pd.DataFrame(leakage_report_rows)

label_summary_rows = []

for target_col in target_cols:
    if target_col in work.columns:
        y = work[target_col].dropna()

        if len(y) > 0:
            label_summary_rows.append({
                "target": target_col,
                "target_name_kr": target_name_kr.get(target_col, target_col),
                "definition": label_info.get(target_col, ""),
                "n": len(y),
                "positive": int(y.sum()),
                "negative": int((y == 0).sum()),
                "positive_rate": round(float(y.mean()), 4)
            })

label_summary_df = pd.DataFrame(label_summary_rows)

processed_data_path = OUT_DIR / "knhanes_processed_for_clean_model.csv"
work.to_csv(processed_data_path, index=False, encoding="utf-8-sig")

excel_path = OUT_DIR / "clean_model_training_results.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    performance_df.to_excel(writer, index=False, sheet_name="performance")
    feature_importance_df.to_excel(writer, index=False, sheet_name="feature_importance")
    model_schema_df.to_excel(writer, index=False, sheet_name="model_schema")
    label_summary_df.to_excel(writer, index=False, sheet_name="label_summary")
    leakage_report_df.to_excel(writer, index=False, sheet_name="leakage_report")

print("\n==============================")
print("누수 제거 모델 학습 전체 완료")
print("==============================")
print("성능 결과 저장:", excel_path)
print("전처리 데이터 저장:", processed_data_path)
print("모델 저장 폴더:", MODEL_DIR)

print("\n[생성된 누수 제거 모델 파일]")
for p in MODEL_DIR.glob("target_*_clean.pkl"):
    print("-", p.name)

print("\n다음 단계:")
print("1) output\\clean_model_training_results.xlsx에서 performance 시트 확인")
print("2) AUROC가 1.0에서 현실적인 값으로 낮아졌는지 확인")
print("3) model 폴더의 *_clean.pkl 모델을 app.py에 연결")

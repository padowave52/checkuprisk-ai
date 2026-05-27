from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "knhanes"
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

candidate_files = []
for ext in ["*.sas7bdat", "*.csv", "*.xlsx", "*.xls"]:
    candidate_files.extend(DATA_DIR.glob(ext))

if len(candidate_files) == 0:
    raise FileNotFoundError("data\\knhanes 폴더 안에 데이터 파일이 없습니다.")

DATA_PATH = candidate_files[0]
print("선택된 데이터 파일:", DATA_PATH)

suffix = DATA_PATH.suffix.lower()

if suffix == ".sas7bdat":
    try:
        df = pd.read_sas(DATA_PATH, format="sas7bdat", encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_sas(DATA_PATH, format="sas7bdat", encoding="utf-8")
elif suffix == ".csv":
    try:
        df = pd.read_csv(DATA_PATH, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding="utf-8")
elif suffix in [".xlsx", ".xls"]:
    df = pd.read_excel(DATA_PATH)
else:
    raise ValueError("지원하지 않는 파일 형식입니다.")

print("데이터 크기:", df.shape)

keywords = {
    "age": ["age", "연령", "나이"],
    "sex": ["sex", "성별"],
    "BMI": ["bmi", "체질량"],
    "waist": ["waist", "허리"],
    "systolic_bp": ["수축", "sbp", "sys"],
    "diastolic_bp": ["이완", "dbp", "dia"],
    "fasting_glucose": ["glucose", "glu", "혈당", "공복"],
    "total_cholesterol": ["chol", "콜레스테롤", "총콜"],
    "HDL": ["hdl"],
    "triglyceride": ["trig", "tg", "중성"],
    "LDL": ["ldl"],
    "AST": ["ast"],
    "ALT": ["alt"],
    "GGT": ["ggt", "gtp", "감마"],
    "creatinine": ["creatinine", "cr", "크레아티닌"],
    "eGFR": ["egfr", "gfr", "사구체"],
}

rows = []
for model_name, key_list in keywords.items():
    for col in df.columns:
        col_lower = str(col).lower()
        if any(k.lower() in col_lower for k in key_list):
            rows.append({
                "model_variable": model_name,
                "candidate_column": col,
                "dtype": df[col].dtype,
                "missing_count": df[col].isna().sum(),
                "missing_rate": round(df[col].isna().mean() * 100, 2),
                "example_values": df[col].dropna().head(5).tolist(),
            })

candidate_df = pd.DataFrame(rows)
output_path = OUT_DIR / "candidate_variables.xlsx"
candidate_df.to_excel(output_path, index=False)

print("후보 변수 저장 완료:")
print(output_path)

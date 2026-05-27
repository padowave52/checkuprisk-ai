from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "knhanes"
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("KNHANES 데이터 폴더:", DATA_DIR)

files = list(DATA_DIR.glob("*"))
print("\n[data\\knhanes 폴더 안 파일 목록]")
for f in files:
    print("-", f.name)

candidate_files = []
for ext in ["*.sas7bdat", "*.csv", "*.xlsx", "*.xls"]:
    candidate_files.extend(DATA_DIR.glob(ext))

if len(candidate_files) == 0:
    raise FileNotFoundError("data\\knhanes 폴더 안에 국민건강영양조사 원시자료 파일이 없습니다.")

DATA_PATH = candidate_files[0]
print("\n선택된 데이터 파일:", DATA_PATH.name)
print("전체 경로:", DATA_PATH)

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

wrong_columns = {"pdf_id", "item_name", "value", "unit", "model_variable"}
if wrong_columns.issubset(set(df.columns)):
    raise ValueError("현재 읽은 파일은 국민건강영양조사 원시자료가 아니라 건강검진표 PDF 추출 결과 파일입니다.")

print("\n데이터 크기:", df.shape)
print("\n앞부분 미리보기:")
print(df.head())

cols = pd.DataFrame({
    "column": df.columns,
    "dtype": [df[col].dtype for col in df.columns],
    "missing_count": [df[col].isna().sum() for col in df.columns],
    "missing_rate": [round(df[col].isna().mean() * 100, 2) for col in df.columns]
})

output_path = OUT_DIR / "knhanes_columns.xlsx"
cols.to_excel(output_path, index=False)

print("\n변수명 저장 완료:")
print(output_path)

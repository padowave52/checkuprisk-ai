# CheckupRisk AI

건강검진표 PDF를 업로드하면 주요 검진 수치를 자동 추출하고, 성별·나이 기반 해석과 KNHANES 기반 AI 질병 위험 예측확률을 제공하는 Streamlit 웹앱입니다.

## 주요 기능

- 건강검진표 PDF 자동 텍스트 추출
- 키, 몸무게, BMI, 허리둘레, 혈압, 혈색소, 공복혈당, 지질, 간기능, 신장기능 등 주요 항목 추출
- 정상/주의/위험/확인필요 등급 분류
- 영역별 건강점수 및 관리 우선순위 제공
- Random Forest 기반 질병 위험 예측모델 연결
- Streamlit 웹앱 실행 및 엑셀 다운로드 지원

## 프로젝트 구조

```text
checkuprisk-ai
├─ code
│  ├─ app.py
│  ├─ 01_check_knhanes_columns.py
│  ├─ 02_find_candidate_variables.py
│  ├─ 03_train_clean_prediction_models.py
│  └─ 04_optimize_model_thresholds.py
├─ model
│  ├─ README_model_files.txt
│  ├─ target_*_clean.pkl                  # 직접 생성/추가 필요
│  └─ clean_model_thresholds.json          # 직접 생성/추가 필요
├─ data
│  └─ knhanes                              # 원시자료 로컬 보관용, GitHub 업로드 제외
├─ output                                  # 분석 결과 로컬 저장용, GitHub 업로드 제외
├─ .streamlit
│  └─ config.toml
├─ requirements.txt
├─ README.md
└─ .gitignore
```

## 로컬 실행 방법

### 1. 프로젝트 폴더로 이동

```bash
cd checkuprisk-ai
```

### 2. 가상환경 생성 및 실행

Windows PowerShell 기준:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Git Bash 또는 macOS/Linux 기준:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. 패키지 설치

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 앱 실행

이미 `model` 폴더에 `.pkl` 모델 파일과 `clean_model_thresholds.json`이 있다면 아래 명령만 실행하면 됩니다.

```bash
python -m streamlit run code/app.py
```

## 모델을 처음부터 다시 학습하는 경우

국민건강영양조사 원시자료는 `data/knhanes` 폴더에 직접 넣고, 파일명/확장자가 코드에서 인식되는지 확인해야 합니다.

```bash
python code/01_check_knhanes_columns.py
python code/02_find_candidate_variables.py
python code/03_train_clean_prediction_models.py
python code/04_optimize_model_thresholds.py
python -m streamlit run code/app.py
```

실행 후 생성되는 주요 파일:

- `output/knhanes_columns.xlsx`
- `output/candidate_variables.xlsx`
- `output/clean_model_training_results.xlsx`
- `output/clean_model_thresholds.xlsx`
- `output/knhanes_processed_for_clean_model.csv`
- `model/target_*_clean.pkl`
- `model/clean_model_thresholds.json`

## GitHub 업로드 전 확인

GitHub에 올려도 되는 파일:

- `code/`
- `model/*.pkl`
- `model/clean_model_thresholds.json`
- `model/README_model_files.txt`
- `.streamlit/config.toml`
- `requirements.txt`
- `README.md`
- `.gitignore`

GitHub에 올리면 안 되는 파일:

- 개인 건강검진표 PDF
- 국민건강영양조사 원시자료
- `data/` 폴더 내용
- `output/` 폴더 내용
- 개인정보, 인증키, 비밀번호 등 민감정보

## Streamlit Community Cloud 배포

1. GitHub에 새 저장소를 만듭니다.
2. 이 프로젝트 폴더 내용을 업로드합니다.
3. Streamlit Community Cloud에서 새 앱을 생성합니다.
4. 설정값은 아래처럼 지정합니다.

```text
Repository: GitHub에 올린 checkuprisk-ai 저장소
Branch: main
Main file path: code/app.py
```

## 주의사항

- 현재 패키지에 모델 파일이 없으면 앱은 실행되더라도 AI 예측 영역에서 모델 파일 확인 경고가 표시됩니다.
- Streamlit Cloud에서 AI 예측까지 보이게 하려면 `model` 폴더에 학습된 `.pkl` 파일들과 `clean_model_thresholds.json`이 포함되어야 합니다.
- 의료 진단 서비스가 아니라 건강검진 수치 기반 참고용 위험도 안내 서비스로 설명하는 것이 안전합니다.

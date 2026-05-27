# model 폴더 안내

Streamlit 앱에서 AI 예측모델을 사용하려면 아래 파일들이 이 폴더에 있어야 합니다.

필수 파일:
- clean_model_thresholds.json
- target_diabetes_clean.pkl
- target_hypertension_clean.pkl
- target_dyslipidemia_clean.pkl
- target_metabolic_syndrome_clean.pkl
- target_liver_dysfunction_clean.pkl
- target_ckd_clean.pkl
- target_anemia_clean.pkl
- target_obesity_clean.pkl

주의:
- 현재 GitHub-ready 패키지에는 모델 실파일(.pkl, .json)이 포함되어 있지 않으면 앱의 AI 예측 영역이 경고로 표시됩니다.
- 모델 파일은 03_train_clean_prediction_models.py와 04_optimize_model_thresholds.py 실행 후 생성됩니다.
- 개인정보가 포함된 PDF, 국민건강영양조사 원시자료, output 파일은 GitHub에 올리지 마세요.

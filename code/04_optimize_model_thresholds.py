from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")


# =========================
# 경로 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "model"

PROCESSED_DATA_PATH = OUT_DIR / "knhanes_processed_for_clean_model.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("OUT_DIR:", OUT_DIR)
print("MODEL_DIR:", MODEL_DIR)
print("전처리 데이터:", PROCESSED_DATA_PATH)


# =========================
# 입력 파일 확인
# =========================

if not PROCESSED_DATA_PATH.exists():
    raise FileNotFoundError(
        "knhanes_processed_for_clean_model.csv 파일이 없습니다.\n"
        "먼저 03_train_clean_prediction_models.py를 실행해서 전처리 데이터와 clean 모델을 생성하세요."
    )

clean_model_files = sorted(MODEL_DIR.glob("target_*_clean.pkl"))

if len(clean_model_files) == 0:
    raise FileNotFoundError(
        "model 폴더에 target_*_clean.pkl 파일이 없습니다.\n"
        "먼저 03_train_clean_prediction_models.py를 실행하세요."
    )

print("\n[찾은 clean 모델 파일]")
for f in clean_model_files:
    print("-", f.name)


# =========================
# 데이터 불러오기
# =========================

work = pd.read_csv(PROCESSED_DATA_PATH)

print("\n전처리 데이터 크기:", work.shape)


# =========================
# 유틸 함수
# =========================

def risk_grade_by_probability(prob):
    """
    앱 화면용 확률 기반 등급.
    threshold 최적화 결과와 별개로 사용자가 이해하기 쉽게 3단계로 표시.
    """
    if prob >= 0.70:
        return "높음"
    elif prob >= 0.40:
        return "주의"
    else:
        return "낮음"


def safe_divide(a, b):
    if b == 0:
        return 0
    return a / b


def calculate_metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "threshold": round(float(threshold), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "specificity": round(safe_divide(tn, tn + fp), 4),
    }

    return metrics


def find_best_thresholds(y_true, y_prob):
    """
    threshold 후보들을 돌면서 여러 기준의 최적 threshold를 찾음.
    - best_f1: F1 최대
    - recall_70: recall 0.70 이상 중 precision 최대
    - recall_80: recall 0.80 이상 중 precision 최대
    - balanced: sensitivity와 specificity 균형
    """

    thresholds = np.arange(0.05, 0.951, 0.01)

    rows = []

    for threshold in thresholds:
        m = calculate_metrics_at_threshold(y_true, y_prob, threshold)
        rows.append(m)

    threshold_df = pd.DataFrame(rows)

    # 1) F1 최대 threshold
    best_f1_row = threshold_df.sort_values(
        ["f1", "recall", "precision"],
        ascending=False
    ).iloc[0].to_dict()

    # 2) recall 0.70 이상 중 precision 최대
    recall70_df = threshold_df[threshold_df["recall"] >= 0.70].copy()

    if len(recall70_df) > 0:
        recall70_row = recall70_df.sort_values(
            ["precision", "f1", "threshold"],
            ascending=False
        ).iloc[0].to_dict()
    else:
        recall70_row = best_f1_row.copy()

    # 3) recall 0.80 이상 중 precision 최대
    recall80_df = threshold_df[threshold_df["recall"] >= 0.80].copy()

    if len(recall80_df) > 0:
        recall80_row = recall80_df.sort_values(
            ["precision", "f1", "threshold"],
            ascending=False
        ).iloc[0].to_dict()
    else:
        recall80_row = recall70_row.copy()

    # 4) sensitivity와 specificity 균형
    threshold_df["balance_gap"] = (threshold_df["recall"] - threshold_df["specificity"]).abs()
    balanced_row = threshold_df.sort_values(
        ["balance_gap", "f1"],
        ascending=[True, False]
    ).iloc[0].to_dict()

    # 참고용: precision_recall_curve도 계산해서 PR 곡선 기반 후보 확인
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)

    pr_rows = []
    for i, threshold in enumerate(pr_thresholds):
        p = precision[i]
        r = recall[i]
        f1 = 0 if (p + r) == 0 else 2 * p * r / (p + r)

        pr_rows.append({
            "threshold": round(float(threshold), 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
        })

    pr_curve_df = pd.DataFrame(pr_rows)

    return {
        "threshold_grid": threshold_df,
        "pr_curve": pr_curve_df,
        "best_f1": best_f1_row,
        "recall_70": recall70_row,
        "recall_80": recall80_row,
        "balanced": balanced_row,
    }


def choose_recommended_threshold(target, best_result):
    """
    앱에 넣을 추천 threshold 선택 기준.
    - 위험군을 놓치면 안 되는 질환: recall_70 또는 recall_80 선호
    - 비만/대사증후군처럼 성능 좋은 모델: best_f1 사용
    """

    if target in [
        "target_diabetes",
        "target_hypertension",
        "target_ckd",
    ]:
        # 놓치면 안 되는 질환은 recall을 조금 더 중요하게
        selected = best_result["recall_70"]
        strategy = "recall_70"

    elif target in [
        "target_anemia",
        "target_liver_dysfunction",
    ]:
        # 참고용 모델은 균형형으로
        selected = best_result["balanced"]
        strategy = "balanced"

    else:
        # 비만, 대사증후군, 이상지질혈증은 F1 중심
        selected = best_result["best_f1"]
        strategy = "best_f1"

    return selected, strategy


# =========================
# 모델별 threshold 최적화
# =========================

summary_rows = []
all_threshold_rows = []
all_pr_rows = []
config = {}

for model_path in clean_model_files:
    print("\n==============================")
    print("모델:", model_path.name)

    saved = joblib.load(model_path)

    target = saved["target"]
    target_name_kr = saved.get("target_name_kr", target)
    definition = saved.get("definition", "")
    features = saved["features"]
    model = saved["model"]
    removed_features = saved.get("leakage_removed_features", [])

    print("target:", target)
    print("질병명:", target_name_kr)
    print("사용 변수:", features)
    print("제거된 누수 변수:", removed_features)

    if target not in work.columns:
        print("target 컬럼이 전처리 데이터에 없어 생략:", target)
        continue

    missing_features = [f for f in features if f not in work.columns]

    if len(missing_features) > 0:
        print("전처리 데이터에 없는 feature가 있어 생략:", missing_features)
        continue

    model_df = work[features + [target]].copy()

    # target 없는 행 제거
    model_df = model_df.dropna(subset=[target])

    # 입력변수가 전부 결측인 행 제거
    model_df = model_df.dropna(how="all", subset=features)

    if model_df.empty:
        print("사용 가능한 데이터가 없어 생략")
        continue

    X = model_df[features]
    y = model_df[target].astype(int)

    if y.nunique() < 2:
        print("target이 한 클래스만 있어 생략")
        continue

    # 03_train_clean_prediction_models.py와 동일한 split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    y_prob = model.predict_proba(X_test)[:, 1]

    best_result = find_best_thresholds(y_test.values, y_prob)
    recommended, strategy = choose_recommended_threshold(target, best_result)

    # 기본 0.5 threshold 성능도 같이 저장
    default_05 = calculate_metrics_at_threshold(y_test.values, y_prob, 0.5)

    print("기본 threshold 0.5 F1:", default_05["f1"], "/ Recall:", default_05["recall"])
    print("추천 전략:", strategy)
    print("추천 threshold:", recommended["threshold"])
    print("추천 F1:", recommended["f1"])
    print("추천 Recall:", recommended["recall"])
    print("추천 Precision:", recommended["precision"])

    summary_rows.append({
        "target": target,
        "target_name_kr": target_name_kr,
        "definition": definition,
        "recommended_strategy": strategy,
        "recommended_threshold": recommended["threshold"],
        "recommended_precision": recommended["precision"],
        "recommended_recall": recommended["recall"],
        "recommended_f1": recommended["f1"],
        "recommended_accuracy": recommended["accuracy"],
        "recommended_specificity": recommended["specificity"],
        "default_05_precision": default_05["precision"],
        "default_05_recall": default_05["recall"],
        "default_05_f1": default_05["f1"],
        "best_f1_threshold": best_result["best_f1"]["threshold"],
        "best_f1": best_result["best_f1"]["f1"],
        "recall70_threshold": best_result["recall_70"]["threshold"],
        "recall70_precision": best_result["recall_70"]["precision"],
        "recall70_recall": best_result["recall_70"]["recall"],
        "recall70_f1": best_result["recall_70"]["f1"],
        "recall80_threshold": best_result["recall_80"]["threshold"],
        "recall80_precision": best_result["recall_80"]["precision"],
        "recall80_recall": best_result["recall_80"]["recall"],
        "recall80_f1": best_result["recall_80"]["f1"],
        "balanced_threshold": best_result["balanced"]["threshold"],
        "balanced_precision": best_result["balanced"]["precision"],
        "balanced_recall": best_result["balanced"]["recall"],
        "balanced_f1": best_result["balanced"]["f1"],
        "n_test": len(y_test),
        "positive_rate_test": round(float(y_test.mean()), 4),
        "used_features": ", ".join(features),
        "removed_features": ", ".join(removed_features),
    })

    threshold_grid = best_result["threshold_grid"].copy()
    threshold_grid.insert(0, "target_name_kr", target_name_kr)
    threshold_grid.insert(0, "target", target)
    all_threshold_rows.append(threshold_grid)

    pr_curve = best_result["pr_curve"].copy()
    pr_curve.insert(0, "target_name_kr", target_name_kr)
    pr_curve.insert(0, "target", target)
    all_pr_rows.append(pr_curve)

    config[target] = {
        "target_name_kr": target_name_kr,
        "definition": definition,
        "recommended_strategy": strategy,
        "threshold": float(recommended["threshold"]),
        "features": features,
        "removed_features": removed_features,
        "model_file": str(model_path),
        "display_grade_rule": {
            "high": 0.70,
            "moderate": 0.40,
            "low": 0.0
        }
    }


# =========================
# 결과 저장
# =========================

summary_df = pd.DataFrame(summary_rows)

if all_threshold_rows:
    threshold_detail_df = pd.concat(all_threshold_rows, ignore_index=True)
else:
    threshold_detail_df = pd.DataFrame()

if all_pr_rows:
    pr_curve_df = pd.concat(all_pr_rows, ignore_index=True)
else:
    pr_curve_df = pd.DataFrame()

excel_path = OUT_DIR / "clean_model_thresholds.xlsx"
json_path = MODEL_DIR / "clean_model_thresholds.json"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    summary_df.to_excel(writer, index=False, sheet_name="summary")
    threshold_detail_df.to_excel(writer, index=False, sheet_name="threshold_grid")
    pr_curve_df.to_excel(writer, index=False, sheet_name="pr_curve")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)


print("\n==============================")
print("threshold 최적화 완료")
print("==============================")
print("엑셀 저장:", excel_path)
print("JSON 저장:", json_path)

print("\n[추천 threshold 요약]")
if len(summary_df) > 0:
    print(summary_df[[
        "target_name_kr",
        "recommended_strategy",
        "recommended_threshold",
        "recommended_precision",
        "recommended_recall",
        "recommended_f1",
        "default_05_f1"
    ]])
else:
    print("저장된 결과가 없습니다.")

print("\n다음 단계:")
print("1) output\\clean_model_thresholds.xlsx의 summary 시트 확인")
print("2) recommended_threshold와 recommended_f1 확인")
print("3) model\\clean_model_thresholds.json을 app.py에서 불러와 AI 위험도 출력에 사용")

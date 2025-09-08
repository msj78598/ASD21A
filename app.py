# -*- coding: utf-8 -*-
# لا تضع Docstring هنا لتجنب مشاكل \U في مسارات ويندوز

import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import joblib
import matplotlib.pyplot as plt

# نماذج التعزيز
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# ===== المسارات الافتراضية المطلوبة منك =====
PROJECT_DIR_DEFAULT = Path(r"C:\Users\78598\Documents\asd21")
DATA_PATH_DEFAULT   = Path(r"C:\Users\78598\Documents\asd21\data.xlsx")

# ===== أدوات مساعدة =====
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_excel_biggest_sheet(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    best_name, best_rows = None, -1
    for name in xl.sheet_names:
        df_tmp = xl.parse(name)
        if df_tmp.shape[0] > best_rows:
            best_rows, best_name = df_tmp.shape[0], name
    df = xl.parse(best_name)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def auto_detect_target(df: pd.DataFrame) -> str:
    if "Target" in df.columns:
        return "Target"
    kwords = ["target","label","y","loss","defect","fault","anomaly",
              "فاقد","خسارة","هدر","عطل","اعطال","تسرب","تسريب","سرقة","عيب"]
    scores = []
    for c in df.columns:
        s = 0
        name = str(c).lower()
        for kw in kwords:
            if kw in name: s += 2
        u = df[c].dropna().unique()
        if len(u) <= 4:
            try:
                vv = pd.Series(df[c]).astype(str).str.lower().str.strip()
                if set(vv.unique()).issubset({"0","1","true","false","yes","no","نعم","لا"}):
                    s += 3
            except: pass
        if s > 0: scores.append((s, c))
    if scores:
        scores.sort(key=lambda x: (-x[0], x[1]))
        return scores[0][1]
    return df.columns[-1]

def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    m = {}
    try: m["roc_auc"] = roc_auc_score(y_true, y_prob)
    except: m["roc_auc"] = np.nan
    try: m["pr_auc"] = average_precision_score(y_true, y_prob)
    except: m["pr_auc"] = np.nan
    m["accuracy"]  = accuracy_score(y_true, y_pred)
    m["precision"] = precision_score(y_true, y_pred, zero_division=0)
    m["recall"]    = recall_score(y_true, y_pred, zero_division=0)
    m["f1"]        = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    m["tn"], m["fp"], m["fn"], m["tp"] = int(tn), int(fp), int(fn), int(tp)
    return m

def plot_curves(y_true, y_prob, out_prefix: Path):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.grid(True, alpha=0.3)
    plt.savefig(out_prefix.with_suffix(".roc.png"), bbox_inches="tight", dpi=160)
    plt.close()
    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec, label="PR")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.grid(True, alpha=0.3)
    plt.savefig(out_prefix.with_suffix(".pr.png"), bbox_inches="tight", dpi=160)
    plt.close()

# ===== تدريب عبر K-Fold لكل نموذج =====
def train_cv_all_models(
    df: pd.DataFrame,
    target_col: str,
    project_dir: Path,
    n_splits: int = 5,
    random_state: int = 42,
):
    models_dir  = project_dir / "models"
    results_dir = project_dir / "results"
    ensure_dir(models_dir); ensure_dir(results_dir)

    # تجهيز الهدف
    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])

    if y_raw.dtype.kind in "if":
        y = (y_raw.astype(float) > 0).astype(int).values
    else:
        mapper = {"1":1,"0":0,"true":1,"false":0,"yes":1,"no":0,"نعم":1,"لا":0}
        y = y_raw.astype(str).str.lower().str.strip().map(mapper)
        if y.isna().any():
            y = pd.Categorical(y_raw).codes
            if len(np.unique(y)) != 2:
                raise ValueError("العمود الهدف ليس ثنائيًا، رجاءً وفّر Target ثنائي.")
        y = y.values

    # تحويل ميزات نصية إلى أرقام إن وجدت
    X_all = X_raw.copy()
    for c in X_all.columns:
        if X_all[c].dtype == object:
            X_all[c] = pd.to_numeric(X_all[c], errors="coerce")
    num_cols = X_all.columns.tolist()

    # تعريف النماذج
    model_defs = {
        "LightGBM": lgb.LGBMClassifier(
            objective="binary", learning_rate=0.05, n_estimators=2000,
            num_leaves=63, subsample=0.9, colsample_bytree=0.9,
            random_state=random_state, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            objective="binary:logistic", learning_rate=0.05, n_estimators=3000,
            max_depth=6, min_child_weight=3, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, tree_method="hist",
            eval_metric="logloss",  # لثبات الإصدارات
            n_jobs=-1, random_state=random_state
        ),
        "CatBoost": CatBoostClassifier(
            iterations=3000, learning_rate=0.05, depth=6,
            loss_function="Logloss", eval_metric="AUC",
            random_seed=random_state, verbose=False
        ),
        "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
        "RandomForest": RandomForestClassifier(n_estimators=600, random_state=random_state, n_jobs=-1),
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    leaderboard_rows = []
    best_global = {"model": None, "auc": -1, "prob_full": None, "y_full": None}

    for model_name, base_model in model_defs.items():
        fold_rows, best_iters = [], []

        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X_all, y), start=1):
            X_tr = X_all.iloc[train_idx].copy()
            X_va = X_all.iloc[valid_idx].copy()
            y_tr = y[train_idx]; y_va = y[valid_idx]

            imputer = SimpleImputer(strategy="median")
            X_tr_imp = imputer.fit_transform(X_tr[num_cols])
            X_va_imp = imputer.transform(X_va[num_cols])

            use_scaler = (model_name == "LogisticRegression")
            if use_scaler:
                scaler = StandardScaler()
                X_tr_imp = scaler.fit_transform(X_tr_imp)
                X_va_imp = scaler.transform(X_va_imp)

            import copy
            model = copy.deepcopy(base_model)

            # ===== تدريب حسب النموذج =====
            if model_name == "LightGBM":
                callbacks = [
                    lgb.early_stopping(stopping_rounds=100),
                    lgb.log_evaluation(period=0)  # كتم الطباعة
                ]
                model.fit(
                    X_tr_imp, y_tr,
                    eval_set=[(X_va_imp, y_va)],
                    eval_metric="auc",
                    callbacks=callbacks
                )
                if hasattr(model, "best_iteration_") and model.best_iteration_ is not None:
                    best_iters.append(int(model.best_iteration_))
                y_prob = model.predict_proba(X_va_imp)[:, 1]

            elif model_name == "XGBoost":
                # دعم كل الإصدارات: جرّب الوسيطة القديمة، وإن فشلت استخدم callbacks، وإن فشلت درّب بدون إيقاف مبكر
                fitted = False
                try:
                    model.fit(
                        X_tr_imp, y_tr,
                        eval_set=[(X_va_imp, y_va)],
                        early_stopping_rounds=100
                    )
                    fitted = True
                except TypeError:
                    try:
                        try:
                            from xgboost.callback import EarlyStopping as XgbEarlyStopping
                            cb = [XgbEarlyStopping(rounds=100, save_best=True)]
                        except Exception:
                            cb = [xgb.callback.EarlyStopping(rounds=100, save_best=True)]
                        model.fit(
                            X_tr_imp, y_tr,
                            eval_set=[(X_va_imp, y_va)],
                            callbacks=cb
                        )
                        fitted = True
                    except Exception:
                        fitted = False
                if not fitted:
                    model.fit(X_tr_imp, y_tr)

                best_it = getattr(model, "best_iteration", None)
                if best_it is None:
                    # بعض الإصدارات تستخدم best_ntree_limit
                    best_it = getattr(model, "best_ntree_limit", None)
                if best_it is not None:
                    try: best_iters.append(int(best_it))
                    except: pass

                y_prob = model.predict_proba(X_va_imp)[:, 1]

            elif model_name == "CatBoost":
                model.fit(X_tr_imp, y_tr, eval_set=(X_va_imp, y_va), use_best_model=True, verbose=False)
                try:
                    bi = model.get_best_iteration()
                    if bi is not None: best_iters.append(int(bi))
                except: pass
                y_prob = model.predict_proba(X_va_imp)[:, 1]

            else:
                model.fit(X_tr_imp, y_tr)
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_va_imp)[:, 1]
                else:
                    s = model.decision_function(X_va_imp)
                    y_prob = (s - s.min())/(s.max()-s.min()+1e-9)

            # مقاييس الطية
            m = compute_metrics(y_va, y_prob, thr=0.5)
            m["fold"] = fold_idx
            fold_rows.append(m)

        # حفظ نتائج الطيات
        fold_df = pd.DataFrame(fold_rows)
        (project_dir / "results").mkdir(exist_ok=True)
        fold_df.to_csv(project_dir / "results" / f"metrics_cv_{model_name}.csv", index=False, encoding="utf-8-sig")

        # متوسطات للـ leaderboard
        mean_metrics = fold_df.mean(numeric_only=True).to_dict()
        std_metrics  = fold_df.std(numeric_only=True).to_dict()
        leaderboard_rows.append({
            "model": model_name,
            "roc_auc_mean": mean_metrics.get("roc_auc", np.nan),
            "roc_auc_std":  std_metrics.get("roc_auc", np.nan),
            "pr_auc_mean":  mean_metrics.get("pr_auc", np.nan),
            "accuracy_mean":mean_metrics.get("accuracy", np.nan),
            "precision_mean":mean_metrics.get("precision", np.nan),
            "recall_mean":  mean_metrics.get("recall", np.nan),
            "f1_mean":      mean_metrics.get("f1", np.nan)
        })

        # ===== إعادة التدريب على كامل البيانات وحفظ النموذج =====
        imputer_full = SimpleImputer(strategy="median")
        X_full = imputer_full.fit_transform(X_all[num_cols])

        scaler_full = None
        if model_name == "LogisticRegression":
            scaler_full = StandardScaler()
            X_full = scaler_full.fit_transform(X_full)

        import copy
        final_model = copy.deepcopy(base_model)
        if model_name == "LightGBM" and best_iters:
            final_model.set_params(n_estimators=int(np.mean(best_iters)))
        if model_name == "XGBoost" and best_iters:
            final_model.set_params(n_estimators=int(np.mean(best_iters)))
        if model_name == "CatBoost" and best_iters:
            final_model.set_params(iterations=int(np.mean(best_iters)))

        final_model.fit(X_full, y)

        bundle = {
            "imputer": imputer_full,
            "scaler": scaler_full,
            "model": final_model,
            "num_cols": num_cols,
            "target_col": target_col,
            "columns": X_all.columns.tolist()
        }
        (project_dir / "models").mkdir(exist_ok=True)
        joblib.dump(bundle, project_dir / "models" / f"{model_name}.joblib")

        # اختيار أفضل نموذج (مرجعيًا على كامل البيانات)
        if hasattr(final_model, "predict_proba"):
            try:
                y_prob_full = final_model.predict_proba(X_full)[:, 1]
                auc_full = roc_auc_score(y, y_prob_full)
            except Exception:
                y_prob_full, auc_full = None, -1
        else:
            y_prob_full, auc_full = None, -1

        try:
            if auc_full > best_global["auc"]:
                best_global = {"model": model_name, "auc": auc_full, "prob_full": y_prob_full, "y_full": y}
        except: pass

    # حفظ الـ leaderboard
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("roc_auc_mean", ascending=False)
    leaderboard_path = project_dir / "results" / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False, encoding="utf-8-sig")

    # منحنيات ROC/PR لأفضل نموذج على كامل البيانات (مرجعيّة)
    if best_global["model"] is not None and best_global["prob_full"] is not None:
        out_prefix = project_dir / "results" / f"best_{best_global['model']}"
        plot_curves(best_global["y_full"], best_global["prob_full"], out_prefix)
        with open(project_dir / "results" / "best_model.txt", "w", encoding="utf-8") as f:
            f.write(f"Best model (by full-data ROC proxy): {best_global['model']}\nAUC={best_global['auc']:.6f}\n")

    return leaderboard

# ===== واجهة Streamlit (اختياري) =====
def run_ui(default_project_dir: Path, default_data_path: Path):
    import streamlit as st
    st.set_page_config(page_title="تدريب نماذج الفاقد", layout="wide")
    st.title("تدريب نماذج كشف/تصنيف حالات الفاقد")

    project_dir = Path(st.text_input("مسار مجلد المشروع:", str(default_project_dir)))
    data_path   = Path(st.text_input("مسار ملف البيانات (Excel):", str(default_data_path)))
    target_col  = st.text_input("اسم عمود الهدف (اتركه فارغ لاكتشاف تلقائي):", "")

    col1, col2 = st.columns(2)
    with col1:
        n_splits = st.number_input("عدد الطيات (K-Fold)", min_value=3, max_value=10, value=5, step=1)
    with col2:
        random_state = st.number_input("Random State", min_value=0, max_value=999999, value=42, step=1)

    if st.button("🚀 درّب النماذج وحفظ النتائج"):
        if not data_path.exists():
            st.error("ملف البيانات غير موجود. تأكد من المسار."); st.stop()
        df = load_excel_biggest_sheet(data_path)
        tgt = target_col.strip() if target_col.strip() else auto_detect_target(df)
        if tgt not in df.columns:
            st.error(f"لم يتم العثور على العمود الهدف: {tgt}"); st.stop()
        with st.spinner("جاري التدريب عبر جميع النماذج..."):
            leaderboard = train_cv_all_models(df, tgt, project_dir, n_splits=int(n_splits), random_state=int(random_state))
        st.success("اكتمل التدريب! تم حفظ النماذج والنتائج.")
        st.subheader("🏆 النتائج المجمّعة (Leaderboard)")
        st.dataframe(leaderboard, use_container_width=True)
        st.caption(f"النتائج: {project_dir / 'results'}/leaderboard.csv")
        st.caption(f"النماذج: {project_dir / 'models'}/*.joblib")

# ===== نقطة الدخول =====
def main():
    parser = argparse.ArgumentParser(description="تدريب نماذج الفاقد وحفظ النتائج")
    parser.add_argument("--train", action="store_true", help="تشغيل وضع التدريب (بدون واجهة)")
    parser.add_argument("--project_dir", type=str, default=str(PROJECT_DIR_DEFAULT), help="مسار مجلد المشروع")
    parser.add_argument("--data_path", type=str, default=str(DATA_PATH_DEFAULT), help="مسار ملف Excel")
    parser.add_argument("--target", type=str, default="", help="اسم عمود الهدف (افتراضي اكتشاف تلقائي)")
    parser.add_argument("--kfold", type=int, default=5, help="عدد الطيات")
    parser.add_argument("--seed", type=int, default=42, help="Random state")
    args, _ = parser.parse_known_args()

    project_dir = Path(args.project_dir)
    data_path   = Path(args.data_path)

    if args.train:
        if not data_path.exists():
            print(f"[خطأ] ملف البيانات غير موجود: {data_path}"); sys.exit(1)
        df = load_excel_biggest_sheet(data_path)
        tgt = args.target.strip() if args.target.strip() else auto_detect_target(df)
        if tgt not in df.columns:
            print(f"[خطأ] لم يتم العثور على العمود الهدف: {tgt}"); sys.exit(1)
        print(f"[معلومة] استخدام العمود الهدف: {tgt}")
        leaderboard = train_cv_all_models(df, tgt, project_dir, n_splits=args.kfold, random_state=args.seed)
        print("[تمّ] اكتمل التدريب. أفضل النماذج حسب ROC-AUC:")
        print(leaderboard.head(10).to_string(index=False))
    else:
        try:
            import streamlit as st
        except Exception:
            print("[تنبيه] Streamlit غير متاح. ثبّت الحزم أو استخدم --train."); sys.exit(1)
        run_ui(PROJECT_DIR_DEFAULT, DATA_PATH_DEFAULT)

if __name__ == "__main__":
    main()

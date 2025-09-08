# -*- coding: utf-8 -*-
# واجهة Streamlit لتشغيل النماذج المُدرّبة وعرض ملخص على الشاشة
# وإخراج النتائج إلى Excel مع نسبة الثقة لكل حالة

from pathlib import Path
import io
import time
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

# مسارات المشروع كما زوّدتني
# مسارات المشروع (نسبيّة للمستودع/التطبيق)
REPO_DIR   = Path(__file__).resolve().parent
MODELS_DIR = REPO_DIR / "models"
RESULTS_DIR = REPO_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ========== أدوات مساعدة ==========
def load_excel_biggest_sheet(path_or_buf):
    """قراءة أكبر ورقة من ملف Excel (يدعم رفع ملف عبر Streamlit أو مسار)."""
    xl = pd.ExcelFile(path_or_buf)
    best_name, best_rows = None, -1
    for name in xl.sheet_names:
        df_tmp = xl.parse(name)
        if df_tmp.shape[0] > best_rows:
            best_name, best_rows = name, df_tmp.shape[0]
    df = xl.parse(best_name)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def to_binary_y(series: pd.Series) -> np.ndarray:
    """تحويل عمود الهدف إلى 0/1 عند توفّره (اختياري)."""
    if series.dtype.kind in "if":
        return (series.astype(float) > 0).astype(int).values
    mapper = {"1":1,"0":0,"true":1,"false":0,"yes":1,"no":0,"نعم":1,"لا":0}
    y = series.astype(str).str.lower().str.strip().map(mapper)
    if y.isna().any():
        y = pd.Categorical(series).codes
        if len(np.unique(y)) != 2:
            raise ValueError("Target في ملف الاختبار ليس ثنائيًا.")
    return y.values

def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """تحويل الأعمدة النصية إلى أرقام إن أمكن (غير القابلة تبقى NaN)."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def predict_with_bundle(bundle_path: Path, test_df: pd.DataFrame, threshold: float = 0.5):
    """تشغيل نموذج محفوظ (bundle يحتوي imputer/scaler/model/columns/target_col) على DataFrame."""
    bundle = joblib.load(bundle_path)
    model    = bundle["model"]
    imputer  = bundle["imputer"]
    scaler   = bundle["scaler"]
    columns  = bundle["columns"]
    target_col = bundle.get("target_col", "Target")

    # هدف حقيقي (اختياري للمقاييس)
    y_true = None
    if target_col in test_df.columns:
        try:
            y_true = to_binary_y(test_df[target_col])
        except Exception:
            y_true = None

    # تجهيز الميزات بنفس ترتيب التدريب
    X_df = test_df.drop(columns=[target_col], errors="ignore")
    for c in columns:
        if c not in X_df.columns:
            X_df[c] = np.nan  # أعمدة ناقصة
    X_df = X_df[columns]     # تجاهل أي أعمدة زائدة
    X_df = coerce_numeric_df(X_df)

    # نفس المعالجات
    X = imputer.transform(X_df.values)
    if scaler is not None:
        X = scaler.transform(X)

    # احتمالات تنبؤ الفئة الإيجابية
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        s = model.decision_function(X)
        prob = (s - s.min())/(s.max()-s.min()+1e-9)

    # ملصق متنبأ + نسبة ثقة التنبؤ
    pred = (prob >= float(threshold)).astype(int)
    # "نسبة الثقة" = احتمال الفئة المتنبأ بها
    confidence = np.where(pred == 1, prob, 1.0 - prob)
    confidence_pct = (confidence * 100.0).round(2)

    return prob, pred, confidence, confidence_pct, y_true, target_col

def build_summary_df(test_df: pd.DataFrame, y_true, pred, prob, threshold: float):
    """إنشاء ملخص عربي للعرض/التصدير."""
    n = len(test_df)
    detected = int((pred == 1).sum())
    not_detected = int((pred == 0).sum())

    rows = [
        ["إجمالي السجلات", n],
        ["العتبة المستخدمة (threshold)", threshold],
        ["عدد الحالات المكتشفة (pred=1)", detected],
        ["نسبة الحالات المكتشفة من الإجمالي", round(detected / n, 4) if n else 0.0],
        ["عدد الحالات غير المكتشفة (pred=0)", not_detected],
        ["نسبة الحالات غير المكتشفة من الإجمالي", round(not_detected / n, 4) if n else 0.0],
    ]

    if y_true is not None:
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
            acc  = accuracy_score(y_true, pred)
            prec = precision_score(y_true, pred, zero_division=0)
            rec  = recall_score(y_true, pred, zero_division=0)
            f1   = f1_score(y_true, pred, zero_division=0)
            try:
                roc_auc = roc_auc_score(y_true, prob)
            except Exception:
                roc_auc = np.nan
            try:
                pr_auc = average_precision_score(y_true, prob)
            except Exception:
                pr_auc = np.nan
            # مقاييس للفئة السالبة:
            npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            rows += [
                ["الدقة الكلية (Accuracy)", round(acc, 4)],
                ["ROC-AUC", round(float(roc_auc), 4) if roc_auc == roc_auc else ""],
                ["PR-AUC", round(float(pr_auc), 4) if pr_auc == pr_auc else ""],
                ["دقة الفئة الإيجابية (Precision1)", round(prec, 4)],
                ["استدعاء الفئة الإيجابية (Recall1)", round(rec, 4)],
                ["F1 للفئة الإيجابية", round(f1, 4)],
                ["دقة الفئة السالبة (NPV)", round(npv, 4)],
                ["استدعاء الفئة السالبة (Specificity)", round(spec, 4)],
                ["TP (صحيح إيجابي)", tp],
                ["FP (خاطئ إيجابي)", fp],
                ["TN (صحيح سلبي)", tn],
                ["FN (خاطئ سلبي)", fn],
            ]
        except Exception:
            pass

    return pd.DataFrame(rows, columns=["المؤشر", "القيمة"])

def make_excel_bytes(pred_df: pd.DataFrame, summary_df: pd.DataFrame) -> bytes:
    """تجهيز ملف Excel بصفحتين: Predictions + Summary وإرجاعه كـ bytes للتنزيل."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pred_df.to_excel(writer, index=False, sheet_name="Predictions")
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
    buf.seek(0)
    return buf.read()

# ========== واجهة Streamlit ==========
st.set_page_config(page_title="تشغيل النماذج المُدرّبة (Excel)", layout="wide")
st.title("تشغيل/اختبار النماذج المُدرّبة وإخراج النتائج إلى Excel")

# اختيار النموذج المحفوظ
model_files = sorted(MODELS_DIR.glob("*.joblib"))
if not model_files:
    st.error("لا توجد نماذج محفوظة داخل مجلد models/. قم بالتدريب أولًا عبر app.py.")
    st.stop()

default_idx = 0
for i, p in enumerate(model_files):
    if p.name.lower().startswith("lightgbm"):  # اختيار LightGBM كافتراضي إن وجد
        default_idx = i

model_path = st.selectbox(
    "اختر النموذج:",
    model_files,
    index=default_idx,
    format_func=lambda p: p.name
)

# رفع ملف الاختبار
upl = st.file_uploader("ارفع ملف Excel المراد تشغيله (xlsx):", type=["xlsx"])

# العتبة
thr = st.slider("عتبة القرار (threshold)", 0.0, 1.0, 0.50, 0.01)

# زر التشغيل
if st.button("🚀 تشغيل التنبؤ وإنتاج ملف Excel"):
    if upl is None:
        st.warning("الرجاء رفع ملف Excel أولًا.")
        st.stop()

    with st.spinner("جارٍ التشغيل..."):
        try:
            test_df = load_excel_biggest_sheet(upl)

            prob, pred, conf, conf_pct, y_true, target_col = predict_with_bundle(
                model_path, test_df, threshold=thr
            )

            # بناء DataFrame النتائج + نسبة الثقة
            out_df = test_df.copy()
            out_df["prob_1"] = prob
            out_df["pred_label"] = pred
            out_df["confidence"] = conf                    # 0..1
            out_df["confidence_pct"] = conf_pct            # 0..100 %

            # ملخص للعرض على الشاشة + التصدير
            summary_df = build_summary_df(test_df, y_true, pred, prob, threshold=thr)

            # عرض الملخص في شاشة التشغيل
            st.subheader("📊 الملخص")
            st.dataframe(summary_df, use_container_width=True)

            # حفظ ملف Excel على القرص + زر تنزيل
            ts = time.strftime("%Y%m%d-%H%M%S")
            out_name = f"pred_{model_path.stem}_thr{thr:.2f}_{ts}.xlsx"
            out_path = RESULTS_DIR / out_name

            # بُنْيَة Excel بصفحتين
            excel_bytes = make_excel_bytes(out_df, summary_df)
            with open(out_path, "wb") as f:
                f.write(excel_bytes)

            st.success(f"تم التشغيل وحفظ ملف Excel في: {out_path}")

            # تمكين التنزيل مباشرة من الواجهة
            st.download_button(
                label="⬇️ تنزيل النتيجة كملف Excel",
                data=excel_bytes,
                file_name=out_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # عرض عيّنة من النتائج
            st.subheader("🔎 عينة من النتائج")
            st.dataframe(out_df.head(30), use_container_width=True)

        except Exception as e:
            st.error(f"حدث خطأ أثناء التشغيل: {e}")


    st.markdown("---")
st.markdown("👨‍💻 **تطوير :** مشهور العباس | 00966553339838 | ")

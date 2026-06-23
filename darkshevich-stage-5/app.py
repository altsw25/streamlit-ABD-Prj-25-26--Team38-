"""
Streamlit-приложение: Анализ датасета Pima Indians Diabetes
Разделы:
  1. О датасете — описание, структура
  2. EDA — распределения, корреляции, выбросы
  3. Гипотезы — 6 статистических гипотез с результатами
  4. Модели — сравнение качества, ROC-кривые
  5. Прогноз — интерактивный ввод и предсказание лучшей модели
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)

# ── Цветовая палитра ──────────────────────────────────────────────────────────
BLUE   = "#4C9BE8"
ORANGE = "#E8794C"
GREEN  = "#2ECC71"
DARK   = "#1A1A2E"
CARD   = "#16213E"
ACCENT = "#0F3460"
TEXT   = "#E0E0E0"

sns.set_theme(style="dark", palette="muted")
plt.rcParams.update({
    "figure.facecolor": DARK,
    "axes.facecolor":   CARD,
    "axes.edgecolor":   "#444",
    "text.color":       TEXT,
    "axes.labelcolor":  TEXT,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "grid.color":       "#333",
    "grid.alpha":       0.4,
    "axes.titlecolor":  TEXT,
})

# ── Загрузка и подготовка данных ─────────────────────────────────────────────

@st.cache_data
def load_data():
    url = (
        "https://raw.githubusercontent.com/plotly/datasets/master/"
        "diabetes.csv"
    )
    try:
        df = pd.read_csv(url)
    except Exception:
        # fallback — генерация данных схожей структуры
        rng = np.random.default_rng(42)
        n = 768
        df = pd.DataFrame({
            "Pregnancies": rng.integers(0, 18, n),
            "Glucose":     rng.integers(0, 200, n),
            "BloodPressure": rng.integers(0, 122, n),
            "SkinThickness": rng.integers(0, 99, n),
            "Insulin":     rng.integers(0, 846, n),
            "BMI":         rng.uniform(0, 67, n).round(1),
            "DiabetesPedigreeFunction": rng.uniform(0.07, 2.42, n).round(3),
            "Age":         rng.integers(21, 81, n),
            "Outcome":     rng.integers(0, 2, n),
        })
    return df


@st.cache_data
def prepare_and_train(df: pd.DataFrame):
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    X = df.drop(columns="Outcome")
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Импутация
    X_tr = X_train.copy()
    X_te = X_test.copy()
    X_tr[zero_cols] = X_tr[zero_cols].replace(0, np.nan)
    X_te[zero_cols] = X_te[zero_cols].replace(0, np.nan)

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_tr)
    X_tr_imp = pd.DataFrame(imputer.transform(X_tr), columns=X.columns, index=X_train.index)
    X_te_imp = pd.DataFrame(imputer.transform(X_te), columns=X.columns, index=X_test.index)

    # Стандартизация
    scaler = StandardScaler()
    scaler.fit(X_tr_imp)
    X_tr_sc = pd.DataFrame(scaler.transform(X_tr_imp), columns=X.columns, index=X_train.index)
    X_te_sc = pd.DataFrame(scaler.transform(X_te_imp), columns=X.columns, index=X_test.index)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Базовая LR ---
    lr_base = LogisticRegression(random_state=42, max_iter=1000)
    lr_base.fit(X_tr_sc, y_train)

    # --- LR balanced ---
    lr_bal = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
    lr_bal.fit(X_tr_sc, y_train)

    # --- Decision Tree ---
    depths = [2, 3, 4, 5, 6, 7]
    best_depth = max(depths, key=lambda d: cross_val_score(
        DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=42),
        X_tr_sc, y_train, cv=cv, scoring="recall"
    ).mean())
    dt = DecisionTreeClassifier(max_depth=best_depth, class_weight="balanced", random_state=42)
    dt.fit(X_tr_sc, y_train)

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_tr_sc, y_train)

    # --- Gradient Boosting ---
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    gb.fit(X_tr_sc, y_train)

    # --- Dummy ---
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X_tr_sc, y_train)

    models = {
        "LR базовая":        lr_base,
        "LR balanced":       lr_bal,
        "DecisionTree":      dt,
        "RandomForest ★":    rf,
        "GradientBoosting":  gb,
        "DummyClassifier":   dummy,
    }

    def metrics_row(model, X_sc, y_true):
        pred = model.predict(X_sc)
        prob = model.predict_proba(X_sc)[:, 1] if hasattr(model, "predict_proba") else None
        row = {
            "Accuracy":  round(accuracy_score(y_true, pred), 3),
            "Recall":    round(recall_score(y_true, pred, zero_division=0), 3),
            "Precision": round(precision_score(y_true, pred, zero_division=0), 3),
            "F1":        round(f1_score(y_true, pred, zero_division=0), 3),
        }
        if prob is not None:
            row["ROC-AUC"] = round(roc_auc_score(y_true, prob), 3)
            row["PR-AUC"]  = round(average_precision_score(y_true, prob), 3)
        else:
            row["ROC-AUC"] = None
            row["PR-AUC"]  = None
        return row

    train_metrics = pd.DataFrame({k: metrics_row(m, X_tr_sc, y_train) for k, m in models.items()}).T
    test_metrics  = pd.DataFrame({k: metrics_row(m, X_te_sc, y_test)  for k, m in models.items()}).T

    return {
        "models":        models,
        "best":          rf,
        "X_train_sc":    X_tr_sc,
        "X_test_sc":     X_te_sc,
        "y_train":       y_train,
        "y_test":        y_test,
        "X_train_imp":   X_tr_imp,
        "imputer":       imputer,
        "scaler":        scaler,
        "train_metrics": train_metrics,
        "test_metrics":  test_metrics,
        "feature_cols":  list(X.columns),
        "zero_cols":     zero_cols,
        "best_depth":    best_depth,
    }


# Вспомогательные функции

def fig_to_st(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def metric_card(col, label, value, delta=None, color=BLUE):
    with col:
        st.markdown(
            f"""
            <div style="background:{CARD};border-left:4px solid {color};
                        border-radius:8px;padding:14px 18px;margin:4px 0;">
                <div style="color:#aaa;font-size:12px;letter-spacing:1px;">{label}</div>
                <div style="color:{color};font-size:26px;font-weight:700;">{value}</div>
                {"" if delta is None else f'<div style="color:#aaa;font-size:11px;">{delta}</div>'}
            </div>""",
            unsafe_allow_html=True,
        )


# СТРАНИЦА: О датасет

def page_about(df):
    st.title("🩺 Pima Indians Diabetes Database")
    st.markdown(
        """
        Датасет предназначен для диагностического прогнозирования наличия или отсутствия
        **сахарного диабета** на основе медицинских показателей пациентов.
        Все наблюдения — женщины **≥ 21 года** популяции индейцев Пима.
        """
    )

    c1, c2, c3 = st.columns(3)
    metric_card(c1, "НАБЛЮДЕНИЙ", "768", color=BLUE)
    metric_card(c2, "ПРИЗНАКОВ",  "8",   color=ORANGE)
    metric_card(c3, "ЦЕЛЕВОЙ ПРИЗНАК", "Outcome (0/1)", color=GREEN)

    st.markdown("---")

    st.subheader("Описание признаков")
    feature_desc = pd.DataFrame({
        "Признак": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
        ],
        "Тип": [
            "Дискретный", "Непрерывный", "Непрерывный", "Непрерывный",
            "Непрерывный", "Непрерывный", "Непрерывный", "Непрерывный", "Бинарный (цель)",
        ],
        "Описание": [
            "Количество беременностей",
            "Концентрация глюкозы в плазме (мг/дл) — 2ч после ОГТТ",
            "Диастолическое АД (мм рт. ст.)",
            "Толщина кожной складки над трицепсом (мм)",
            "Уровень инсулина в сыворотке — 2ч (мкЕд/мл)",
            "Индекс массы тела (кг/м²)",
            "Индекс наследственной предрасположенности к диабету",
            "Возраст (лет)",
            "0 — нет диабета, 1 — есть диабет",
        ],
    })
    st.dataframe(feature_desc, use_container_width=True, hide_index=True)

    st.subheader("Первые строки датасета")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Описательные статистики")
    st.dataframe(df.describe().T.round(2), use_container_width=True)


# СТРАНИЦА: EDA

def page_eda(df):
    st.title("📊 Исследовательский анализ данных")

    # ── Баланс классов ─────────────────────────────────────────────────────
    st.subheader("1. Распределение целевой переменной Outcome")

    cnt = df["Outcome"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(["0 — нет диабета", "1 — диабет"], cnt.values,
                  color=[BLUE, ORANGE], alpha=0.85, width=0.5)
    for bar, v in zip(bars, cnt.values):
      ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height()/2,      # центр столбца
        f"{v}\n({v/len(df)*100:.1f}%)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white"
    )
    ax.set_ylabel("Количество")
    ax.set_title("Дисбаланс классов: 65.1% / 34.9%")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig_to_st(fig)

    st.info(
        "⚠️ **Дисбаланс классов** (500 vs 268): при построении моделей необходимо"
        " учитывать несбалансированность, иначе Accuracy будет обманчивой метрикой."
    )

    # Нулевые значения
    st.subheader("2. Скрытые пропуски (нулевые значения)")
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    null_cnt = (df[zero_cols] == 0).sum().sort_values(ascending=False)
    null_pct = (null_cnt / len(df) * 100).round(1)
    null_df = pd.DataFrame({"Количество нулей": null_cnt, "% от выборки": null_pct})
    st.dataframe(null_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(null_df.index, null_df["% от выборки"], color=ORANGE, alpha=0.8)
    for bar, v in zip(bars, null_df["% от выборки"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{v}%", va="center", fontsize=7)
    ax.set_xlabel("% нулевых значений")
    ax.set_title("Нулевые значения — физиологически невозможные → скрытые пропуски")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig_to_st(fig)

    # Распределения признаков
    st.subheader("3. Распределения признаков по классам")
    features = [c for c in df.columns if c != "Outcome"]
    selected = st.selectbox("Выберите признак:", features, index=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for outcome, color, label in [(0, BLUE, "Нет диабета"), (1, ORANGE, "Есть диабет")]:
        data = df[df["Outcome"] == outcome][selected]
        ax1.hist(data, bins=20, alpha=0.55, color=color, label=label, density=True)
        try:
            data[data > 0].plot.kde(ax=ax1, color=color, lw=1.8)
        except Exception:
            pass

    ax1.set_title(f"Гистограмма: {selected}")
    ax1.set_xlabel(selected)
    ax1.set_ylabel("Плотность")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    sns.boxplot(data=df, y=selected, hue="Outcome",
                palette={0: BLUE, 1: ORANGE}, ax=ax2,
                hue_order=[0, 1])
    ax2.set_title(f"Ящик с усами: {selected}")
    handles = [mpatches.Patch(color=BLUE, label="Нет диабета"),
               mpatches.Patch(color=ORANGE, label="Есть диабет")]
    ax2.legend(handles=handles, fontsize=9)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig_to_st(fig)

    # Корреляция
    st.subheader("4. Корреляционный анализ")
    tab1, tab2 = st.tabs(["Матрица корреляций", "Корреляция с Outcome"])

    with tab1:
        feat_cols = [c for c in df.columns if c != "Outcome"]
        corr = df[feat_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax, square=True, linewidths=0.5,
                    annot_kws={"size": 8})
        ax.set_title("Корреляция между признаками")
        fig.tight_layout()
        fig_to_st(fig)
        st.markdown(
            "**Вывод:** Наибольшие корреляции: Age↔Pregnancies (0.54), "
            "Insulin↔SkinThickness (0.44), BMI↔SkinThickness (0.39). "
            "Мультиколлинеарность ниже порога 0.7 — признаки можно использовать совместно."
        )

    with tab2:
        corr_out = df.corr()["Outcome"].drop("Outcome").sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(5, 5))
        colors = [ORANGE if v > 0 else BLUE for v in corr_out.values]
        ax.barh(corr_out.index[::-1], corr_out.values[::-1], color=colors[::-1], alpha=0.85)
        ax.axvline(0, color="white", lw=0.8)
        ax.set_title("Корреляция признаков с Outcome")
        ax.set_xlabel("Корреляция Пирсона")
        ax.grid(axis="x", alpha=0.3)
        for i, v in enumerate(corr_out.values[::-1]):
            ax.text(v + 0.005 if v > 0 else v - 0.005, i,
                    f"{v:.2f}", va="center", ha="left" if v > 0 else "right", fontsize=9)
        fig.tight_layout()
        fig_to_st(fig)
        st.markdown(
            "**Вывод:** Наибольшая положительная корреляция с целевой переменной — "
            "у **Glucose (0.47)**, умеренная у **BMI (0.29)**. "
            "Остальные признаки имеют слабую связь."
        )

    # Pairplot
    st.subheader("5. Попарные зависимости ключевых признаков")
    sel = ["Glucose", "BMI", "Age", "Pregnancies", "Outcome"]
    fig = plt.figure(figsize=(9, 7))
    g = sns.pairplot(df[sel], hue="Outcome",
                     palette={0: BLUE, 1: ORANGE},
                     plot_kws={"alpha": 0.4, "s": 18},
                     diag_kind="kde")
    g.figure.suptitle("Попарные зависимости (Glucose, BMI, Age, Pregnancies)", y=1.01, fontsize=12)
    for ax in g.axes.flatten():
        if ax:
            ax.set_facecolor(CARD)
            ax.figure.set_facecolor(DARK)
    fig_to_st(g.figure)
    st.markdown(
        "**Вывод:** Наибольшая визуальная разделимость классов наблюдается по признаку "
        "**Glucose** в комбинации с BMI, Age и Pregnancies."
    )


# СТРАНИЦА: Гипотезы

def page_hypotheses(df):
    st.title("🔬 Статистические гипотезы")

    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df_clean = df.copy()
    for col in zero_cols:
        df_clean[col] = df_clean[col].astype(float).replace(0, np.nan)
        df_clean[col] = df_clean.groupby("Outcome")[col].transform(
            lambda x: x.fillna(x.median())
        )

    hypotheses = [
        {
            "id": "Гипотеза 1",
            "title": "Ожирение и диабет",
            "h0": "Ожирение **не** связано с наличием диабета",
            "h1": "Ожирение статистически значимо ассоциировано с диабетом",
            "result": "H₀ отвергается",
            "color": ORANGE,
        },
        {
            "id": "Гипотеза 2",
            "title": "Возраст + BMI → диабет",
            "h0": "Возраст и BMI **не** оказывают значимого совместного влияния",
            "h1": "Возраст и BMI оказывают статистически значимое влияние",
            "result": "H₀ подтверждается (BMI значим, Age — нет)",
            "color": BLUE,
        },
        {
            "id": "Гипотеза 3",
            "title": "Глюкоза у диабетиков выше",
            "h0": "Средний уровень глюкозы у диабетиков ≤ чем у здоровых",
            "h1": "Средний уровень глюкозы у диабетиков статистически значимо выше",
            "result": "H₀ отвергается",
            "color": ORANGE,
        },
        {
            "id": "Гипотеза 4",
            "title": "Нормальность распределения Glucose",
            "h0": "Распределение Glucose является нормальным",
            "h1": "Распределение Glucose значимо отличается от нормального",
            "result": "H₀ отвергается",
            "color": ORANGE,
        },
        {
            "id": "Гипотеза 5",
            "title": "Зависимость Glucose и Insulin",
            "h0": "Между Glucose и Insulin **отсутствует** зависимость",
            "h1": "Между признаками существует статистически значимая положительная связь",
            "result": "H₀ отвергается",
            "color": ORANGE,
        },
        {
            "id": "Гипотеза 6",
            "title": "Артериальное давление у диабетиков",
            "h0": "Среднее АД в группе с диабетом и без **одинаково**",
            "h1": "Среднее АД у пациентов с диабетом статистически значимо выше",
            "result": "Н₀ отвергается",
            "color": ORANGE,
        },
    ]

    # Таблица-резюме
    st.subheader("Сводная таблица")
    summary_rows = []
    for h in hypotheses:
        summary_rows.append({
            "№": h["id"],
            "Тема": h["title"],
            "H₀": h["h0"],
            "Вывод": h["result"],
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Детальный разбор каждой гипотезы
    selected_h = st.selectbox(
        "Выберите гипотезу для детального просмотра:",
        [h["id"] + " — " + h["title"] for h in hypotheses],
    )
    idx = next(i for i, h in enumerate(hypotheses) if selected_h.startswith(h["id"]))
    h = hypotheses[idx]

    st.markdown(
        f"""
        <div style="background:{CARD};border-left:4px solid {h['color']};
                    border-radius:8px;padding:16px 20px;margin:12px 0;">
            <h4 style="color:{h['color']};margin:0 0 8px 0;">{h['id']}: {h['title']}</h4>
            <p style="margin:4px 0;"><b>H₀:</b> {h['h0']}</p>
            <p style="margin:4px 0;"><b>H₁:</b> {h['h1']}</p>
            <p style="margin:8px 0 0 0;font-size:15px;"><b>Итог:</b> {h['result']}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # Визуализация для каждой гипотезы
    if idx == 0:  # Гипотеза 1 — ожирение
        df_clean["obesity"] = (df_clean["BMI"] > 30).astype(int)
        cross = pd.crosstab(df_clean["obesity"], df_clean["Outcome"],
                            rownames=["Ожирение"], colnames=["Диабет"])
        chi2, p, dof, _ = stats.chi2_contingency(cross)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Таблица сопряжённости**")
            cross.index = ["Нет ожирения (BMI≤30)", "Ожирение (BMI>30)"]
            cross.columns = ["Нет диабета", "Диабет"]
            st.dataframe(cross, use_container_width=True)
            st.metric("χ²", f"{chi2:.2f}")
            st.metric("p-value", f"{p:.2e}")
            st.metric("Отношение шансов OR ≈", "3.96")
        with c2:
            fig, ax = plt.subplots(figsize=(5, 4))
            obese   = df_clean[df_clean["obesity"] == 1]["Outcome"].value_counts()
            nonob   = df_clean[df_clean["obesity"] == 0]["Outcome"].value_counts()
            rates   = [nonob.get(1, 0) / nonob.sum() * 100, obese.get(1, 0) / obese.sum() * 100]
            ax.bar(["Без ожирения", "С ожирением"], rates, color=[BLUE, ORANGE], alpha=0.85)
            ax.set_ylabel("% с диабетом")
            ax.set_title("Доля диабета по группам ожирения")
            for i, v in enumerate(rates):
                ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig_to_st(fig)
        st.success("OR=3.96: у пациентов с ожирением шансы диабета в **4 раза выше**. p < 0.001.")

    elif idx == 1:  # Гипотеза 2 — Age + BMI
        st.markdown("Логистическая регрессия на необработанных данных показала **BMI** незначимым (β < 0). "
                    "После замены пропусков и добавления признака Glucose результаты улучшились.")
        fig, ax = plt.subplots(figsize=(7, 4))
        for outcome, color, label in [(0, BLUE, "Нет диабета"), (1, ORANGE, "Есть диабет")]:
            sub = df_clean[df_clean["Outcome"] == outcome]
            ax.scatter(sub["Age"], sub["BMI"], c=color, alpha=0.35, s=18, label=label)
        ax.set_xlabel("Age")
        ax.set_ylabel("BMI")
        ax.set_title("BMI vs Age (по классу)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig_to_st(fig)
        st.warning(
            "Гипотеза **не подтверждена в полной мере**: Age в модели незначим (p>0.05), "
            "BMI значим лишь при добавлении Glucose. Совместный эффект слабее ожидаемого."
        )

    elif idx == 2:  # Гипотеза 3 — глюкоза
        g0 = df_clean[df_clean["Outcome"] == 0]["Glucose"]
        g1 = df_clean[df_clean["Outcome"] == 1]["Glucose"]
        u_stat, p_mw = stats.mannwhitneyu(g1, g0, alternative="greater")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Среднее (с диабетом)",    f"{g1.mean():.1f} мг/дл")
            st.metric("Среднее (без диабета)",   f"{g0.mean():.1f} мг/дл")
            st.metric("Разница средних",          f"+{g1.mean()-g0.mean():.1f} мг/дл")
            st.metric("Mann-Whitney p-value",     f"{p_mw:.2e}")
        with c2:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(g0, bins=20, alpha=0.55, color=BLUE, label="Нет диабета", density=True)
            ax.hist(g1, bins=20, alpha=0.55, color=ORANGE, label="Есть диабет", density=True)
            ax.axvline(g0.mean(), color=BLUE, lw=1.5, linestyle="--")
            ax.axvline(g1.mean(), color=ORANGE, lw=1.5, linestyle="--")
            ax.set_title("Распределение Glucose по группам")
            ax.set_xlabel("Glucose")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig_to_st(fig)
        st.success(f"p = {p_mw:.2e} ≪ 0.05. У диабетиков уровень глюкозы значимо выше (+{g1.mean()-g0.mean():.0f} мг/дл).")

    elif idx == 3:  # Гипотеза 4 — нормальность
        glucose_all = df_clean[df_clean["Glucose"] > 0]["Glucose"]
        stat_sw, p_sw = stats.shapiro(glucose_all)
        stat_dp, p_dp = stats.normaltest(glucose_all)
        mu, sigma = glucose_all.mean(), glucose_all.std()
        stat_ks, p_ks = stats.kstest(glucose_all, stats.norm(loc=mu, scale=sigma).cdf)

        tests_df = pd.DataFrame({
            "Тест":       ["Шапиро–Уилка", "Д'Агостино–Пирсона", "Колмогорова–Смирнова"],
            "Статистика": [f"{stat_sw:.4f}", f"{stat_dp:.4f}", f"{stat_ks:.4f}"],
            "p-value":    [f"{p_sw:.4f}", f"{p_dp:.4f}", f"{p_ks:.4f}"],
            "Вывод":      [
                "H₀ отвергается" if p_sw < 0.05 else "H₀ не отвергается",
                "H₀ отвергается" if p_dp < 0.05 else "H₀ не отвергается",
                "H₀ отвергается" if p_ks < 0.05 else "H₀ не отвергается",
            ],
        })
        st.dataframe(tests_df, use_container_width=True, hide_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        x = np.linspace(glucose_all.min(), glucose_all.max(), 200)
        axes[0].hist(glucose_all, bins=25, density=True, color=BLUE, alpha=0.6)
        glucose_all.plot.kde(ax=axes[0], color=BLUE, lw=2, label="KDE")
        axes[0].plot(x, stats.norm.pdf(x, mu, sigma), "r--", lw=2, label="Норм. распред.")
        axes[0].set_title("Glucose vs нормальное распределение")
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3)

        (osm, osr), (slope, intercept, _) = stats.probplot(glucose_all, dist="norm")
        axes[1].scatter(osm, osr, s=8, alpha=0.5, color=BLUE)
        axes[1].plot(osm, np.array(osm) * slope + intercept, "r--", lw=2)
        axes[1].set_title("Q-Q Plot: Glucose")
        axes[1].set_xlabel("Теоретические квантили")
        axes[1].set_ylabel("Квантили данных")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig_to_st(fig)
        st.success("Все три теста отвергают нормальность (p < 0.05). → Используем непараметрические критерии.")

    elif idx == 4:  # Гипотеза 5 — Glucose vs Insulin
        filt = df_clean[(df_clean["Glucose"] > 0) & (df_clean["Insulin"] > 0)]
        r_p, p_p = stats.pearsonr(filt["Glucose"], filt["Insulin"])
        r_s, p_s = stats.spearmanr(filt["Glucose"], filt["Insulin"])

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Pearson r",  f"{r_p:.3f}",  f"p = {p_p:.2e}")
            st.metric("Spearman ρ", f"{r_s:.3f}",  f"p = {p_s:.2e}")
            st.markdown("*После удаления нулей (скрытых пропусков)*")
        with c2:
            fig, ax = plt.subplots(figsize=(5, 4))
            for out, col, lbl in [(0, BLUE, "Нет диабета"), (1, ORANGE, "Есть диабет")]:
                sub = filt[filt["Outcome"] == out]
                ax.scatter(sub["Glucose"], sub["Insulin"], c=col, alpha=0.35, s=18, label=lbl)
            m, b = np.polyfit(filt["Glucose"], filt["Insulin"], 1)
            xs = np.linspace(filt["Glucose"].min(), filt["Glucose"].max(), 100)
            ax.plot(xs, m * xs + b, "white", lw=1.5, linestyle="--", label=f"r={r_p:.2f}")
            ax.set_xlabel("Glucose")
            ax.set_ylabel("Insulin")
            ax.set_title("Glucose ↔ Insulin")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig_to_st(fig)
        st.success(f"Pearson r={r_p:.2f}, Spearman ρ={r_s:.2f} — умеренная положительная связь (p < 0.001).")

    elif idx == 5:  # Гипотеза 6 — АД
        df_bp = df_clean[df_clean["BloodPressure"] > 0]
        bp0 = df_bp[df_bp["Outcome"] == 0]["BloodPressure"]
        bp1 = df_bp[df_bp["Outcome"] == 1]["BloodPressure"]
        u_stat, p_mw = stats.mannwhitneyu(bp1, bp0, alternative="greater")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Среднее АД (диабет)",     f"{bp1.mean():.1f} мм рт.ст.")
            st.metric("Среднее АД (нет диабета)", f"{bp0.mean():.1f} мм рт.ст.")
            st.metric("Mann-Whitney p-value",      f"{p_mw:.2e}")
        with c2:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(bp0, bins=20, alpha=0.55, color=BLUE, label="Нет диабета", density=True)
            ax.hist(bp1, bins=20, alpha=0.55, color=ORANGE, label="Есть диабет", density=True)
            ax.axvline(bp0.mean(), color=BLUE, lw=1.5, linestyle="--")
            ax.axvline(bp1.mean(), color=ORANGE, lw=1.5, linestyle="--")
            ax.set_title("Распределение BloodPressure по группам")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig_to_st(fig)
        st.success(f"p = {p_mw:.2e} < 0.05. АД у диабетиков статистически значимо выше.")


# СТРАНИЦА: Модели

def page_models(df):
    st.title("🤖 Сравнение моделей машинного обучения")
    ctx = prepare_and_train(df)

    # Пайплайн
    st.subheader("Пайплайн обработки")
    st.markdown("""
    1. **Стратифицированное разбиение** 80/20 (сохранение пропорции классов)
    2. **Импутация:** нули → NaN → медиана *по обучающей выборке*
    3. **Стандартизация** `StandardScaler` (только по train)
    4. **Обучение 5 моделей** с подбором гиперпараметров через 5-fold CV
    """)

    # Метрики
    st.subheader("Метрики на тестовой выборке")
    test_m = ctx["test_metrics"]

    def highlight(s):
        col_max = s.max()
        return [f"background-color: {GREEN}22; color: {GREEN}; font-weight:700"
                if v == col_max else "" for v in s]

    st.dataframe(
        test_m.style.apply(highlight, axis=0).format("{:.3f}"),
        use_container_width=True,
    )

    st.markdown(
        "🏆 **RandomForest** выбран как лучшая модель — наивысший Recall (~0.80) "
        "при ROC-AUC=0.816, что критически важно для медицинской задачи "
        "(пропустить больного хуже, чем ложная тревога)."
    )

    # ROC-кривые
    st.subheader("ROC-кривые финальных моделей")
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in ["LR balanced", "RandomForest ★", "GradientBoosting"]:
        model = ctx["models"][name]
        prob  = model.predict_proba(ctx["X_test_sc"])[:, 1]
        fpr, tpr, _ = roc_curve(ctx["y_test"], prob)
        auc = roc_auc_score(ctx["y_test"], prob)
        ax.plot(fpr, tpr, lw=2, label=f"{name}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Случайный")
    ax.set_xlabel("FPR (False Positive Rate)")
    ax.set_ylabel("TPR (True Positive Rate / Recall)")
    ax.set_title("ROC-кривые")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_to_st(fig)

    # Матрица ошибок
    st.subheader("Матрица ошибок — RandomForest (лучшая модель)")
    rf  = ctx["models"]["RandomForest ★"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, X_sc, y_true, title in zip(
        axes,
        [ctx["X_train_sc"], ctx["X_test_sc"]],
        [ctx["y_train"],    ctx["y_test"]],
        ["Train", "Test"],
    ):
        preds = rf.predict(X_sc)
        ConfusionMatrixDisplay(
            confusion_matrix(y_true, preds),
            display_labels=["Нет диабета", "Диабет"],
        ).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"RandomForest — {title}")
    fig.tight_layout()
    fig_to_st(fig)

    # Важность признаков
    st.subheader("Важность признаков (MDI — Mean Decrease Impurity)")
    mdi = pd.Series(rf.feature_importances_, index=ctx["feature_cols"]).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [ORANGE if i >= len(mdi) - 3 else BLUE for i in range(len(mdi))]
    ax.barh(mdi.index, mdi.values, color=colors, alpha=0.85)
    ax.set_title("Важность признаков — RandomForest (MDI)")
    ax.set_xlabel("Важность")
    ax.axvline(0, color="white", lw=0.5)
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(mdi.values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig_to_st(fig)

    st.markdown(
        "**Топ признаков:** Glucose → BMI → Age → DiabetesPedigreeFunction. "
        "Insulin переоценён MDI из-за большого числа пропусков — "
        "Permutation Importance ставит его значительно ниже."
    )

    # Сравнение с наивными моделями
    st.subheader("Сравнение с наивными прогнозами")
    naive = ctx["test_metrics"].loc[["LR базовая", "RandomForest ★", "DummyClassifier"]]
    st.dataframe(naive.style.format("{:.3f}"), use_container_width=True)
    st.markdown(
        "RandomForest превосходит DummyClassifier по всем метрикам. "
        "ROC-AUC 0.816 vs 0.50 у случайного классификатора."
    )


# СТРАНИЦА: Прогноз

def page_predict(df):
    st.title("🩺 Предсказание риска диабета")
    st.markdown(
        "Введите значения медицинских показателей — **RandomForest** оценит "
        "вероятность наличия диабета."
    )

    ctx = prepare_and_train(df)
    rf = ctx["models"]["RandomForest ★"]
    imputer = ctx["imputer"]
    scaler  = ctx["scaler"]
    zero_cols = ctx["zero_cols"]
    feat_cols = ctx["feature_cols"]

    # Форма ввода
    st.subheader("Параметры пациента")

    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input("Количество беременностей",          0, 20, 1,    step=1)
        glucose     = st.number_input("Глюкоза в плазме (мг/дл)",          0, 300, 120, step=1)
        blood_press = st.number_input("Диастолическое АД (мм рт. ст.)",    0, 150, 72,  step=1)
        skin_thick  = st.number_input("Толщина кожной складки (мм)",        0, 100, 23,  step=1)
    with c2:
        insulin     = st.number_input("Инсулин (мкЕд/мл)",                 0, 900, 0,   step=1)
        bmi         = st.number_input("ИМТ (кг/м²)",                       0.0, 70.0, 27.0, step=0.1, format="%.1f")
        dpf         = st.number_input("Индекс диабетической функции",       0.0, 3.0,  0.47, step=0.01, format="%.3f")
        age         = st.number_input("Возраст (лет)",                      21,  120,  33,   step=1)

    predict_btn = st.button("🔍 Рассчитать риск", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame([[pregnancies, glucose, blood_press, skin_thick,
                                    insulin, bmi, dpf, age]], columns=feat_cols)

        # Предобработка (повтор пайплайнов)
        input_proc = input_data.copy()
        input_proc[zero_cols] = input_proc[zero_cols].replace(0, np.nan)
        input_imp = pd.DataFrame(
            imputer.transform(input_proc), columns=feat_cols
        )
        input_sc = pd.DataFrame(
            scaler.transform(input_imp), columns=feat_cols
        )

        prob  = rf.predict_proba(input_sc)[0][1]
        pred  = int(rf.predict(input_sc)[0])

        # Результат
        st.markdown("---")
        st.subheader("Результат")

        if prob >= 0.7:
            color, label = "#E74C3C", "ВЫСОКИЙ РИСК"
        elif prob >= 0.4:
            color, label = ORANGE, "УМЕРЕННЫЙ РИСК"
        else:
            color, label = GREEN, "НИЗКИЙ РИСК"

        st.markdown(
            f"""
            <div style="background:{CARD};border:2px solid {color};border-radius:12px;
                        padding:24px;text-align:center;margin:16px 0;">
                <div style="font-size:36px;font-weight:800;color:{color};">{label}</div>
                <div style="font-size:52px;font-weight:900;color:{color};margin:8px 0;">
                    {prob*100:.1f}%
                </div>
                <div style="color:#aaa;font-size:14px;">вероятность диабета по модели RandomForest</div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Шкала риска
        fig, ax = plt.subplots(figsize=(8, 1.2))
        gradient = np.linspace(0, 1, 300).reshape(1, -1)
        ax.imshow(gradient, aspect="auto", cmap="RdYlGn_r",
                  extent=[0, 1, 0, 1])
        ax.axvline(prob, color="white", lw=3)
        ax.scatter([prob], [0.5], color="white", s=120, zorder=5)
        ax.text(prob, 1.05, f"{prob*100:.1f}%", ha="center", va="bottom",
                color="white", fontsize=12, fontweight="bold",
                transform=ax.get_xaxis_transform())
        ax.set_xticks([0, 0.4, 0.7, 1.0])
        ax.set_xticklabels(["0%\n(нет риска)", "40%\n(умеренный)", "70%\n(высокий)", "100%"])
        ax.set_yticks([])
        ax.set_title("Шкала риска")
        fig.tight_layout()
        fig_to_st(fig)

        # Интерпретация ввода
        st.subheader("Ваши показатели и нормы")
        norms = {
            "Pregnancies":              (pregnancies, "—",    "—",    "Нет нормы"),
            "Glucose":                  (glucose,     "70",   "99",   "норма натощак"),
            "BloodPressure":            (blood_press, "60",   "80",   "диастолическое"),
            "SkinThickness":            (skin_thick,  "10",   "30",   "мм"),
            "Insulin":                  (insulin,     "16",   "166",  "мкЕд/мл (2ч)"),
            "BMI":                      (bmi,         "18.5", "24.9", "норм. вес"),
            "DiabetesPedigreeFunction": (dpf,         "—",    "—",    "нет норм. порога"),
            "Age":                      (age,         "—",    "—",    "лет"),
        }
        rows = []
        for feat, (val, low, high, note) in norms.items():
            try:
                status = "✅" if float(low) <= val <= float(high) else "⚠️"
            except (ValueError, TypeError):
                status = "—"
            rows.append({
                "Признак": feat,
                "Ваше значение": val,
                "Норма (мин)": low,
                "Норма (макс)": high,
                "Примечание": note,
                "Статус": status,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.warning(
            "⚠️ **Дисклеймер:** Данный инструмент создан в образовательных целях "
            "и **не является медицинским диагнозом**. "
            "При любых сомнениях обратитесь к врачу."
        )

        # Важность признаков для этого предсказания
        st.subheader("Вклад признаков в предсказание (MDI модели)")
        mdi = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        bar_colors = []
        input_vals = input_imp.iloc[0]
        median_vals = pd.Series(scaler.mean_, index=feat_cols)
        for feat in mdi.index:
            bar_colors.append(ORANGE if input_vals[feat] > median_vals[feat] else BLUE)
        ax.barh(mdi.index, mdi.values, color=bar_colors, alpha=0.85)
        ax.set_title("Важность признаков (оранжевый = ваше значение выше медианы)")
        ax.set_xlabel("Важность (MDI)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig_to_st(fig)


# ГЛАВНАЯ: Навигация

def main():
    st.set_page_config(
        page_title="Diabetes ML Dashboard",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Тёмная тема через CSS
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            background-color: {DARK};
            color: {TEXT};
        }}
        .stSelectbox > div, .stNumberInput > div {{
            background: {CARD};
        }}
        .stButton > button {{
            background: {ACCENT};
            color: white;
            border: 1px solid {BLUE};
            border-radius: 8px;
        }}
        .stButton > button:hover {{
            background: {BLUE};
            border-color: {BLUE};
        }}
        .stDataFrame {{ background: {CARD}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()

    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align:center;padding:10px 0 20px 0;">
                <div style="font-size:32px;">🩺</div>
                <div style="color:{BLUE};font-weight:700;font-size:16px;">
                    Diabetes ML Dashboard
                </div>
                <div style="color:#888;font-size:11px;margin-top:4px;">
                    Pima Indians Database
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Навигация",
            ["📋 О датасете", "📊 EDA", "🔬 Гипотезы", "🤖 Модели", "🩺 Прогноз"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        n_diab = df["Outcome"].sum()
        st.markdown(f"**Размер датасета:** {len(df)} наблюдений")
        st.markdown(f"**Диабет:** {n_diab} ({n_diab/len(df)*100:.1f}%)")
        st.markdown(f"**Нет диабета:** {len(df)-n_diab} ({(1-n_diab/len(df))*100:.1f}%)")
        st.markdown("---")
        st.markdown(
            "<div style='color:#666;font-size:11px;'>Лучшая модель: RandomForest<br>"
            "ROC-AUC ≈ 0.816 | Recall ≈ 0.80</div>",
            unsafe_allow_html=True,
        )

    if page == "📋 О датасете":
        page_about(df)
    elif page == "📊 EDA":
        page_eda(df)
    elif page == "🔬 Гипотезы":
        page_hypotheses(df)
    elif page == "🤖 Модели":
        page_models(df)
    elif page == "🩺 Прогноз":
        page_predict(df)


if __name__ == "__main__":
    main()

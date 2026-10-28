"""
UAP Students — Midterm Grades & Study Habits
Descriptive statistics + parametric / non-parametric hypothesis testing in Streamlit.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(
    page_title="UAP Study Habits — Statistical Report",
    page_icon="📊",
    layout="wide",
)

DATA_FILE = Path(__file__).parent / "uap_survey.xlsx"

PALETTE = ["#2E5EAA", "#E07A2E", "#3A9A6B", "#B8466B", "#7A5CB0", "#8C8C3A"]
px.defaults.color_discrete_sequence = PALETTE
px.defaults.template = "plotly_white"

# ----------------------------------------------------------------------------
# 1. DATA LOADING AND CLEANING
# ----------------------------------------------------------------------------
# Each survey question is matched by the beginning of its (long) column header.
QUESTIONS = {
    "semester": "What semester of university",
    "midterm": "What was your average grade on the midterm",
    "gpa_prev": "What was your overall GPA",
    "courses": "How many courses are you currently taking",
    "attendance": "What is your average attendance percentage",
    "study_hours_day": "On average, how many hours do you study per day",
    "study_days_week": "How many days per week on average",
    "days_in_advance": "How far in advance did you start studying",
    "preparation": "How would you describe your level of preparation",
    "job": "Do you currently have a job",
    "work_hours": "If you work, how many hours do you work per week",
    "interference": "To what extent do your work or personal",
    "program": "Do you currently participate in a scholarship",
    "scholarship_hours": "If applicable, how many scholarship service hours",
    "sleep": "How would you rate the quality of your sleep",
    "method": "Which study method did you primarily use",
    "satisfaction": "How satisfied are you with your performance",
    "difficulty": "How difficult did you find the exams",
}

LABELS = {
    "semester": "Semester",
    "midterm": "Midterm average grade",
    "gpa_prev": "Previous-semester GPA",
    "courses": "Courses taken",
    "attendance_pct": "Attendance (%)",
    "study_hours_day": "Study hours per day",
    "study_days_week": "Study days per week",
    "weekly_study_hours": "Weekly study hours (hours × days)",
    "days_in_advance": "Days studied in advance",
    "preparation": "Self-rated preparation (1–5)",
    "work_hours": "Work hours per week",
    "interference": "Responsibilities interfere (1–5)",
    "scholarship_hours": "Scholarship service hours / semester",
    "satisfaction": "Satisfaction with performance (1–5)",
    "difficulty": "Perceived exam difficulty (1–5)",
    "grade_change": "Midterm − previous GPA",
    "job": "Has a job",
    "program": "Scholarship / team participation",
    "in_scholarship": "In scholarship service program",
    "attendance": "Attendance",
    "sleep": "Sleep quality (exam week)",
    "sleep_3": "Sleep quality (3 groups)",
    "method": "Main study method",
    "study_level": "Daily study time",
    "prep_level": "Preparation level",
    "semester_stage": "Semester stage",
}

SLEEP_ORDER = ["Very bad", "Bad", "Normal", "Good", "Very good"]
ATT_ORDER = ["Less than 70%", "80%", "90%", "100%"]


def first_number(x):
    """Extract the first number from free-text answers such as '9 courses' or '40 hours'."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    m = re.search(r"\d+(\.\d+)?", str(x))
    return float(m.group()) if m else np.nan


@st.cache_data
def load_data(path):
    raw = pd.read_excel(path)
    raw.columns = [str(c).replace("\xa0", " ").strip() for c in raw.columns]

    df = pd.DataFrame()
    for key, prefix in QUESTIONS.items():
        match = [c for c in raw.columns if c.startswith(prefix)]
        df[key] = raw[match[0]].values
    df.insert(0, "id", raw["ID"].values)

    notes = []

    # Midterm grade: one answer was a range ("8 to 10") -> treated as missing
    bad = df["midterm"].astype(str).str.contains("to", case=False)
    if bad.any():
        notes.append(f"{bad.sum()} midterm grade answered as a range ('8 to 10') → treated as missing.")
    df.loc[bad, "midterm"] = np.nan
    df["midterm"] = pd.to_numeric(df["midterm"], errors="coerce")

    # Previous GPA: 0 means no previous semester (first-semester students)
    zero_gpa = (df["gpa_prev"] == 0).sum()
    if zero_gpa:
        notes.append(f"{zero_gpa} previous GPA of 0 (no previous semester) → treated as missing.")
    df.loc[df["gpa_prev"] == 0, "gpa_prev"] = np.nan

    # Free-text numeric answers
    for col in ["courses", "scholarship_hours"]:
        df[col] = df[col].apply(first_number)
    notes.append("Free-text numbers cleaned ('9 courses' → 9, '40 hours' → 40, '50 ' → 50).")

    # Scholarship hours only make sense for students in a scholarship program
    df["in_scholarship"] = np.where(
        df["program"].str.contains("scholarship|both", case=False), "Yes", "No"
    )
    df.loc[df["in_scholarship"] == "No", "scholarship_hours"] = np.nan

    # Categorical clean-up
    df["job"] = df["job"].str.strip().str.capitalize()
    df["program"] = df["program"].replace(
        {
            "Yes, I participate in a scholarship service program.": "Scholarship service",
            "Yes, I am a member of a university representative team": "Representative team",
            "Yes, both": "Both",
            "No, neither": "Neither",
        }
    )
    df["method"] = df["method"].str.strip().replace({"Practice excercises": "Practice exercises"})
    df["sleep"] = pd.Categorical(df["sleep"].str.strip(), SLEEP_ORDER, ordered=True)
    df["attendance"] = pd.Categorical(df["attendance"].str.strip(), ATT_ORDER, ordered=True)
    df["attendance_pct"] = df["attendance"].map(
        {"Less than 70%": 65, "80%": 80, "90%": 90, "100%": 100}
    ).astype(float)

    # Derived variables
    df["weekly_study_hours"] = df["study_hours_day"] * df["study_days_week"]
    df["grade_change"] = df["midterm"] - df["gpa_prev"]
    df["sleep_3"] = pd.Categorical(
        df["sleep"].map({"Very bad": "Bad / Very bad", "Bad": "Bad / Very bad",
                         "Normal": "Normal", "Good": "Good / Very good",
                         "Very good": "Good / Very good"}),
        ["Bad / Very bad", "Normal", "Good / Very good"], ordered=True,
    )
    df["study_level"] = pd.Categorical(
        np.where(df["study_hours_day"] >= 2, "2+ h/day", "0–1 h/day"),
        ["0–1 h/day", "2+ h/day"], ordered=True,
    )
    df["prep_level"] = pd.Categorical(
        pd.cut(df["preparation"], [0, 2, 3, 5], labels=["Low (1–2)", "Medium (3)", "High (4–5)"]),
        ["Low (1–2)", "Medium (3)", "High (4–5)"], ordered=True,
    )
    df["semester_stage"] = pd.Categorical(
        pd.cut(df["semester"], [-1, 4, 6, 12], labels=["Early (0–4)", "Middle (5–6)", "Late (7–9)"]),
        ["Early (0–4)", "Middle (5–6)", "Late (7–9)"], ordered=True,
    )
    return df, notes


df, cleaning_notes = load_data(DATA_FILE)

NUMERIC = [
    "midterm", "gpa_prev", "grade_change", "semester", "courses", "attendance_pct",
    "study_hours_day", "study_days_week", "weekly_study_hours", "days_in_advance",
    "preparation", "work_hours", "interference", "scholarship_hours",
    "satisfaction", "difficulty",
]
CATEGORICAL = [
    "job", "in_scholarship", "program", "attendance", "sleep", "sleep_3",
    "method", "study_level", "prep_level", "semester_stage",
]
BINARY = [c for c in CATEGORICAL if df[c].nunique() == 2]
MULTI = [c for c in CATEGORICAL if df[c].nunique() >= 3]


def lab(c):
    return LABELS.get(c, c)


# ----------------------------------------------------------------------------
# 2. HELPERS
# ----------------------------------------------------------------------------
def describe_numeric(data, cols):
    rows = []
    for c in cols:
        s = data[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        rows.append({
            "Variable": lab(c),
            "n": len(s),
            "Missing": data[c].isna().sum(),
            "Mean": s.mean(),
            "Std. dev.": s.std(),
            "Min": s.min(),
            "Q1": q1,
            "Median": s.median(),
            "Q3": q3,
            "Max": s.max(),
            "IQR": q3 - q1,
            "CV (%)": 100 * s.std() / s.mean() if s.mean() != 0 else np.nan,
            "Skewness": stats.skew(s, bias=False),
            "Kurtosis": stats.kurtosis(s, bias=False),
        })
    return pd.DataFrame(rows).set_index("Variable")


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def p_eq(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def verdict(p, alpha):
    return ("✅ **Reject H₀**", True) if p < alpha else ("❌ **Fail to reject H₀**", False)


def shapiro(s):
    s = pd.Series(s).dropna()
    if len(s) < 3 or s.nunique() < 3:
        return np.nan, np.nan
    return stats.shapiro(s)


def ordered_levels(series):
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [c for c in series.cat.categories if c in set(series.dropna())]
    return sorted(series.dropna().unique())


def result_table(rows):
    t = pd.DataFrame(rows, columns=["Test", "Type", "Statistic", "p-value", "Effect size", "Decision"])
    return t


def show_decision_box(param_p, nonparam_p, alpha, param_name, nonparam_name, normal_ok):
    rec = param_name if normal_ok else nonparam_name
    agree = (param_p < alpha) == (nonparam_p < alpha)
    msg = (
        f"**Recommended test:** {rec} "
        f"({'normality and assumptions hold' if normal_ok else 'normality is violated in at least one group, so the rank-based test is more reliable'})."
    )
    if agree:
        msg += " Both tests reach the **same decision**, so the conclusion is robust."
    else:
        msg += " ⚠️ The tests **disagree**; rely on the recommended test."
    st.info(msg)


# ----------------------------------------------------------------------------
# 3. SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.title("📊 UAP Survey")
st.sidebar.caption("Midterm grades & study habits")
page = st.sidebar.radio(
    "Section",
    ["Overview", "Descriptive statistics", "Hypothesis testing", "Conclusions"],
)
alpha = st.sidebar.select_slider("Significance level (α)", [0.01, 0.05, 0.10], value=0.05)
st.sidebar.divider()
st.sidebar.markdown(f"**Sample size:** {len(df)} students")
st.sidebar.markdown(f"**Collection date:** {pd.Timestamp('2026-10-22').strftime('%b %Y')}")
st.sidebar.markdown("**Tools:** Python · pandas · SciPy · Plotly")


# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == "Overview":
    st.title("Midterm Grades & Study Habits of UAP Students")
    st.markdown(
        "A survey was applied to **UAP students** to collect information about their midterm "
        "results and the habits that may explain them: study time, attendance, sleep, work and "
        "scholarship-service commitments. This app presents (1) a **descriptive statistics "
        "report** and (2) **parametric and non-parametric hypothesis tests** with visual results."
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Respondents", len(df))
    k2.metric("Mean midterm grade", f"{df['midterm'].mean():.2f}")
    k3.metric("Mean previous GPA", f"{df['gpa_prev'].mean():.2f}")
    k4.metric("Have a job", f"{(df['job'] == 'Yes').mean():.0%}")
    k5.metric("In scholarship service", f"{(df['in_scholarship'] == 'Yes').mean():.0%}")

    st.subheader("Research questions")
    st.markdown(
        """
1. Is the average midterm grade different from a reference grade of 8.0?
2. Did students perform differently on the midterms than in their previous semester?
3. Do students who **work** get different midterm grades from those who don't?
4. Does **sleep quality** during exam week relate to midterm grades?
5. Is **study time** or **preparation** correlated with the midterm grade?
6. Is having a job associated with responsibilities **interfering** with studies?
"""
    )

    st.subheader("Variables")
    var_info = pd.DataFrame([
        ("Midterm average grade", "Quantitative (continuous)", "0–10"),
        ("Previous-semester GPA", "Quantitative (continuous)", "0–10"),
        ("Semester, courses, study hours/day, study days/week, days in advance", "Quantitative (discrete)", "counts"),
        ("Work hours per week, scholarship service hours", "Quantitative (discrete)", "hours"),
        ("Preparation, interference, satisfaction, difficulty", "Ordinal (Likert)", "1–5"),
        ("Attendance, sleep quality", "Ordinal (categorical)", "levels"),
        ("Has a job, scholarship / team participation, study method", "Nominal", "categories"),
    ], columns=["Variable(s)", "Type", "Scale"])
    st.dataframe(var_info, hide_index=True, use_container_width=True)

    st.subheader("Data cleaning")
    for n in cleaning_notes:
        st.markdown(f"- {n}")
    st.markdown(
        "- Scholarship hours are only considered for students in a scholarship program.\n"
        "- Derived variables: **weekly study hours** = hours/day × days/week; "
        "**grade change** = midterm − previous GPA; grouped versions of sleep, study time, "
        "preparation and semester for group comparisons."
    )

    with st.expander("View the clean dataset"):
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download clean data (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            "uap_survey_clean.csv",
            "text/csv",
        )


# ============================================================================
# PAGE: DESCRIPTIVE STATISTICS
# ============================================================================
elif page == "Descriptive statistics":
    st.title("Descriptive Statistics Report")

    t1, t2, t3, t4, t5 = st.tabs(
        ["Summary table", "Distributions", "Categorical variables", "Group comparison", "Correlations"]
    )

    # --- Summary table
    with t1:
        st.subheader("Measures of central tendency, dispersion and shape")
        desc = describe_numeric(df, NUMERIC)
        st.dataframe(desc.style.format("{:.2f}", subset=desc.columns[2:]), use_container_width=True)
        st.caption(
            "CV = coefficient of variation. Skewness > 0 → long right tail; < 0 → long left tail. "
            "Kurtosis is excess kurtosis (0 = normal)."
        )
        m = df["midterm"].dropna()
        g = df["gpa_prev"].dropna()
        st.markdown(
            f"""
**Highlights**
- The average midterm grade is **{m.mean():.2f}** (median {m.median():.2f}, SD {m.std():.2f}); grades range from {m.min():.1f} to {m.max():.1f}.
- The previous-semester GPA averages **{g.mean():.2f}** (SD {g.std():.2f}), a little higher than the midterm grade.
- Students study a median of **{df['study_hours_day'].median():.0f} h/day** on **{df['study_days_week'].median():.0f} days/week**; {(df['study_hours_day'] == 0).sum()} students report 0 h/day.
- Work hours and scholarship hours are strongly right-skewed: most students don't work, while a few work 30–55 h/week.
"""
        )

    # --- Distributions
    with t2:
        st.subheader("Distribution of a quantitative variable")
        c1, c2 = st.columns([1, 3])
        with c1:
            var = st.selectbox("Variable", NUMERIC, format_func=lab, key="dist_var")
            bins = st.slider("Number of bins", 5, 30, 12)
            s = df[var].dropna()
            st.metric("Mean", f"{s.mean():.2f}")
            st.metric("Median", f"{s.median():.2f}")
            mode = s.mode()
            st.metric("Mode", ", ".join(f"{v:g}" for v in mode.iloc[:3]))
            st.metric("Std. deviation", f"{s.std():.2f}")
        with c2:
            fig = px.histogram(df, x=var, nbins=bins, marginal="box",
                               labels={var: lab(var)}, opacity=0.85)
            fig.add_vline(s.mean(), line_dash="dash", line_color=PALETTE[1],
                          annotation_text=f"mean {s.mean():.2f}")
            fig.add_vline(s.median(), line_dash="dot", line_color=PALETTE[2],
                          annotation_text=f"median {s.median():.2f}", annotation_position="bottom right")
            fig.update_layout(yaxis_title="Students", bargap=0.05, height=460)
            st.plotly_chart(fig, use_container_width=True)

        # Outliers
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        out = s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)]
        if len(out):
            st.warning(f"**{len(out)} outlier(s)** by the 1.5 × IQR rule: {', '.join(f'{v:g}' for v in sorted(out))}")
        else:
            st.success("No outliers by the 1.5 × IQR rule.")

        st.subheader("Midterm grade vs. previous GPA")
        long = df.melt(id_vars="id", value_vars=["gpa_prev", "midterm"], var_name="Measure", value_name="Grade")
        long["Measure"] = long["Measure"].map(lab)
        fig = px.box(long, x="Measure", y="Grade", color="Measure", points="all")
        fig.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    # --- Categorical
    with t3:
        st.subheader("Frequency tables")
        cat = st.selectbox("Categorical / ordinal variable",
                           ["job", "program", "attendance", "sleep", "method",
                            "preparation", "interference", "satisfaction", "difficulty", "semester"],
                           format_func=lab)
        counts = df[cat].value_counts(sort=False).sort_index()
        if isinstance(df[cat].dtype, pd.CategoricalDtype):
            counts = df[cat].value_counts(sort=False).reindex(df[cat].cat.categories)
        freq = pd.DataFrame({"Frequency": counts})
        freq["Relative (%)"] = 100 * freq["Frequency"] / freq["Frequency"].sum()
        freq["Cumulative (%)"] = freq["Relative (%)"].cumsum()
        freq.index = freq.index.astype(str)
        freq.index.name = lab(cat)
        c1, c2 = st.columns([2, 3])
        with c1:
            st.dataframe(freq.style.format({"Relative (%)": "{:.1f}", "Cumulative (%)": "{:.1f}"}),
                         use_container_width=True)
            st.markdown(f"**Mode:** {freq['Frequency'].idxmax()} ({freq['Frequency'].max()} students)")
        with c2:
            fig = px.bar(freq.reset_index(), x=lab(cat), y="Frequency", text="Frequency")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=400, xaxis_type="category")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Likert-scale items (1–5)")
        likert = ["preparation", "interference", "satisfaction", "difficulty"]
        lk = (df[likert].apply(lambda s: s.value_counts(normalize=True)).T * 100).fillna(0)
        lk = lk.reindex(columns=[1, 2, 3, 4, 5], fill_value=0)
        lk.index = [lab(c) for c in lk.index]
        colors = ["#B8466B", "#E3A1B4", "#D9D9D9", "#9DB8E0", "#2E5EAA"]
        fig = go.Figure()
        for i, lvl in enumerate([1, 2, 3, 4, 5]):
            fig.add_bar(y=lk.index, x=lk[lvl], name=str(lvl), orientation="h",
                        marker_color=colors[i], text=lk[lvl].round(0).astype(int).astype(str) + "%",
                        textposition="inside")
        fig.update_layout(barmode="stack", height=320, xaxis_title="% of students",
                          legend_title="Score", legend_orientation="h", legend_y=-0.25)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df[likert].agg(["mean", "median", "std"]).T.rename(index=lab).round(2),
                     use_container_width=True)

    # --- Group comparison
    with t4:
        st.subheader("A quantitative variable across groups")
        c1, c2 = st.columns(2)
        y = c1.selectbox("Quantitative variable", NUMERIC, format_func=lab, key="grp_y")
        g = c2.selectbox("Grouping variable", CATEGORICAL, format_func=lab, key="grp_g")
        order = ordered_levels(df[g])
        fig = px.box(df.dropna(subset=[y, g]), x=g, y=y, color=g, points="all",
                     category_orders={g: order}, labels={g: lab(g), y: lab(y)})
        fig.update_layout(showlegend=False, height=440)
        st.plotly_chart(fig, use_container_width=True)
        tab = df.groupby(g, observed=True)[y].agg(["count", "mean", "median", "std", "min", "max"]).reindex(order)
        tab.index.name = lab(g)
        st.dataframe(tab.round(2), use_container_width=True)

    # --- Correlations
    with t5:
        st.subheader("Correlation matrix (Spearman)")
        corr_vars = ["midterm", "gpa_prev", "study_hours_day", "study_days_week",
                     "days_in_advance", "preparation", "attendance_pct", "courses",
                     "work_hours", "interference", "satisfaction", "difficulty"]
        cm = df[corr_vars].corr(method="spearman")
        cm.index = cm.columns = [lab(c) for c in corr_vars]
        fig = px.imshow(cm, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1,
                        aspect="auto")
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Spearman is used because most variables are discrete or ordinal (Likert scales).")

        mcorr = cm[lab("midterm")].drop(lab("midterm")).sort_values()
        fig = px.bar(mcorr, orientation="h", labels={"value": "Spearman ρ with midterm grade", "index": ""})
        fig.update_traces(marker_color=[PALETTE[3] if v < 0 else PALETTE[0] for v in mcorr])
        fig.update_layout(showlegend=False, height=420, title="Which variables move with the midterm grade?")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: HYPOTHESIS TESTING
# ============================================================================
elif page == "Hypothesis testing":
    st.title("Hypothesis Testing")
    st.markdown(
        f"Each question is tested with a **parametric** test (assumes normality) and its "
        f"**non-parametric** counterpart (rank-based, no normality assumption). "
        f"Significance level **α = {alpha}**. You can change the variables in each tab."
    )
    with st.expander("Which test is which?"):
        st.dataframe(pd.DataFrame([
            ("One sample vs. a value", "One-sample t-test", "Wilcoxon signed-rank"),
            ("Two related measures", "Paired t-test", "Wilcoxon signed-rank (paired)"),
            ("Two independent groups", "Welch t-test", "Mann–Whitney U"),
            ("Three or more groups", "One-way ANOVA (+ Tukey HSD)", "Kruskal–Wallis (+ Mann–Whitney, Bonferroni)"),
            ("Association of two quantitative variables", "Pearson r", "Spearman ρ"),
            ("Association of two categorical variables", "—", "Chi-square / Fisher exact"),
        ], columns=["Situation", "Parametric", "Non-parametric"]), hide_index=True, use_container_width=True)

    tabs = st.tabs(["0 · Normality", "1 · One sample", "2 · Paired", "3 · Two groups",
                    "4 · Three+ groups", "5 · Correlation", "6 · Chi-square"])

    # ---------------- 0. Normality
    with tabs[0]:
        st.subheader("Shapiro–Wilk normality test")
        st.markdown("H₀: the variable follows a normal distribution · H₁: it does not.")
        rows = []
        for c in NUMERIC:
            w, p = shapiro(df[c])
            if not np.isnan(p):
                rows.append({"Variable": lab(c), "n": df[c].notna().sum(), "W": w, "p-value": p,
                             "Normal?": "Yes" if p >= alpha else "No"})
        nt = pd.DataFrame(rows)
        st.dataframe(nt.style.format({"W": "{:.3f}", "p-value": "{:.4f}"})
                     .map(lambda v: "color: #B8466B; font-weight: 600" if v == "No" else "", subset=["Normal?"]),
                     hide_index=True, use_container_width=True)
        st.markdown(
            f"**{(nt['Normal?'] == 'No').sum()} of {len(nt)}** variables are **not normal** at α = {alpha}. "
            "Survey data are often discrete, bounded and skewed, which is why each parametric test below is "
            "paired with a non-parametric alternative."
        )
        c1, c2 = st.columns(2)
        v = c1.selectbox("Q-Q plot for", NUMERIC, format_func=lab, key="qq")
        s = df[v].dropna().sort_values()
        (osm, osr), (slope, intercept, _) = stats.probplot(s, dist="norm")
        fig = go.Figure()
        fig.add_scatter(x=osm, y=osr, mode="markers", name="Data", marker_color=PALETTE[0])
        fig.add_scatter(x=osm, y=slope * np.array(osm) + intercept, mode="lines",
                        name="Normal reference", line_color=PALETTE[1])
        fig.update_layout(title=f"Normal Q-Q plot · {lab(v)}", xaxis_title="Theoretical quantiles",
                          yaxis_title="Sample quantiles", height=420)
        c1.plotly_chart(fig, use_container_width=True)
        fig = px.histogram(s, nbins=12, histnorm="probability density", opacity=0.75)
        xs = np.linspace(s.min(), s.max(), 200)
        fig.add_scatter(x=xs, y=stats.norm.pdf(xs, s.mean(), s.std()), mode="lines",
                        name="Normal curve", line_color=PALETTE[1])
        fig.update_layout(title="Histogram with fitted normal curve", showlegend=False,
                          xaxis_title=lab(v), height=420)
        c2.plotly_chart(fig, use_container_width=True)

    # ---------------- 1. One sample
    with tabs[1]:
        st.subheader("Is the average midterm grade different from a reference value?")
        c1, c2 = st.columns(2)
        v = c1.selectbox("Variable", ["midterm", "gpa_prev", "study_hours_day", "preparation",
                                      "satisfaction", "difficulty"], format_func=lab, key="os_v")
        default = {"midterm": 8.0, "gpa_prev": 8.0, "study_hours_day": 2.0}.get(v, 3.0)
        mu0 = c2.number_input("Reference value μ₀", value=default, step=0.1)
        s = df[v].dropna()
        st.markdown(f"**H₀:** μ = {mu0}  ·  **H₁:** μ ≠ {mu0}  (non-parametric: median = {mu0})")

        t, p_t = stats.ttest_1samp(s, mu0)
        d = (s.mean() - mu0) / s.std()
        diff = s - mu0
        diff = diff[diff != 0]
        w, p_w = stats.wilcoxon(diff)
        n = len(diff)
        r_rb = 1 - 2 * w / (n * (n + 1) / 2) if n else np.nan
        r_rb = r_rb if (s.median() >= mu0) else -abs(r_rb)
        ci = stats.t.interval(1 - alpha, len(s) - 1, s.mean(), stats.sem(s))
        _, p_norm = shapiro(s)

        st.dataframe(result_table([
            ("One-sample t-test", "Parametric", f"t = {t:.3f}", fmt_p(p_t), f"Cohen's d = {d:.2f}", verdict(p_t, alpha)[0].replace("*", "")),
            ("Wilcoxon signed-rank", "Non-parametric", f"W = {w:.1f}", fmt_p(p_w), f"r = {abs(r_rb):.2f}", verdict(p_w, alpha)[0].replace("*", "")),
        ]), hide_index=True, use_container_width=True)
        st.markdown(f"Mean = **{s.mean():.2f}**, median = **{s.median():.2f}**, "
                    f"{int((1-alpha)*100)}% CI for the mean: **[{ci[0]:.2f}, {ci[1]:.2f}]**. "
                    f"Shapiro–Wilk " + p_eq(p_norm) + ".")
        show_decision_box(p_t, p_w, alpha, "One-sample t-test", "Wilcoxon signed-rank", p_norm >= alpha)

        fig = px.histogram(s, nbins=14, opacity=0.8, labels={"value": lab(v)})
        fig.add_vline(mu0, line_color=PALETTE[3], line_width=3, annotation_text=f"μ₀ = {mu0}")
        fig.add_vline(s.mean(), line_dash="dash", line_color=PALETTE[1], annotation_text=f"mean {s.mean():.2f}",
                      annotation_position="top left")
        fig.add_vrect(ci[0], ci[1], fillcolor=PALETTE[1], opacity=0.15, line_width=0)
        fig.update_layout(showlegend=False, yaxis_title="Students", height=400,
                          title="Sample distribution, mean with confidence interval (shaded) and μ₀")
        st.plotly_chart(fig, use_container_width=True)
        dec = "is significantly different from" if p_w < alpha and p_t < alpha else (
            "is not significantly different from" if p_w >= alpha and p_t >= alpha else "is borderline compared to")
        st.success(f"**Conclusion:** At α = {alpha}, the average {lab(v).lower()} ({s.mean():.2f}) {dec} {mu0}.")

    # ---------------- 2. Paired
    with tabs[2]:
        st.subheader("Midterm grade vs. previous-semester GPA (same students)")
        st.markdown("**H₀:** the mean (median) difference midterm − GPA is 0 · **H₁:** it is not 0.")
        pair = df[["midterm", "gpa_prev"]].dropna()
        dif = pair["midterm"] - pair["gpa_prev"]
        t, p_t = stats.ttest_rel(pair["midterm"], pair["gpa_prev"])
        dz = dif.mean() / dif.std()
        nz = dif[dif != 0]
        w, p_w = stats.wilcoxon(nz)
        n = len(nz)
        rrb = 1 - 2 * w / (n * (n + 1) / 2)
        _, p_norm = shapiro(dif)
        st.dataframe(result_table([
            ("Paired t-test", "Parametric", f"t = {t:.3f}", fmt_p(p_t), f"Cohen's dz = {dz:.2f}", verdict(p_t, alpha)[0].replace("*", "")),
            ("Wilcoxon signed-rank", "Non-parametric", f"W = {w:.1f}", fmt_p(p_w), f"r = {abs(rrb):.2f}", verdict(p_w, alpha)[0].replace("*", "")),
        ]), hide_index=True, use_container_width=True)
        st.markdown(f"n = {len(pair)} students with both grades. Mean difference = **{dif.mean():+.2f}** points, "
                    f"median difference = **{dif.median():+.2f}**. "
                    f"{(dif < 0).sum()} students dropped, {(dif > 0).sum()} improved, {(dif == 0).sum()} stayed equal. "
                    f"Shapiro–Wilk on the differences: {p_eq(p_norm)}.")
        show_decision_box(p_t, p_w, alpha, "Paired t-test", "Wilcoxon signed-rank", p_norm >= alpha)

        c1, c2 = st.columns(2)
        fig = go.Figure()
        for _, r in pair.iterrows():
            fig.add_scatter(x=["Previous GPA", "Midterm"], y=[r["gpa_prev"], r["midterm"]], mode="lines+markers",
                            line=dict(color=PALETTE[3] if r["midterm"] < r["gpa_prev"] else PALETTE[2], width=1),
                            marker_size=5, opacity=0.5, showlegend=False)
        fig.add_scatter(x=["Previous GPA", "Midterm"], y=[pair["gpa_prev"].mean(), pair["midterm"].mean()],
                        mode="lines+markers", line=dict(color="black", width=4), marker_size=10, name="Mean")
        fig.update_layout(title="Each line is one student (red = dropped, green = improved)", height=440,
                          yaxis_title="Grade")
        c1.plotly_chart(fig, use_container_width=True)
        fig = px.histogram(dif, nbins=15, opacity=0.8, labels={"value": "Midterm − previous GPA"})
        fig.add_vline(0, line_color="black", line_width=2)
        fig.add_vline(dif.mean(), line_dash="dash", line_color=PALETTE[1], annotation_text=f"mean {dif.mean():+.2f}")
        fig.update_layout(showlegend=False, title="Distribution of differences", height=440, yaxis_title="Students")
        c2.plotly_chart(fig, use_container_width=True)
        word = "lower" if dif.mean() < 0 else "higher"
        if p_t < alpha or p_w < alpha:
            st.success(f"**Conclusion:** Midterm grades are significantly {word} than the previous-semester GPA "
                       f"(mean change {dif.mean():+.2f} points).")
        else:
            st.success("**Conclusion:** There is no significant difference between midterm grades and the previous GPA.")

    # ---------------- 3. Two groups
    with tabs[3]:
        st.subheader("Comparing two independent groups")
        c1, c2 = st.columns(2)
        y = c1.selectbox("Outcome (quantitative)", NUMERIC, format_func=lab, key="tg_y")
        g = c2.selectbox("Groups", BINARY, format_func=lab, key="tg_g")
        sub = df[[y, g]].dropna()
        lv = ordered_levels(sub[g])
        a, b = sub.loc[sub[g] == lv[0], y], sub.loc[sub[g] == lv[1], y]
        st.markdown(f"**H₀:** {lab(y)} is the same for '{lv[0]}' and '{lv[1]}' · **H₁:** it differs.")

        t, p_t = stats.ttest_ind(a, b, equal_var=False)
        lev, p_lev = stats.levene(a, b, center="median")
        sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
        d = (a.mean() - b.mean()) / sp if sp else np.nan
        u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
        rrb = 1 - 2 * u / (len(a) * len(b))
        _, pa = shapiro(a)
        _, pb = shapiro(b)
        normal_ok = (pa >= alpha) and (pb >= alpha)

        st.dataframe(result_table([
            ("Welch t-test", "Parametric", f"t = {t:.3f}", fmt_p(p_t), f"Cohen's d = {d:.2f}", verdict(p_t, alpha)[0].replace("*", "")),
            ("Mann–Whitney U", "Non-parametric", f"U = {u:.1f}", fmt_p(p_u), f"rank-biserial r = {rrb:.2f}", verdict(p_u, alpha)[0].replace("*", "")),
        ]), hide_index=True, use_container_width=True)
        summ = sub.groupby(g, observed=True)[y].agg(["count", "mean", "median", "std"]).reindex(lv)
        summ["Shapiro p"] = [pa, pb]
        summ.index.name = lab(g)
        st.dataframe(summ.style.format("{:.3f}", subset=["mean", "median", "std", "Shapiro p"]), use_container_width=True)
        st.caption(f"Levene test for equal variances: {p_eq(p_lev)} "
                   f"({'equal' if p_lev >= alpha else 'unequal'} variances). Welch's t-test does not require equal variances.")
        show_decision_box(p_t, p_u, alpha, "Welch t-test", "Mann–Whitney U", normal_ok)

        c1, c2 = st.columns(2)
        fig = px.box(sub, x=g, y=y, color=g, points="all", category_orders={g: lv},
                     labels={g: lab(g), y: lab(y)})
        fig.update_layout(showlegend=False, height=420, title="Box plot by group")
        c1.plotly_chart(fig, use_container_width=True)
        means = sub.groupby(g, observed=True)[y].agg(["mean", "sem"]).reindex(lv).reset_index()
        means["ci"] = means["sem"] * stats.t.ppf(1 - alpha / 2, summ["count"].values - 1)
        fig = px.bar(means, x=g, y="mean", error_y="ci", color=g, labels={g: lab(g), "mean": f"Mean {lab(y).lower()}"})
        fig.update_layout(showlegend=False, height=420, title=f"Means with {int((1-alpha)*100)}% CI")
        c2.plotly_chart(fig, use_container_width=True)
        p_main = p_t if normal_ok else p_u
        if p_main < alpha:
            hi = lv[0] if a.median() > b.median() or (a.median() == b.median() and a.mean() > b.mean()) else lv[1]
            st.success(f"**Conclusion:** {lab(y)} differs significantly between groups; the '{hi}' group scores higher.")
        else:
            st.success(f"**Conclusion:** There is no significant difference in {lab(y).lower()} between "
                       f"'{lv[0]}' and '{lv[1]}' at α = {alpha}.")

    # ---------------- 4. Three or more groups
    with tabs[4]:
        st.subheader("Comparing three or more groups")
        c1, c2 = st.columns(2)
        y = c1.selectbox("Outcome (quantitative)", NUMERIC, format_func=lab, key="kg_y")
        g = c2.selectbox("Groups", MULTI, index=MULTI.index("sleep_3"), format_func=lab, key="kg_g")
        sub = df[[y, g]].dropna()
        lv = ordered_levels(sub[g])
        groups = [sub.loc[sub[g] == l, y] for l in lv]
        small = [l for l, s in zip(lv, groups) if len(s) < 5]
        if small:
            st.warning(f"Small groups (n < 5): {', '.join(map(str, small))}. Results for these groups are less reliable; "
                       "consider using the grouped version of the variable.")
        st.markdown(f"**H₀:** {lab(y)} is the same in all groups · **H₁:** at least one group differs.")

        f, p_f = stats.f_oneway(*groups)
        h, p_h = stats.kruskal(*groups)
        N, k = len(sub), len(groups)
        ss_b = sum(len(s) * (s.mean() - sub[y].mean()) ** 2 for s in groups)
        ss_t = ((sub[y] - sub[y].mean()) ** 2).sum()
        eta2 = ss_b / ss_t if ss_t else np.nan
        eps2 = h / ((N ** 2 - 1) / (N + 1))
        lev, p_lev = stats.levene(*groups, center="median")
        sh = [shapiro(s)[1] for s in groups]
        normal_ok = all((np.isnan(p) or p >= alpha) for p in sh) and p_lev >= alpha

        st.dataframe(result_table([
            ("One-way ANOVA", "Parametric", f"F = {f:.3f}", fmt_p(p_f), f"η² = {eta2:.3f}", verdict(p_f, alpha)[0].replace("*", "")),
            ("Kruskal–Wallis H", "Non-parametric", f"H = {h:.3f}", fmt_p(p_h), f"ε² = {eps2:.3f}", verdict(p_h, alpha)[0].replace("*", "")),
        ]), hide_index=True, use_container_width=True)
        summ = sub.groupby(g, observed=True)[y].agg(["count", "mean", "median", "std"]).reindex(lv)
        summ["Shapiro p"] = sh
        summ.index.name = lab(g)
        st.dataframe(summ.style.format("{:.3f}", subset=["mean", "median", "std", "Shapiro p"]), use_container_width=True)
        st.caption(f"Levene test (homogeneity of variances): {p_eq(p_lev)}.")
        show_decision_box(p_f, p_h, alpha, "One-way ANOVA", "Kruskal–Wallis", normal_ok)

        c1, c2 = st.columns(2)
        fig = px.box(sub, x=g, y=y, color=g, points="all", category_orders={g: lv},
                     labels={g: lab(g), y: lab(y)})
        fig.update_layout(showlegend=False, height=430, title="Distribution by group")
        c1.plotly_chart(fig, use_container_width=True)
        means = summ.reset_index()
        means["ci"] = sub.groupby(g, observed=True)[y].sem().reindex(lv).values * \
            stats.t.ppf(1 - alpha / 2, np.maximum(summ["count"].values - 1, 1))
        fig = px.scatter(means, x=lab(g), y="mean", error_y="ci", labels={"mean": f"Mean {lab(y).lower()}"})
        fig.update_traces(marker_size=14, marker_color=PALETTE[0])
        fig.update_layout(height=430, title=f"Means with {int((1-alpha)*100)}% CI", xaxis_type="category")
        c2.plotly_chart(fig, use_container_width=True)

        st.markdown("**Post-hoc comparisons**")
        pairs = []
        tk = stats.tukey_hsd(*groups)
        m = k * (k - 1) // 2
        for i in range(k):
            for j in range(i + 1, k):
                u, pu = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
                pairs.append({
                    "Comparison": f"{lv[i]} vs {lv[j]}",
                    "Mean diff.": groups[i].mean() - groups[j].mean(),
                    "Tukey HSD p": tk.pvalue[i, j],
                    "Mann–Whitney p (Bonferroni)": min(pu * m, 1.0),
                })
        ph = pd.DataFrame(pairs)
        st.dataframe(ph.style.format({"Mean diff.": "{:+.3f}", "Tukey HSD p": "{:.4f}",
                                      "Mann–Whitney p (Bonferroni)": "{:.4f}"}),
                     hide_index=True, use_container_width=True)
        p_main = p_f if normal_ok else p_h
        if p_main < alpha:
            best = summ["median"].idxmax()
            st.success(f"**Conclusion:** {lab(y)} differs significantly across {lab(g).lower()} groups "
                       f"(highest median in '{best}'). See the post-hoc table for which pairs differ.")
        else:
            st.success(f"**Conclusion:** No significant differences in {lab(y).lower()} across "
                       f"{lab(g).lower()} groups at α = {alpha}.")

    # ---------------- 5. Correlation
    with tabs[5]:
        st.subheader("Association between two quantitative variables")
        c1, c2 = st.columns(2)
        x = c1.selectbox("X variable", NUMERIC, index=NUMERIC.index("study_hours_day"), format_func=lab, key="co_x")
        y = c2.selectbox("Y variable", NUMERIC, index=NUMERIC.index("midterm"), format_func=lab, key="co_y")
        sub = df[[x, y]].dropna()
        if x == y:
            st.error("Choose two different variables.")
        else:
            st.markdown("**H₀:** no correlation (ρ = 0) · **H₁:** there is a correlation (ρ ≠ 0).")
            r, p_r = stats.pearsonr(sub[x], sub[y])
            rho, p_s = stats.spearmanr(sub[x], sub[y])
            tau, p_k = stats.kendalltau(sub[x], sub[y])
            normal_ok = shapiro(sub[x])[1] >= alpha and shapiro(sub[y])[1] >= alpha

            def strength(v):
                v = abs(v)
                return "negligible" if v < 0.1 else "weak" if v < 0.3 else "moderate" if v < 0.5 else "strong"

            st.dataframe(result_table([
                ("Pearson r", "Parametric", f"r = {r:.3f}", fmt_p(p_r), f"r² = {r**2:.3f} ({strength(r)})", verdict(p_r, alpha)[0].replace("*", "")),
                ("Spearman ρ", "Non-parametric", f"ρ = {rho:.3f}", fmt_p(p_s), strength(rho), verdict(p_s, alpha)[0].replace("*", "")),
                ("Kendall τ", "Non-parametric", f"τ = {tau:.3f}", fmt_p(p_k), strength(tau), verdict(p_k, alpha)[0].replace("*", "")),
            ]), hide_index=True, use_container_width=True)
            show_decision_box(p_r, p_s, alpha, "Pearson r", "Spearman ρ", normal_ok)

            slope, intercept = np.polyfit(sub[x], sub[y], 1)
            jitter = st.checkbox("Add jitter (overlapping points on discrete scales)", value=True)
            rng = np.random.default_rng(1)
            px_ = sub[x] + (rng.uniform(-0.15, 0.15, len(sub)) if jitter else 0)
            fig = px.scatter(x=px_, y=sub[y], labels={"x": lab(x), "y": lab(y)}, opacity=0.75)
            xs = np.linspace(sub[x].min(), sub[x].max(), 50)
            fig.add_scatter(x=xs, y=slope * xs + intercept, mode="lines", name=f"OLS: y = {intercept:.2f} + {slope:.3f}x",
                            line_color=PALETTE[1])
            fig.update_traces(marker=dict(size=10, color=PALETTE[0]), selector=dict(mode="markers"))
            fig.update_layout(height=460, title=f"r = {r:.2f}, ρ = {rho:.2f}", legend_orientation="h", legend_y=-0.2)
            st.plotly_chart(fig, use_container_width=True)
            p_main = p_r if normal_ok else p_s
            coef = r if normal_ok else rho
            if p_main < alpha:
                st.success(f"**Conclusion:** There is a significant {strength(coef)} "
                           f"{'positive' if coef > 0 else 'negative'} association between {lab(x).lower()} and {lab(y).lower()}.")
            else:
                st.success(f"**Conclusion:** No significant association between {lab(x).lower()} and {lab(y).lower()} at α = {alpha}.")

    # ---------------- 6. Chi-square
    with tabs[6]:
        st.subheader("Independence between two categorical variables")
        cat_opts = CATEGORICAL + ["interference_hi"]
        df["interference_hi"] = np.where(df["interference"] >= 3, "Interferes (3–5)", "Little (1–2)")
        LABELS["interference_hi"] = "Responsibilities interfere"
        c1, c2 = st.columns(2)
        a_ = c1.selectbox("Variable A", cat_opts, index=cat_opts.index("job"), format_func=lab, key="ch_a")
        b_ = c2.selectbox("Variable B", cat_opts, index=cat_opts.index("interference_hi"), format_func=lab, key="ch_b")
        if a_ == b_:
            st.error("Choose two different variables.")
        else:
            st.markdown(f"**H₀:** {lab(a_)} and {lab(b_)} are independent · **H₁:** they are associated.")
            ct = pd.crosstab(df[a_], df[b_])
            ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
            chi2, p_c, dof, exp = stats.chi2_contingency(ct, correction=False)
            n = ct.values.sum()
            v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
            low = (exp < 5).mean()
            rows = [("Pearson Chi-square", "Non-parametric", f"χ² = {chi2:.3f} (df = {dof})", fmt_p(p_c),
                     f"Cramér's V = {v:.2f}", verdict(p_c, alpha)[0].replace("*", ""))]
            if ct.shape == (2, 2):
                chi2y, p_y, _, _ = stats.chi2_contingency(ct, correction=True)
                orr, p_fi = stats.fisher_exact(ct)
                rows.append(("Chi-square with Yates correction", "Non-parametric", f"χ² = {chi2y:.3f}", fmt_p(p_y), "", verdict(p_y, alpha)[0].replace("*", "")))
                rows.append(("Fisher exact test", "Non-parametric (exact)", f"OR = {orr:.2f}", fmt_p(p_fi), f"Odds ratio = {orr:.2f}", verdict(p_fi, alpha)[0].replace("*", "")))
            st.dataframe(result_table(rows), hide_index=True, use_container_width=True)
            if low > 0.2:
                st.warning(f"{low:.0%} of expected counts are below 5 — the χ² approximation may be unreliable"
                           + ("; use **Fisher's exact test**." if ct.shape == (2, 2) else "; consider grouping categories."))

            c1, c2 = st.columns(2)
            c1.markdown("**Observed counts**")
            c1.dataframe(ct, use_container_width=True)
            c2.markdown("**Expected counts under H₀**")
            c2.dataframe(pd.DataFrame(exp, index=ct.index, columns=ct.columns).round(1), use_container_width=True)

            prop = (ct.div(ct.sum(axis=1), axis=0) * 100).reset_index().melt(id_vars=a_, value_name="%")
            prop[a_] = prop[a_].astype(str)
            prop[b_] = prop[b_].astype(str)
            fig = px.bar(prop, x=a_, y="%", color=b_, text=prop["%"].round(0).astype(int).astype(str) + "%",
                         labels={a_: lab(a_), b_: lab(b_)})
            fig.update_layout(barmode="stack", height=440, title=f"{lab(b_)} within each level of {lab(a_)}",
                              yaxis_title="% of row")
            st.plotly_chart(fig, use_container_width=True)
            p_main = p_fi if (ct.shape == (2, 2) and low > 0) else p_c
            if p_main < alpha:
                st.success(f"**Conclusion:** {lab(a_)} and {lab(b_)} are significantly associated (Cramér's V = {v:.2f}).")
            else:
                st.success(f"**Conclusion:** No significant association between {lab(a_).lower()} and {lab(b_).lower()} at α = {alpha}.")


# ============================================================================
# PAGE: CONCLUSIONS
# ============================================================================
elif page == "Conclusions":
    st.title("Summary of Findings")
    st.markdown(f"Results of the main research questions at **α = {alpha}**. "
                "The *recommended* test is the parametric one only when its normality assumption holds.")

    res = []

    def add(q, ptest, pp, ntest, npv, normal_ok, effect):
        rec_p = pp if normal_ok else npv
        res.append({
            "Research question": q,
            "Parametric test": f"{ptest} ({p_eq(pp)})" if ptest else "—",
            "Non-parametric test": f"{ntest} ({p_eq(npv)})",
            "Recommended": (ptest if normal_ok and ptest else ntest),
            "Effect size": effect,
            "Significant?": "Yes" if rec_p < alpha else "No",
            "_p": rec_p,
        })

    # Q1 one-sample
    s = df["midterm"].dropna()
    _, p1 = stats.ttest_1samp(s, 8.0)
    _, p1w = stats.wilcoxon((s - 8)[s != 8])
    add("Q1. Mean midterm grade ≠ 8.0", "One-sample t", p1, "Wilcoxon", p1w, shapiro(s)[1] >= alpha,
        f"mean = {s.mean():.2f}")
    # Q2 paired
    pr = df[["midterm", "gpa_prev"]].dropna()
    dif = pr["midterm"] - pr["gpa_prev"]
    _, p2 = stats.ttest_rel(pr["midterm"], pr["gpa_prev"])
    _, p2w = stats.wilcoxon(dif[dif != 0])
    add("Q2. Midterm ≠ previous GPA", "Paired t", p2, "Wilcoxon signed-rank", p2w, shapiro(dif)[1] >= alpha,
        f"mean diff = {dif.mean():+.2f}")
    # Q3 job
    a = df.loc[df["job"] == "Yes", "midterm"].dropna()
    b = df.loc[df["job"] == "No", "midterm"].dropna()
    _, p3 = stats.ttest_ind(a, b, equal_var=False)
    u, p3u = stats.mannwhitneyu(a, b, alternative="two-sided")
    add("Q3. Midterm grade: job vs. no job", "Welch t", p3, "Mann–Whitney U", p3u,
        shapiro(a)[1] >= alpha and shapiro(b)[1] >= alpha,
        f"means {a.mean():.2f} vs {b.mean():.2f}")
    # Q4 sleep
    lv = ordered_levels(df["sleep_3"])
    gr = [df.loc[df["sleep_3"] == l, "midterm"].dropna() for l in lv]
    _, p4 = stats.f_oneway(*gr)
    _, p4k = stats.kruskal(*gr)
    add("Q4. Midterm grade by sleep quality", "ANOVA", p4, "Kruskal–Wallis", p4k,
        all(shapiro(g_)[1] >= alpha for g_ in gr),
        " / ".join(f"{g_.mean():.2f}" for g_ in gr) + " (bad / normal / good)")
    # Q5 correlations
    for xv in ["study_hours_day", "preparation", "gpa_prev"]:
        sub = df[[xv, "midterm"]].dropna()
        r, p5 = stats.pearsonr(sub[xv], sub["midterm"])
        rho, p5s = stats.spearmanr(sub[xv], sub["midterm"])
        add(f"Q5. {lab(xv)} ↔ midterm grade", "Pearson r", p5, "Spearman ρ", p5s,
            shapiro(sub[xv])[1] >= alpha and shapiro(sub["midterm"])[1] >= alpha,
            f"r = {r:.2f}, ρ = {rho:.2f}")
    # Q6 chi-square
    ih = np.where(df["interference"] >= 3, "High", "Low")
    ct = pd.crosstab(df["job"], ih)
    chi2, p6, _, exp = stats.chi2_contingency(ct, correction=False)
    _, p6f = stats.fisher_exact(ct)
    v = np.sqrt(chi2 / ct.values.sum())
    res.append({"Research question": "Q6. Job ↔ responsibilities interfere",
                "Parametric test": "—",
                "Non-parametric test": f"χ² ({p_eq(p6)}); Fisher ({p_eq(p6f)})",
                "Recommended": "Fisher exact" if (exp < 5).any() else "Chi-square",
                "Effect size": f"Cramér's V = {v:.2f}",
                "Significant?": "Yes" if min(p6, p6f) < alpha and max(p6, p6f) < alpha else "No",
                "_p": p6f if (exp < 5).any() else p6})

    out = pd.DataFrame(res)
    st.dataframe(
        out.drop(columns="_p").style.map(
            lambda v: "background-color: #DCEBDD; font-weight: 600" if v == "Yes" else "", subset=["Significant?"]),
        hide_index=True, use_container_width=True,
    )

    out["-log10(p)"] = -np.log10(out["_p"].clip(lower=1e-10))
    out["Short"] = out["Research question"].str.split(".").str[0] + ": " + \
        out["Research question"].str.split(". ", n=1).str[1].str.slice(0, 38)
    fig = px.bar(out.sort_values("-log10(p)"), x="-log10(p)", y="Short", orientation="h",
                 color="Significant?", color_discrete_map={"Yes": PALETTE[2], "No": "#B0B0B0"},
                 labels={"Short": ""})
    fig.add_vline(-np.log10(alpha), line_dash="dash", annotation_text=f"α = {alpha}", annotation_position="bottom right")
    fig.update_layout(height=420, title="Strength of evidence for each hypothesis (recommended test)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bars that pass the dashed line are statistically significant.")

    st.subheader("Interpretation")
    sig = out[out["Significant?"] == "Yes"]["Research question"].tolist()
    nsig = out[out["Significant?"] == "No"]["Research question"].tolist()
    st.markdown("**Significant results:** " + ("; ".join(sig) if sig else "none") + ".")
    st.markdown("**Not significant:** " + ("; ".join(nsig) if nsig else "none") + ".")
    st.markdown(
        f"""
- Midterm grades (mean **{s.mean():.2f}**) were on average **{abs(dif.mean()):.2f} points {'lower' if dif.mean() < 0 else 'higher'}**
  than students' previous-semester GPA.
- Most variables are **not normally distributed** (bounded grades, Likert and count data), so the
  **non-parametric tests are generally the more reliable choice**; where both tests agree, the conclusion is robust.
- Self-rated preparation and previous GPA are the variables most associated with the midterm grade,
  while daily study time on its own shows a weaker relationship.

**Limitations.** The sample is small (n = {len(df)}) and voluntary, grades are self-reported,
some groups are small (e.g. *Study group*, *Videos*), and correlation does not imply causation.
"""
    )

st.sidebar.divider()
st.sidebar.caption("Statistical report · UAP students survey · 2026")

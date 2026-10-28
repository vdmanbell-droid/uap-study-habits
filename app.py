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


def decision(p, alpha):
    return "✅ Reject H₀" if p < alpha else "❌ Fail to reject H₀"


def ordered_levels(series):
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [c for c in series.cat.categories if c in set(series.dropna())]
    return sorted(series.dropna().unique())


def result_table(rows):
    return pd.DataFrame(rows, columns=["Test", "Type", "Statistic", "p-value", "Decision"])


ALT_SIGN = {"two-sided": "≠", "greater": ">", "less": "<"}
ALT_LABEL = {"two-sided": "Two-sided (≠)", "greater": "Right-tailed (>)", "less": "Left-tailed (<)"}


def alt_select(key, container=st):
    return container.radio("Alternative hypothesis H₁", list(ALT_LABEL), format_func=ALT_LABEL.get,
                           key=key, horizontal=True)


def p_from_dist(dist, stat, alt):
    """p-value for a statistic with a continuous reference distribution."""
    if alt == "greater":
        return float(dist.sf(stat))
    if alt == "less":
        return float(dist.cdf(stat))
    if dist.dist.name in ("norm", "t"):
        return float(2 * dist.sf(abs(stat)))
    return float(min(1.0, 2 * min(dist.cdf(stat), dist.sf(stat))))


def ks_normal(s, mu=None, sigma=None):
    """Kolmogorov–Smirnov test against a normal distribution (Unit 4.4)."""
    x = pd.Series(s).dropna().to_numpy(float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan, np.nan
    mu = x.mean() if mu is None else mu
    sigma = x.std(ddof=1) if sigma is None else sigma
    r = stats.kstest(x, lambda v: stats.norm.cdf(v, mu, sigma))
    return float(r.statistic), float(r.pvalue)


def ks_caption(groups, alpha):
    parts = []
    for name, s in groups:
        _, p = ks_normal(s)
        if not np.isnan(p):
            parts.append(f"{name}: p = {p:.3f} ({'normal' if p >= alpha else 'not normal'})")
    if parts:
        st.caption("Normality assumption (Kolmogorov–Smirnov, Unit 4.4) · " + " · ".join(parts))


def dist_plot(dist, stat, alt, alpha, name, title):
    lo, hi = float(dist.ppf(0.0005)), float(dist.ppf(0.9995))
    lo, hi = min(lo, stat - 0.5), max(hi, stat + 0.5)
    xs = np.linspace(lo, hi, 500)
    ys = dist.pdf(xs)
    fig = go.Figure()
    fig.add_scatter(x=xs, y=ys, mode="lines", line_color=PALETTE[0], name=f"{name} distribution under H₀")
    if alt == "two-sided":
        regions = [(lo, float(dist.ppf(alpha / 2))), (float(dist.ppf(1 - alpha / 2)), hi)]
    elif alt == "greater":
        regions = [(float(dist.ppf(1 - alpha)), hi)]
    else:
        regions = [(lo, float(dist.ppf(alpha)))]
    for i, (a, b) in enumerate(regions):
        m = (xs >= a) & (xs <= b)
        if m.sum() < 2:
            continue
        fig.add_scatter(x=np.r_[a, xs[m], b], y=np.r_[0, ys[m], 0], fill="toself", mode="none",
                        fillcolor="rgba(184,70,107,0.35)", name=f"Rejection region (α = {alpha})",
                        showlegend=(i == 0))
    fig.add_vline(stat, line_color=PALETTE[1], line_width=3, annotation_text=f"observed {name} = {stat:.2f}")
    fig.update_layout(height=380, title=title, xaxis_title=name, yaxis_title="Density",
                      legend_orientation="h", legend_y=-0.25)
    return fig


def conclude(p, alpha, if_reject, if_not):
    st.success(f"**Conclusion (α = {alpha}):** " + (if_reject if p < alpha else if_not))


def two_groups(data, y, g):
    sub = data[[y, g]].dropna()
    lv = ordered_levels(sub[g])
    return sub, lv, sub.loc[sub[g] == lv[0], y], sub.loc[sub[g] == lv[1], y]


def sign_test(diff, alt):
    d = pd.Series(diff).dropna()
    n_pos, n_neg, ties = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    n = n_pos + n_neg
    p = stats.binomtest(n_pos, n, 0.5, alternative=alt).pvalue if n else np.nan
    z = (n_pos - n / 2) / np.sqrt(n / 4) if n else np.nan
    p_z = p_from_dist(stats.norm(), z, alt) if n else np.nan
    return n_pos, n_neg, ties, float(p), float(z), float(p_z)


def signed_rank(diff, alt):
    d = pd.Series(diff).dropna()
    d = d[d != 0]
    r = stats.rankdata(d.abs())
    w_pos, w_neg = float(r[d.to_numpy() > 0].sum()), float(r[d.to_numpy() < 0].sum())
    n = len(d)
    z = (w_pos - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    p = float(stats.wilcoxon(d, alternative=alt).pvalue)
    return d, r, w_pos, w_neg, n, float(z), p


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
        "report** and (2) the **parametric (Unit 3) and non-parametric (Unit 4) hypothesis tests** of the course, with visual results."
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
1. Is the mean (median) midterm grade above 8.0?
2. Did students perform differently on the midterms than in their previous semester?
3. Do students who **work** get different midterm grades from those who don't?
4. Does **sleep quality** during exam week relate to midterm grades?
5. Is the variability of midterm grades small (standard deviation below 1 point)?
6. Do fewer than half of the students study **2 or more hours per day**?
7. Do midterm grades follow a **normal distribution**?
8. Are the four **study methods** equally popular?
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
        st.subheader("Correlation matrix (descriptive)")
        corr_vars = ["midterm", "gpa_prev", "study_hours_day", "study_days_week",
                     "days_in_advance", "preparation", "attendance_pct", "courses",
                     "work_hours", "interference", "satisfaction", "difficulty"]
        cm = df[corr_vars].corr(method="pearson")
        cm.index = cm.columns = [lab(c) for c in corr_vars]
        fig = px.imshow(cm, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1,
                        aspect="auto")
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Pearson correlation coefficients, shown as a descriptive measure of association only (no hypothesis test)." )

        mcorr = cm[lab("midterm")].drop(lab("midterm")).sort_values()
        fig = px.bar(mcorr, orientation="h", labels={"value": "Correlation with midterm grade", "index": ""})
        fig.update_traces(marker_color=[PALETTE[3] if v < 0 else PALETTE[0] for v in mcorr])
        fig.update_layout(showlegend=False, height=420, title="Which variables move with the midterm grade?")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: HYPOTHESIS TESTING
# ============================================================================
elif page == "Hypothesis testing":
    st.title("Hypothesis Testing")
    st.markdown(
        f"Only the tests from the course units are used: **Unit 3 – Parametric tests** and "
        f"**Unit 4 – Non-parametric tests**. Significance level **α = {alpha}** (change it in the sidebar). "
        "In every tab you can change the variables, the reference value and the alternative hypothesis."
    )
    with st.expander("Tests included in this app"):
        st.dataframe(pd.DataFrame([
            ("3.1 Mean tests", "Z test (known variance) · t test (unknown variance) for one mean", "Parametric"),
            ("3.1 Mean tests", "Two means: Z test (known variances) · pooled t (equal variances) · Welch t (unequal variances)", "Parametric"),
            ("3.2 Tests for the variance", "χ² test for one variance · F test for two variances", "Parametric"),
            ("3.3 Test of proportions", "Z test for one proportion", "Parametric"),
            ("3.3 ANOVA", "One-way ANOVA · confidence intervals · Tukey HSD", "Parametric"),
            ("4.1 Sign test", "One-sample median · paired samples", "Non-parametric"),
            ("4.3 Wilcoxon", "Wilcoxon signed-rank (paired) · Wilcoxon rank-sum / Mann–Whitney (two independent samples)", "Non-parametric"),
            ("4.4 Goodness of fit", "χ² goodness of fit · Kolmogorov–Smirnov", "Non-parametric"),
        ], columns=["Course unit", "Test", "Type"]), hide_index=True, use_container_width=True)

    st.markdown("#### Unit 3 · Parametric tests")
    tp = st.tabs(["3.1 One mean", "3.1 Two means", "3.2 Variances", "3.3 Proportion", "3.3 ANOVA"])
    st.markdown("#### Unit 4 · Non-parametric tests")
    tn = st.tabs(["4.1 Sign test", "4.3 Wilcoxon signed-rank", "4.3 Wilcoxon rank-sum / Mann–Whitney",
                  "4.4 χ² goodness of fit", "4.4 Kolmogorov–Smirnov"])

    # ---------------- 3.1 One mean
    with tp[0]:
        st.subheader("Test for one mean")
        st.markdown("Known variance → **Z test**; unknown variance → **t test** with n − 1 degrees of freedom. "
                    "Choose *Midterm − previous GPA* with μ₀ = 0 to compare midterm grades with the previous GPA "
                    "of the same students (paired data).")
        c1, c2, c3 = st.columns([2, 1, 3])
        v = c1.selectbox("Variable", NUMERIC, format_func=lab, key="m1_v")
        mu_def = {"midterm": 8.0, "gpa_prev": 8.0, "grade_change": 0.0, "study_hours_day": 2.0}.get(v, round(float(df[v].mean()), 1))
        mu0 = c2.number_input("μ₀", value=float(mu_def), step=0.1, key=f"m1_mu_{v}")
        alt = alt_select("m1_alt", c3)
        s = df[v].dropna()
        n, xbar, sd = len(s), s.mean(), s.std()
        known = st.checkbox("The population variance is known → use the Z test", key="m1_known")
        sigma = st.number_input("Known σ", value=float(round(sd, 2)), min_value=0.01, step=0.05,
                                key=f"m1_sigma_{v}", disabled=not known)
        st.markdown(f"**H₀:** μ = {mu0}  ·  **H₁:** μ {ALT_SIGN[alt]} {mu0}")

        t = (xbar - mu0) / (sd / np.sqrt(n))
        p_t = p_from_dist(stats.t(n - 1), t, alt)
        rows = [("t test (unknown variance)", "Parametric", f"t = {t:.3f}, df = {n - 1}", fmt_p(p_t), decision(p_t, alpha))]
        main_dist, main_stat, main_name, main_p = stats.t(n - 1), t, "t", p_t
        if known:
            z = (xbar - mu0) / (sigma / np.sqrt(n))
            p_z = p_from_dist(stats.norm(), z, alt)
            rows.insert(0, (f"Z test (known σ = {sigma})", "Parametric", f"z = {z:.3f}", fmt_p(p_z), decision(p_z, alpha)))
            main_dist, main_stat, main_name, main_p = stats.norm(), z, "z", p_z
        st.dataframe(result_table(rows), hide_index=True, use_container_width=True)
        ci = stats.t.interval(1 - alpha, n - 1, xbar, sd / np.sqrt(n))
        st.markdown(f"n = {n} · x̄ = **{xbar:.3f}** · s = {sd:.3f} · {int((1 - alpha) * 100)}% confidence interval for μ: "
                    f"**[{ci[0]:.3f}, {ci[1]:.3f}]**")
        ks_caption([(lab(v), s)], alpha)

        c1, c2 = st.columns(2)
        fig = px.histogram(s, nbins=14, opacity=0.8, labels={"value": lab(v)})
        fig.add_vrect(ci[0], ci[1], fillcolor=PALETTE[1], opacity=0.15, line_width=0)
        fig.add_vline(mu0, line_color=PALETTE[3], line_width=3, annotation_text=f"μ₀ = {mu0}")
        fig.add_vline(xbar, line_dash="dash", line_color=PALETTE[1], annotation_text=f"x̄ = {xbar:.2f}",
                      annotation_position="top left")
        fig.update_layout(showlegend=False, yaxis_title="Students", height=380,
                          title="Sample, mean and confidence interval (shaded)")
        c1.plotly_chart(fig, use_container_width=True)
        c2.plotly_chart(dist_plot(main_dist, main_stat, alt, alpha, main_name,
                                  f"{main_name} statistic and rejection region"), use_container_width=True)
        conclude(main_p, alpha,
                 f"the mean {lab(v).lower()} is significantly {ALT_SIGN[alt] if alt != 'two-sided' else 'different from'} {mu0} (x̄ = {xbar:.2f}).",
                 f"there is not enough evidence that the mean {lab(v).lower()} is {ALT_SIGN[alt] if alt != 'two-sided' else 'different from'} {mu0} (x̄ = {xbar:.2f}).")

    # ---------------- 3.1 Two means
    with tp[1]:
        st.subheader("Equality of means of two normal populations")
        st.markdown("The **F test** decides if the variances can be considered equal: equal → **pooled t test**, "
                    "unequal → **Welch t test**. If both variances are known, the **Z test** applies.")
        c1, c2, c3 = st.columns([2, 2, 3])
        y = c1.selectbox("Variable", NUMERIC, format_func=lab, key="m2_y")
        g = c2.selectbox("Groups", BINARY, format_func=lab, key="m2_g")
        alt = alt_select("m2_alt", c3)
        sub, lv, a, b = two_groups(df, y, g)
        n1, n2 = len(a), len(b)
        st.markdown(f"**H₀:** μ({lv[0]}) = μ({lv[1]})  ·  **H₁:** μ({lv[0]}) {ALT_SIGN[alt]} μ({lv[1]})")

        F = a.var() / b.var()
        p_f = p_from_dist(stats.f(n1 - 1, n2 - 1), F, "two-sided")
        equal = p_f >= alpha
        tp_res = stats.ttest_ind(a, b, equal_var=True, alternative=alt)
        tw_res = stats.ttest_ind(a, b, equal_var=False, alternative=alt)
        rows = [
            ("F test for equal variances", "Parametric", f"F = {F:.3f}, df = ({n1 - 1}, {n2 - 1})", fmt_p(p_f), decision(p_f, alpha)),
            ("Pooled t test" + (" ← recommended" if equal else ""), "Parametric", f"t = {tp_res.statistic:.3f}, df = {n1 + n2 - 2}", fmt_p(tp_res.pvalue), decision(tp_res.pvalue, alpha)),
            ("Welch t test" + ("" if equal else " ← recommended"), "Parametric", f"t = {tw_res.statistic:.3f}, df = {tw_res.df:.1f}", fmt_p(tw_res.pvalue), decision(tw_res.pvalue, alpha)),
        ]
        known = st.checkbox("Both population variances are known → add the Z test", key="m2_known")
        if known:
            k1, k2 = st.columns(2)
            s1 = k1.number_input(f"Known σ ({lv[0]})", value=float(round(a.std(), 2)), min_value=0.01, step=0.05, key=f"m2_s1_{y}_{g}")
            s2 = k2.number_input(f"Known σ ({lv[1]})", value=float(round(b.std(), 2)), min_value=0.01, step=0.05, key=f"m2_s2_{y}_{g}")
            z = (a.mean() - b.mean()) / np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
            p_z = p_from_dist(stats.norm(), z, alt)
            rows.insert(1, ("Z test (known variances)", "Parametric", f"z = {z:.3f}", fmt_p(p_z), decision(p_z, alpha)))
        st.dataframe(result_table(rows), hide_index=True, use_container_width=True)
        summ = sub.groupby(g, observed=True)[y].agg(["count", "mean", "std", "var"]).reindex(lv)
        summ.index.name = lab(g)
        st.dataframe(summ.style.format("{:.3f}", subset=["mean", "std", "var"]), use_container_width=True)
        ks_caption([(str(lv[0]), a), (str(lv[1]), b)], alpha)
        rec = tp_res if equal else tw_res
        c1, c2 = st.columns(2)
        fig = px.box(sub, x=g, y=y, color=g, points="all", category_orders={g: lv}, labels={g: lab(g), y: lab(y)})
        fig.update_layout(showlegend=False, height=380, title="Distribution by group")
        c1.plotly_chart(fig, use_container_width=True)
        dfree = n1 + n2 - 2 if equal else rec.df
        c2.plotly_chart(dist_plot(stats.t(dfree), rec.statistic, alt, alpha, "t",
                                  f"{'Pooled' if equal else 'Welch'} t statistic and rejection region"), use_container_width=True)
        conclude(rec.pvalue, alpha,
                 f"the mean {lab(y).lower()} differs significantly between '{lv[0]}' ({a.mean():.2f}) and '{lv[1]}' ({b.mean():.2f}).",
                 f"no significant difference in mean {lab(y).lower()} between '{lv[0]}' ({a.mean():.2f}) and '{lv[1]}' ({b.mean():.2f}).")

    # ---------------- 3.2 Variances
    with tp[2]:
        st.subheader("Tests for the variance")
        st.markdown("**χ² test for one variance:** χ² = (n − 1)s² / σ₀², with n − 1 degrees of freedom.")
        c1, c2, c3 = st.columns([2, 1, 3])
        v = c1.selectbox("Variable", NUMERIC, format_func=lab, key="v1_v")
        s = df[v].dropna()
        n, s2 = len(s), s.var()
        sig0 = c2.number_input("σ₀", value=1.0 if v == "midterm" else float(round(np.sqrt(s2), 1) or 1.0),
                               min_value=0.01, step=0.1, key=f"v1_s0_{v}")
        alt = alt_select("v1_alt", c3)
        st.markdown(f"**H₀:** σ² = {sig0 ** 2:.3f} (σ = {sig0})  ·  **H₁:** σ² {ALT_SIGN[alt]} {sig0 ** 2:.3f}")
        chi = (n - 1) * s2 / sig0 ** 2
        p_c = p_from_dist(stats.chi2(n - 1), chi, alt)
        st.dataframe(result_table([("χ² test for one variance", "Parametric", f"χ² = {chi:.3f}, df = {n - 1}", fmt_p(p_c), decision(p_c, alpha))]),
                     hide_index=True, use_container_width=True)
        lo_ci = (n - 1) * s2 / stats.chi2.ppf(1 - alpha / 2, n - 1)
        hi_ci = (n - 1) * s2 / stats.chi2.ppf(alpha / 2, n - 1)
        st.markdown(f"s² = **{s2:.4f}** (s = {np.sqrt(s2):.3f}) · {int((1 - alpha) * 100)}% CI for σ: "
                    f"**[{np.sqrt(lo_ci):.3f}, {np.sqrt(hi_ci):.3f}]**")
        ks_caption([(lab(v), s)], alpha)
        st.plotly_chart(dist_plot(stats.chi2(n - 1), chi, alt, alpha, "χ²", "χ² statistic and rejection region"),
                        use_container_width=True)
        conclude(p_c, alpha,
                 f"the standard deviation of {lab(v).lower()} is significantly {ALT_SIGN[alt] if alt != 'two-sided' else 'different from'} {sig0} (s = {np.sqrt(s2):.3f}).",
                 f"no evidence that the standard deviation of {lab(v).lower()} is {ALT_SIGN[alt] if alt != 'two-sided' else 'different from'} {sig0} (s = {np.sqrt(s2):.3f}).")

        st.divider()
        st.markdown("**F test for the equality of variances of two normal populations:** F = s₁² / s₂², "
                    "with (n₁ − 1, n₂ − 1) degrees of freedom.")
        c1, c2, c3 = st.columns([2, 2, 3])
        y = c1.selectbox("Variable", NUMERIC, format_func=lab, key="v2_y")
        g = c2.selectbox("Groups", BINARY, format_func=lab, key="v2_g")
        alt2 = alt_select("v2_alt", c3)
        sub, lv, a, b = two_groups(df, y, g)
        n1, n2 = len(a), len(b)
        F = a.var() / b.var()
        p_f = p_from_dist(stats.f(n1 - 1, n2 - 1), F, alt2)
        st.markdown(f"**H₀:** σ²({lv[0]}) = σ²({lv[1]})  ·  **H₁:** σ²({lv[0]}) {ALT_SIGN[alt2]} σ²({lv[1]})")
        st.dataframe(result_table([("F test for two variances", "Parametric", f"F = {F:.3f}, df = ({n1 - 1}, {n2 - 1})", fmt_p(p_f), decision(p_f, alpha))]),
                     hide_index=True, use_container_width=True)
        st.markdown(f"s²({lv[0]}) = {a.var():.4f} (n = {n1}) · s²({lv[1]}) = {b.var():.4f} (n = {n2})")
        c1, c2 = st.columns(2)
        fig = px.box(sub, x=g, y=y, color=g, points="all", category_orders={g: lv}, labels={g: lab(g), y: lab(y)})
        fig.update_layout(showlegend=False, height=380, title="Spread by group")
        c1.plotly_chart(fig, use_container_width=True)
        c2.plotly_chart(dist_plot(stats.f(n1 - 1, n2 - 1), F, alt2, alpha, "F", "F statistic and rejection region"),
                        use_container_width=True)
        conclude(p_f, alpha, f"the variances of {lab(y).lower()} are significantly different between the groups.",
                 f"the variances of {lab(y).lower()} can be considered equal between '{lv[0]}' and '{lv[1]}'.")

    # ---------------- 3.3 Proportion
    with tp[3]:
        st.subheader("Test of proportion")
        events = {
            "Midterm grade of 9 or more": df["midterm"].dropna() >= 9,
            "Studies 2 or more hours per day": df["study_hours_day"].dropna() >= 2,
            "Has a job": df["job"] == "Yes",
            "Does scholarship service": df["in_scholarship"] == "Yes",
            "Slept badly or very badly in exam week": df["sleep"].dropna().isin(["Bad", "Very bad"]),
            "Attendance of 90% or more": df["attendance_pct"].dropna() >= 90,
        }
        c1, c2, c3 = st.columns([2, 1, 3])
        ev = c1.selectbox("Proportion of students who…", list(events), key="pr_ev")
        p0 = c2.number_input("p₀", value=0.5, min_value=0.01, max_value=0.99, step=0.05, key="pr_p0")
        alt = alt_select("pr_alt", c3)
        x_ev = events[ev]
        n, k = len(x_ev), int(x_ev.sum())
        ph = k / n
        st.markdown(f"**H₀:** p = {p0}  ·  **H₁:** p {ALT_SIGN[alt]} {p0}")
        z = (ph - p0) / np.sqrt(p0 * (1 - p0) / n)
        p_z = p_from_dist(stats.norm(), z, alt)
        st.dataframe(result_table([("Z test for one proportion", "Parametric", f"z = {z:.3f}", fmt_p(p_z), decision(p_z, alpha))]),
                     hide_index=True, use_container_width=True)
        zc = stats.norm.ppf(1 - alpha / 2)
        half = zc * np.sqrt(ph * (1 - ph) / n)
        st.markdown(f"{k} of {n} students · p̂ = **{ph:.3f}** · {int((1 - alpha) * 100)}% CI for p: "
                    f"**[{max(ph - half, 0):.3f}, {min(ph + half, 1):.3f}]**")
        if n * p0 < 5 or n * (1 - p0) < 5:
            st.warning("n·p₀ or n·(1 − p₀) is below 5: the normal approximation may not be valid.")
        else:
            st.caption(f"Normal approximation valid: n·p₀ = {n * p0:.1f} ≥ 5 and n·(1 − p₀) = {n * (1 - p0):.1f} ≥ 5.")
        c1, c2 = st.columns(2)
        fig = go.Figure()
        fig.add_bar(x=["Sample proportion p̂"], y=[ph], error_y=dict(type="data", array=[half]),
                    marker_color=PALETTE[0], width=0.4, text=[f"{ph:.1%}"], textposition="outside")
        fig.add_hline(p0, line_dash="dash", line_color=PALETTE[3], annotation_text=f"p₀ = {p0}")
        fig.update_layout(height=380, yaxis_range=[0, 1], yaxis_tickformat=".0%",
                          title=f"{ev}: p̂ with {int((1 - alpha) * 100)}% CI")
        c1.plotly_chart(fig, use_container_width=True)
        c2.plotly_chart(dist_plot(stats.norm(), z, alt, alpha, "z", "z statistic and rejection region"),
                        use_container_width=True)
        conclude(p_z, alpha,
                 f"the proportion of students who {ev[0].lower() + ev[1:]} ({ph:.1%}) is significantly {ALT_SIGN[alt] if alt != 'two-sided' else 'different from'} {p0:.0%}.",
                 f"there is not enough evidence that the proportion ({ph:.1%}) is {ALT_SIGN[alt] if alt != 'two-sided' else 'different from'} {p0:.0%}.")

    # ---------------- 3.3 ANOVA
    with tp[4]:
        st.subheader("One-way ANOVA (one factor)")
        st.markdown("Assumptions: independent random observations, normal data in each group and a common variance.")
        c1, c2 = st.columns(2)
        y = c1.selectbox("Variable", NUMERIC, format_func=lab, key="an_y")
        g = c2.selectbox("Factor (groups)", MULTI, index=MULTI.index("sleep_3"), format_func=lab, key="an_g")
        sub = df[[y, g]].dropna()
        lv = ordered_levels(sub[g])
        dropped = [str(l) for l in lv if (sub[g] == l).sum() < 2]
        lv = [l for l in lv if (sub[g] == l).sum() >= 2]
        groups = [sub.loc[sub[g] == l, y] for l in lv]
        if dropped:
            st.info(f"Groups with fewer than 2 observations were excluded: {', '.join(dropped)}.")
        small = [str(l) for l, s_ in zip(lv, groups) if len(s_) < 5]
        if small:
            st.warning(f"Small groups (n < 5): {', '.join(small)}. Consider the grouped version of the factor.")
        st.markdown(f"**H₀:** all group means of {lab(y).lower()} are equal  ·  **H₁:** at least one mean is different")
        N, k = len(sub), len(groups)
        gm = sub[y].mean()
        ssb = sum(len(s_) * (s_.mean() - gm) ** 2 for s_ in groups)
        ssw = sum(((s_ - s_.mean()) ** 2).sum() for s_ in groups)
        msb, msw = ssb / (k - 1), ssw / (N - k)
        F = msb / msw
        p_a = float(stats.f.sf(F, k - 1, N - k))
        st.markdown("**ANOVA table**")
        st.dataframe(pd.DataFrame({
            "Source": ["Between groups", "Within groups (error)", "Total"],
            "SS": [ssb, ssw, ssb + ssw], "df": [k - 1, N - k, N - 1],
            "MS": [msb, msw, np.nan], "F": [F, np.nan, np.nan], "p-value": [p_a, np.nan, np.nan],
        }).style.format({"SS": "{:.4f}", "MS": "{:.4f}", "F": "{:.3f}", "p-value": "{:.4f}"}, na_rep=""),
            hide_index=True, use_container_width=True)
        st.markdown(f"**Decision:** {decision(p_a, alpha)}")
        summ = sub.groupby(g, observed=True)[y].agg(["count", "mean", "std"]).reindex(lv)
        tq = stats.t.ppf(1 - alpha / 2, np.maximum(summ["count"] - 1, 1))
        summ["CI low"] = summ["mean"] - tq * summ["std"] / np.sqrt(summ["count"])
        summ["CI high"] = summ["mean"] + tq * summ["std"] / np.sqrt(summ["count"])
        summ.index.name = lab(g)
        st.markdown(f"**Group means with {int((1 - alpha) * 100)}% confidence intervals**")
        st.dataframe(summ.style.format("{:.3f}", subset=["mean", "std", "CI low", "CI high"]), use_container_width=True)
        ks_caption([(str(l), s_) for l, s_ in zip(lv, groups)], alpha)
        c1, c2 = st.columns(2)
        fig = px.box(sub, x=g, y=y, color=g, points="all", category_orders={g: lv}, labels={g: lab(g), y: lab(y)})
        fig.update_layout(showlegend=False, height=400, title="Distribution by group")
        c1.plotly_chart(fig, use_container_width=True)
        m_ = summ.reset_index()
        fig = go.Figure()
        fig.add_scatter(x=m_[lab(g)].astype(str), y=m_["mean"], mode="markers",
                        error_y=dict(type="data", symmetric=False, array=m_["CI high"] - m_["mean"],
                                     arrayminus=m_["mean"] - m_["CI low"]),
                        marker=dict(size=14, color=PALETTE[0]))
        fig.update_layout(height=400, title=f"Means with {int((1 - alpha) * 100)}% CI", yaxis_title=f"Mean {lab(y).lower()}")
        c2.plotly_chart(fig, use_container_width=True)
        st.markdown("**Tukey's HSD (honestly significant difference)**")
        tk = stats.tukey_hsd(*groups)
        tci = tk.confidence_interval(1 - alpha)
        pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                pairs.append({"Comparison": f"{lv[i]} − {lv[j]}", "Mean difference": groups[i].mean() - groups[j].mean(),
                              "CI low": tci.low[i, j], "CI high": tci.high[i, j], "p-value": tk.pvalue[i, j],
                              "Significant?": "Yes" if tk.pvalue[i, j] < alpha else "No"})
        st.dataframe(pd.DataFrame(pairs).style.format({"Mean difference": "{:+.3f}", "CI low": "{:.3f}",
                                                       "CI high": "{:.3f}", "p-value": "{:.4f}"}),
                     hide_index=True, use_container_width=True)
        conclude(p_a, alpha,
                 f"the mean {lab(y).lower()} differs across {lab(g).lower()} groups; Tukey's HSD shows which pairs differ.",
                 f"no significant differences in mean {lab(y).lower()} across {lab(g).lower()} groups.")

    # ---------------- 4.1 Sign test
    with tn[0]:
        st.subheader("Sign test")
        st.markdown("Counts how many observations fall above (+) and below (−) the hypothesized median; under H₀ "
                    "the number of + signs follows a Binomial(n, 0.5). Ties are discarded. Alternative to the "
                    "one-sample t test and the paired t test.")
        mode = st.radio("Design", ["Paired: midterm grade vs previous GPA", "One sample: median vs m₀"],
                        horizontal=True, key="sg_mode")
        c1, c2, c3 = st.columns([2, 1, 3])
        if mode.startswith("One"):
            v = c1.selectbox("Variable", NUMERIC, format_func=lab, key="sg_v")
            m0 = c2.number_input("m₀", value=8.0 if v == "midterm" else float(df[v].median()), step=0.1, key=f"sg_m0_{v}")
            alt = alt_select("sg_alt", c3)
            diff = df[v].dropna() - m0
            h0, h1 = f"median = {m0}", f"median {ALT_SIGN[alt]} {m0}"
            what = f"the median {lab(v).lower()}"
            ref = f"{m0}"
        else:
            alt = alt_select("sg_alt2", c3)
            diff = (df["midterm"] - df["gpa_prev"]).dropna()
            h0, h1 = "median difference (midterm − GPA) = 0", f"median difference {ALT_SIGN[alt]} 0"
            what = "the median difference midterm − previous GPA"
            ref = "0"
        st.markdown(f"**H₀:** {h0}  ·  **H₁:** {h1}")
        n_pos, n_neg, ties, p_b, z, p_z = sign_test(diff, alt)
        st.dataframe(result_table([
            ("Sign test (exact binomial)", "Non-parametric", f"n₊ = {n_pos}, n₋ = {n_neg}, n = {n_pos + n_neg}", fmt_p(p_b), decision(p_b, alpha)),
            ("Sign test (normal approximation)", "Non-parametric", f"z = {z:.3f}", fmt_p(p_z), decision(p_z, alpha)),
        ]), hide_index=True, use_container_width=True)
        st.caption(f"{ties} tie(s) discarded.")
        c1, c2 = st.columns(2)
        fig = px.bar(x=["+ (above)", "− (below)", "Ties"], y=[n_pos, n_neg, ties], text=[n_pos, n_neg, ties],
                     labels={"x": "", "y": "Count"}, color=["+", "−", "0"],
                     color_discrete_map={"+": PALETTE[2], "−": PALETTE[3], "0": "#B0B0B0"})
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=380, title="Signs of the differences")
        c1.plotly_chart(fig, use_container_width=True)
        nn = n_pos + n_neg
        ks_ = np.arange(nn + 1)
        pmf = stats.binom.pmf(ks_, nn, 0.5)
        fig = go.Figure()
        fig.add_bar(x=ks_, y=pmf, marker_color=PALETTE[0], name="Binomial(n, 0.5)")
        fig.add_vline(n_pos, line_color=PALETTE[1], line_width=3, annotation_text=f"observed n₊ = {n_pos}")
        fig.update_layout(height=380, title="Distribution of n₊ under H₀", xaxis_title="Number of + signs",
                          yaxis_title="Probability", showlegend=False)
        c2.plotly_chart(fig, use_container_width=True)
        conclude(p_b, alpha, f"{what} is significantly {ALT_SIGN[alt] if alt != 'two-sided' else 'different from'} {ref}.",
                 f"no evidence that {what} is {ALT_SIGN[alt] if alt != 'two-sided' else 'different from'} {ref}.")

    # ---------------- 4.3 Wilcoxon signed-rank
    with tn[1]:
        st.subheader("Wilcoxon signed-rank test")
        st.markdown("Compares two matched (paired) samples using the ranks of the absolute differences. "
                    "Unlike the sign test, it also uses the size of each difference.")
        mode = st.radio("Design", ["Paired: midterm grade vs previous GPA", "One sample: median vs m₀"],
                        horizontal=True, key="wsr_mode")
        c1, c2, c3 = st.columns([2, 1, 3])
        if mode.startswith("One"):
            v = c1.selectbox("Variable", NUMERIC, format_func=lab, key="wsr_v")
            m0 = c2.number_input("m₀", value=8.0 if v == "midterm" else float(df[v].median()), step=0.1, key=f"wsr_m0_{v}")
            alt = alt_select("wsr_alt", c3)
            diff = df[v].dropna() - m0
            h0, h1, ref, what = f"median = {m0}", f"median {ALT_SIGN[alt]} {m0}", f"{m0}", f"the median {lab(v).lower()}"
        else:
            alt = alt_select("wsr_alt2", c3)
            diff = (df["midterm"] - df["gpa_prev"]).dropna()
            h0, h1 = "median of the differences (midterm − GPA) = 0", f"median of the differences {ALT_SIGN[alt]} 0"
            ref, what = "0", "the median difference midterm − previous GPA"
        st.markdown(f"**H₀:** {h0}  ·  **H₁:** {h1}")
        d, r, w_pos, w_neg, n, z, p_w = signed_rank(diff, alt)
        st.dataframe(result_table([
            ("Wilcoxon signed-rank", "Non-parametric", f"W₊ = {w_pos:.1f}, W₋ = {w_neg:.1f}, n = {n}", fmt_p(p_w), decision(p_w, alpha)),
            ("Normal approximation", "Non-parametric", f"z = {z:.3f}", fmt_p(p_from_dist(stats.norm(), z, alt)),
             decision(p_from_dist(stats.norm(), z, alt), alpha)),
        ]), hide_index=True, use_container_width=True)
        st.caption(f"Zero differences discarded: {int((pd.Series(diff).dropna() == 0).sum())}. "
                   f"Under H₀, E[W₊] = n(n+1)/4 = {n * (n + 1) / 4:.1f}.")
        sr = pd.DataFrame({"difference": d.to_numpy(), "signed rank": np.sign(d.to_numpy()) * r})
        sr = sr.sort_values("signed rank").reset_index(drop=True)
        sr["sign"] = np.where(sr["signed rank"] > 0, "Positive", "Negative")
        fig = px.bar(sr, x=sr.index, y="signed rank", color="sign", hover_data=["difference"],
                     color_discrete_map={"Positive": PALETTE[2], "Negative": PALETTE[3]},
                     labels={"x": "Observations (sorted)", "signed rank": "Signed rank"})
        fig.update_layout(height=380, title=f"Signed ranks: W₊ = {w_pos:.0f} vs W₋ = {w_neg:.0f}", xaxis_title="Observations (sorted)")
        st.plotly_chart(fig, use_container_width=True)
        conclude(p_w, alpha, f"{what} is significantly {ALT_SIGN[alt] if alt != 'two-sided' else 'different from'} {ref}.",
                 f"no evidence that {what} is {ALT_SIGN[alt] if alt != 'two-sided' else 'different from'} {ref}.")

    # ---------------- 4.3 Rank-sum
    with tn[2]:
        st.subheader("Wilcoxon rank-sum test / Mann–Whitney U")
        st.markdown("Non-parametric alternative to the two-sample t test, based only on the order (ranks) of the "
                    "observations of two independent samples.")
        c1, c2, c3 = st.columns([2, 2, 3])
        y = c1.selectbox("Variable", NUMERIC, format_func=lab, key="rs_y")
        g = c2.selectbox("Groups", BINARY, format_func=lab, key="rs_g")
        alt = alt_select("rs_alt", c3)
        sub, lv, a, b = two_groups(df, y, g)
        n1, n2 = len(a), len(b)
        st.markdown(f"**H₀:** both groups come from the same distribution  ·  **H₁:** values of '{lv[0]}' tend to be "
                    f"{ {'two-sided': 'different from', 'greater': 'larger than', 'less': 'smaller than'}[alt] } '{lv[1]}'")
        ranks = stats.rankdata(np.r_[a.to_numpy(), b.to_numpy()])
        r1 = ranks[:n1].sum()
        u = r1 - n1 * (n1 + 1) / 2
        mw = stats.mannwhitneyu(a, b, alternative=alt)
        rsz = stats.ranksums(a, b, alternative=alt)
        st.dataframe(result_table([
            ("Wilcoxon rank-sum", "Non-parametric", f"W = R₁ = {r1:.1f}, z = {rsz.statistic:.3f}", fmt_p(rsz.pvalue), decision(rsz.pvalue, alpha)),
            ("Mann–Whitney U", "Non-parametric", f"U = {u:.1f}", fmt_p(mw.pvalue), decision(mw.pvalue, alpha)),
        ]), hide_index=True, use_container_width=True)
        st.caption(f"Mean rank '{lv[0]}' = {r1 / n1:.1f} (n = {n1}) · mean rank '{lv[1]}' = {ranks[n1:].mean():.1f} (n = {n2}). "
                   "Mann–Whitney p-value includes the tie correction.")
        rk = pd.DataFrame({g: [lv[0]] * n1 + [lv[1]] * n2, "Rank": ranks, y: np.r_[a.to_numpy(), b.to_numpy()]})
        c1, c2 = st.columns(2)
        fig = px.box(sub, x=g, y=y, color=g, points="all", category_orders={g: lv}, labels={g: lab(g), y: lab(y)})
        fig.update_layout(showlegend=False, height=380, title="Values by group")
        c1.plotly_chart(fig, use_container_width=True)
        fig = px.strip(rk, x=g, y="Rank", color=g, category_orders={g: lv}, labels={g: lab(g)})
        for i, l in enumerate(lv):
            fig.add_scatter(x=[l], y=[rk.loc[rk[g] == l, "Rank"].mean()], mode="markers",
                            marker=dict(symbol="line-ew-open", size=60, color="black", line_width=3),
                            name="Mean rank", showlegend=(i == 0))
        fig.update_layout(height=380, title="Ranks by group (black line = mean rank)")
        c2.plotly_chart(fig, use_container_width=True)
        conclude(mw.pvalue, alpha, f"{lab(y).lower()} differs significantly between '{lv[0]}' and '{lv[1]}'.",
                 f"no significant difference in {lab(y).lower()} between '{lv[0]}' and '{lv[1]}'.")

    # ---------------- 4.4 Chi-square goodness of fit
    with tn[3]:
        st.subheader("χ² goodness of fit test")
        st.markdown("χ² = Σ (Oᵢ − Eᵢ)² / Eᵢ.  **H₀:** the data follow the specified distribution · "
                    "**H₁:** they do not. Every expected count should be at least 5.")
        mode = st.radio("Specified distribution", ["Categorical variable: equal proportions",
                                                   "Quantitative variable: normal distribution"],
                        horizontal=True, key="gof_mode")
        if mode.startswith("Categorical"):
            cat = st.selectbox("Variable", ["method", "sleep", "sleep_3", "attendance", "program", "semester_stage", "prep_level"],
                               format_func=lab, key="gof_cat")
            counts = df[cat].value_counts(sort=False)
            if isinstance(df[cat].dtype, pd.CategoricalDtype):
                counts = counts.reindex(df[cat].cat.categories).fillna(0)
            obs = counts.to_numpy(float)
            k, N = len(obs), obs.sum()
            exp = np.full(k, N / k)
            labels_ = counts.index.astype(str).tolist()
            dfree = k - 1
            st.markdown(f"**H₀:** every category of {lab(cat).lower()} has proportion 1/{k}")
        else:
            v = st.selectbox("Variable", NUMERIC, format_func=lab, key="gof_v")
            x = df[v].dropna().to_numpy(float)
            N = len(x)
            k = st.slider("Number of classes k (equal-probability classes under the fitted normal)", 4, max(4, min(10, N // 5)), min(6, max(4, N // 5)), key="gof_k")
            mu_, sd_ = x.mean(), x.std(ddof=1)
            edges = stats.norm.ppf(np.linspace(0, 1, k + 1), mu_, sd_)
            idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, k - 1)
            obs = np.bincount(idx, minlength=k).astype(float)
            cdf = stats.norm.cdf(edges, mu_, sd_)
            exp = N * np.diff(cdf)
            fmt_edge = lambda e: "−∞" if np.isinf(e) and e < 0 else ("∞" if np.isinf(e) else f"{e:.2f}")
            labels_ = [f"({fmt_edge(edges[i])}, {fmt_edge(edges[i + 1])}]" for i in range(k)]
            dfree = k - 3
            st.markdown(f"**H₀:** {lab(v).lower()} follows a normal distribution with μ = x̄ = {mu_:.3f} and σ = s = {sd_:.3f}. "
                        f"Eᵢ = N·(F(Yᵤ) − F(Yₗ)). Two parameters are estimated, so c = 3 and df = k − c = {dfree}.")
        chi = float(((obs - exp) ** 2 / exp).sum())
        p_g = float(stats.chi2.sf(chi, dfree)) if dfree >= 1 else np.nan
        crit = stats.chi2.ppf(1 - alpha, dfree) if dfree >= 1 else np.nan
        st.dataframe(result_table([("χ² goodness of fit", "Non-parametric", f"χ² = {chi:.3f}, df = {dfree}, critical value = {crit:.3f}",
                                    fmt_p(p_g), decision(p_g, alpha))]), hide_index=True, use_container_width=True)
        tab = pd.DataFrame({"Class": labels_, "Observed Oᵢ": obs.astype(int), "Expected Eᵢ": exp,
                            "(Oᵢ − Eᵢ)² / Eᵢ": (obs - exp) ** 2 / exp})
        if (exp < 5).any():
            st.warning("Some expected counts are below 5; the χ² approximation may be unreliable.")
        c1, c2 = st.columns([2, 3])
        c1.dataframe(tab.style.format({"Expected Eᵢ": "{:.2f}", "(Oᵢ − Eᵢ)² / Eᵢ": "{:.3f}"}), hide_index=True, use_container_width=True)
        fig = go.Figure()
        fig.add_bar(x=labels_, y=obs, name="Observed", marker_color=PALETTE[0])
        fig.add_bar(x=labels_, y=exp, name="Expected under H₀", marker_color=PALETTE[1], opacity=0.7)
        fig.update_layout(barmode="group", height=400, title="Observed vs expected frequencies",
                          yaxis_title="Students", xaxis_type="category", legend_orientation="h", legend_y=-0.3)
        c2.plotly_chart(fig, use_container_width=True)
        conclude(p_g, alpha, "the data do NOT follow the specified distribution.",
                 "the data are consistent with the specified distribution.")

    # ---------------- 4.4 Kolmogorov–Smirnov
    with tn[4]:
        st.subheader("Kolmogorov–Smirnov goodness of fit test")
        st.markdown("D = max |Fₙ(x) − F(x)|: the maximum distance between the empirical distribution function "
                    "and the normal cumulative distribution function. **H₀:** the data follow the specified "
                    "distribution · **H₁:** they do not.")
        c1, c2 = st.columns([2, 3])
        v = c1.selectbox("Variable", NUMERIC, format_func=lab, key="ks_v")
        x = np.sort(df[v].dropna().to_numpy(float))
        par = c2.radio("Normal parameters", ["Estimated from the sample (x̄, s)", "Fully specified"],
                       horizontal=True, key="ks_par")
        if par.startswith("Fully"):
            k1, k2 = st.columns(2)
            mu_ = k1.number_input("μ", value=float(round(x.mean(), 2)), step=0.1, key=f"ks_mu_{v}")
            sd_ = k2.number_input("σ", value=float(round(x.std(ddof=1), 2)), min_value=0.01, step=0.05, key=f"ks_sd_{v}")
        else:
            mu_, sd_ = x.mean(), x.std(ddof=1)
            st.warning("The parameters are estimated from the data, so the standard K–S critical values are no longer "
                       "exact (the test becomes conservative). For an exact test the distribution must be fully specified.")
        D, p_k = ks_normal(x, mu_, sd_)
        n = len(x)
        st.markdown(f"**H₀:** {lab(v).lower()} ~ Normal(μ = {mu_:.3f}, σ = {sd_:.3f})")
        st.dataframe(result_table([("Kolmogorov–Smirnov", "Non-parametric", f"D = {D:.4f}, n = {n}, critical value ≈ {stats.kstwo.ppf(1 - alpha, n):.4f}",
                                    fmt_p(p_k), decision(p_k, alpha))]), hide_index=True, use_container_width=True)
        F_ = stats.norm.cdf(x, mu_, sd_)
        ecdf_hi = np.arange(1, n + 1) / n
        ecdf_lo = np.arange(0, n) / n
        gaps = np.maximum(ecdf_hi - F_, F_ - ecdf_lo)
        i = int(np.argmax(gaps))
        top, bot = (ecdf_hi[i], F_[i]) if ecdf_hi[i] - F_[i] >= F_[i] - ecdf_lo[i] else (F_[i], ecdf_lo[i])
        xs = np.linspace(x.min() - sd_, x.max() + sd_, 300)
        fig = go.Figure()
        fig.add_scatter(x=np.r_[x.min() - sd_, x, x.max() + sd_], y=np.r_[0, ecdf_hi, 1], mode="lines",
                        line_shape="hv", name="Empirical Fₙ(x)", line_color=PALETTE[0])
        fig.add_scatter(x=xs, y=stats.norm.cdf(xs, mu_, sd_), mode="lines", name="Normal F(x)", line_color=PALETTE[1])
        fig.add_scatter(x=[x[i], x[i]], y=[bot, top], mode="lines+markers", name=f"D = {D:.3f}",
                        line=dict(color=PALETTE[3], width=4))
        fig.update_layout(height=430, title="Empirical vs normal cumulative distribution", xaxis_title=lab(v),
                          yaxis_title="Cumulative probability", legend_orientation="h", legend_y=-0.2)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**K–S normality check for all quantitative variables** (parameters estimated)")
        allks = []
        for c in NUMERIC:
            Dc, pc = ks_normal(df[c])
            if not np.isnan(pc):
                allks.append({"Variable": lab(c), "n": int(df[c].notna().sum()), "D": Dc, "p-value": pc,
                              "Normal?": "Yes" if pc >= alpha else "No"})
        st.dataframe(pd.DataFrame(allks).style.format({"D": "{:.4f}", "p-value": "{:.4f}"}),
                     hide_index=True, use_container_width=True)
        conclude(p_k, alpha, f"{lab(v).lower()} does NOT follow the specified normal distribution.",
                 f"{lab(v).lower()} is consistent with a normal distribution.")


# ============================================================================
# PAGE: CONCLUSIONS
# ============================================================================
elif page == "Conclusions":
    st.title("Summary of Findings")
    st.markdown(f"Results of the research questions at **α = {alpha}**, using only the tests of Units 3 and 4.")

    res = []

    def add(q, ptest, pp, ntest, npv, result):
        ps = [p for p in (pp, npv) if p is not None]
        sig = all(p < alpha for p in ps)
        mixed = (not sig) and any(p < alpha for p in ps)
        res.append({"Research question": q,
                    "Parametric (Unit 3)": f"{ptest} ({p_eq(pp)})" if ptest else "—",
                    "Non-parametric (Unit 4)": f"{ntest} ({p_eq(npv)})" if ntest else "—",
                    "Result": result,
                    "Significant?": "Yes" if sig else ("Mixed" if mixed else "No"),
                    "_p": max(ps)})

    m = df["midterm"].dropna()
    d = (df["midterm"] - df["gpa_prev"]).dropna()

    # Q1 one mean / sign test
    t1 = (m.mean() - 8) / (m.std() / np.sqrt(len(m)))
    p1 = p_from_dist(stats.t(len(m) - 1), t1, "greater")
    p1s = sign_test(m - 8, "greater")[3]
    add("Q1. Is the mean (median) midterm grade above 8.0?", "t test, one mean", p1, "Sign test", p1s,
        f"mean {m.mean():.2f}, median {m.median():.1f}")
    # Q2 paired
    t2 = d.mean() / (d.std() / np.sqrt(len(d)))
    p2 = p_from_dist(stats.t(len(d) - 1), t2, "two-sided")
    p2s = sign_test(d, "two-sided")[3]
    p2w = signed_rank(d, "two-sided")[6]
    add("Q2. Midterm grade vs previous GPA (paired)", "t test on differences", p2,
        f"Sign test ({p_eq(p2s)}); Wilcoxon signed-rank", p2w, f"mean change {d.mean():+.2f}; {int((d < 0).sum())} of {len(d)} dropped")
    # Q3 job
    a = df.loc[df["job"] == "Yes", "midterm"].dropna()
    b = df.loc[df["job"] == "No", "midterm"].dropna()
    Fj = a.var() / b.var()
    pF = p_from_dist(stats.f(len(a) - 1, len(b) - 1), Fj, "two-sided")
    tt = stats.ttest_ind(a, b, equal_var=pF >= alpha)
    p3w = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    add("Q3. Midterm grade: job vs no job", f"F test ({p_eq(pF)}) → {'pooled' if pF >= alpha else 'Welch'} t", tt.pvalue,
        "Wilcoxon rank-sum / Mann–Whitney", p3w, f"means {a.mean():.2f} vs {b.mean():.2f}; {'equal' if pF >= alpha else 'unequal'} variances")
    # Q4 ANOVA sleep
    lvs = ordered_levels(df["sleep_3"])
    gr = [df.loc[df["sleep_3"] == l, "midterm"].dropna() for l in lvs]
    p4 = float(stats.f_oneway(*gr).pvalue)
    add("Q4. Midterm grade by sleep quality (3 groups)", "One-way ANOVA", p4, None, None,
        " / ".join(f"{g_.mean():.2f}" for g_ in gr) + " (bad / normal / good)")
    # Q5 variance
    chi5 = (len(m) - 1) * m.var() / 1.0
    p5 = p_from_dist(stats.chi2(len(m) - 1), chi5, "less")
    add("Q5. Is the SD of midterm grades below 1 point?", "χ² test, one variance", p5, None, None, f"s = {m.std():.2f}")
    # Q6 proportion
    st2 = (df["study_hours_day"] >= 2)
    ph = st2.mean()
    z6 = (ph - 0.5) / np.sqrt(0.25 / len(st2))
    p6 = p_from_dist(stats.norm(), z6, "less")
    add("Q6. Do fewer than half study 2+ hours per day?", "Z test, one proportion", p6, None, None, f"p̂ = {ph:.1%}")
    # Q7 normality of grades
    _, p7k = ks_normal(m)
    xx = m.to_numpy(float)
    edges = stats.norm.ppf(np.linspace(0, 1, 7), xx.mean(), xx.std(ddof=1))
    obs = np.bincount(np.clip(np.searchsorted(edges, xx, side="right") - 1, 0, 5), minlength=6)
    exp = np.full(6, len(xx) / 6)
    p7c = float(stats.chi2.sf(((obs - exp) ** 2 / exp).sum(), 6 - 3))
    add("Q7. Do midterm grades follow a normal distribution?", None, None,
        f"χ² GOF, k = 6 ({p_eq(p7c)}); Kolmogorov–Smirnov", p7k,
        "not normal" if min(p7c, p7k) < alpha else "consistent with normal")
    res[-1]["Significant?"] = "Yes" if (p7c < alpha and p7k < alpha) else ("Mixed" if min(p7c, p7k) < alpha else "No")
    res[-1]["_p"] = max(p7c, p7k)
    # Q8 study methods uniform
    cnt = df["method"].value_counts()
    p8 = float(stats.chisquare(cnt.to_numpy()).pvalue)
    add("Q8. Are the four study methods equally popular?", None, None, "χ² goodness of fit", p8,
        ", ".join(f"{k_} {v_}" for k_, v_ in cnt.items()))

    out = pd.DataFrame(res)
    st.dataframe(out.drop(columns="_p").style.map(
        lambda v: "background-color: #DCEBDD; font-weight: 600" if v == "Yes" else
        ("background-color: #FBE7D3" if v == "Mixed" else ""), subset=["Significant?"]),
        hide_index=True, use_container_width=True)
    st.caption("“Significant?” = Yes when every test reported for the question rejects H₀; Mixed when only some do. "
               "Q1 and Q6 are one-sided (right and left tailed); Q5 is left-tailed.")

    out["-log10(p)"] = -np.log10(out["_p"].clip(lower=1e-10))
    out["Short"] = out["Research question"].str.split(". ", n=1).str[0] + ": " + \
        out["Research question"].str.split(". ", n=1).str[1].str.slice(0, 42)
    fig = px.bar(out.sort_values("-log10(p)"), x="-log10(p)", y="Short", orientation="h",
                 color="Significant?", color_discrete_map={"Yes": PALETTE[2], "Mixed": PALETTE[1], "No": "#B0B0B0"},
                 labels={"Short": ""})
    fig.add_vline(-np.log10(alpha), line_dash="dash", annotation_text=f"α = {alpha}", annotation_position="bottom right")
    fig.update_layout(height=460, title="Strength of evidence (largest p-value among the tests of each question)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bars that pass the dashed line are statistically significant.")

    st.subheader("Interpretation")
    st.markdown(
        f"""
- **Level of grades:** the mean midterm grade ({m.mean():.2f}) is significantly above 8.0 (t test and sign test).
- **Change vs previous semester:** grades were on average **{abs(d.mean()):.2f} points {'lower' if d.mean() < 0 else 'higher'}**
  than the previous GPA; {int((d < 0).sum())} of {len(d)} students dropped (t test on differences, sign test, Wilcoxon signed-rank).
- **Job and sleep:** having a job (pooled t, Wilcoxon rank-sum) and sleep quality (ANOVA) show **no significant differences** in grades.
- **Spread and habits:** grades are concentrated (s = {m.std():.2f}); {ph:.0%} of students study 2+ hours per day.
- **Distributions:** practice exercises and notes dominate the study methods; normality of grades is checked with the χ² and K–S
  goodness-of-fit tests, which is why each parametric test is paired with a non-parametric alternative.

**Limitations.** Small voluntary sample (n = {len(df)}), self-reported grades, small groups for some categories,
and K–S with estimated parameters is only approximate.
"""
    )

st.sidebar.divider()
st.sidebar.caption("Statistical report · UAP students survey · 2026")

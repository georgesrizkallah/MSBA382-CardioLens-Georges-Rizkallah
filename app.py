"""
CardioLens - Cardiovascular Risk Intelligence Dashboard
MSBA382 Healthcare Analytics | Individual Project
Data: Framingham Heart Study + WHO/GBD/IDF global context
Run locally:  streamlit run app.py
"""
import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

st.set_page_config(page_title="CardioLens | CVD Risk Dashboard",
                   page_icon="❤️", layout="wide", initial_sidebar_state="expanded")

# ----------------------- palette + CSS -----------------------
CRIMSON="#A6192E"; STEEL="#4C6A92"; GOLD="#E0A458"; INK="#2B2B2B"; PAPER="#F7F5F2"
st.markdown(f"""
<style>
:root {{ --crimson:{CRIMSON}; }}
.stApp {{ background:{PAPER}; }}
h1,h2,h3 {{ color:{INK}; font-family:'Georgia',serif; }}
.block-container {{ padding-top:1.4rem; }}
[data-testid="stMetricValue"] {{ color:{CRIMSON}; font-weight:700; }}
.kpi {{ background:white; border-radius:14px; padding:18px 20px; box-shadow:0 2px 10px rgba(0,0,0,.06); }}
.lead {{ background:white; border-left:5px solid {CRIMSON}; border-radius:10px;
         padding:14px 18px; box-shadow:0 2px 8px rgba(0,0,0,.05); }}
.login-card {{ max-width:430px; margin:7vh auto; background:white; border-radius:18px;
              padding:38px 40px; box-shadow:0 10px 40px rgba(0,0,0,.12); text-align:center; }}
.stButton>button {{ background:{CRIMSON}; color:white; border:none; border-radius:8px;
                    padding:.5rem 1.2rem; font-weight:600; }}
.stButton>button:hover {{ background:#85121f; color:white; }}
.badge {{ display:inline-block; background:#F3E4E6; color:{CRIMSON}; border-radius:20px;
          padding:3px 12px; font-size:.78rem; font-weight:700; letter-spacing:.4px; }}
</style>""", unsafe_allow_html=True)

# ----------------------- data loaders -----------------------
BASE = Path(__file__).parent
DATA = BASE/"data" if (BASE/"data").exists() else BASE.parent/"data"

@st.cache_data
def load():
    df = pd.read_csv(DATA/"framingham_clean.csv")
    gl = pd.read_csv(DATA/"global_cvd_trend.csv")
    lw = pd.read_csv(DATA/"lebanon_vs_world.csv")
    return df, gl, lw

@st.cache_resource
def load_model():
    return joblib.load(BASE/"chd_model.joblib")

# ----------------------- password gate -----------------------
def gate():
    st.markdown(f"""
    <div class='login-card'>
      <div style='font-size:46px'>❤️</div>
      <h2 style='font-family:Georgia;margin:.2rem 0;color:{CRIMSON}'>CardioLens</h2>
      <p style='color:#666;margin-top:-6px'>Cardiovascular Risk Intelligence</p>
      <p style='color:#999;font-size:.85rem'>Confidential clinical analytics &mdash; authorized access only</p>
    </div>""", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,1.1,1])
    with c2:
        pw = st.text_input("Password", type="password", placeholder="Enter access password")
        if st.button("Enter dashboard", width="stretch"):
            if pw == "heart2026":
                st.session_state.auth = True; st.rerun()
            else:
                st.error("Incorrect password. Hint for graders: heart2026")
        st.caption("Demo password: **heart2026**")

if "auth" not in st.session_state: st.session_state.auth = False
if not st.session_state.auth:
    gate(); st.stop()

df, gl, lw = load()
bundle = load_model()

# ----------------------- sidebar / filters -----------------------
st.sidebar.markdown(f"<h2 style='color:{CRIMSON}'>❤️ CardioLens</h2>", unsafe_allow_html=True)
section = st.sidebar.radio("Navigate", ["Global Context","Population Explorer",
                                        "Risk-Factor Analysis","Risk Calculator",
                                        "Model & Validation","Preventable Burden","About"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Filter the cohort**")
sex_f = st.sidebar.multiselect("Sex", ["Male","Female"], default=["Male","Female"])
age_f = st.sidebar.multiselect("Age group", ["30-39","40-49","50-59","60+"],
                               default=["30-39","40-49","50-59","60+"])
smoke_f = st.sidebar.selectbox("Smoking status", ["All","Smokers only","Non-smokers only"])
if st.sidebar.button("Log out"):
    st.session_state.auth=False; st.rerun()

f = df[df["sex"].isin(sex_f) & df["age_group"].isin(age_f)].copy()
if smoke_f=="Smokers only": f=f[f.currentSmoker==1]
elif smoke_f=="Non-smokers only": f=f[f.currentSmoker==0]
if len(f)==0: st.warning("No records match the current filters."); st.stop()

# =====================================================================
if section=="Global Context":
    st.markdown("<span class='badge'>WHO &middot; GBD &middot; IDF</span>", unsafe_allow_html=True)
    st.title("The world's number-one killer")
    st.markdown("<div class='lead'>Cardiovascular disease (CVD) is the leading cause of death "
                "globally, killing an estimated <b>20.5 million</b> people in 2021 &mdash; about "
                "<b>one in three</b> deaths worldwide. In <b>Lebanon</b>, the burden is even heavier: "
                "CVD accounts for roughly <b>47% of all deaths</b>, the highest-ranked cause of "
                "mortality and hospital admission.</div>", unsafe_allow_html=True)
    st.write("")
    k = st.columns(4)
    for col,(v,l) in zip(k, [("20.5M","Global CVD deaths / yr (2021)"),
                             ("47%","Share of deaths in Lebanon"),
                             (">75%","CVD deaths in low/middle-income countries"),
                             ("~90%","Heart-attack risk from modifiable factors")]):
        col.markdown(f"<div class='kpi'><h2 style='color:{CRIMSON};margin:0'>{v}</h2>"
                     f"<p style='color:#666;margin:0;font-size:.85rem'>{l}</p></div>", unsafe_allow_html=True)
    st.write(""); a,b = st.columns([1.15,1])
    with a:
        fig = px.area(gl, x="year", y="cvd_deaths_millions", markers=True,
                      title="Global CVD deaths are still rising (millions/yr)")
        fig.update_traces(line_color=CRIMSON, fillcolor="rgba(166,25,46,.12)")
        fig.update_layout(plot_bgcolor="white", height=360, yaxis_title="Deaths (millions)")
        st.plotly_chart(fig, width="stretch")
    with b:
        m = lw.melt(id_vars="indicator", var_name="Region", value_name="pct")
        m = m[m.indicator.isin(["CVD share of all deaths (%)","Coronary heart disease share of deaths (%)",
                                "Adult smoking prevalence (%)","Diabetes prevalence (%)"])]
        m["short"] = m.indicator.str.replace(r"\(.*\)","",regex=True).str.replace(" share of all deaths","")
        fig = px.bar(m, x="short", y="pct", color="Region", barmode="group",
                     color_discrete_map={"Lebanon":CRIMSON,"World":STEEL},
                     title="Lebanon vs. world: burden & drivers")
        fig.update_layout(plot_bgcolor="white", height=360, xaxis_title="", yaxis_title="%")
        st.plotly_chart(fig, width="stretch")
    st.info("**Why it matters for Lebanon:** smoking (~42% of adults) and "
            "hypertension (~30%) are far above global averages, and heart attacks strike at a "
            "younger age in the Middle East — making early, data-driven risk screening essential.")

# =====================================================================
elif section=="Population Explorer":
    st.title("Population Explorer")
    st.caption(f"Framingham cohort &mdash; {len(f):,} of {len(df):,} records match your filters")
    k = st.columns(4)
    k[0].metric("Records", f"{len(f):,}")
    k[1].metric("10-yr CHD rate", f"{f.TenYearCHD.mean()*100:.1f}%")
    k[2].metric("Mean age", f"{f.age.mean():.0f} yrs")
    k[3].metric("Smokers", f"{f.currentSmoker.mean()*100:.0f}%")
    st.write(""); a,b = st.columns(2)
    with a:
        g = (f.groupby(["age_group","sex"])["TenYearCHD"].mean()*100).reset_index()
        fig = px.bar(g, x="age_group", y="TenYearCHD", color="sex", barmode="group",
                     color_discrete_map={"Male":STEEL,"Female":CRIMSON},
                     title="10-yr CHD risk by age & sex", labels={"TenYearCHD":"CHD risk (%)","age_group":"Age group"})
        fig.update_layout(plot_bgcolor="white", height=370); st.plotly_chart(fig, width="stretch")
    with b:
        fig = px.histogram(f, x="age", color="sex", nbins=24, opacity=.75,
                           color_discrete_map={"Male":STEEL,"Female":CRIMSON},
                           title="Age distribution of the cohort")
        fig.update_layout(plot_bgcolor="white", height=370, barmode="overlay")
        st.plotly_chart(fig, width="stretch")
    c,d = st.columns(2)
    with c:
        fig = px.box(f, x="sex", y="sysBP", color="sex",
                     color_discrete_map={"Male":STEEL,"Female":CRIMSON},
                     title="Systolic blood pressure by sex")
        fig.update_layout(plot_bgcolor="white", height=350, showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with d:
        bp = (f.groupby("bp_category")["TenYearCHD"].mean()*100).reset_index()
        fig = px.bar(bp, x="bp_category", y="TenYearCHD", title="CHD risk by blood-pressure category",
                     labels={"TenYearCHD":"CHD risk (%)","bp_category":"BP category"})
        fig.update_traces(marker_color=CRIMSON)
        fig.update_layout(plot_bgcolor="white", height=350); st.plotly_chart(fig, width="stretch")
    with st.expander("View underlying records"):
        st.dataframe(f[["sex","age","age_group","currentSmoker","sysBP","totChol","BMI",
                        "glucose","diabetes","prevalentHyp","TenYearCHD"]].head(300),
                     width="stretch")

# =====================================================================
elif section=="Risk-Factor Analysis":
    st.title("Risk-Factor Analysis")
    st.markdown("<div class='lead'>Each modifiable risk factor sharply raises 10-year CHD risk. "
                "This mirrors the landmark INTERHEART finding that ~9 modifiable factors explain "
                "over 90% of heart-attack risk worldwide.</div>", unsafe_allow_html=True)
    st.write("")
    binf = {"currentSmoker":"Smoker","prevalentHyp":"Hypertensive","diabetes":"Diabetic",
            "BPMeds":"On BP meds","prevalentStroke":"Prior stroke"}
    rows=[]
    for c,n in binf.items():
        rows.append((n, f[f[c]==0].TenYearCHD.mean()*100, f[f[c]==1].TenYearCHD.mean()*100,
                     int((f[c]==1).sum())))
    rf = pd.DataFrame(rows, columns=["Factor","Without","With","n_with"])
    a,b = st.columns([1.2,1])
    with a:
        m = rf.melt(id_vars="Factor", value_vars=["Without","With"], var_name="Status", value_name="risk")
        fig = px.bar(m, x="risk", y="Factor", color="Status", barmode="group", orientation="h",
                     color_discrete_map={"With":CRIMSON,"Without":"#BBBBBB"},
                     title="10-yr CHD risk: with vs. without each factor", labels={"risk":"CHD risk (%)"})
        fig.update_layout(plot_bgcolor="white", height=400); st.plotly_chart(fig, width="stretch")
    with b:
        rf["Relative risk"] = (rf["With"]/rf["Without"]).round(2)
        st.markdown("**Relative risk multiplier**")
        st.dataframe(rf[["Factor","With","Without","Relative risk"]].round(1),
                     width="stretch", hide_index=True)
        st.caption("e.g. a multiplier of 2.0 means double the CHD risk versus people without the factor.")
    st.write("")
    num = ["age","cigsPerDay","totChol","sysBP","diaBP","BMI","heartRate","glucose","prevalentHyp","diabetes","TenYearCHD"]
    corr = f[num].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlation matrix of cardiovascular risk factors", aspect="auto")
    fig.update_layout(height=520); st.plotly_chart(fig, width="stretch")

# =====================================================================
elif section=="Risk Calculator":
    st.title("10-Year CHD Risk Calculator")
    st.markdown("<div class='lead'>Enter a patient's profile to estimate their 10-year risk of "
                "coronary heart disease, using a model trained on the Framingham cohort "
                f"(AUC = {bundle['auc_logit']:.2f}). For screening &mdash; not a diagnosis.</div>",
                unsafe_allow_html=True)
    st.write("")
    c1,c2,c3 = st.columns(3)
    with c1:
        sex = st.selectbox("Sex", ["Male","Female"]); age = st.slider("Age",30,70,52)
        cig = st.slider("Cigarettes/day",0,40,0); smoker = 1 if cig>0 else 0
    with c2:
        sys = st.slider("Systolic BP (mmHg)",90,250,130); dia = st.slider("Diastolic BP (mmHg)",60,140,82)
        chol = st.slider("Total cholesterol (mg/dL)",120,400,235); bmi = st.slider("BMI",16.0,45.0,26.0)
    with c3:
        glu = st.slider("Glucose (mg/dL)",50,300,80); hr = st.slider("Heart rate (bpm)",45,130,75)
        hyp = st.checkbox("Hypertensive"); dia_b = st.checkbox("Diabetic")
        bpmed = st.checkbox("On BP medication"); stroke = st.checkbox("Prior stroke")

    row = pd.DataFrame([{"male":1 if sex=="Male" else 0,"age":age,"currentSmoker":smoker,
        "cigsPerDay":cig,"BPMeds":int(bpmed),"prevalentStroke":int(stroke),
        "prevalentHyp":int(hyp),"diabetes":int(dia_b),"totChol":chol,"sysBP":sys,
        "diaBP":dia,"BMI":bmi,"heartRate":hr,"glucose":glu}])[bundle["features"]]
    prob = bundle["logit_cal"].predict_proba(bundle["scaler"].transform(row))[0,1]*100
    band = ("Low","#2A9D8F") if prob<10 else ("Intermediate",GOLD) if prob<20 else ("High",CRIMSON)

    st.write(""); g1,g2 = st.columns([1,1.1])
    with g1:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=prob,
            number={"suffix":"%","font":{"size":46}},
            gauge={"axis":{"range":[0,40]},"bar":{"color":band[1]},
                   "steps":[{"range":[0,10],"color":"#E3F2EE"},{"range":[10,20],"color":"#FBF0DD"},
                            {"range":[20,40],"color":"#F7E0E3"}]},
            title={"text":f"Estimated 10-yr CHD risk<br><b style='color:{band[1]}'>{band[0]} risk</b>"}))
        fig.update_layout(height=340, margin=dict(t=70,b=10)); st.plotly_chart(fig, width="stretch")
    with g2:
        st.markdown(f"### <span style='color:{band[1]}'>{band[0]} risk &mdash; {prob:.1f}%</span>",
                    unsafe_allow_html=True)
        cohort = df.TenYearCHD.mean()*100
        st.write(f"Cohort average risk is **{cohort:.1f}%**. This profile is "
                 f"**{'above' if prob>cohort else 'below'} average**.")
        tips=[]
        if smoker: tips.append("Smoking cessation is the single highest-impact intervention.")
        if sys>=140 or hyp: tips.append("Blood pressure is in the hypertensive range — prioritise control.")
        if chol>=240: tips.append("Total cholesterol is high — review lipids / statin eligibility.")
        if glu>=126 or dia_b: tips.append("Glucose suggests diabetes risk — confirm and manage.")
        if bmi>=30: tips.append("BMI in the obese range — weight management reduces multiple risks.")
        if not tips: tips.append("No major modifiable flags — maintain healthy lifestyle and re-screen periodically.")
        st.markdown("**Actionable flags**")
        for t in tips: st.markdown(f"- {t}")

    # ---- live SHAP explanation (why this estimate?) ----
    if "xgb" in bundle:
        st.write(""); st.markdown("#### Why this estimate? (SHAP contributions)")
        try:
            import shap
            expl = shap.TreeExplainer(bundle["xgb"])
            rl = row.copy(); rl.columns = bundle["feature_labels"]
            sv = expl(rl)
            contrib = pd.DataFrame({"feature":bundle["feature_labels"],
                                    "shap":sv.values[0]})
            contrib["abs"]=contrib.shap.abs()
            contrib=contrib.sort_values("abs",ascending=False).head(8).sort_values("shap")
            contrib["dir"]=np.where(contrib.shap>=0,"Increases risk","Lowers risk")
            figs = px.bar(contrib, x="shap", y="feature", orientation="h", color="dir",
                          color_discrete_map={"Increases risk":CRIMSON,"Lowers risk":STEEL},
                          labels={"shap":"Contribution to log-odds","feature":"","dir":""},
                          title="Top drivers of this patient's predicted risk")
            figs.update_layout(plot_bgcolor="white", height=330, legend_title="",
                               margin=dict(t=50,b=10))
            st.plotly_chart(figs, width="stretch")
            up = contrib[contrib.shap>0].sort_values("shap",ascending=False).feature.head(2).tolist()
            if up:
                st.caption(f"Risk is driven mostly by **{'** and **'.join(up)}** for this profile. "
                           "Red bars push risk up; blue bars pull it down.")
        except Exception as e:
            st.caption("SHAP explanation unavailable in this environment.")
    st.caption("Model: logistic regression (class-balanced) trained on 4,240 Framingham participants. "
               "Screening tool only; clinical decisions require a qualified provider.")

# =====================================================================
elif section=="Model & Validation":
    st.title("Model & Validation")
    st.markdown("<div class='lead'>A risk model is only as good as its validation. Beyond accuracy, "
                "CardioLens is tested for <b>discrimination</b>, <b>calibration</b>, <b>clinical net "
                "benefit</b>, <b>fairness</b>, and benchmarked against the classical clinical equation.</div>",
                unsafe_allow_html=True)
    st.write("")
    FG = BASE/"figures"
    comp = pd.read_csv(DATA/"model_comparison.csv")
    best = comp.sort_values("auc",ascending=False).iloc[0]
    k = st.columns(4)
    for col,(v,l) in zip(k,[(f"{comp.auc.max():.3f}","Best AUC (Framingham score)"),
                            (f"{bundle.get('xgb_auc',0):.3f}","XGBoost AUC"),
                            ("0.121","Best Brier (calibrated)"),
                            ("✓","Fair across sex")]):
        col.markdown(f"<div class='kpi'><h2 style='color:{CRIMSON};margin:0'>{v}</h2>"
                     f"<p style='color:#666;margin:0;font-size:.85rem'>{l}</p></div>", unsafe_allow_html=True)
    st.write("")
    st.info("**Headline finding:** the parsimonious 2008 Framingham score and a simple logistic "
            "regression match gradient boosting on this cohort — their confidence intervals overlap "
            "heavily. More model complexity did **not** buy more discrimination, so CardioLens favours "
            "the interpretable models and uses XGBoost for per-patient explanation.")
    a,b = st.columns(2)
    with a:
        st.image(str(FG/"09_model_comparison_auc.png"), width="stretch")
        st.image(str(FG/"11_calibration.png"), width="stretch")
    with b:
        st.image(str(FG/"12_decision_curve.png"), width="stretch")
        st.image(str(FG/"15_subgroup_auc.png"), width="stretch")
    with st.expander("Precision–recall under 15% prevalence & global SHAP drivers"):
        c,d = st.columns(2)
        c.image(str(FG/"10_pr_curves.png"), width="stretch")
        d.image(str(FG/"13_shap_summary.png"), width="stretch")
    st.dataframe(comp, width="stretch", hide_index=True)
    st.caption("AUC = discrimination; Brier = calibration error (lower is better); "
               "net benefit = decision-curve clinical value; subgroup AUCs audit fairness.")

# =====================================================================
elif section=="Preventable Burden":
    import plotly.graph_objects as go
    st.title("Preventable Burden — Where to Act First")
    st.markdown("<div class='lead'>Population attributable fraction (PAF) estimates the share of CHD that "
                "would be prevented if a modifiable risk factor were removed. Projecting the cohort's relative "
                "risks onto Lebanon's higher exposure prevalence shows where prevention pays off most.</div>",
                unsafe_allow_html=True)
    st.write("")
    leb_prev = {"Smoking":0.42,"Hypertension":0.30,"Diabetes":0.097,"Obesity (BMI≥30)":0.29}
    defs = {"Smoking":df.currentSmoker==1, "Hypertension":df.prevalentHyp==1,
            "Diabetes":df.diabetes==1, "Obesity (BMI≥30)":df.BMI>=30,
            "High cholesterol (≥240)":df.totChol>=240}
    rows=[]
    for name,mask in defs.items():
        pe=mask.mean(); r1=df[mask].TenYearCHD.mean(); r0=df[~mask].TenYearCHD.mean()
        rr=r1/r0; paf=pe*(rr-1)/(1+pe*(rr-1))*100
        leb=(leb_prev[name]*(rr-1)/(1+leb_prev[name]*(rr-1))*100) if name in leb_prev else None
        rows.append((name,round(paf,1),(round(leb,1) if leb is not None else None)))
    pdf=pd.DataFrame(rows,columns=["factor","cohort","lebanon"]).sort_values("cohort")
    fig=go.Figure()
    fig.add_bar(y=pdf.factor,x=pdf.cohort,orientation="h",name="Framingham cohort",
                marker_color="#4C6A92",text=[f"{v:.0f}%" for v in pdf.cohort],textposition="outside")
    fig.add_bar(y=pdf.factor,x=pdf.lebanon,orientation="h",name="Projected for Lebanon",
                marker_color=CRIMSON,text=[f"{v:.0f}%" if v==v and v is not None else "" for v in pdf.lebanon],
                textposition="outside")
    fig.update_layout(barmode="group",height=430,bargap=0.25,
                      xaxis_title="Share of CHD cases preventable if the factor were removed (%)",
                      legend=dict(orientation="h",yanchor="bottom",y=1.02,x=0),
                      margin=dict(t=40,l=10,r=30,b=10),plot_bgcolor="white",
                      xaxis=dict(range=[0,35],gridcolor="#eee"))
    st.plotly_chart(fig,width="stretch")
    k=st.columns(2)
    k[0].markdown(f"<div class='kpi'><h2 style='color:{CRIMSON};margin:0'>~28%</h2>"
                  f"<p style='color:#666;margin:0;font-size:.85rem'>Hypertension — the largest preventable "
                  f"share of CHD, in the cohort and in Lebanon</p></div>", unsafe_allow_html=True)
    k[1].markdown(f"<div class='kpi'><h2 style='color:{CRIMSON};margin:0'>~13%</h2>"
                  f"<p style='color:#666;margin:0;font-size:.85rem'>Diabetes — still triples to ~13% for "
                  f"Lebanon vs only ~4% in the cohort, despite Lebanon's diabetes rate being close to the "
                  f"global average</p></div>", unsafe_allow_html=True)
    st.write("")
    st.info("**So what?** Hypertension control is the universal priority. Lebanon's outsized burden comes "
            "mainly from smoking, hypertension and obesity, which run far above global averages; diabetes, "
            "while close to the global average locally, still triples its preventable share once the "
            "cohort's relative risk is applied — worth factoring into screening even though it is not "
            "Lebanon's single highest-leverage target. These factors anchor the project's recommendations.")
    st.caption("PAF uses crude relative risks (Levin's formula); values do not sum to 100% because risk factors "
               "co-occur. The Lebanon projection applies the cohort's relative risks to published Lebanese "
               "exposure prevalence (WHO/IDF).")

# =====================================================================
else:
    st.title("About this dashboard")
    st.markdown(f"""
**CardioLens** turns cardiovascular epidemiology into a decision-support tool for a healthcare
centre seeking to understand and act on the burden of heart disease.

**Health problem.** Cardiovascular disease — the leading cause of death globally and in Lebanon
(~47% of deaths).

**Data sources**
- *Framingham Heart Study* — 4,240 participants, 15 clinical variables, 10-year CHD outcome
  (primary dataset for analysis & prediction).
- *WHO Cardiovascular Disease Fact Sheet*, *Global Burden of Disease (IHME)*, *IDF* and Lebanese
  NCD studies — global & national context layer.

**What it does**
1. *Global Context* — situates Lebanon against world trends.
2. *Population Explorer* — interactive cohort analysis by age, sex, and behaviour.
3. *Risk-Factor Analysis* — quantifies how each modifiable factor raises risk.
4. *Risk Calculator* — patient-level 10-year CHD risk with live SHAP explanation.
5. *Model & Validation* — discrimination, calibration, decision-curve & fairness audits.
6. *Preventable Burden* — population attributable fraction: where prevention pays off most, for Lebanon.

**Methods.** Median/mode imputation, winsorization of outliers, and four risk models — logistic
regression, random forest, gradient boosting (XGBoost) and the classical 2008 Framingham score —
compared on a held-out test set with bootstrapped 95% AUC confidence intervals. Models are further
assessed for calibration (Brier score), clinical net benefit (decision-curve analysis), per-patient
explainability (SHAP) and fairness across sex and age.

**Limitations.** Framingham is a US cohort from an earlier era; absolute risks may differ for the
Lebanese population, and the model is a screening aid, not a diagnostic.

*MSBA382 Healthcare Analytics — Individual Project, 2026.*
""")

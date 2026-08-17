import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pakistani Emigration Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1448 40%, #0d1b4b 100%);
    color: #e8e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    border-right: 1px solid rgba(100, 180, 255, 0.15);
}
[data-testid="stSidebar"] * { color: #c8d8f0 !important; }

/* Header */
.hero-header {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(100,180,255,0.08) 0%, rgba(180,100,255,0.08) 100%);
    border-radius: 16px;
    border: 1px solid rgba(100,180,255,0.2);
    margin-bottom: 1.5rem;
}
.hero-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #64b4ff, #c87aff, #64b4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-header p {
    font-family: 'DM Sans', sans-serif;
    color: #8899bb;
    font-size: 1rem;
    margin: 0;
}

/* KPI Cards */
.kpi-grid { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 150px;
    background: linear-gradient(135deg, rgba(100,180,255,0.1), rgba(180,100,255,0.06));
    border: 1px solid rgba(100,180,255,0.25);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #64b4ff;
    line-height: 1;
}
.kpi-label {
    font-size: 0.75rem;
    color: #7788aa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.4rem;
}

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #a0c8ff;
    margin: 1.2rem 0 0.6rem;
    letter-spacing: 0.5px;
}

/* Divider label */
.section-divider {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #c87aff;
    margin: 2rem 0 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(200,122,255,0.25);
    letter-spacing: 0.5px;
}

/* Plotly chart containers */
.chart-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(100,180,255,0.12);
    border-radius: 14px;
    padding: 0.5rem;
    margin-bottom: 1rem;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(100,180,255,0.3); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Plotly Theme ─────────────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#c8d8f0", size=12),
    title_font=dict(family="Syne, sans-serif", size=15, color="#a0c8ff"),
)
AXIS_STYLE = dict(
    gridcolor="rgba(100,180,255,0.08)",
    linecolor="rgba(100,180,255,0.2)",
    tickfont=dict(size=11),
)
LEGEND_STYLE = dict(
    bgcolor="rgba(15,12,41,0.7)",
    bordercolor="rgba(100,180,255,0.2)",
    borderwidth=1,
)

def layout(**kwargs):
    base = dict(
        **PLOTLY_BASE,
        xaxis=dict(**AXIS_STYLE),
        yaxis=dict(**AXIS_STYLE),
        legend=dict(**LEGEND_STYLE, font=dict(size=11)),
        margin=dict(l=60, r=30, t=50, b=60),
    )
    base.update(kwargs)
    return base

COLOR_SEQ = px.colors.qualitative.Plotly + px.colors.qualitative.Bold
# Distinct warm palette for countries to visually separate from profession charts
COUNTRY_COLORS = [
    "#f7c948", "#ff7043", "#ab47bc", "#26c6da", "#66bb6a",
    "#ef5350", "#42a5f5", "#ff8a65", "#d4e157", "#26a69a",
]
ACCENT = "#64b4ff"

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_profession_data():
    df = pd.read_csv(
        "number-of-pakistani-emigrants-profession-wise-1981-2026.csv",
        header=None,
        names=["Profession", "Year", "Count"],
        encoding="utf-8-sig",
    )
    df["Profession"] = df["Profession"].str.strip()
    df["Profession"] = df["Profession"].replace(
        {"Others.": "Others", "Others. ": "Others"}
    )
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)
    df = df.groupby(["Profession", "Year"], as_index=False)["Count"].sum()
    return df

@st.cache_data
def load_country_data():
    df = pd.read_csv(
        "number-of-pakistani-emigrants-country-wise-1981-2026.csv",
        encoding="utf-8-sig",
    )
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Destination Country": "Country",
        "No. of Emigrants": "Count",
        "Destination Region": "Region",
        "Dest. Country Latitude": "Lat",
        "Dest. Country Longitude": "Lon",
    })
    df["Country"] = df["Country"].str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)
    df = df.groupby(["Country", "Region", "Year"], as_index=False)["Count"].sum()
    return df

df = load_profession_data()
dfc = load_country_data()

all_years = sorted(df["Year"].unique())
all_professions = sorted(df["Profession"].unique())
all_countries = sorted(dfc["Country"].unique())

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filters")

    year_range = st.slider(
        "Year Range",
        min_value=int(min(all_years)),
        max_value=int(max(all_years)),
        value=(1981, 2026),
        step=1,
    )

    selected_profs = st.multiselect(
        "Professions (for trend chart)",
        options=all_professions,
        default=["Labourer", "Driver", "Technician", "Engineer", "Doctor"],
        help="Select professions to compare in the trend line chart.",
    )

    top_n = st.slider("Top N Professions (bar/pie charts)", 5, 20, 10)

    st.markdown("---")

    # Country-specific filter: top N for bar/pie
    top_n_countries = st.slider("Top N Countries (bar/pie charts)", 5, 20, 10)

    # Countries for the multi-line trend
    # Pre-select top 5 by overall total
    default_countries = (
        dfc.groupby("Country")["Count"].sum()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )
    selected_countries = st.multiselect(
        "Countries (for trend chart)",
        options=all_countries,
        default=default_countries,
        help="Select destination countries to compare in the trend line chart.",
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#556'>Data source: Bureau of Emigration & Overseas Employment, Pakistan · 1981–2026</small>",
        unsafe_allow_html=True,
    )

# ─── Filter Data ──────────────────────────────────────────────────────────────
mask = (df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])
dff = df[mask].copy()

maskc = (dfc["Year"] >= year_range[0]) & (dfc["Year"] <= year_range[1])
dffc = dfc[maskc].copy()

total_emigrants = dff["Count"].sum()
total_years = year_range[1] - year_range[0] + 1
peak_year_row = dff.groupby("Year")["Count"].sum().idxmax()
peak_year_count = dff.groupby("Year")["Count"].sum().max()
top_country = dffc.groupby("Country")["Count"].sum().idxmax()
top_country_count = dffc.groupby("Country")["Count"].sum().max()

# ─── Hero Header ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-header">
  <h1>🌍 Pakistani Emigration Dashboard</h1>
  <p>Profession-wise &amp; Country-wise emigration trends · 1981–2026 &nbsp;·&nbsp; Bureau of Emigration &amp; Overseas Employment</p>
</div>
""", unsafe_allow_html=True)

# ─── KPI Row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">{total_emigrants/1_000_000:.2f}M</div>
        <div class="kpi-label">Total Emigrants</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">{peak_year_row}</div>
        <div class="kpi-label">Peak Year ({peak_year_count:,})</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">{total_emigrants//total_years:,}</div>
        <div class="kpi-label">Avg / Year</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">{len(all_professions)}</div>
        <div class="kpi-label">Profession Categories</div>
    </div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">{top_country}</div>
        <div class="kpi-label">Top Destination ({top_country_count/1_000_000:.1f}M)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — PROFESSION-WISE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider">👔 Profession-wise Analysis</div>', unsafe_allow_html=True)

# ─── Chart 1: Annual Total Emigration ────────────────────────────────────────
st.markdown('<div class="section-title">📈 Annual Total Emigration</div>', unsafe_allow_html=True)

annual = dff.groupby("Year")["Count"].sum().reset_index()

fig_annual = go.Figure()
fig_annual.add_trace(go.Scatter(
    x=annual["Year"], y=annual["Count"],
    mode="lines",
    line=dict(color=ACCENT, width=2.5),
    fill="tozeroy",
    fillcolor="rgba(100,180,255,0.08)",
    name="Total Emigrants",
    hovertemplate="<b>%{x}</b><br>Emigrants: %{y:,}<extra></extra>",
))
fig_annual.update_layout(**layout(
    title="Total Pakistani Emigrants per Year",
    height=320,
    xaxis=dict(**AXIS_STYLE, dtick=5),
    yaxis=dict(**AXIS_STYLE, tickformat=","),
))
st.plotly_chart(fig_annual, use_container_width=True)

# ─── Chart 2 & 3: Bar + Pie ───────────────────────────────────────────────────
col_bar, col_pie = st.columns(2)

prof_totals = (
    dff.groupby("Profession")["Count"].sum()
    .sort_values(ascending=False)
    .reset_index()
)
top_df = prof_totals.head(top_n)

with col_bar:
    st.markdown(f'<div class="section-title">🏆 Top {top_n} Professions by Total</div>', unsafe_allow_html=True)
    fig_bar = go.Figure(go.Bar(
        y=top_df["Profession"],
        x=top_df["Count"],
        orientation="h",
        marker=dict(color=top_df["Count"], colorscale="Blues", showscale=False),
        text=top_df["Count"].apply(lambda x: f"{x/1000:.0f}K" if x < 1_000_000 else f"{x/1_000_000:.2f}M"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Total: %{x:,}<extra></extra>",
    ))
    fig_bar.update_layout(**layout(
        height=420,
        xaxis=dict(**AXIS_STYLE, tickformat=",", title=""),
        yaxis=dict(**AXIS_STYLE, autorange="reversed", title=""),
        margin=dict(l=110, r=70, t=20, b=40),
    ))
    st.plotly_chart(fig_bar, use_container_width=True)

with col_pie:
    st.markdown(f'<div class="section-title">🥧 Proportion — Top {top_n}</div>', unsafe_allow_html=True)
    rest = prof_totals.iloc[top_n:]["Count"].sum()
    pie_df = top_df.copy()
    if rest > 0:
        pie_df = pd.concat([pie_df, pd.DataFrame([{"Profession": "Others (Rest)", "Count": rest}])], ignore_index=True)

    fig_pie = go.Figure(go.Pie(
        labels=pie_df["Profession"],
        values=pie_df["Count"],
        hole=0.45,
        textinfo="percent",
        textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        marker=dict(
            colors=(px.colors.qualitative.Bold + px.colors.qualitative.Pastel) * 3,
            line=dict(color="rgba(15,12,41,0.8)", width=2),
        ),
    ))
    fig_pie.update_layout(**layout(
        height=420,
        showlegend=True,
        legend=dict(**LEGEND_STYLE, orientation="v", x=1.02, y=0.5, font=dict(size=10)),
        margin=dict(l=10, r=140, t=20, b=20),
        annotations=[dict(
            text=f"Top {top_n}", x=0.5, y=0.5, font_size=14, showarrow=False,
            font=dict(color="#a0c8ff", family="Syne"),
        )],
    ))
    st.plotly_chart(fig_pie, use_container_width=True)

# ─── Chart 4: Multi-line Profession Trend ─────────────────────────────────────
st.markdown('<div class="section-title">📊 Profession-wise Emigration Trends</div>', unsafe_allow_html=True)

if not selected_profs:
    st.info("Select at least one profession in the sidebar to view trends.")
else:
    trend_df = dff[dff["Profession"].isin(selected_profs)]
    trend_pivot = trend_df.pivot_table(index="Year", columns="Profession", values="Count", aggfunc="sum").fillna(0)

    fig_trend = go.Figure()
    for i, col in enumerate(trend_pivot.columns):
        fig_trend.add_trace(go.Scatter(
            x=trend_pivot.index,
            y=trend_pivot[col],
            mode="lines+markers",
            name=col,
            line=dict(width=2, color=COLOR_SEQ[i % len(COLOR_SEQ)]),
            marker=dict(size=4),
            hovertemplate=f"<b>{col}</b><br>Year: %{{x}}<br>Count: %{{y:,}}<extra></extra>",
        ))
    fig_trend.update_layout(**layout(
        title="Year-wise Trend by Profession",
        height=400,
        xaxis=dict(**AXIS_STYLE, dtick=5),
        yaxis=dict(**AXIS_STYLE, tickformat=","),
        legend=dict(**LEGEND_STYLE, orientation="h", x=0, y=-0.25, traceorder="normal"),
        margin=dict(l=60, r=30, t=50, b=120),
    ))
    st.plotly_chart(fig_trend, use_container_width=True)

# ─── Chart 5: Heatmap ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔥 Emigration Heatmap — Decade View</div>', unsafe_allow_html=True)

top15 = prof_totals.head(15)["Profession"].tolist()
heat_df = dff[dff["Profession"].isin(top15)].copy()
heat_df["Decade"] = (heat_df["Year"] // 10 * 10).astype(str) + "s"
heat_pivot = heat_df.pivot_table(index="Profession", columns="Decade", values="Count", aggfunc="sum").fillna(0)
heat_pivot = heat_pivot.reindex(top15)

fig_heat = go.Figure(go.Heatmap(
    z=heat_pivot.values,
    x=heat_pivot.columns.tolist(),
    y=heat_pivot.index.tolist(),
    colorscale="Blues",
    hoverongaps=False,
    hovertemplate="<b>%{y}</b><br>%{x}<br>Emigrants: %{z:,}<extra></extra>",
    text=[[f"{v/1000:.0f}K" if v < 1_000_000 else f"{v/1_000_000:.1f}M" for v in row] for row in heat_pivot.values],
    texttemplate="%{text}",
    textfont=dict(size=10, color="white"),
))
fig_heat.update_layout(**layout(
    title="Emigrant Count by Profession & Decade",
    height=480,
    yaxis=dict(**AXIS_STYLE, autorange="reversed"),
    xaxis=dict(**AXIS_STYLE),
    margin=dict(l=130, r=30, t=50, b=50),
))
st.plotly_chart(fig_heat, use_container_width=True)

# ─── Chart 6: Stacked Area ────────────────────────────────────────────────────
st.markdown('<div class="section-title">📦 Stacked Composition Over Time (Top 8)</div>', unsafe_allow_html=True)

top8 = prof_totals.head(8)["Profession"].tolist()
stack_df = dff[dff["Profession"].isin(top8)]
stack_pivot = stack_df.pivot_table(index="Year", columns="Profession", values="Count", aggfunc="sum").fillna(0)

fig_stack = go.Figure()
for i, col in enumerate(stack_pivot.columns):
    fig_stack.add_trace(go.Scatter(
        x=stack_pivot.index,
        y=stack_pivot[col],
        mode="lines",
        stackgroup="one",
        name=col,
        line=dict(width=0.5, color=COLOR_SEQ[i % len(COLOR_SEQ)]),
        fillcolor=COLOR_SEQ[i % len(COLOR_SEQ)].replace("rgb", "rgba").replace(")", ", 0.7)") if "rgb" in COLOR_SEQ[i % len(COLOR_SEQ)] else COLOR_SEQ[i % len(COLOR_SEQ)],
        hovertemplate=f"<b>{col}</b><br>Year: %{{x}}<br>Count: %{{y:,}}<extra></extra>",
    ))
fig_stack.update_layout(**layout(
    title="Cumulative Emigration Composition — Top 8 Professions",
    height=380,
    xaxis=dict(**AXIS_STYLE, dtick=5),
    yaxis=dict(**AXIS_STYLE, tickformat=","),
    legend=dict(**LEGEND_STYLE, orientation="h", x=0, y=-0.3),
    margin=dict(l=60, r=30, t=50, b=120),
))
st.plotly_chart(fig_stack, use_container_width=True)

# ─── Chart 7: Year-over-Year Growth ───────────────────────────────────────────
st.markdown('<div class="section-title">📉 Year-over-Year Growth Rate</div>', unsafe_allow_html=True)

annual_sorted = annual.sort_values("Year")
annual_sorted["YoY_%"] = annual_sorted["Count"].pct_change() * 100

fig_yoy = go.Figure()
colors_yoy = ["#ff6b6b" if v < 0 else "#64b4ff" for v in annual_sorted["YoY_%"].fillna(0)]
fig_yoy.add_trace(go.Bar(
    x=annual_sorted["Year"],
    y=annual_sorted["YoY_%"],
    marker_color=colors_yoy,
    hovertemplate="<b>%{x}</b><br>Growth: %{y:.1f}%<extra></extra>",
    name="YoY Growth %",
))
fig_yoy.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
fig_yoy.update_layout(**layout(
    title="Year-over-Year Emigration Growth Rate (%)",
    height=320,
    xaxis=dict(**AXIS_STYLE, dtick=5),
    yaxis=dict(**AXIS_STYLE, ticksuffix="%"),
    showlegend=False,
    margin=dict(l=60, r=30, t=50, b=60),
))
st.plotly_chart(fig_yoy, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — COUNTRY-WISE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider">🌐 Country-wise Analysis</div>', unsafe_allow_html=True)

# Pre-compute country totals for the filtered period
country_totals = (
    dffc.groupby("Country")["Count"].sum()
    .sort_values(ascending=False)
    .reset_index()
)
top_countries_df = country_totals.head(top_n_countries)

# ─── Chart C1 & C2: Country Bar + Pie side by side ───────────────────────────
col_cbar, col_cpie = st.columns(2)

with col_cbar:
    st.markdown(f'<div class="section-title">🏆 Top {top_n_countries} Destination Countries</div>', unsafe_allow_html=True)
    fig_cbar = go.Figure(go.Bar(
        y=top_countries_df["Country"],
        x=top_countries_df["Count"],
        orientation="h",
        marker=dict(
            color=top_countries_df["Count"],
            colorscale="Purples",
            showscale=False,
        ),
        text=top_countries_df["Count"].apply(lambda x: f"{x/1000:.0f}K" if x < 1_000_000 else f"{x/1_000_000:.2f}M"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Total: %{x:,}<extra></extra>",
    ))
    fig_cbar.update_layout(**layout(
        height=420,
        xaxis=dict(**AXIS_STYLE, tickformat=",", title=""),
        yaxis=dict(**AXIS_STYLE, autorange="reversed", title=""),
        margin=dict(l=140, r=80, t=20, b=40),
    ))
    st.plotly_chart(fig_cbar, use_container_width=True)

with col_cpie:
    st.markdown(f'<div class="section-title">🥧 Country Share — Top {top_n_countries}</div>', unsafe_allow_html=True)
    crest = country_totals.iloc[top_n_countries:]["Count"].sum()
    cpie_df = top_countries_df.copy()
    if crest > 0:
        cpie_df = pd.concat(
            [cpie_df, pd.DataFrame([{"Country": "Others (Rest)", "Count": crest}])],
            ignore_index=True,
        )

    fig_cpie = go.Figure(go.Pie(
        labels=cpie_df["Country"],
        values=cpie_df["Count"],
        hole=0.45,
        textinfo="percent",
        textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        marker=dict(
            colors=(COUNTRY_COLORS * 3),
            line=dict(color="rgba(15,12,41,0.8)", width=2),
        ),
    ))
    fig_cpie.update_layout(**layout(
        height=420,
        showlegend=True,
        legend=dict(**LEGEND_STYLE, orientation="v", x=1.02, y=0.5, font=dict(size=10)),
        margin=dict(l=10, r=150, t=20, b=20),
        annotations=[dict(
            text=f"Top {top_n_countries}", x=0.5, y=0.5, font_size=13, showarrow=False,
            font=dict(color="#c87aff", family="Syne"),
        )],
    ))
    st.plotly_chart(fig_cpie, use_container_width=True)

# ─── Chart C3: Multi-line Country Trend (MAIN NEW CHART) ─────────────────────
st.markdown('<div class="section-title">📊 Country-wise Emigration Trends Over Time</div>', unsafe_allow_html=True)

if not selected_countries:
    st.info("Select at least one country in the sidebar to view trends.")
else:
    ctry_trend_df = dffc[dffc["Country"].isin(selected_countries)]
    ctry_pivot = ctry_trend_df.pivot_table(
        index="Year", columns="Country", values="Count", aggfunc="sum"
    ).fillna(0)

    fig_ctry_trend = go.Figure()
    for i, col in enumerate(ctry_pivot.columns):
        fig_ctry_trend.add_trace(go.Scatter(
            x=ctry_pivot.index,
            y=ctry_pivot[col],
            mode="lines+markers",
            name=col,
            line=dict(width=2.5, color=COUNTRY_COLORS[i % len(COUNTRY_COLORS)]),
            marker=dict(size=5),
            hovertemplate=f"<b>{col}</b><br>Year: %{{x}}<br>Emigrants: %{{y:,}}<extra></extra>",
        ))
    fig_ctry_trend.update_layout(**layout(
        title="Year-wise Emigration Trend by Destination Country",
        height=440,
        xaxis=dict(**AXIS_STYLE, dtick=5, title="Year"),
        yaxis=dict(**AXIS_STYLE, tickformat=",", title="Emigrants"),
        legend=dict(**LEGEND_STYLE, orientation="h", x=0, y=-0.22, traceorder="normal"),
        margin=dict(l=70, r=30, t=55, b=120),
    ))
    st.plotly_chart(fig_ctry_trend, use_container_width=True)

# ─── Chart C4: Country Stacked Area ──────────────────────────────────────────
st.markdown('<div class="section-title">📦 Country Composition Over Time (Top 8)</div>', unsafe_allow_html=True)

top8_countries = country_totals.head(8)["Country"].tolist()
cstack_df = dffc[dffc["Country"].isin(top8_countries)]
cstack_pivot = cstack_df.pivot_table(
    index="Year", columns="Country", values="Count", aggfunc="sum"
).fillna(0)

fig_cstack = go.Figure()
for i, col in enumerate(cstack_pivot.columns):
    fig_cstack.add_trace(go.Scatter(
        x=cstack_pivot.index,
        y=cstack_pivot[col],
        mode="lines",
        stackgroup="one",
        name=col,
        line=dict(width=0.5, color=COUNTRY_COLORS[i % len(COUNTRY_COLORS)]),
        hovertemplate=f"<b>{col}</b><br>Year: %{{x}}<br>Count: %{{y:,}}<extra></extra>",
    ))
fig_cstack.update_layout(**layout(
    title="Cumulative Emigration Composition — Top 8 Destination Countries",
    height=400,
    xaxis=dict(**AXIS_STYLE, dtick=5),
    yaxis=dict(**AXIS_STYLE, tickformat=","),
    legend=dict(**LEGEND_STYLE, orientation="h", x=0, y=-0.28),
    margin=dict(l=70, r=30, t=55, b=130),
))
st.plotly_chart(fig_cstack, use_container_width=True)

# ─── Chart C5: Country Heatmap by Decade ─────────────────────────────────────
st.markdown('<div class="section-title">🔥 Country Heatmap — Decade View</div>', unsafe_allow_html=True)

top12_countries = country_totals.head(12)["Country"].tolist()
cheat_df = dffc[dffc["Country"].isin(top12_countries)].copy()
cheat_df["Decade"] = (cheat_df["Year"] // 10 * 10).astype(str) + "s"
cheat_pivot = cheat_df.pivot_table(
    index="Country", columns="Decade", values="Count", aggfunc="sum"
).fillna(0)
cheat_pivot = cheat_pivot.reindex(top12_countries)

fig_cheat = go.Figure(go.Heatmap(
    z=cheat_pivot.values,
    x=cheat_pivot.columns.tolist(),
    y=cheat_pivot.index.tolist(),
    colorscale="RdPu",
    hoverongaps=False,
    hovertemplate="<b>%{y}</b><br>%{x}<br>Emigrants: %{z:,}<extra></extra>",
    text=[[f"{v/1000:.0f}K" if v < 1_000_000 else f"{v/1_000_000:.1f}M" for v in row] for row in cheat_pivot.values],
    texttemplate="%{text}",
    textfont=dict(size=10, color="white"),
))
fig_cheat.update_layout(**layout(
    title="Emigrant Count by Destination Country & Decade",
    height=460,
    yaxis=dict(**AXIS_STYLE, autorange="reversed"),
    xaxis=dict(**AXIS_STYLE),
    margin=dict(l=150, r=30, t=55, b=50),
))
st.plotly_chart(fig_cheat, use_container_width=True)

# ─── Chart C6: Region-wise Pie ────────────────────────────────────────────────
st.markdown('<div class="section-title">🗺️ Emigration by Destination Region</div>', unsafe_allow_html=True)

region_totals = dffc.groupby("Region")["Count"].sum().reset_index().sort_values("Count", ascending=False)

fig_region = go.Figure(go.Pie(
    labels=region_totals["Region"],
    values=region_totals["Count"],
    hole=0.4,
    textinfo="label+percent",
    textfont=dict(size=11),
    hovertemplate="<b>%{label}</b><br>Total: %{value:,}<br>Share: %{percent}<extra></extra>",
    marker=dict(
        colors=COUNTRY_COLORS * 2,
        line=dict(color="rgba(15,12,41,0.8)", width=2),
    ),
))
fig_region.update_layout(**layout(
    title="Share of Emigrants by Destination Region",
    height=380,
    showlegend=True,
    legend=dict(**LEGEND_STYLE, orientation="h", x=0.1, y=-0.15),
    margin=dict(l=30, r=30, t=55, b=80),
))
st.plotly_chart(fig_region, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════════════════
col_t1, col_t2 = st.columns(2)

with col_t1:
    with st.expander("📋 Full Data Table — Profession Totals"):
        table_df = (
            dff.groupby("Profession")["Count"]
            .sum().reset_index()
            .sort_values("Count", ascending=False)
            .rename(columns={"Count": "Total Emigrants"})
        )
        table_df["Share %"] = (table_df["Total Emigrants"] / table_df["Total Emigrants"].sum() * 100).round(2)
        st.dataframe(
            table_df.style.format({"Total Emigrants": "{:,}", "Share %": "{:.2f}%"}),
            use_container_width=True,
            height=400,
        )

with col_t2:
    with st.expander("📋 Full Data Table — Country Totals"):
        ctable_df = (
            dffc.groupby("Country")["Count"]
            .sum().reset_index()
            .sort_values("Count", ascending=False)
            .rename(columns={"Count": "Total Emigrants"})
        )
        ctable_df["Share %"] = (ctable_df["Total Emigrants"] / ctable_df["Total Emigrants"].sum() * 100).round(2)
        st.dataframe(
            ctable_df.style.format({"Total Emigrants": "{:,}", "Share %": "{:.2f}%"}),
            use_container_width=True,
            height=400,
        )

st.markdown(
    "<br><center><small style='color:#445566'>Built with Streamlit & Plotly · Data: Bureau of Emigration & Overseas Employment, Pakistan</small></center>",
    unsafe_allow_html=True,
)

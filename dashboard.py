import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# ===================================================================
# 🛠️ CONFIGURATION — EDIT THESE TO CUSTOMIZE
# ===================================================================
CURRENCY_SYMBOL = "$"          # 👈 Change to "€", "₹", "£", etc.
REFRESH_INTERVAL_SEC = 30     # Auto-refresh every N seconds

# ===================================================================
# 🔐 Load Google Sheet URL from Streamlit Secrets
# ===================================================================
try:
    GOOGLE_SHEET_CSV_URL = st.secrets["google_sheet"]["csv_url"]
except KeyError:
    st.error("🔐 Missing Google Sheet URL in Streamlit Secrets.")
    st.stop()

st.set_page_config(
    page_title="MATQCODE IBKR Dashboard",
    page_icon="💼",
    layout="wide"
)

# ===================================================================
# 🧮 Helper Functions
# ===================================================================
def mask_account(acc: str) -> str:
    if not isinstance(acc, str) or len(acc) < 5:
        return "N/A"
    return acc[:2] + "*****" + acc[-2:]

def format_currency(val):
    if pd.isna(val) or val == 0:
        return f"{CURRENCY_SYMBOL}0.00"
    return f"{CURRENCY_SYMBOL}{val:,.2f}"

def format_percent(val):
    if pd.isna(val):
        return "—"
    return f"{val:+.2f}%"

# ===================================================================
# 📥 Load & Clean Data (Remove ALL empty/invalid rows)
# ===================================================================
@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def load_data(url):
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"❌ Failed to load data: {str(e)[:150]}...")
        return pd.DataFrame()

df_raw = load_data(GOOGLE_SHEET_CSV_URL)

if df_raw.empty:
    st.warning("📭 No data received.")
    st.stop()

# Required columns (your exact structure)
required_cols = {
    'Strategy Name', 'Account', 'Symbol', 'SecType',
    'Currency', 'Position', 'AvgCost', 'MarketPrice'
}
if not required_cols.issubset(df_raw.columns):
    st.error(f"⚠️ Missing columns: {required_cols - set(df_raw.columns)}")
    st.stop()

# Select relevant columns
df = df_raw[list(required_cols)].copy()

# Convert to numeric — errors='coerce' turns bad values to NaN
for col in ['Position', 'AvgCost', 'MarketPrice']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 🔥 CRITICAL: Remove ALL non-meaningful rows
df = df.dropna(subset=['Symbol', 'Position', 'AvgCost', 'MarketPrice'])  # Must have these
df = df[df['Position'] != 0]                                             # Exclude zero positions
df = df[df['Symbol'].str.strip() != '']                                 # Exclude blank symbols
df = df[df['Strategy Name'].str.strip() != '']                          # Exclude blank strategies

# Final reset
df = df.reset_index(drop=True)

if df.empty:
    st.info("📭 No valid positions found.")
    st.stop()

# ===================================================================
# 📊 Compute P&L (Long/Short Aware)
# ===================================================================
def calculate_pnl(row):
    qty = row['Position']
    avg = row['AvgCost']
    mp = row['MarketPrice']
    if qty > 0:
        return (mp - avg) * qty          # Long
    else:
        return (avg - mp) * abs(qty)     # Short

df['UnrealizedPnL'] = df.apply(calculate_pnl, axis=1)
df['CostBasis'] = df['Position'].abs() * df['AvgCost']
df['UnrealizedPnL%'] = np.where(
    df['CostBasis'] != 0,
    (df['UnrealizedPnL'] / df['CostBasis']) * 100,
    0
)
df['Long/Short'] = df['Position'].apply(lambda x: 'Long' if x > 0 else 'Short')
df['MarketValue'] = df['Position'] * df['MarketPrice']

# Totals
total_pnl = df['UnrealizedPnL'].sum()
total_exposure = df['MarketValue'].abs().sum()
total_cost = df['CostBasis'].sum()
total_pnl_pct = (total_pnl / total_cost * 100) if total_cost != 0 else 0

# Mask account
df['Account'] = df['Account'].apply(mask_account)

# Sort by absolute P&L (largest moves first)
df = df.iloc[df['UnrealizedPnL'].abs().argsort()[::-1]].reset_index(drop=True)

# ===================================================================
# 🎨 UI — Professional Dashboard
# ===================================================================
st.markdown(
    '<div style="text-align:center; font-size:2.2rem; font-weight:700; margin-bottom:0.5rem;">'
    '🏦 Portfolio Dashboard</div>',
    unsafe_allow_html=True
)
# st.caption("Live P&L • Long/Short aware • Strategy-based view")

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    color = "green" if total_pnl >= 0 else "red"
    st.metric(
        "Total P&L",
        format_currency(total_pnl),
        delta=format_percent(total_pnl_pct),
        delta_color="normal"
    )
with col2:
    st.metric("Total Exposure", format_currency(total_exposure))
with col3:
    st.metric("Total Cost", format_currency(total_cost))
with col4:
    st.metric("Positions", len(df))

st.caption(f"Last Updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.divider()

# ===================================================================
# 📋 Position Table — Color-Coded, No Blanks
# ===================================================================
st.subheader("📋 Open Positions")

display_df = df[[
    'Strategy Name', 'Account', 'Symbol', 'SecType', 'Long/Short',
    'Position', 'AvgCost', 'MarketPrice', 'UnrealizedPnL', 'UnrealizedPnL%'
]].copy()

# Apply formatting
display_df['AvgCost'] = display_df['AvgCost'].apply(format_currency)
display_df['MarketPrice'] = display_df['MarketPrice'].apply(format_currency)
display_df['UnrealizedPnL'] = display_df['UnrealizedPnL'].apply(format_currency)
display_df['UnrealizedPnL%'] = display_df['UnrealizedPnL%'].apply(format_percent)

# 💡 Use Pandas Styler to color P&L cells directly in table
def color_pnl(val):
    if isinstance(val, str):
        if "−" in val or (val.startswith(f"{CURRENCY_SYMBOL}-") or (val.count("-") and not val.startswith(f"{CURRENCY_SYMBOL}"))):
            return "color: red; font-weight: bold;"
        elif val.startswith(f"{CURRENCY_SYMBOL}") or "+" in val:
            return "color: green; font-weight: bold;"
    return ""

def color_percent(val):
    if isinstance(val, str) and val.endswith("%"):
        if val.startswith("-") or "−" in val:
            return "color: red; font-weight: bold;"
        elif val.startswith("+"):
            return "color: green; font-weight: bold;"
    return ""

styled_df = display_df.style \
    .applymap(color_pnl, subset=['UnrealizedPnL']) \
    .applymap(color_percent, subset=['UnrealizedPnL%'])

st.dataframe(styled_df, use_container_width=True, height=520)

# ===================================================================
# 📈 Charts
# ===================================================================
c1, c2 = st.columns(2)

# P&L by Strategy
with c1:
    st.subheader("🎯 P&L by Strategy")
    pnl_strat = df.groupby('Strategy Name')['UnrealizedPnL'].sum().reset_index()
    fig1 = px.bar(
        pnl_strat,
        x='UnrealizedPnL',
        y='Strategy Name',
        orientation='h',
        color='UnrealizedPnL',
        color_continuous_scale=['red', 'lightgray', 'green'],
        color_continuous_midpoint=0
    )
    fig1.update_layout(height=400, showlegend=False, xaxis_title=f"P&L ({CURRENCY_SYMBOL})")
    st.plotly_chart(fig1, use_container_width=True)

# Allocation
with c2:
    st.subheader("🌍 Exposure Allocation")
    alloc_df = df.copy()
    alloc_df['AbsExposure'] = alloc_df['MarketValue'].abs()
    fig2 = px.pie(
        alloc_df,
        values='AbsExposure',
        names='Symbol',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    st.plotly_chart(fig2, use_container_width=True)

# ===================================================================
# ℹ️ Footer
# ===================================================================
st.divider()

# Auto-refresh
st.markdown(
    f"""
    <script>
    setTimeout(() => window.location.reload(), {REFRESH_INTERVAL_SEC * 1000});
    </script>
    """,
    unsafe_allow_html=True
)

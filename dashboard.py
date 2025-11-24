import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# ===================================================================
# 🏦 IBKR Portfolio Dashboard — Strategy-Based View
# ===================================================================

# 🔧 CONFIG: Auto-refresh interval (in seconds)
REFRESH_INTERVAL_SEC = 60  # Change this to 30, 120, etc.

# 🔐 Load Google Sheet URL from secrets
try:
    GOOGLE_SHEET_CSV_URL = st.secrets["google_sheet"]["csv_url"]
except KeyError:
    st.error("🔐 Missing Google Sheet URL in Streamlit Secrets.")
    st.stop()

st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="💼",
    layout="wide"
)

# ===================================================================
# 🛠️ Helpers
# ===================================================================
def mask_account(acc: str) -> str:
    if not isinstance(acc, str) or len(acc) < 5:
        return "N/A"
    return acc[:2] + "*****" + acc[-2:]

def format_currency(val, currency="USD"):
    if pd.isna(val) or val == 0:
        return f"{currency} 0.00"
    return f"{currency} {val:,.2f}"

def format_percent(val):
    if pd.isna(val):
        return "—"
    return f"{val:+.2f}%"

# ===================================================================
# 📥 Load & Clean Data
# ===================================================================
@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def load_data(url):
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"❌ Load failed: {str(e)[:150]}...")
        return pd.DataFrame()

df_raw = load_data(GOOGLE_SHEET_CSV_URL)

if df_raw.empty:
    st.warning("📭 No data from Google Sheet.")
    st.stop()

# Required columns (your new structure)
required = {'Strategy Name', 'Account', 'Symbol', 'SecType', 'Currency', 'Position', 'AvgCost', 'MarketPrice'}
if not required.issubset(df_raw.columns):
    st.error(f"⚠️ Missing columns: {required - set(df_raw.columns)}")
    st.stop()

# Select and clean
df = df_raw[list(required)].copy()

# Convert numeric
for col in ['Position', 'AvgCost', 'MarketPrice']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 🔥 Keep only valid, non-zero positions
df = df.dropna(subset=['Symbol', 'Position', 'AvgCost', 'MarketPrice'])
df = df[df['Position'] != 0]
df = df[df['Symbol'].str.strip() != '']
df = df.reset_index(drop=True)

if df.empty:
    st.info("📭 No active positions.")
    st.stop()

# ===================================================================
# 📊 Compute P&L (Handles Long & Short Correctly)
# ===================================================================
def calculate_pnl(row):
    qty = row['Position']
    avg = row['AvgCost']
    price = row['MarketPrice']
    
    if qty > 0:
        # Long: (Current - Avg) * Qty
        return (price - avg) * qty
    else:
        # Short: (Avg - Current) * |Qty|
        return (avg - price) * abs(qty)

df['UnrealizedPnL'] = df.apply(calculate_pnl, axis=1)
df['UnrealizedPnL%'] = np.where(
    (df['Position'] * df['AvgCost']) != 0,
    (df['UnrealizedPnL'] / (df['Position'] * df['AvgCost'].abs())) * 100,
    0
)
df['Long/Short'] = df['Position'].apply(lambda x: 'Long' if x > 0 else 'Short')
df['MarketValue'] = df['Position'] * df['MarketPrice']

# Totals
total_pnl = df['UnrealizedPnL'].sum()
total_mv = df['MarketValue'].abs().sum()  # Absolute for exposure
total_cost = (df['Position'].abs() * df['AvgCost']).sum()
total_pnl_pct = (total_pnl / total_cost * 100) if total_cost != 0 else 0
portfolio_currency = df['Currency'].iloc[0] if df['Currency'].nunique() == 1 else "MULTI"

# Mask account
df['Account'] = df['Account'].apply(mask_account)

# Sort by |P&L| descending
df = df.iloc[df['UnrealizedPnL'].abs().argsort()[::-1]].reset_index(drop=True)

# ===================================================================
# 🎨 UI
# ===================================================================
st.markdown('<div style="text-align:center; font-size:2.2rem; font-weight:700; margin-bottom:1rem;">🏦 Portfolio Dashboard</div>', unsafe_allow_html=True)
st.caption("Live P&L by strategy • Long/Short aware • Auto-refresh enabled")

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total P&L", format_currency(total_pnl, portfolio_currency), delta=format_percent(total_pnl_pct))
with col2:
    st.metric("Total Exposure", format_currency(total_mv, portfolio_currency))
with col3:
    st.metric("Total Cost", format_currency(total_cost, portfolio_currency))
with col4:
    st.metric("Positions", len(df))

st.divider()

# Position Table
st.subheader("📋 Positions (Sorted by |P&L|)")

display_df = df[[
    'Strategy Name', 'Account', 'Symbol', 'SecType', 'Long/Short',
    'Position', 'AvgCost', 'MarketPrice', 'UnrealizedPnL', 'UnrealizedPnL%'
]].copy()

# Formatting
display_df['AvgCost'] = display_df['AvgCost'].apply(lambda x: format_currency(x, portfolio_currency))
display_df['MarketPrice'] = display_df['MarketPrice'].apply(lambda x: format_currency(x, portfolio_currency))
display_df['UnrealizedPnL'] = display_df['UnrealizedPnL'].apply(lambda x: format_currency(x, portfolio_currency))
display_df['UnrealizedPnL%'] = display_df['UnrealizedPnL%'].apply(format_percent)

# Style P&L
def style_pnl(val):
    if isinstance(val, str) and ("+" in val or "−" in val or "-" in val):
        color = "green" if "+" in val else "red"
        return f"color: {color}; font-weight: bold;"
    return ""

styled_df = display_df.style.applymap(
    lambda x: style_pnl(x),
    subset=['UnrealizedPnL', 'UnrealizedPnL%']
)

st.dataframe(styled_df, use_container_width=True, height=500)

# ===================================================================
# 📈 Charts
# ===================================================================
c1, c2 = st.columns(2)

# P&L by Strategy
with c1:
    st.subheader("🎯 P&L by Strategy")
    pnl_by_strat = df.groupby('Strategy Name')['UnrealizedPnL'].sum().reset_index()
    fig1 = px.bar(
        pnl_by_strat,
        x='UnrealizedPnL',
        y='Strategy Name',
        orientation='h',
        color='UnrealizedPnL',
        color_continuous_scale=['red', 'lightgray', 'green'],
        color_continuous_midpoint=0
    )
    fig1.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

# Allocation by Symbol
with c2:
    st.subheader("🌍 Exposure Allocation")
    alloc = df.copy()
    alloc['AbsExposure'] = alloc['MarketValue'].abs()
    fig2 = px.pie(
        alloc,
        values='AbsExposure',
        names='Symbol',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig2, use_container_width=True)

# ===================================================================
# ℹ️ Footer
# ===================================================================
st.divider()
st.caption(f"✅ Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Refresh every {REFRESH_INTERVAL_SEC} sec")

# Auto-refresh
st.markdown(
    f"""
    <script>
    setTimeout(() => window.location.reload(), {REFRESH_INTERVAL_SEC * 1000});
    </script>
    """,
    unsafe_allow_html=True
)

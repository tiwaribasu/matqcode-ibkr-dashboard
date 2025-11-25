import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# ===================================================================
# 🛠️ CONFIGURATION
# ===================================================================
CURRENCY_SYMBOL = "$"
REFRESH_INTERVAL_SEC = 5  # Data auto-refreshes every 30 sec (no page reload needed)

# ===================================================================
# 🔐 Load Google Sheet URL from Streamlit Secrets
# ===================================================================
try:
    GOOGLE_SHEET_CSV_URL = st.secrets["google_sheet"]["csv_url"]
except KeyError:
    st.error("🔐 Missing Google Sheet URL in Streamlit Secrets.")
    st.stop()

st.set_page_config(
    page_title="BITQCODE IBKR Dashboard",
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
# 📥 Load & Clean Data — STRICTEST FILTERING
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

required_cols = {
    'Strategy Name', 'Account', 'Symbol', 'SecType',
    'Currency', 'Position', 'AvgCost', 'MarketPrice'
}
if not required_cols.issubset(df_raw.columns):
    st.error(f"⚠️ Missing columns: {required_cols - set(df_raw.columns)}")
    st.stop()

# Select only needed columns
df = df_raw[list(required_cols)].copy()

# Convert to numeric
for col in ['Position', 'AvgCost', 'MarketPrice']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 🔥 ULTRA-AGGRESSIVE CLEANING: Remove ALL invalid/empty rows
# Remove rows with NaN in any critical column
df = df.dropna(subset=['Strategy Name', 'Symbol', 'SecType', 'Position', 'AvgCost', 'MarketPrice'])

# Remove rows with empty strings or whitespace only in text columns
text_columns = ['Strategy Name', 'Symbol', 'SecType', 'Account']
for col in text_columns:
    if col in df.columns:
        df = df[df[col].astype(str).str.strip() != '']
        df = df[df[col].astype(str).str.strip() != 'nan']  # Remove 'nan' strings
        df = df[df[col].notna()]  # Double check for NaN

# Remove zero positions and invalid numeric values
df = df[df['Position'] != 0]
df = df[df['AvgCost'] > 0]  # AvgCost should be positive
df = df[df['MarketPrice'] > 0]  # MarketPrice should be positive

# Reset index after all filtering
df = df.reset_index(drop=True)

if df.empty:
    st.info("📭 No valid positions found.")
    st.stop()

# ===================================================================
# 📊 Compute P&L
# ===================================================================
def calculate_pnl(row):
    qty = row['Position']
    avg = row['AvgCost']
    mp = row['MarketPrice']
    return (mp - avg) * qty if qty > 0 else (avg - mp) * abs(qty)

df['UnrealizedPnL'] = df.apply(calculate_pnl, axis=1)
df['CostBasis'] = df['Position'].abs() * df['AvgCost']
df['UnrealizedPnL%'] = np.where(
    df['CostBasis'] != 0,
    (df['UnrealizedPnL'] / df['CostBasis']) * 100,
    0
)
df['Long/Short'] = df['Position'].apply(lambda x: 'Long' if x > 0 else 'Short')
df['MarketValue'] = df['Position'] * df['MarketPrice']

# Calculate volume metrics
total_long_volume = df[df['Long/Short'] == 'Long']['MarketValue'].abs().sum()
total_short_volume = df[df['Long/Short'] == 'Short']['MarketValue'].abs().sum()
total_volume = total_long_volume + total_short_volume

# Totals for P&L
total_pnl = df['UnrealizedPnL'].sum()
total_pnl_pct = (total_pnl / df['CostBasis'].sum() * 100) if df['CostBasis'].sum() != 0 else 0

df['Account'] = df['Account'].apply(mask_account)
df = df.iloc[df['UnrealizedPnL'].abs().argsort()[::-1]].reset_index(drop=True)

# ===================================================================
# 🎯 BIG BOLD TOTAL P&L
# ===================================================================
pnl_color = "green" if total_pnl >= 0 else "red"
pnl_symbol = "▲" if total_pnl >= 0 else "▼"
st.markdown(
    f"""
    <div style="text-align: center; margin-bottom: 1.2rem;">
        <span style="font-size: 2.4rem; font-weight: 800; color: {pnl_color};">
            {format_currency(total_pnl)}
        </span>
        <br>
        <span style="font-size: 1.1rem; color: #666;">
            {pnl_symbol} {format_percent(total_pnl_pct)}
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# ===================================================================
# 📊 NEW METRICS: Volume Breakdown
# ===================================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Long Volume", 
        format_currency(total_long_volume),
        delta=None,
        delta_color="normal"
    )

with col2:
    st.metric(
        "Total Short Volume", 
        format_currency(total_short_volume),
        delta=None,
        delta_color="normal"
    )

with col3:
    st.metric(
        "Total Volume", 
        format_currency(total_volume),
        delta=None,
        delta_color="normal"
    )

with col4:
    st.metric("Positions", len(df))

# ===================================================================
# 🎨 CUSTOM CSS FOR METRIC COLORS
# ===================================================================
st.markdown("""
<style>
    /* Blue for Long Volume */
    [data-testid="stMetricValue"]:first-child {
        color: #1f77b4 !important;
    }
    /* Red for Short Volume */
    [data-testid="stMetricValue"]:nth-child(2) {
        color: #ff4b4b !important;
    }
    /* Black for Total Volume */
    [data-testid="stMetricValue"]:nth-child(3) {
        color: #000000 !important;
    }
    /* Keep Positions as default color */
</style>
""", unsafe_allow_html=True)

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.divider()

# ===================================================================
# 📋 Open Positions — ONLY VALID ROWS
# ===================================================================
st.subheader("📋 Open Positions")

# Final display columns
display_df = df[[
    'Strategy Name', 'Account', 'Symbol', 'SecType', 'Long/Short',
    'Position', 'AvgCost', 'MarketPrice', 'UnrealizedPnL', 'UnrealizedPnL%'
]].copy()

# Formatting
display_df['AvgCost'] = display_df['AvgCost'].apply(format_currency)
display_df['MarketPrice'] = display_df['MarketPrice'].apply(format_currency)
display_df['UnrealizedPnL'] = display_df['UnrealizedPnL'].apply(format_currency)
display_df['UnrealizedPnL%'] = display_df['UnrealizedPnL%'].apply(format_percent)

# Color styling
def color_pnl(val):
    if isinstance(val, str):
        if "−" in val or f"{CURRENCY_SYMBOL}-" in val or (val.startswith("-") and CURRENCY_SYMBOL not in val):
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

# ✅ Use .map() instead of .applymap()
styled_df = display_df.style \
    .map(color_pnl, subset=['UnrealizedPnL']) \
    .map(color_percent, subset=['UnrealizedPnL%'])

st.dataframe(styled_df, use_container_width=True, height=500)

# ===================================================================
# 📈 Charts - 4 PROFESSIONAL PLOTS (2 per row)
# ===================================================================
# Common chart configuration
chart_height = 400
color_scale = ['#FF4B4B', '#E0E0E0', '#00D4AA']  # Red-Gray-Green professional colors

# First row: Strategy and Long/Short
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("🎯 P&L by Strategy")
    pnl_strat = df.groupby('Strategy Name')['UnrealizedPnL'].sum().reset_index()
    pnl_strat = pnl_strat.sort_values('UnrealizedPnL', ascending=True)  # Sort for better visualization
    
    fig1 = px.bar(
        pnl_strat,
        x='UnrealizedPnL',
        y='Strategy Name',
        orientation='h',
        color='UnrealizedPnL',
        color_continuous_scale=color_scale,
        color_continuous_midpoint=0
    )
    fig1.update_layout(
        height=chart_height, 
        showlegend=False, 
        xaxis_title=f"P&L ({CURRENCY_SYMBOL})",
        yaxis_title="",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig1, use_container_width=True)

with row1_col2:
    st.subheader("📊 P&L by Long/Short")
    pnl_long_short = df.groupby('Long/Short')['UnrealizedPnL'].sum().reset_index()
    
    fig2 = px.bar(
        pnl_long_short,
        x='Long/Short',
        y='UnrealizedPnL',
        color='UnrealizedPnL',
        color_continuous_scale=color_scale,
        color_continuous_midpoint=0
    )
    fig2.update_layout(
        height=chart_height, 
        showlegend=False, 
        xaxis_title="Position Type", 
        yaxis_title=f"P&L ({CURRENCY_SYMBOL})",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig2, use_container_width=True)

# Second row: SecType and Exposure Allocation
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("🔧 P&L by Security Type")
    pnl_sectype = df.groupby('SecType')['UnrealizedPnL'].sum().reset_index()
    pnl_sectype = pnl_sectype.sort_values('UnrealizedPnL', ascending=True)
    
    fig3 = px.bar(
        pnl_sectype,
        x='UnrealizedPnL',
        y='SecType',
        orientation='h',
        color='UnrealizedPnL',
        color_continuous_scale=color_scale,
        color_continuous_midpoint=0
    )
    fig3.update_layout(
        height=chart_height, 
        showlegend=False, 
        xaxis_title=f"P&L ({CURRENCY_SYMBOL})",
        yaxis_title="",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig3, use_container_width=True)

with row2_col2:
    st.subheader("🌍 Exposure Allocation")
    alloc_df = df.copy()
    alloc_df['AbsExposure'] = alloc_df['MarketValue'].abs()
    
    # Use professional color palette for pie chart
    fig4 = px.pie(
        alloc_df,
        values='AbsExposure',
        names='Symbol',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig4.update_layout(
        height=chart_height,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.1)
    )
    st.plotly_chart(fig4, use_container_width=True)

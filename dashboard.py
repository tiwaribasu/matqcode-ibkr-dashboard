import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# ==============================
# IBKR Portfolio Dashboard
# Reads from a public Google Sheet (CSV)
# Account IDs are masked for privacy
# Google Sheet URL stored in Streamlit Secrets
# ==============================

# 🔐 Load Google Sheet URL from secrets
try:
    GOOGLE_SHEET_CSV_URL = st.secrets["google_sheet"]["csv_url"]
except KeyError:
    st.error(
        "🚨 Missing Google Sheet URL in Streamlit Secrets.\n\n"
        "To fix this:\n"
        "- Local: Create `.streamlit/secrets.toml`\n"
        "- Cloud: Add secret in Streamlit Cloud dashboard"
    )
    st.stop()

st.set_page_config(
    page_title="MATQCODE-IBKR Dashboard",
    page_icon="📈",
    layout="wide"
)

# ------------------------------
# Helper Functions
# ------------------------------

def mask_account(acc: str) -> str:
    """Mask account ID as DU*****67"""
    if not isinstance(acc, str) or len(acc) < 5:
        return str(acc) if pd.notna(acc) else "N/A"
    return acc[:2] + "*****" + acc[-2:]

def format_money(val, currency="USD"):
    if pd.isna(val) or val == 0:
        return f"{currency} 0.00"
    return f"{currency} {val:,.2f}"

@st.cache_data(ttl=60)
def load_portfolio_data(url):
    """Load and validate portfolio data from public Google Sheet CSV"""
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"❌ Failed to load data from Google Sheet:\n{str(e)[:180]}...")
        return pd.DataFrame()

# ------------------------------
# Main App
# ------------------------------

st.title("🏦 IBKR Portfolio Dashboard")
st.caption("Live positions from Trader Workstation • Account IDs masked for privacy")

# Load data
df_raw = load_portfolio_data(GOOGLE_SHEET_CSV_URL)

if df_raw.empty:
    st.warning("No data available. Ensure your Google Sheet is published as CSV.")
    st.stop()

# Validate required columns
required_cols = {'Symbol', 'Position', 'AvgCost', 'MarketPrice', 'MarketValue', 'Currency'}
if not required_cols.issubset(df_raw.columns):
    missing = required_cols - set(df_raw.columns)
    st.error(f"Missing required columns in Google Sheet: {missing}")
    st.stop()

# Clean and prepare data
df = df_raw.copy()
numeric_cols = ['Position', 'AvgCost', 'MarketPrice', 'MarketValue']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['Symbol', 'Position'])
df = df[df['Position'] != 0].copy()

if df.empty:
    st.info("No active positions found.")
    st.stop()

# Mask account IDs
if 'Account' in df.columns:
    df['AccountMasked'] = df['Account'].apply(mask_account)
    account_col = 'AccountMasked'
else:
    df['AccountMasked'] = 'N/A'
    account_col = 'AccountMasked'

# Compute financials
df['CostBasis'] = df['Position'] * df['AvgCost']
df['UnrealizedPnL'] = df['MarketValue'] - df['CostBasis']
df['UnrealizedPnL%'] = np.where(
    df['CostBasis'] != 0,
    (df['UnrealizedPnL'] / df['CostBasis']) * 100,
    0
)

# Portfolio summary
total_mv = df['MarketValue'].sum()
total_cost = df['CostBasis'].sum()
total_pnl = total_mv - total_cost
total_pnl_pct = (total_pnl / total_cost * 100) if total_cost != 0 else 0
portfolio_currency = df['Currency'].iloc[0] if df['Currency'].nunique() == 1 else "MULTI"

# ------------------------------
# Dashboard UI
# ------------------------------

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Market Value", format_money(total_mv, portfolio_currency))
col2.metric("Total Cost Basis", format_money(total_cost, portfolio_currency))
col3.metric("Unrealized P&L", format_money(total_pnl, portfolio_currency), delta=f"{total_pnl_pct:+.2f}%")
col4.metric("Active Positions", len(df))

# Position table
st.subheader("📋 Position Summary")
display_df = df[[
    account_col, 'Symbol', 'SecType', 'Position',
    'AvgCost', 'MarketPrice', 'MarketValue', 'UnrealizedPnL', 'UnrealizedPnL%'
]].rename(columns={account_col: "Account"})

# Format numbers
display_df['AvgCost'] = display_df['AvgCost'].apply(lambda x: f"{portfolio_currency} {x:.4f}" if pd.notna(x) else "-")
display_df['MarketPrice'] = display_df['MarketPrice'].apply(lambda x: f"{portfolio_currency} {x:.4f}" if pd.notna(x) else "-")
display_df['MarketValue'] = display_df['MarketValue'].apply(lambda x: format_money(x, portfolio_currency))
display_df['UnrealizedPnL'] = display_df['UnrealizedPnL'].apply(lambda x: format_money(x, portfolio_currency))
display_df['UnrealizedPnL%'] = display_df['UnrealizedPnL%'].apply(lambda x: f"{x:+.2f}%")

st.dataframe(display_df, use_container_width=True, height=420)

# Charts
c1, c2 = st.columns(2)
with c1:
    st.subheader("🌍 Allocation by Symbol")
    alloc_df = df.groupby('Symbol')['MarketValue'].sum().reset_index()
    fig1 = px.pie(alloc_df, values='MarketValue', names='Symbol', hole=0.35)
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.subheader("💰 P&L by Position")
    pnl_df = df.sort_values('UnrealizedPnL', key=abs, ascending=False)
    colors = ['green' if x >= 0 else 'red' for x in pnl_df['UnrealizedPnL']]
    fig2 = go.Figure(go.Bar(
        x=pnl_df['UnrealizedPnL'],
        y=pnl_df['Symbol'],
        orientation='h',
        marker_color=colors
    ))
    fig2.update_layout(
        xaxis_title=f"Unrealized P&L ({portfolio_currency})",
        yaxis_title="Symbol",
        height=450
    )
    st.plotly_chart(fig2, use_container_width=True)

# Footer
st.divider()
st.caption(f"✅ Data synced from IBKR • Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Auto-refresh every 60 seconds (client-side)
st.markdown(
    """
    <script>
    setTimeout(() => window.location.reload(), 60000);
    </script>
    """,
    unsafe_allow_html=True
)

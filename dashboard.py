import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import pytz

# ===================================================================
# 🛠️ CONFIGURATION
# ===================================================================
REFRESH_INTERVAL_SEC = 10  # Data refreshes every 10 seconds

# ===================================================================
# 🔐 Load Google Sheet URL from Streamlit Secrets
# ===================================================================
try:
    GOOGLE_SHEET_CSV_URL = st.secrets["google_sheet"]["csv_url"]
except KeyError:
    st.error("🔐 Missing Google Sheet URL in Streamlit Secrets.")
    st.stop()

st.set_page_config(
    page_title="BITQCODE Dashboard",
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

def format_currency(val, currency_symbol="$"):
    if pd.isna(val) or val == 0:
        return f"{currency_symbol}0.00"
    return f"{currency_symbol}{val:,.2f}"

def format_percent(val):
    if pd.isna(val):
        return "—"
    return f"{val:+.2f}%"

def get_time_with_timezone(region):
    """Get current time with appropriate timezone"""
    if region == "INDIA":
        # IST timezone (UTC+5:30)
        tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(tz)
        return now.strftime('%Y-%m-%d %H:%M:%S IST')
    else:  # GLOBAL - Use US/Eastern or UTC
        # Using US Eastern Time (ET)
        tz = pytz.timezone('US/Eastern')
        now = datetime.now(tz)
        return now.strftime('%Y-%m-%d %H:%M:%S ET')

# ===================================================================
# 📥 Load & Clean Data — WITH AUTOMATIC REFRESH
# ===================================================================
@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def load_sheet_data(sheet_gid="0"):
    """Load specific sheet from Google Sheets using gid parameter"""
    try:
        # Construct URL with gid parameter for specific sheet
        if "export?format=csv" in GOOGLE_SHEET_CSV_URL:
            # Replace or add gid parameter
            if "gid=" in GOOGLE_SHEET_CSV_URL:
                url = GOOGLE_SHEET_CSV_URL.split("&gid=")[0] + f"&gid={sheet_gid}"
            else:
                url = GOOGLE_SHEET_CSV_URL + f"&gid={sheet_gid}"
        else:
            url = GOOGLE_SHEET_CSV_URL + f"?gid={sheet_gid}&format=csv"
        
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"❌ Failed to load sheet {sheet_gid}: {str(e)[:150]}...")
        return pd.DataFrame()

@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def process_data(df_raw, region_name, currency_symbol="$"):
    """Process data with caching"""
    if df_raw.empty:
        return pd.DataFrame()
    
    # Check for required columns
    required_cols = {
        'Strategy Name', 'Account', 'Symbol', 'SecType',
        'Currency', 'Position', 'AvgCost', 'MarketPrice'
    }
    
    # Check which required columns exist
    missing_cols = required_cols - set(df_raw.columns)
    if missing_cols:
        return pd.DataFrame()  # Silent fail - will be handled in dashboard
    
    # Select only needed columns
    df = df_raw[list(required_cols)].copy()
    
    # Convert to numeric
    for col in ['Position', 'AvgCost', 'MarketPrice']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 🔥 ULTRA-AGGRESSIVE CLEANING
    df = df.dropna(subset=['Strategy Name', 'Symbol', 'SecType', 'Position', 'AvgCost', 'MarketPrice'])
    
    # Remove rows with empty strings or whitespace only
    text_columns = ['Strategy Name', 'Symbol', 'SecType', 'Account']
    for col in text_columns:
        if col in df.columns:
            df = df[df[col].astype(str).str.strip() != '']
            df = df[df[col].astype(str).str.strip() != 'nan']
            df = df[df[col].notna()]
    
    # Remove zero positions and invalid numeric values
    df = df[df['Position'] != 0]
    df = df[df['AvgCost'] > 0]
    df = df[df['MarketPrice'] > 0]
    
    # Reset index after all filtering
    df = df.reset_index(drop=True)
    
    if df.empty:
        return df
    
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
    
    df['Account'] = df['Account'].apply(mask_account)
    df = df.iloc[df['UnrealizedPnL'].abs().argsort()[::-1]].reset_index(drop=True)
    
    return df

def create_dashboard_tab(df, region_name, currency_symbol="$"):
    """Create dashboard for a specific region"""
    
    if df.empty:
        st.info(f"📭 No valid positions found for {region_name}.")
        return
    
    # Calculate volume metrics
    total_long_volume = df[df['Long/Short'] == 'Long']['MarketValue'].abs().sum()
    total_short_volume = df[df['Long/Short'] == 'Short']['MarketValue'].abs().sum()
    total_volume = total_long_volume + total_short_volume
    
    # Totals for P&L
    total_pnl = df['UnrealizedPnL'].sum()
    total_pnl_pct = (total_pnl / df['CostBasis'].sum() * 100) if df['CostBasis'].sum() != 0 else 0
    
    # ===================================================================
    # 🎯 BIG BOLD TOTAL P&L
    # ===================================================================
    pnl_color = "green" if total_pnl >= 0 else "red"
    pnl_symbol = "▲" if total_pnl >= 0 else "▼"
    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 1.2rem;">
            <span style="font-size: 2.4rem; font-weight: 800; color: {pnl_color};">
                {format_currency(total_pnl, currency_symbol)}
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
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.95rem; font-weight: 600; color: #1f77b4; margin-bottom: 0.2rem;">Total Long Exposure</div>
                <div style="font-size: 1.75rem; font-weight: 700; color: #1f77b4;">{format_currency(total_long_volume, currency_symbol)}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.95rem; font-weight: 600; color: #ff4b4b; margin-bottom: 0.2rem;">Total Short Exposure</div>
                <div style="font-size: 1.75rem; font-weight: 700; color: #ff4b4b;">{format_currency(total_short_volume, currency_symbol)}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.95rem; font-weight: 600; color: #000000; margin-bottom: 0.2rem;">Total Exposure</div>
                <div style="font-size: 1.75rem; font-weight: 700; color: #000000;">{format_currency(total_volume, currency_symbol)}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.95rem; font-weight: 600; color: #000000; margin-bottom: 0.2rem;">Positions</div>
                <div style="font-size: 1.75rem; font-weight: 700; color: #000000;">{len(df)}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Show appropriate timezone based on region
    st.caption(f"Last updated: {get_time_with_timezone(region_name.split()[0])}")
    
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
    display_df['AvgCost'] = display_df['AvgCost'].apply(lambda x: format_currency(x, currency_symbol))
    display_df['MarketPrice'] = display_df['MarketPrice'].apply(lambda x: format_currency(x, currency_symbol))
    display_df['UnrealizedPnL'] = display_df['UnrealizedPnL'].apply(lambda x: format_currency(x, currency_symbol))
    display_df['UnrealizedPnL%'] = display_df['UnrealizedPnL%'].apply(format_percent)
    
    # Color styling
    def color_pnl(val):
        if isinstance(val, str):
            if "−" in val or f"{currency_symbol}-" in val or (val.startswith("-") and currency_symbol not in val):
                return "color: red; font-weight: bold;"
            elif val.startswith(f"{currency_symbol}") or "+" in val:
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
        pnl_strat = pnl_strat.sort_values('UnrealizedPnL', ascending=True)
        
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
            xaxis_title=f"P&L ({currency_symbol})",
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
            yaxis_title=f"P&L ({currency_symbol})",
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
            xaxis_title=f"P&L ({currency_symbol})",
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

# ===================================================================
# 🏠 MAIN APP - SIMPLE AND CLEAN
# ===================================================================

# Load data for both sheets using caching with TTL
df_global_raw = load_sheet_data(sheet_gid="5320120")  # GLOBAL sheet
df_india_raw = load_sheet_data(sheet_gid="649765105")  # INDIA sheet

# Process data
df_global = process_data(df_global_raw, "GLOBAL", "$")
df_india = process_data(df_india_raw, "INDIA", "₹")

# ===================================================================
# 🎨 CSS for Bigger Tabs
# ===================================================================
st.markdown("""
<style>
    /* Make tabs larger and more prominent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        padding-top: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-size: 18px;
        font-weight: 600;
        padding: 10px 24px;
        border-radius: 8px 8px 0 0;
        background-color: #f0f2f6;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-bottom: 3px solid #FF4B4B;
    }
    
    /* Remove default Streamlit tab styling */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid #e0e0e0;
    }
    
    /* Make the tab content area cleaner */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
    
    /* Remove extra spacing in main container */
    .main .block-container {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ===================================================================
# 📊 Create Tabs with Cleaner Layout
# ===================================================================
tab1, tab2 = st.tabs([
    "🌍 **GLOBAL DASHBOARD**", 
    "🇮🇳 **INDIA DASHBOARD**"
])

with tab1:
    create_dashboard_tab(df_global, "GLOBAL DASHBOARD", "$")

with tab2:
    create_dashboard_tab(df_india, "INDIA DASHBOARD", "₹")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date
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

def format_inr(val):
    """Format Indian Rupees"""
    if pd.isna(val) or val == 0:
        return "₹0.00"
    return f"₹{val:,.2f}"

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
def process_live_pnl_data(df_raw):
    """Process Live PnL data - filter for today's date only"""
    if df_raw.empty:
        return pd.DataFrame()
    
    # Clean column names
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    
    # Check for required columns
    required_cols = ['DateTime', 'Total PnL']
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()
    
    # Convert DateTime to datetime and Total PnL to numeric
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df['Total PnL'] = pd.to_numeric(df['Total PnL'], errors='coerce')
    
    # Drop rows with invalid dates or PnL
    df = df.dropna(subset=['DateTime', 'Total PnL'])
    
    if df.empty:
        return df
    
    # Get today's date in IST (Asia/Kolkata)
    ist_tz = pytz.timezone('Asia/Kolkata')
    today_ist = datetime.now(ist_tz).date()
    
    # Filter for today's data only (based on date part, ignoring time)
    df['Date'] = df['DateTime'].dt.date
    df_today = df[df['Date'] == today_ist].copy()
    
    # Sort by DateTime
    df_today = df_today.sort_values('DateTime')
    
    return df_today

@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def process_global_data(df_raw):
    """Process GLOBAL data with caching"""
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

@st.cache_data(ttl=REFRESH_INTERVAL_SEC)
def process_india_data(df_raw):
    """Process INDIA data with the new format"""
    if df_raw.empty:
        return {
            'open_positions': pd.DataFrame(),
            'closed_positions': pd.DataFrame(),
            'summary': {}
        }
    
    # Create a copy to avoid modifying the original
    df = df_raw.copy()
    
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
    
    # Define expected columns for INDIA format
    expected_cols = [
        's_no', 'tradingsymbol', 'buy_value', 'buy_price', 
        'buy_quantity', 'sell_quantity', 'sell_price', 
        'sell_value', 'last_price', 'pnl'
    ]
    
    # Check if we have the expected columns
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        st.warning(f"Missing expected columns in INDIA data: {missing_cols}")
        return {
            'open_positions': pd.DataFrame(),
            'closed_positions': pd.DataFrame(),
            'summary': {}
        }
    
    # Convert numeric columns
    numeric_cols = ['buy_value', 'buy_price', 'buy_quantity', 'sell_quantity', 
                   'sell_price', 'sell_value', 'last_price', 'pnl']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean data
    df = df.dropna(subset=['tradingsymbol'])
    df = df[df['tradingsymbol'].astype(str).str.strip() != '']
    
    # ===================================================================
    # SEPARATE OPEN AND CLOSED POSITIONS
    # ===================================================================
    
    # Closed positions: buy_quantity == sell_quantity (fully closed)
    closed_mask = (df['buy_quantity'] > 0) & (df['sell_quantity'] > 0) & (df['buy_quantity'] == df['sell_quantity'])
    closed_df = df[closed_mask].copy()
    
    # Open positions: buy_quantity != sell_quantity or sell_quantity == 0
    open_mask = ~closed_mask
    open_df = df[open_mask].copy()
    
    # Calculate additional metrics for open positions
    if not open_df.empty:
        # Calculate net quantity
        open_df['net_quantity'] = open_df['buy_quantity'] - open_df['sell_quantity']
        
        # Calculate average price for open positions
        open_df['avg_price'] = np.where(
            open_df['net_quantity'] != 0,
            (open_df['buy_value'] - open_df['sell_value']) / open_df['net_quantity'],
            0
        )
        
        # Calculate unrealized P&L for open positions
        open_df['unrealized_pnl'] = (open_df['last_price'] - open_df['avg_price']) * open_df['net_quantity']
        
        # Calculate open exposure
        open_df['open_exposure'] = open_df['net_quantity'] * open_df['last_price']
        
        # Determine position type
        open_df['position_type'] = open_df['net_quantity'].apply(lambda x: 'Long' if x > 0 else 'Short' if x < 0 else 'Flat')
        
        # Sort by unrealized P&L
        open_df = open_df.sort_values('unrealized_pnl', ascending=False)
    
    # Calculate summary metrics
    total_traded_volume = df['buy_value'].sum() + df['sell_value'].sum()
    total_closed_pnl = closed_df['pnl'].sum() if not closed_df.empty else 0
    total_unrealized_pnl = open_df['unrealized_pnl'].sum() if not open_df.empty else 0
    total_open_exposure = open_df['open_exposure'].abs().sum() if not open_df.empty else 0
    
    # Calculate number of positions
    open_positions_count = len(open_df)
    closed_positions_count = len(closed_df)
    
    return {
        'open_positions': open_df,
        'closed_positions': closed_df,
        'summary': {
            'total_traded_volume': total_traded_volume,
            'total_closed_pnl': total_closed_pnl,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_open_exposure': total_open_exposure,
            'open_positions_count': open_positions_count,
            'closed_positions_count': closed_positions_count,
            'total_pnl': total_closed_pnl + total_unrealized_pnl
        }
    }

def create_global_dashboard(df):
    """Create GLOBAL dashboard"""
    
    if df.empty:
        st.info("📭 No valid positions found for GLOBAL.")
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
                {format_currency(total_pnl, "$")}
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
                <div style="font-size: 1.75rem; font-weight: 700; color: #1f77b4;">{format_currency(total_long_volume, "$")}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.95rem; font-weight: 600; color: #ff4b4b; margin-bottom: 0.2rem;">Total Short Exposure</div>
                <div style="font-size: 1.75rem; font-weight: 700; color: #ff4b4b;">{format_currency(total_short_volume, "$")}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.95rem; font-weight: 600; color: #000000; margin-bottom: 0.2rem;">Total Exposure</div>
                <div style="font-size: 1.75rem; font-weight: 700; color: #000000;">{format_currency(total_volume, "$")}</div>
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
    
    # Show appropriate timezone
    st.caption(f"Last updated: {get_time_with_timezone('GLOBAL')}")
    
    st.divider()
    
    # ===================================================================
    # 📋 Open Positions — CLEAN TABLE WITHOUT INDEX
    # ===================================================================
    st.subheader("📋 Open Positions")
    
    # Final display columns
    display_df = df[[
        'Strategy Name', 'Account', 'Symbol', 'SecType', 'Long/Short',
        'Position', 'AvgCost', 'MarketPrice', 'UnrealizedPnL', 'UnrealizedPnL%'
    ]].copy()
    
    # Formatting
    display_df['AvgCost'] = display_df['AvgCost'].apply(lambda x: format_currency(x, "$"))
    display_df['MarketPrice'] = display_df['MarketPrice'].apply(lambda x: format_currency(x, "$"))
    display_df['UnrealizedPnL'] = display_df['UnrealizedPnL'].apply(lambda x: format_currency(x, "$"))
    display_df['UnrealizedPnL%'] = display_df['UnrealizedPnL%'].apply(format_percent)
    
    # Create HTML table for GLOBAL positions
    html_table = """
    <div style="overflow-x: auto;">
    <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
        <thead>
            <tr style="background-color: #f2f2f2;">
                <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Strategy Name</th>
                <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Account</th>
                <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Symbol</th>
                <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">SecType</th>
                <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Long/Short</th>
                <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Position</th>
                <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Avg Cost</th>
                <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Market Price</th>
                <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Unrealized P&L</th>
                <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Unrealized P&L%</th>
            </tr>
        </thead>
        <tbody>
    """
    
    # Add rows with color coding for P&L
    for _, row in display_df.iterrows():
        html_table += '<tr>'
        
        # Strategy Name
        html_table += f'<td style="padding: 8px; border-bottom: 1px solid #ddd;">{row["Strategy Name"]}</td>'
        
        # Account
        html_table += f'<td style="padding: 8px; border-bottom: 1px solid #ddd;">{row["Account"]}</td>'
        
        # Symbol
        html_table += f'<td style="padding: 8px; border-bottom: 1px solid #ddd;">{row["Symbol"]}</td>'
        
        # SecType
        html_table += f'<td style="padding: 8px; border-bottom: 1px solid #ddd;">{row["SecType"]}</td>'
        
        # Long/Short
        html_table += f'<td style="padding: 8px; border-bottom: 1px solid #ddd;">{row["Long/Short"]}</td>'
        
        # Position
        html_table += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["Position"]}</td>'
        
        # AvgCost
        html_table += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["AvgCost"]}</td>'
        
        # MarketPrice
        html_table += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["MarketPrice"]}</td>'
        
        # UnrealizedPnL - with color coding
        pnl_style = "padding: 8px; border-bottom: 1px solid #ddd; text-align: right; font-weight: bold;"
        pnl_value = row["UnrealizedPnL"]
        if "$-" in str(pnl_value) or "-" in str(pnl_value) or "−" in str(pnl_value):
            pnl_style += "color: red;"
        else:
            pnl_style += "color: green;"
        html_table += f'<td style="{pnl_style}">{pnl_value}</td>'
        
        # UnrealizedPnL% - with color coding
        pnl_pct_style = "padding: 8px; border-bottom: 1px solid #ddd; text-align: right; font-weight: bold;"
        pnl_pct_value = row["UnrealizedPnL%"]
        if "-" in str(pnl_pct_value) or "−" in str(pnl_pct_value):
            pnl_pct_style += "color: red;"
        else:
            pnl_pct_style += "color: green;"
        html_table += f'<td style="{pnl_pct_style}">{pnl_pct_value}</td>'
        
        html_table += '</tr>'
    
    html_table += """
        </tbody>
    </table>
    </div>
    """
    
    st.markdown(html_table, unsafe_allow_html=True)
    
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
            xaxis_title="P&L ($)",
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
            yaxis_title="P&L ($)",
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
            xaxis_title="P&L ($)",
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

def create_india_dashboard(data_dict, live_pnl_df):
    """Create INDIA dashboard with new format"""
    
    open_df = data_dict['open_positions']
    closed_df = data_dict['closed_positions']
    summary = data_dict['summary']
    
    if open_df.empty and closed_df.empty:
        st.info("📭 No positions found for INDIA.")
        return
    
    # ===================================================================
    # 🎯 BIG BOLD TOTAL P&L (Closed + Unrealized)
    # ===================================================================
    total_pnl = summary.get('total_pnl', 0)
    capital = 1000000  # Fixed capital amount
    pnl_percentage = (total_pnl / capital) * 100 if capital > 0 else 0
    
    # Determine arrow and color
    if total_pnl > 0:
        pnl_color = "green"
        pnl_symbol = "▲"
        change_text = f"{pnl_symbol} +{abs(pnl_percentage):.2f}%"
    elif total_pnl < 0:
        pnl_color = "red"
        pnl_symbol = "▼"
        change_text = f"{pnl_symbol} -{abs(pnl_percentage):.2f}%"
    else:
        pnl_color = "gray"
        change_text = "0.00%"
    
    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 1.2rem;">
            <span style="font-size: 2.4rem; font-weight: 800; color: {pnl_color};">
                {format_inr(total_pnl)}
            </span>
            <br>
            <span style="font-size: 1.1rem; color: {pnl_color}; font-weight: 600;">
                {change_text}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # ===================================================================
    # 📊 KEY METRICS
    # ===================================================================
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #1f77b4; margin-bottom: 0.2rem;">Closed P&L</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1f77b4;">{format_inr(summary.get('total_closed_pnl', 0))}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #ff7f0e; margin-bottom: 0.2rem;">Unrealized P&L</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #ff7f0e;">{format_inr(summary.get('total_unrealized_pnl', 0))}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #2ca02c; margin-bottom: 0.2rem;">Traded Volume</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2ca02c;">{format_inr(summary.get('total_traded_volume', 0))}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #d62728; margin-bottom: 0.2rem;">Open Exposure</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #d62728;">{format_inr(summary.get('total_open_exposure', 0))}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col5:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #9467bd; margin-bottom: 0.2rem;">Open Positions</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #9467bd;">{summary.get('open_positions_count', 0)}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col6:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #8c564b; margin-bottom: 0.2rem;">Closed Positions</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #8c564b;">{summary.get('closed_positions_count', 0)}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Show appropriate timezone
    st.caption(f"Last updated: {get_time_with_timezone('INDIA')}")
    
    # ===================================================================
    # 📈 LIVE P&L CHART (Today's P&L)
    # ===================================================================
    if not live_pnl_df.empty:
        st.divider()
        st.subheader("📈 Today's Live P&L Trend")
        
        # Get today's date for display
        ist_tz = pytz.timezone('Asia/Kolkata')
        today_date = datetime.now(ist_tz).strftime('%Y-%m-%d')
        
        # Calculate stats for display
        if len(live_pnl_df) > 0:
            latest_pnl = live_pnl_df['Total PnL'].iloc[-1]
            highest_pnl = live_pnl_df['Total PnL'].max()
            lowest_pnl = live_pnl_df['Total PnL'].min()
            start_pnl = live_pnl_df['Total PnL'].iloc[0] if len(live_pnl_df) > 0 else 0
            current_change = latest_pnl - start_pnl
            
            # Create metrics row
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                current_color = "green" if latest_pnl >= 0 else "red"
                st.metric(
                    label="Current P&L",
                    value=format_inr(latest_pnl),
                    delta=format_inr(current_change)
                )
            
            with metric_col2:
                st.metric(
                    label="Today's High",
                    value=format_inr(highest_pnl),
                    delta=None
                )
            
            with metric_col3:
                st.metric(
                    label="Today's Low",
                    value=format_inr(lowest_pnl),
                    delta=None
                )
            
            with metric_col4:
                data_points = len(live_pnl_df)
                st.metric(
                    label="Data Points",
                    value=f"{data_points}",
                    delta=None
                )
        
        # Create professional line chart for Live P&L
        fig = go.Figure()
        
        # Add the main line
        fig.add_trace(go.Scatter(
            x=live_pnl_df['DateTime'],
            y=live_pnl_df['Total PnL'],
            mode='lines+markers',
            name='Live P&L',
            line=dict(color='#00D4AA', width=3),
            marker=dict(size=6, color='#00D4AA'),
            hovertemplate='<b>Time:</b> %{x|%H:%M:%S}<br><b>P&L:</b> ₹%{y:,.2f}<extra></extra>'
        ))
        
        # Add zero line reference
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            line_width=1,
            opacity=0.5
        )
        
        # Add fill for positive/negative areas
        fig.add_trace(go.Scatter(
            x=live_pnl_df['DateTime'],
            y=live_pnl_df['Total PnL'].where(live_pnl_df['Total PnL'] >= 0),
            mode='none',
            fill='tozeroy',
            fillcolor='rgba(0, 212, 170, 0.2)',
            name='Positive',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=live_pnl_df['DateTime'],
            y=live_pnl_df['Total PnL'].where(live_pnl_df['Total PnL'] < 0),
            mode='none',
            fill='tozeroy',
            fillcolor='rgba(255, 75, 75, 0.2)',
            name='Negative',
            showlegend=False
        ))
        
        # Update layout for professional look
        fig.update_layout(
            height=400,
            title=f"Live P&L Trend ({today_date})",
            title_font=dict(size=20, color='#333'),
            xaxis_title="Time (IST)",
            yaxis_title="P&L (₹)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", size=12, color="#333"),
            hovermode='x unified',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickformat='%H:%M',
                title_font=dict(size=14)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickprefix='₹',
                title_font=dict(size=14)
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data summary
        if len(live_pnl_df) > 1:
            time_range = live_pnl_df['DateTime'].iloc[-1] - live_pnl_df['DateTime'].iloc[0]
            avg_interval = time_range / (len(live_pnl_df) - 1) if len(live_pnl_df) > 1 else pd.Timedelta(0)
            st.caption(f"📊 Data from {live_pnl_df['DateTime'].iloc[0].strftime('%H:%M:%S')} to {live_pnl_df['DateTime'].iloc[-1].strftime('%H:%M:%S')} | Average interval: {avg_interval.seconds // 60} min {avg_interval.seconds % 60} sec")
    
    # ===================================================================
    # 📋 OPEN POSITIONS
    # ===================================================================
    if not open_df.empty:
        st.divider()
        st.subheader("📈 Open Positions")
        
        # Prepare display dataframe for open positions
        open_display_df = open_df[[
            'tradingsymbol', 'position_type', 'net_quantity',
            'avg_price', 'last_price', 'unrealized_pnl', 'open_exposure'
        ]].copy()
        
        # Rename columns for better display
        open_display_df = open_display_df.rename(columns={
            'tradingsymbol': 'Symbol',
            'position_type': 'Position',
            'net_quantity': 'Quantity',
            'avg_price': 'Avg Price',
            'last_price': 'Last Price',
            'unrealized_pnl': 'Unrealized P&L',
            'open_exposure': 'Open Exposure'
        })
        
        # Format columns
        open_display_df['Avg Price'] = open_display_df['Avg Price'].apply(format_inr)
        open_display_df['Last Price'] = open_display_df['Last Price'].apply(format_inr)
        open_display_df['Unrealized P&L'] = open_display_df['Unrealized P&L'].apply(format_inr)
        open_display_df['Open Exposure'] = open_display_df['Open Exposure'].apply(format_inr)
        
        # Create HTML table for open positions
        html_table_open = """
        <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
            <thead>
                <tr style="background-color: #f2f2f2;">
                    <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Symbol</th>
                    <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Position</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Quantity</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Avg Price</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Last Price</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Unrealized P&L</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Open Exposure</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Add rows with color coding for P&L
        for _, row in open_display_df.iterrows():
            html_table_open += '<tr>'
            
            # Symbol
            html_table_open += f'<td style="padding: 8px; border-bottom: 1px solid #ddd;">{row["Symbol"]}</td>'
            
            # Position
            html_table_open += f'<td style="padding: 8px; border-bottom: 1px solid #ddd;">{row["Position"]}</td>'
            
            # Quantity
            html_table_open += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["Quantity"]}</td>'
            
            # Avg Price
            html_table_open += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["Avg Price"]}</td>'
            
            # Last Price
            html_table_open += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["Last Price"]}</td>'
            
            # Unrealized P&L - with color coding
            pnl_style = "padding: 8px; border-bottom: 1px solid #ddd; text-align: right; font-weight: bold;"
            pnl_value = row["Unrealized P&L"]
            if "₹-" in str(pnl_value) or "-" in str(pnl_value) or "−" in str(pnl_value):
                pnl_style += "color: red;"
            else:
                pnl_style += "color: green;"
            html_table_open += f'<td style="{pnl_style}">{pnl_value}</td>'
            
            # Open Exposure
            html_table_open += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["Open Exposure"]}</td>'
            
            html_table_open += '</tr>'
        
        html_table_open += """
            </tbody>
        </table>
        </div>
        """
        
        st.markdown(html_table_open, unsafe_allow_html=True)
    
    # ===================================================================
    # 📋 CLOSED POSITIONS
    # ===================================================================
    if not closed_df.empty:
        st.divider()
        st.subheader("📊 Closed Positions (Today)")
        
        # Prepare display dataframe for closed positions
        closed_display_df = closed_df[[
            'tradingsymbol', 'buy_quantity', 'buy_price',
            'sell_quantity', 'sell_price', 'pnl'
        ]].copy()
        
        # Rename columns for better display
        closed_display_df = closed_display_df.rename(columns={
            'tradingsymbol': 'Symbol',
            'buy_quantity': 'Buy Qty',
            'buy_price': 'Buy Price',
            'sell_quantity': 'Sell Qty',
            'sell_price': 'Sell Price',
            'pnl': 'Realized P&L'
        })
        
        # Format columns
        closed_display_df['Buy Price'] = closed_display_df['Buy Price'].apply(format_inr)
        closed_display_df['Sell Price'] = closed_display_df['Sell Price'].apply(format_inr)
        closed_display_df['Realized P&L'] = closed_display_df['Realized P&L'].apply(format_inr)
        
        # Create HTML table for closed positions
        html_table_closed = """
        <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
            <thead>
                <tr style="background-color: #f2f2f2;">
                    <th style="padding: 10px; text-align: left; border-bottom: 1px solid #ddd;">Symbol</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Buy Qty</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Buy Price</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Sell Qty</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Sell Price</th>
                    <th style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">Realized P&L</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Add rows with color coding for P&L
        for _, row in closed_display_df.iterrows():
            html_table_closed += '<tr>'
            
            # Symbol
            html_table_closed += f'<td style="padding: 8px; border-bottom: 1px solid #ddd;">{row["Symbol"]}</td>'
            
            # Buy Qty
            html_table_closed += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["Buy Qty"]}</td>'
            
            # Buy Price
            html_table_closed += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["Buy Price"]}</td>'
            
            # Sell Qty
            html_table_closed += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["Sell Qty"]}</td>'
            
            # Sell Price
            html_table_closed += f'<td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{row["Sell Price"]}</td>'
            
            # Realized P&L - with color coding
            pnl_style = "padding: 8px; border-bottom: 1px solid #ddd; text-align: right; font-weight: bold;"
            pnl_value = row["Realized P&L"]
            if "₹-" in str(pnl_value) or "-" in str(pnl_value) or "−" in str(pnl_value):
                pnl_style += "color: red;"
            else:
                pnl_style += "color: green;"
            html_table_closed += f'<td style="{pnl_style}">{pnl_value}</td>'
            
            html_table_closed += '</tr>'
        
        html_table_closed += """
            </tbody>
        </table>
        </div>
        """
        
        st.markdown(html_table_closed, unsafe_allow_html=True)
    
    # ===================================================================
    # 📈 OTHER CHARTS FOR INDIA
    # ===================================================================
    if not open_df.empty or not closed_df.empty:
        st.divider()
        st.subheader("📊 Performance Analysis")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # P&L Distribution Chart
            st.subheader("💰 P&L Distribution")
            
            if not open_df.empty and not closed_df.empty:
                # Combine open and closed P&L data
                pnl_data = pd.DataFrame({
                    'Category': ['Closed P&L', 'Unrealized P&L'],
                    'Amount': [summary['total_closed_pnl'], summary['total_unrealized_pnl']]
                })
                
                fig1 = px.bar(
                    pnl_data,
                    x='Category',
                    y='Amount',
                    color='Category',
                    color_discrete_map={'Closed P&L': '#1f77b4', 'Unrealized P&L': '#ff7f0e'},
                    text=[format_inr(x) for x in pnl_data['Amount']]
                )
                fig1.update_layout(
                    height=300,
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title="Amount (₹)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("No P&L data available for chart.")
        
        with chart_col2:
            # Positions Overview Chart
            st.subheader("📊 Positions Overview")
            
            positions_data = pd.DataFrame({
                'Category': ['Open Positions', 'Closed Positions'],
                'Count': [summary['open_positions_count'], summary['closed_positions_count']]
            })
            
            fig2 = px.pie(
                positions_data,
                values='Count',
                names='Category',
                hole=0.4,
                color_discrete_sequence=['#9467bd', '#8c564b']
            )
            fig2.update_layout(
                height=300,
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.1)
            )
            st.plotly_chart(fig2, use_container_width=True)

# ===================================================================
# 🏠 MAIN APP - SIMPLE AND CLEAN
# ===================================================================

# Load data for all sheets using caching with TTL
df_global_raw = load_sheet_data(sheet_gid="5320120")  # GLOBAL sheet
df_india_raw = load_sheet_data(sheet_gid="649765105")  # INDIA sheet
df_live_pnl_raw = load_sheet_data(sheet_gid="1065660372")  # LIVE PnL sheet

# Process data
df_global = process_global_data(df_global_raw)
india_data = process_india_data(df_india_raw)
live_pnl_data = process_live_pnl_data(df_live_pnl_raw)

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
    create_global_dashboard(df_global)

with tab2:
    create_india_dashboard(india_data, live_pnl_data)

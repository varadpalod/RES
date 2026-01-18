"""
Streamlit Dashboard
Interactive visualization for Pune Real Estate Sentiment Intelligence
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from src.geospatial import get_locality_coordinates_df
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR, DATA_DIR, MODEL_DIR, DASHBOARD_CONFIG, PUNE_LOCALITIES
from config import validate_config

# Page configuration
st.set_page_config(
    page_title=DASHBOARD_CONFIG['page_title'],
    page_icon=DASHBOARD_CONFIG['page_icon'],
    layout=DASHBOARD_CONFIG['layout'],
    initial_sidebar_state="expanded"
)

# Validate configuration on startup (run once)
if 'config_validated' not in st.session_state:
    st.session_state.config_validated = True
    is_valid, warnings, errors = validate_config()
    
    # Show any critical errors
    if errors:
        st.error("⚠️ **Configuration Errors Detected**")
        for error in errors:
            st.error(error)

# Custom CSS for dark theme and styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid #3d3d5c;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 8px;
    }
    .insight-card {
        background: #1e1e2f;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .alert-warning {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
    }
    .alert-success {
        background: rgba(40, 167, 69, 0.1);
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# Removed @st.cache_data to ensure fresh data loading
def load_data():
    """Load all data files"""
    data = {}
    
    # Load locality sentiment
    sentiment_file = OUTPUT_DIR / 'locality_sentiment.csv'
    if sentiment_file.exists():
        data['sentiment'] = pd.read_csv(sentiment_file)
    else:
        data['sentiment'] = pd.DataFrame()
    
    # Load rankings
    rankings_file = OUTPUT_DIR / 'locality_rankings.csv'
    if rankings_file.exists():
        data['rankings'] = pd.read_csv(rankings_file)
    else:
        data['rankings'] = pd.DataFrame()
    
    # Load full predictions
    predictions_file = OUTPUT_DIR / 'full_predictions.csv'
    if predictions_file.exists():
        data['predictions'] = pd.read_csv(predictions_file)
    else:
        data['predictions'] = pd.DataFrame()
    
    # Load alerts
    alerts_file = OUTPUT_DIR / 'sentiment_alerts.json'
    if alerts_file.exists():
        with open(alerts_file) as f:
            data['alerts'] = json.load(f)
    else:
        data['alerts'] = []
    
    # Load recommendations
    recs_file = OUTPUT_DIR / 'detailed_builder_recommendations.json'
    if not recs_file.exists():
        recs_file = OUTPUT_DIR / 'builder_recommendations.json'
        
    if recs_file.exists():
        with open(recs_file) as f:
            data['recommendations'] = json.load(f)
    else:
        data['recommendations'] = []


    
    # Load model report
    report_file = OUTPUT_DIR / 'model_report.json'
    if report_file.exists():
        with open(report_file) as f:
            data['model_report'] = json.load(f)
    else:
        data['model_report'] = {}
    
    # Load affordability analysis (NEW)
    affordability_file = OUTPUT_DIR / 'affordability_analysis.csv'
    if affordability_file.exists():
        data['affordability'] = pd.read_csv(affordability_file)
    else:
        data['affordability'] = pd.DataFrame()
    
    # Load locality highlights (NEW)
    highlights_file = OUTPUT_DIR / 'locality_highlights.csv'
    if highlights_file.exists():
        data['highlights'] = pd.read_csv(highlights_file)
    else:
        data['highlights'] = pd.DataFrame()
    
    return data


def render_header():
    """Render main header"""
    st.markdown('<p class="main-header">🏠 Pune Real Estate Sentiment Intelligence</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sentiment-aware demand prediction for informed real estate decisions</p>', 
                unsafe_allow_html=True)


def render_overview_metrics(data):
    """Render overview metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not data['predictions'].empty:
            st.metric("Total Properties", f"{len(data['predictions']):,}")
        else:
            st.metric("Total Properties", "N/A")
    
    with col2:
        if not data['sentiment'].empty:
            st.metric("Localities Analyzed", len(data['sentiment']))
        else:
            st.metric("Localities Analyzed", "N/A")
    
    with col3:
        if not data['rankings'].empty:
            top_loc = data['rankings'].iloc[0]['area']
            st.metric("Top Investment Area", top_loc)
        else:
            st.metric("Top Investment Area", "N/A")
    
    with col4:
        st.metric("Sentiment Alerts", len(data['alerts']))


def render_locality_sentiment_chart(data):
    """Render locality sentiment radar chart"""
    if data['sentiment'].empty:
        st.warning("No sentiment data available. Run the pipeline first.")
        return
    
    df = data['sentiment'].head(10)
    
    fig = go.Figure()
    
    categories = ['Price Perception', 'Infrastructure', 'Investment', 'Urgency', 'Overall']
    
    for _, row in df.iterrows():
        values = [
            row.get('price_perception', 0),
            row.get('infrastructure_satisfaction', 0),
            row.get('investment_confidence', 0),
            row.get('buying_urgency', 0) * 2 - 1,  # Scale to -1 to 1
            row.get('overall_sentiment', 0)
        ]
        values.append(values[0])  # Close the radar
        
        fig.add_trace(go.Scatterpolar(
            r=[v * 50 + 50 for v in values],  # Scale to 0-100
            theta=categories + [categories[0]],
            fill='toself',
            name=row['locality'],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True,
        title="Locality Sentiment Profiles",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment Score Legend
    with st.expander("📖 Understanding Sentiment Scores", expanded=False):
        st.markdown("""
        **Sentiment Score** measures public opinion about a locality based on news, reviews, and social media.
        
        | Score Range | Meaning | Interpretation |
        |-------------|---------|----------------|
        | **+0.5 to +1.0** | 🟢 Very Positive | Strong buyer confidence, area in high demand |
        | **+0.1 to +0.5** | 🟡 Mildly Positive | Generally good outlook, minor concerns |
        | **-0.1 to +0.1** | ⚪ Neutral | Mixed opinions, no clear trend |
        | **-0.5 to -0.1** | 🟠 Negative | Concerns about pricing, traffic, or infrastructure |
        | **-1.0 to -0.5** | 🔴 Very Negative | Serious issues, low buyer interest |
        
        **Components:**
        - **Price Perception**: Is the area considered overpriced or good value?
        - **Infrastructure**: Roads, metro, connectivity satisfaction
        - **Investment Confidence**: Expected price appreciation
        - **Buying Urgency**: How urgently are people looking to buy?
        """)


def render_investment_rankings(data):
    """Render investment rankings table with new metrics"""
    if data['rankings'].empty:
        st.warning("No ranking data available. Run the pipeline first.")
        return
    
    df = data['rankings'].copy()
    
    # Format columns for display
    if 'avg_rental_yield' in df.columns:
        df['avg_rental_yield'] = df['avg_rental_yield'].round(2).astype(str) + '%'
    if 'min_annual_income' in df.columns:
        df['min_annual_income'] = (df['min_annual_income'] / 100000).round(1).astype(str) + 'L'
    if 'rera_compliance_pct' in df.columns:
        df['rera_compliance_pct'] = df['rera_compliance_pct'].round(0).astype(str) + '%'
    if 'price' in df.columns:
        df['price'] = '₹' + (df['price'] / 100000).round(1).astype(str) + 'L'
    
    # Color code recommendations
    def color_recommendation(val):
        colors = {
            'Strong Buy': 'background-color: rgba(40, 167, 69, 0.3)',
            'Buy': 'background-color: rgba(23, 162, 184, 0.3)',
            'Hold': 'background-color: rgba(255, 193, 7, 0.3)',
            'Avoid': 'background-color: rgba(220, 53, 69, 0.3)'
        }
        return colors.get(val, '')
    
    # Updated columns to display including new metrics
    display_cols = ['rank', 'area', 'investment_score', 'recommendation', 
                    'avg_rental_yield', 'min_annual_income', 'rera_compliance_pct',
                    'predicted_demand', 'price', 'overall_sentiment']
    available_cols = [c for c in display_cols if c in df.columns]
    
    # Rename columns for display
    rename_map = {
        'avg_rental_yield': 'Rental Yield',
        'min_annual_income': 'Min Income',
        'rera_compliance_pct': 'RERA %',
        'predicted_demand': 'Demand',
        'overall_sentiment': 'Sentiment',
        'investment_score': 'Score'
    }
    
    display_df = df[available_cols].rename(columns=rename_map)
    
    st.dataframe(
        display_df.style.applymap(
            color_recommendation, subset=['recommendation']
        ) if 'recommendation' in display_df.columns else display_df,
        use_container_width=True,
        height=400
    )
    
    # Score Legends
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📖 Understanding Demand Score", expanded=False):
            st.markdown("""
            **Demand Score** (0-100) predicts how likely a property will attract buyers.
            
            | Score | Category | Meaning |
            |-------|----------|---------|
            | **75-100** | 🟢 Very High | Hot property, likely to sell fast |
            | **60-74** | 🟡 High | Good demand, attracts buyers |
            | **40-59** | ⚪ Moderate | Average interest |
            | **0-39** | 🔴 Low | May struggle to find buyers |
            
            **Calculated from:**
            - 35% Price (affordable = higher demand)
            - 35% Sentiment (positive area = higher demand)
            - 30% Quality (new, garage, spacious = higher)
            """)
    
    with col2:
        with st.expander("📖 Understanding Investment Score", expanded=False):
            st.markdown("""
            **Investment Score** (0-100) rates a locality's investment potential.
            
            | Score | Recommendation | Action |
            |-------|----------------|--------|
            | **70+** | 🟢 Strong Buy | Ideal for new projects |
            | **55-69** | 🟡 Buy | Good potential |
            | **40-54** | ⚪ Hold | Wait for improvement |
            | **0-39** | 🔴 Avoid | Not recommended |
            
            **Based on:**
            - Predicted demand (25%)
            - Investment confidence (25%)
            - Overall sentiment (15%)
            - Rental yield bonus (20%)
            - Low volatility bonus (15%)
            """)


def render_alerts(data):
    """Render sentiment alerts"""
    if not data['alerts']:
        st.info("No alerts at this time.")
        return
    
    # Group by type
    warnings = [a for a in data['alerts'] if a['type'] == 'warning']
    opportunities = [a for a in data['alerts'] if a['type'] == 'opportunity']
    cautions = [a for a in data['alerts'] if a['type'] == 'caution']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ Warnings")
        for alert in warnings[:5]:
            st.markdown(f"""
            <div class="alert-warning">
                <strong>{alert['locality']}</strong> - {alert['category']}<br>
                {alert['message']}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🎯 Opportunities")
        for alert in opportunities[:5]:
            st.markdown(f"""
            <div class="alert-success">
                <strong>{alert['locality']}</strong> - {alert['category']}<br>
                {alert['message']}
            </div>
            """, unsafe_allow_html=True)


def render_demand_predictor(data):
    """Interactive demand prediction interface"""
    st.subheader("🔮 Demand Predictor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        locality = st.selectbox("Select Locality", PUNE_LOCALITIES)
        square_feet = st.slider("Square Feet", 500, 3000, 1200)
        bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5], index=1)
    
    with col2:
        bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
        property_age = st.slider("Property Age (years)", 0, 30, 5)
        has_garage = st.checkbox("Has Garage", value=True)
    
    if st.button("Predict Demand", type="primary"):
        # Get sentiment for locality
        if not data['sentiment'].empty:
            loc_sentiment = data['sentiment'][data['sentiment']['locality'] == locality]
            if not loc_sentiment.empty:
                sentiment = loc_sentiment.iloc[0]
                
                # Simple prediction based on features
                base_score = 50
                base_score += sentiment.get('investment_confidence', 0) * 15
                base_score += sentiment.get('overall_sentiment', 0) * 10
                base_score -= property_age * 0.5
                base_score += has_garage * 5
                
                demand_score = max(0, min(100, base_score))  # Deterministic - no random variance
                
                # Calculate additional metrics
                infra_satisfaction = sentiment.get('infrastructure_satisfaction', 0)
                
                # Liquidity Score
                liquidity = demand_score * 0.4 + 15 + (has_garage * 10) - (property_age * 0.5)
                liquidity = max(0, min(100, liquidity))
                
                # Infrastructure Risk
                infra_risk = 50 - (infra_satisfaction * 30) + (property_age * 0.3)
                infra_risk = max(0, min(100, infra_risk))
                
                # Sale Probability
                sale_prob = demand_score * 0.6 + liquidity * 0.3 - infra_risk * 0.1
                sale_prob = max(5, min(95, sale_prob))
                
                # Time to Sell
                time_to_sell = int(180 - (sale_prob * 1.5) + (-30 if demand_score >= 75 else (-15 if demand_score >= 60 else 15)))
                time_to_sell = max(15, min(365, time_to_sell))
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Demand Score", f"{demand_score:.1f}/100")
                
                with col2:
                    category = ("Very High" if demand_score >= 75 else 
                               "High" if demand_score >= 60 else
                               "Moderate" if demand_score >= 40 else "Low")
                    st.metric("Demand Category", category)
                
                with col3:
                    st.metric("Locality Sentiment", 
                             f"{sentiment.get('overall_sentiment', 0):.2f}")
                
                st.markdown("---")
                st.subheader("📈 Sales Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🔄 Liquidity Score", f"{liquidity:.0f}/100",
                             help="How easily the property can be sold (higher = easier)")
                
                with col2:
                    risk_color = "🔴" if infra_risk > 60 else ("🟡" if infra_risk > 40 else "🟢")
                    st.metric(f"{risk_color} Infra Risk", f"{infra_risk:.0f}/100",
                             help="Risk from traffic/water/connectivity issues")
                
                with col3:
                    st.metric("📊 Sale Probability", f"{sale_prob:.0f}%",
                             help="Chance of selling within 90 days")
                
                with col4:
                    st.metric("⏱️ Time to Sell", f"{time_to_sell} days",
                             help="Estimated days to close the deal")
                
                # Summary insight
                if sale_prob >= 60:
                    st.success(f"✅ This property in **{locality}** has **high sale potential**. "
                              f"Expected to sell within **{time_to_sell} days** with {sale_prob:.0f}% probability.")
                elif sale_prob >= 40:
                    st.info(f"ℹ️ This property in **{locality}** has **moderate sale potential**. "
                           f"May take around **{time_to_sell} days** to sell.")
                else:
                    st.warning(f"⚠️ This property in **{locality}** may be **difficult to sell**. "
                              f"Could take **{time_to_sell}+ days** with only {sale_prob:.0f}% probability.")
            else:
                st.warning(f"No sentiment data available for {locality}")
        else:
            st.warning("Run the pipeline first to enable predictions")


def render_model_performance(data):
    """Render model performance metrics"""
    if not data['model_report']:
        st.warning("No model report available. Train the model first.")
        return
    
    report = data['model_report']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", report.get('model_type', 'N/A'))
    
    if 'performance' in report:
        perf = report['performance']
        
        with col2:
            st.metric("Test R² Score", f"{perf.get('test', {}).get('r2', 0):.3f}")
        
        with col3:
            st.metric("CV R² Score", f"{perf.get('cv_r2_mean', 0):.3f}")
    
    # Feature importance
    if 'top_features' in report:
        st.subheader("Feature Importance")
        
        features_df = pd.DataFrame(report['top_features'])
        
        fig = px.bar(
            features_df.head(10),
            x='importance_pct',
            y='feature',
            orientation='h',
            title='Top 10 Features by Importance',
            labels={'importance_pct': 'Importance (%)', 'feature': 'Feature'}
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        st.plotly_chart(fig, use_container_width=True)


def render_builder_recommendations(data):
    """Render detailed builder recommendations"""
    if not data['recommendations']:
        st.info("No recommendations available. Run the pipeline first.")
        return
    
    for rec in data['recommendations']:
        # Color-code the expander based on recommendation
        emoji = "🟢" if rec['recommendation'] == 'Strong Buy' else (
                "🟡" if rec['recommendation'] == 'Buy' else (
                "⚪" if rec['recommendation'] == 'Hold' else "🔴"))
        
        with st.expander(f"{emoji} {rec['locality']} - {rec['recommendation']}", expanded=False):
            # Summary
            st.markdown(f"**{rec.get('summary', 'No summary available')}**")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Investment Score", f"{rec['investment_score']:.1f}")
            with col2:
                demand = rec.get('demand_score', 'N/A')
                st.metric("Demand Score", f"{demand:.1f}" if isinstance(demand, (int, float)) else demand)
            with col3:
                if rec.get('avg_price'):
                    st.metric("Avg. Price", f"₹{float(rec['avg_price'])/100000:.1f}L")
            with col4:
                if rec.get('rental_yield'):
                    yield_val = rec['rental_yield']
                    st.metric("Rental Yield", f"{yield_val:.1f}%" if isinstance(yield_val, (int, float)) else yield_val)
            
            st.markdown("---")
            
            # Strengths and Weaknesses
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**✅ Strengths:**")
                for s in rec.get('strengths', ['N/A']):
                    st.markdown(f"- {s}")
            
            with col2:
                st.markdown("**⚠️ Weaknesses:**")
                for w in rec.get('weaknesses', ['N/A']):
                    st.markdown(f"- {w}")
            
            st.markdown("---")
            
            # Action Items
            st.markdown("**📋 Action Items:**")
            for action in rec.get('action_items', ['No actions available']):
                st.markdown(f"  {action}")
            
            st.markdown("---")
            
            # Target Segment and Pricing in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Target Segment:**")
                target = rec.get('target_segment', 'N/A')
                if isinstance(target, dict):
                    # AI generated dict - handle flexible structure
                    def render_dict(d, level=0):
                        for k, v in d.items():
                            prefix = "  " * level
                            key_formatted = k.replace('_', ' ').title()
                            if isinstance(v, dict):
                                st.markdown(f"{prefix}**{key_formatted}:**")
                                render_dict(v, level + 1)
                            elif isinstance(v, list):
                                st.markdown(f"{prefix}**{key_formatted}:**")
                                for item in v:
                                    st.markdown(f"{prefix}- {item}")
                            else:
                                st.markdown(f"{prefix}**{key_formatted}:** {v}")

                    render_dict(target)
                else:
                    # Old string format
                    st.markdown(target.replace('**Target Segment**:', '').strip() if target != 'N/A' else target)
            
            with col2:
                st.markdown("**💰 Pricing Strategy:**")
                pricing = rec.get('pricing_strategy', 'N/A')
                st.markdown(pricing.replace('**Strategy**:', '').strip() if pricing != 'N/A' else pricing)




def render_geospatial_view(data):
    """Render 3D geospatial heatmap"""
    if data['rankings'].empty:
        st.warning("No ranking data available for map.")
        return
        
    # Get coordinates
    df = get_locality_coordinates_df(data['rankings'])
    
    if df.empty:
        st.warning("Could not map coordinates for localities.")
        return
        
    # Normalize data for visualization
    df['norm_demand'] = df['predicted_demand'] / 100
    df['norm_price'] = df['price'] / df['price'].max()
    
    # 3D Column Layer (Demand Intensity)
    # Using ColumnLayer instead of HexagonLayer for precise locality visualization
    layer = pdk.Layer(
        'ColumnLayer',
        df,
        get_position='[lon, lat]',
        auto_highlight=True,
        elevation_scale=50, # Scale factor
        radius=400,         # Radius of each column in meters
        pickable=True,
        extruded=True,
        get_elevation='predicted_demand',
        get_fill_color='[255, (1 - norm_demand) * 255, 100, 200]'
    )
    
    # Text Layer (Locality Labels)
    text_layer = pdk.Layer(
        "TextLayer",
        df,
        pickable=False,
        get_position='[lon, lat]',
        get_text='area',
        get_size=14,
        get_color=[0, 0, 0, 255],
        get_angle=0,
        get_text_anchor='"middle"',
        get_alignment_baseline='"top"',
        get_pixel_offset=[0, -20] # slightly above the column
    )

    # Tooltip
    tooltip = {
        "html": "<b>{area}</b><br/>Demand Score: {predicted_demand}<br/>Price: ₹{price}",
        "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"}
    }
    
    # View State (Centered on Pune)
    view_state = pdk.ViewState(
        latitude=18.5204,
        longitude=73.8567,
        zoom=10.5,
        pitch=50,
        bearing=0
    )
    
    # Map Style Selector
    map_style_name = st.radio(
        "Map Style",
        ["Streets", "Satellite", "Dark", "Light"],
        horizontal=True,
        index=0
    )
    
    # Map Styles (Carto - does not require Mapbox Token)
    styles = {
        "Streets": pdk.map_styles.CARTO_ROAD,
        "Dark": pdk.map_styles.CARTO_DARK,
        "Light": pdk.map_styles.CARTO_LIGHT,
        "Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    }
    
    selected_style = styles.get(map_style_name)
    
    # Handle Satellite via TileLayer if selected, otherwise use map_style
    if map_style_name == "Satellite":
        # For satellite, we use a TileLayer as the base
        tile_layer = pdk.Layer(
            "TileLayer",
            selected_style,
            get_point_color=[0, 0, 0, 128],
            # Zoom limits for public tile server
            min_zoom=0,
            max_zoom=19,
        )
        layers_list = [tile_layer, layer, text_layer]
        deck_style = None # No base map style, the TileLayer acts as base
    else:
        layers_list = [layer, text_layer]
        deck_style = selected_style

    r = pdk.Deck(
        layers=layers_list,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=deck_style
    )
    
    st.pydeck_chart(r)
    
    # Legend/Info
    st.info("👆 **3D Columns**: Height represents **Predicted Demand**. Taller/Redder = Higher Demand.")


def main():
    """Main dashboard function"""
    # Load data
    data = load_data()
    
    # Render header
    render_header()
    
    # Check if data exists
    if data['predictions'].empty and data['sentiment'].empty:
        st.warning("⚠️ No data found. Please run the pipeline first:")
        st.code("python main.py", language="bash")
        st.info("This will generate synthetic sentiment data, train the model, and create all outputs.")
        return
    
    # Overview metrics
    render_overview_metrics(data)
    
    st.markdown("---")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Market Overview",
        "🗺️ Map View",
        "🏆 Investment Rankings", 
        "💰 Affordability",
        "🔮 Demand Predictor",
        "⚠️ Alerts & Insights",
        "📈 Model Performance"
    ])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            render_locality_sentiment_chart(data)
        
        with col2:
            st.subheader("🏗️ Builder Recommendations")
            render_builder_recommendations(data)
            
    with tab2:
        st.subheader("🗺️ Pune Real Estate Heatmap")
        render_geospatial_view(data)
    
    with tab3: # Investment Rankings
        st.subheader("🏆 Locality Investment Rankings")
        render_investment_rankings(data)
        
        # Visualization
        if not data['rankings'].empty:
            fig = px.bar(
                data['rankings'].head(15),
                x='area',
                y='investment_score',
                color='recommendation',
                title='Investment Score by Locality',
                color_discrete_map={
                    'Strong Buy': '#28a745',
                    'Buy': '#17a2b8',
                    'Hold': '#ffc107',
                    'Avoid': '#dc3545'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # AI Analysis Section
            st.markdown("---")
            st.subheader("Investment Analysis")
            
            # Select locality for analysis
            loc_options = data['rankings']['area'].tolist()
            selected_loc = st.selectbox("Select Locality for Deep Dive Analysis", loc_options)
            
            if st.button("Generate Investment Thesis", type="secondary"):
                with st.spinner(f"Deep Diving Into {selected_loc}..."):
                    try:
                        # Import here to avoid issues if module missing
                        from src.llm_insights import LLMInsights
                        llm = LLMInsights()
                        
                        if llm.available:
                            # Get data for selected locality
                            loc_data = data['rankings'][data['rankings']['area'] == selected_loc].iloc[0].to_dict()
                            
                            # Add highlighted features
                            if 'highlights' in data and not data['highlights'].empty:
                                highlights = data['highlights'][data['highlights']['locality'] == selected_loc]
                                if not highlights.empty:
                                    loc_data.update(highlights.iloc[0].to_dict())
                            
                            analysis = llm.analyze_locality(loc_data)
                            
                            # Clean up potential code block formatting from LLM
                            analysis = analysis.replace("```markdown", "").replace("```", "")
                            
                            st.markdown("### Deep Analysis Report")
                            st.markdown(analysis)
                            # st.success("Analysis generated using OpenAI (GPT-4o)")
                        else:
                            st.error("⚠️ OPENAI_API_KEY not found in .env file.")
                    except Exception as e:
                        st.error(f"Failed to generate analysis: {e}")
    
    with tab4:
        st.subheader("💰 Affordability Analysis")
        st.markdown("**Income requirements and rental yields by locality**")
        
        if not data['affordability'].empty:
            aff_df = data['affordability'].copy()
            
            # Format for display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_income = aff_df['min_annual_income'].mean()
                st.metric("Avg Min Income Needed", f"₹{avg_income/100000:.1f}L/year")
            
            with col2:
                avg_yield = aff_df['avg_rental_yield'].mean()
                st.metric("Avg Rental Yield", f"{avg_yield:.1f}%")
            
            with col3:
                avg_rera = aff_df['rera_compliance_pct'].mean()
                st.metric("Avg RERA Compliance", f"{avg_rera:.0f}%")
            
            st.markdown("---")
            
            # Format dataframe for display
            display_aff = aff_df.copy()
            display_aff['min_annual_income'] = '₹' + (display_aff['min_annual_income']/100000).round(1).astype(str) + 'L'
            display_aff['avg_rental_yield'] = display_aff['avg_rental_yield'].round(1).astype(str) + '%'
            display_aff['rera_compliance_pct'] = display_aff['rera_compliance_pct'].round(0).astype(str) + '%'
            
            # Rename columns for better display
            display_aff = display_aff.rename(columns={
                'area': 'Locality',
                'avg_rental_yield': 'Rental Yield',
                'min_annual_income': 'Min Annual Income',
                'rera_compliance_pct': 'RERA Compliance'
            })
            
            st.dataframe(display_aff, use_container_width=True)
            
            # Bar chart of min income
            fig = px.bar(
                aff_df.sort_values('min_annual_income'),
                x='area',
                y='min_annual_income',
                title='Minimum Annual Income Required by Locality',
                labels={'min_annual_income': 'Min Annual Income (₹)', 'area': 'Locality'},
                color='avg_rental_yield',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No affordability data. Run: `python scripts/enhance_property_data.py`")
    
    with tab5:
        render_demand_predictor(data)
    
    with tab6:
        render_alerts(data)
    
    with tab7:
        render_model_performance(data)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "Pune Real Estate Sentiment Intelligence System | "
        "Built with Streamlit & Python</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

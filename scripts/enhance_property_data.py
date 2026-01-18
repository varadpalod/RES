"""
Enhanced Property Data Generator
Adds RERA status, locality highlights, and rental yield to property data
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, OUTPUT_DIR, PUNE_LOCALITIES


# Locality-specific characteristics (synthetic but realistic for Pune)
LOCALITY_PROFILES = {
    "Koregaon Park": {
        "avg_rental_yield": 3.2,  # % per annum
        "rera_compliance_rate": 0.95,
        "highlights": ["Premium lifestyle", "Restaurants & nightlife", "Central location", "Expat-friendly"],
        "issues": ["Parking shortage", "High maintenance costs"],
        "metro_distance_km": 4.5,
        "it_park_distance_km": 12,
        "avg_commute_time_min": 35
    },
    "Hinjewadi": {
        "avg_rental_yield": 4.5,
        "rera_compliance_rate": 0.88,
        "highlights": ["IT hub proximity", "Modern townships", "Employment center", "Upcoming metro"],
        "issues": ["Heavy traffic", "Limited social infrastructure"],
        "metro_distance_km": 2,
        "it_park_distance_km": 0.5,
        "avg_commute_time_min": 15
    },
    "Pimpri-Chinchwad": {
        "avg_rental_yield": 5.2,
        "rera_compliance_rate": 0.82,
        "highlights": ["Industrial hub", "Affordable housing", "PCMC metro", "Good connectivity"],
        "issues": ["Industrial pollution", "Crowded areas"],
        "metro_distance_km": 1,
        "it_park_distance_km": 8,
        "avg_commute_time_min": 25
    },
    "Viman Nagar": {
        "avg_rental_yield": 3.8,
        "rera_compliance_rate": 0.92,
        "highlights": ["Airport proximity", "Phoenix Mall", "Established locality", "Good schools"],
        "issues": ["Aircraft noise", "Traffic congestion"],
        "metro_distance_km": 3,
        "it_park_distance_km": 10,
        "avg_commute_time_min": 30
    },
    "Kalyani Nagar": {
        "avg_rental_yield": 3.5,
        "rera_compliance_rate": 0.94,
        "highlights": ["Premium locality", "Aga Khan Palace", "Riverfront", "Corporate offices"],
        "issues": ["High property costs", "Limited new inventory"],
        "metro_distance_km": 2.5,
        "it_park_distance_km": 8,
        "avg_commute_time_min": 28
    },
    "Baner": {
        "avg_rental_yield": 4.0,
        "rera_compliance_rate": 0.90,
        "highlights": ["IT corridor", "Balewadi Stadium", "Multiplexes", "Premium societies"],
        "issues": ["Traffic bottlenecks", "Water scarcity"],
        "metro_distance_km": 5,
        "it_park_distance_km": 4,
        "avg_commute_time_min": 22
    },
    "Wakad": {
        "avg_rental_yield": 4.8,
        "rera_compliance_rate": 0.85,
        "highlights": ["IT proximity", "Affordable premium", "Good connectivity", "Rapid development"],
        "issues": ["Infrastructure gaps", "Waterlogging"],
        "metro_distance_km": 3,
        "it_park_distance_km": 3,
        "avg_commute_time_min": 18
    },
    "Kharadi": {
        "avg_rental_yield": 4.2,
        "rera_compliance_rate": 0.91,
        "highlights": ["EON IT Park", "World Trade Center", "Modern infrastructure", "Job hub"],
        "issues": ["Traffic on main road", "Limited public transport"],
        "metro_distance_km": 6,
        "it_park_distance_km": 0.5,
        "avg_commute_time_min": 12
    },
    "Hadapsar": {
        "avg_rental_yield": 5.5,
        "rera_compliance_rate": 0.80,
        "highlights": ["Magarpatta City", "IT parks", "Budget-friendly", "Metro connectivity"],
        "issues": ["Distance from city center", "Traffic issues"],
        "metro_distance_km": 2,
        "it_park_distance_km": 2,
        "avg_commute_time_min": 20
    },
    "Magarpatta": {
        "avg_rental_yield": 3.8,
        "rera_compliance_rate": 0.98,
        "highlights": ["Integrated township", "Self-sufficient", "Premium amenities", "Walk to work"],
        "issues": ["Premium pricing", "Closed ecosystem"],
        "metro_distance_km": 2.5,
        "it_park_distance_km": 0,
        "avg_commute_time_min": 10
    },
    "Aundh": {
        "avg_rental_yield": 3.6,
        "rera_compliance_rate": 0.93,
        "highlights": ["University area", "Bremen Chowk", "Established locality", "Good hospitals"],
        "issues": ["Old constructions", "Parking issues"],
        "metro_distance_km": 4,
        "it_park_distance_km": 6,
        "avg_commute_time_min": 25
    },
    "Shivaji Nagar": {
        "avg_rental_yield": 3.0,
        "rera_compliance_rate": 0.88,
        "highlights": ["City center", "Commercial hub", "FC Road", "Educational institutes"],
        "issues": ["Very crowded", "No parking"],
        "metro_distance_km": 0.5,
        "it_park_distance_km": 10,
        "avg_commute_time_min": 30
    },
    "Deccan": {
        "avg_rental_yield": 2.8,
        "rera_compliance_rate": 0.85,
        "highlights": ["Heritage area", "Cultural hub", "Central location", "Metro station"],
        "issues": ["Old buildings", "Very congested"],
        "metro_distance_km": 0,
        "it_park_distance_km": 10,
        "avg_commute_time_min": 32
    },
    "Kothrud": {
        "avg_rental_yield": 3.4,
        "rera_compliance_rate": 0.92,
        "highlights": ["Family-friendly", "Good schools", "Peaceful", "Established infrastructure"],
        "issues": ["Limited new projects", "Premium pricing"],
        "metro_distance_km": 3,
        "it_park_distance_km": 8,
        "avg_commute_time_min": 28
    },
    "Warje": {
        "avg_rental_yield": 4.5,
        "rera_compliance_rate": 0.86,
        "highlights": ["Affordable", "Near expressway", "Developing area", "Good value"],
        "issues": ["Under development", "Limited amenities"],
        "metro_distance_km": 5,
        "it_park_distance_km": 7,
        "avg_commute_time_min": 25
    },
    "Undri": {
        "avg_rental_yield": 5.8,
        "rera_compliance_rate": 0.78,
        "highlights": ["Most affordable", "Green spaces", "New developments", "Investment potential"],
        "issues": ["Far from city", "Infrastructure gaps"],
        "metro_distance_km": 10,
        "it_park_distance_km": 12,
        "avg_commute_time_min": 40
    },
    "Kondhwa": {
        "avg_rental_yield": 5.0,
        "rera_compliance_rate": 0.82,
        "highlights": ["Affordable housing", "Near NIBM", "Growing area", "Good connectivity"],
        "issues": ["Traffic on main road", "Water issues"],
        "metro_distance_km": 8,
        "it_park_distance_km": 10,
        "avg_commute_time_min": 35
    },
    "NIBM Road": {
        "avg_rental_yield": 4.6,
        "rera_compliance_rate": 0.84,
        "highlights": ["Educational hub", "Affordable premium", "Good societies", "Family area"],
        "issues": ["Single road access", "Traffic jams"],
        "metro_distance_km": 7,
        "it_park_distance_km": 8,
        "avg_commute_time_min": 30
    },
    "Bavdhan": {
        "avg_rental_yield": 4.2,
        "rera_compliance_rate": 0.89,
        "highlights": ["Near expressway", "Scenic location", "Modern townships", "Growing IT hub"],
        "issues": ["Hilly terrain", "Limited public transport"],
        "metro_distance_km": 6,
        "it_park_distance_km": 5,
        "avg_commute_time_min": 22
    },
    "Pashan": {
        "avg_rental_yield": 3.8,
        "rera_compliance_rate": 0.91,
        "highlights": ["University area", "Research institutes", "Peaceful", "Green cover"],
        "issues": ["Limited commercial options", "Far from IT hubs"],
        "metro_distance_km": 5,
        "it_park_distance_km": 7,
        "avg_commute_time_min": 28
    }
}


def generate_rera_status(compliance_rate: float) -> dict:
    """Generate RERA registration status based on compliance rate"""
    is_registered = np.random.random() < compliance_rate
    
    if is_registered:
        # Generate RERA number (format: P52100XXXXXX)
        rera_number = f"P521000{np.random.randint(10000, 99999)}"
        return {
            "rera_registered": 1,
            "rera_number": rera_number,
            "rera_expiry_years": np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.3, 0.25, 0.15])
        }
    else:
        return {
            "rera_registered": 0,
            "rera_number": None,
            "rera_expiry_years": 0
        }


def enhance_property_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RERA status, locality highlights, and rental yield to property data
    
    Args:
        df: Original property DataFrame
        
    Returns:
        Enhanced DataFrame with new features
    """
    enhanced_records = []
    
    for _, row in df.iterrows():
        area = row['area']
        profile = LOCALITY_PROFILES.get(area, LOCALITY_PROFILES.get("Baner"))  # Default fallback
        
        # Get RERA status
        rera_info = generate_rera_status(profile['rera_compliance_rate'])
        
        # Calculate property-specific rental yield with some variance
        base_yield = profile['avg_rental_yield']
        # Newer properties and those with garage have slightly higher yields
        property_age = 2024 - row.get('year_built', 2015)
        age_adjustment = 0.3 if property_age < 5 else (-0.2 if property_age > 15 else 0)
        garage_adjustment = 0.2 if row.get('has_garage', 0) else 0
        
        property_yield = base_yield + age_adjustment + garage_adjustment + np.random.normal(0, 0.3)
        property_yield = max(2.0, min(7.0, property_yield))  # Clamp to realistic range
        
        # Create enhanced record
        enhanced = {
            **row.to_dict(),
            # RERA features
            'rera_registered': rera_info['rera_registered'],
            'rera_expiry_years': rera_info['rera_expiry_years'],
            # Rental yield
            'avg_rental_yield': round(property_yield, 2),
            'estimated_monthly_rent': int(row['price'] * property_yield / 100 / 12),
            # Locality features
            'metro_distance_km': profile['metro_distance_km'],
            'it_park_distance_km': profile['it_park_distance_km'],
            'avg_commute_time_min': profile['avg_commute_time_min'],
            # Locality highlights score (number of positive highlights)
            'locality_highlights_count': len(profile['highlights']),
            'locality_issues_count': len(profile['issues']),
            'locality_net_score': len(profile['highlights']) - len(profile['issues'])
        }
        
        enhanced_records.append(enhanced)
    
    enhanced_df = pd.DataFrame(enhanced_records)
    
    print(f"✓ Enhanced {len(enhanced_df):,} properties with new features")
    print(f"  RERA registered: {enhanced_df['rera_registered'].sum():,} ({enhanced_df['rera_registered'].mean()*100:.1f}%)")
    print(f"  Avg rental yield: {enhanced_df['avg_rental_yield'].mean():.2f}%")
    
    return enhanced_df


def get_locality_highlights_df() -> pd.DataFrame:
    """Create a DataFrame with locality highlights for dashboard"""
    records = []
    
    for locality, profile in LOCALITY_PROFILES.items():
        records.append({
            'locality': locality,
            'highlights': ', '.join(profile['highlights']),
            'issues': ', '.join(profile['issues']),
            'avg_rental_yield': profile['avg_rental_yield'],
            'rera_compliance_rate': profile['rera_compliance_rate'] * 100,
            'metro_distance_km': profile['metro_distance_km'],
            'it_park_distance_km': profile['it_park_distance_km'],
            'avg_commute_time_min': profile['avg_commute_time_min']
        })
    
    return pd.DataFrame(records)


def calculate_affordability(price: float, interest_rate: float = 8.5, 
                           loan_tenure_years: int = 20, 
                           emi_to_income_ratio: float = 0.4) -> dict:
    """
    Calculate affordability metrics including minimum income needed
    
    Args:
        price: Property price in INR
        interest_rate: Annual interest rate (%)
        loan_tenure_years: Loan tenure in years
        emi_to_income_ratio: Max EMI as fraction of income
        
    Returns:
        Dictionary with affordability metrics
    """
    # Assume 80% loan to value
    loan_amount = price * 0.80
    down_payment = price * 0.20
    
    # Calculate EMI using formula: EMI = P × r × (1+r)^n / ((1+r)^n - 1)
    monthly_rate = interest_rate / 100 / 12
    n_months = loan_tenure_years * 12
    
    if monthly_rate > 0:
        emi = loan_amount * monthly_rate * (1 + monthly_rate)**n_months / ((1 + monthly_rate)**n_months - 1)
    else:
        emi = loan_amount / n_months
    
    # Calculate minimum monthly income needed
    min_monthly_income = emi / emi_to_income_ratio
    min_annual_income = min_monthly_income * 12
    
    return {
        'down_payment': int(down_payment),
        'loan_amount': int(loan_amount),
        'monthly_emi': int(emi),
        'min_monthly_income': int(min_monthly_income),
        'min_annual_income': int(min_annual_income),
        'income_bracket': get_income_bracket(min_annual_income)
    }


def get_income_bracket(annual_income: float) -> str:
    """Categorize income into brackets"""
    if annual_income < 500000:
        return "Entry Level (<5L)"
    elif annual_income < 1000000:
        return "Mid Level (5-10L)"
    elif annual_income < 2000000:
        return "Senior Level (10-20L)"
    elif annual_income < 5000000:
        return "Executive (20-50L)"
    else:
        return "Premium (50L+)"


def add_affordability_to_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add affordability metrics to property data"""
    affordability_data = []
    
    for _, row in df.iterrows():
        price = row['price']
        affordability = calculate_affordability(price)
        affordability_data.append(affordability)
    
    affordability_df = pd.DataFrame(affordability_data)
    result = pd.concat([df.reset_index(drop=True), affordability_df], axis=1)
    
    print(f"✓ Added affordability metrics")
    print(f"  Income brackets: {result['income_bracket'].value_counts().to_dict()}")
    
    return result


if __name__ == "__main__":
    # Load existing property data
    property_file = DATA_DIR / 'pune_house_prices.csv'
    
    if property_file.exists():
        print("Loading property data...")
        df = pd.read_csv(property_file)
        
        print("\nEnhancing with new features...")
        enhanced_df = enhance_property_data(df)
        
        print("\nAdding affordability metrics...")
        final_df = add_affordability_to_data(enhanced_df)
        
        # Save enhanced data
        output_file = DATA_DIR / 'pune_house_prices_enhanced.csv'
        final_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved enhanced data to {output_file}")
        
        # Save locality highlights
        highlights_df = get_locality_highlights_df()
        highlights_file = OUTPUT_DIR / 'locality_highlights.csv'
        highlights_df.to_csv(highlights_file, index=False)
        print(f"✓ Saved locality highlights to {highlights_file}")
        
        print("\nSample enhanced data:")
        print(final_df[['area', 'price', 'rera_registered', 'avg_rental_yield', 
                       'min_annual_income', 'income_bracket']].head(10))
    else:
        print(f"Property file not found: {property_file}")

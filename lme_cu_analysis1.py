# Fixed: analyze_copper.py
import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime
import os
from scipy import stats
from statistics import mean, stdev
import warnings

warnings.filterwarnings('ignore')

class CopperPriceAnalyzer:
    def __init__(self):
        self.df = None
        self.analysis_results = {}

    def fetch_data_from_website(self):
        """Fetch CSV data from infilearnai.com"""
        csv_url = 'https://infilearnai.com/LME_Cu_Dashboard/lme_copper_historical_data.csv'
        
        try:
            print(f"🔄 Fetching data from: {csv_url}")
            response = requests.get(csv_url, timeout=30)
            response.raise_for_status()
            
            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            print(f"✅ Raw data loaded. Shape: {df.shape}")
            print(f"📊 Columns: {list(df.columns)}")
            
            # Auto-detect columns
            date_col = self.find_date_column(df)
            price_col = self.find_price_column(df)
            
            if not date_col or not price_col:
                raise ValueError(f"Could not detect date/price columns from: {list(df.columns)}")
            
            print(f"📅 Date column: '{date_col}'")
            print(f"💰 Price column: '{price_col}'")
            
            # Standardize names
            df = df.rename(columns={date_col: 'Date', price_col: 'Price'})
            
            # Clean data
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).sort_values('Date')
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
            df = df.dropna(subset=['Price'])
            df = df[df['Price'] > 0]
            
            # Add time features
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['MonthName'] = df['Date'].dt.month_name()
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['DayName'] = df['Date'].dt.day_name()
            
            self.df = df
            
            print(f"🎯 Final dataset: {len(df)} records")
            print(f"📆 Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"💵 Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return False

    def find_date_column(self, df):
        """Auto-detect date column"""
        for col in df.columns:
            if any(word in col.lower() for word in ['date', 'time', 'day']):
                try:
                    pd.to_datetime(df[col].iloc[:5])
                    return col
                except:
                    continue
        return df.columns[0] if len(df.columns) > 0 else None

    def find_price_column(self, df):
        """Auto-detect price column"""
        for col in df.columns:
            if any(word in col.lower() for word in ['price', 'copper', 'lme', 'settlement', 'cash']):
                try:
                    pd.to_numeric(df[col].iloc[:5])
                    return col
                except:
                    continue
        
        # Find any numeric column
        for col in df.select_dtypes(include=[np.number]).columns:
            return col
        return None

    def safe_json_convert(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.safe_json_convert(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.safe_json_convert(item) for item in obj]
        else:
            return obj

    def analyze_basic_stats(self):
        """Calculate basic statistics"""
        prices = self.df['Price'].tolist()
        
        return {
            'average_price': round(mean(prices), 2),
            'min_price': round(min(prices), 2),
            'max_price': round(max(prices), 2),
            'volatility': round((stdev(prices) / mean(prices)) * 100, 2),
            'total_records': len(prices)
        }

    def analyze_seasonality(self):
        """Analyze seasonal patterns"""
        # Monthly averages
        monthly_avg = self.df.groupby('MonthName')['Price'].mean().round(2)
        best_month = monthly_avg.idxmax()
        
        # Day of week averages
        dow_avg = self.df.groupby('DayName')['Price'].mean().round(2)
        best_day = dow_avg.idxmax()
        
        return {
            'monthly': {'mean': monthly_avg.to_dict()},
            'day_of_week': {'mean': dow_avg.to_dict()},
            'best_month': best_month,
            'best_day': best_day
        }

    def analyze_trends(self):
        """Analyze price trends"""
        prices = self.df['Price'].values
        x = np.arange(len(prices))
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        direction = 'Upward' if slope > 0 else 'Downward' if slope < 0 else 'Flat'
        
        return {
            'slope': round(slope, 6),
            'r_squared': round(r_value**2, 4),
            'trend_direction': direction
        }

    def analyze_monthly_fluctuations(self):
        """Month-over-month analysis"""
        # Create year-month groups
        self.df['YearMonth'] = self.df['Date'].dt.to_period('M')
        monthly_avg = self.df.groupby('YearMonth')['Price'].mean()
        
        # Calculate month-over-month changes
        mom_changes = monthly_avg.pct_change().dropna() * 100
        
        return {
            'mom_changes': mom_changes.round(2).tolist(),
            'mom_dates': [str(d) for d in mom_changes.index],
            'base_price': round(monthly_avg.mean(), 2),
            'volatility': round(mom_changes.std(), 2)
        }

    def analyze_weekly_patterns(self):
        """Weekly pattern analysis - FIXED VERSION"""
        dow_stats = self.df.groupby('DayName')['Price'].agg(['mean', 'count']).round(2)
        best_day = dow_stats['mean'].idxmax()
        overall_avg = self.df['Price'].mean()
        
        # Weekly performance vs baseline
        weekly_performance = {}
        for day in dow_stats.index:
            day_avg = dow_stats.loc[day, 'mean']
            performance = ((day_avg - overall_avg) / overall_avg) * 100
            
            # FIX: Convert numpy.bool_ to Python bool using bool() or .item()
            is_better = day_avg > overall_avg
            weekly_performance[day] = {
                'average_price': round(day_avg, 2),
                'vs_monthly_avg': round(performance, 2),
                'is_better_than_monthly': bool(is_better)  # ✅ FIXED: Convert to Python bool
            }
        
        return {
            'best_day': best_day,
            'weekly_performance': weekly_performance,
            'monthly_baseline': round(overall_avg, 2)
        }

    def run_comprehensive_analysis(self):
        """Run all analysis components"""
        print("🔬 Starting comprehensive analysis...")
        
        # Fetch data
        if not self.fetch_data_from_website():
            print("❌ Failed to fetch data")
            return None
        
        # Run analyses
        print("📊 Calculating basic statistics...")
        basic_stats = self.analyze_basic_stats()
        
        print("🌀 Analyzing seasonality...")
        seasonality = self.analyze_seasonality()
        
        print("📈 Analyzing trends...")
        trends = self.analyze_trends()
        
        print("📅 Analyzing monthly fluctuations...")
        monthly_fluct = self.analyze_monthly_fluctuations()
        
        print("📆 Analyzing weekly patterns...")
        weekly_patterns = self.analyze_weekly_patterns()
        
        # Combine results
        self.analysis_results = {
            'key_metrics': {
                **basic_stats,
                'trend_direction': trends['trend_direction'],
                'best_month': seasonality['best_month'],
                'best_day': seasonality['best_day']
            },
            'seasonality': seasonality,
            'trend': trends,
            'monthly_fluctuations': monthly_fluct,
            'weekly_patterns': weekly_patterns
        }
        
        print("✅ Analysis completed successfully!")
        return self.analysis_results

    def save_results(self):
        """Save results to JSON files for GitHub Pages"""
        os.makedirs('docs', exist_ok=True)
        
        # Prepare complete response data
        response_data = {
            'status': 'success',
            'message': 'Analysis completed successfully',
            'timestamp': datetime.now().isoformat(),
            'data_source': 'https://infilearnai.com/LME_Cu_Dashboard/lme_copper_historical_data.csv',
            'results': self.analysis_results,
            'raw_data': {
                'dates': self.df['Date'].dt.strftime('%Y-%m-%d').tolist(),
                'prices': self.df['Price'].round(2).tolist()
            },
            'metadata': {
                'total_records': len(self.df),
                'date_range': {
                    'start': self.df['Date'].min().strftime('%Y-%m-%d'),
                    'end': self.df['Date'].max().strftime('%Y-%m-%d')
                },
                'analysis_version': '1.1',
                'github_action': True
            }
        }
        
        # Apply safe JSON conversion to handle numpy types
        response_data = self.safe_json_convert(response_data)
        
        # Save main analysis results
        with open('docs/analysis.json', 'w') as f:
            json.dump(response_data, f, indent=2, default=str)
        
        # Save timestamp for cache busting
        with open('docs/last_updated.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'unix_timestamp': int(datetime.now().timestamp())
            }, f)
        
        # Create a simple index.html for GitHub Pages
        index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LME Copper Price Analysis API</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        code { background: #e0e0e0; padding: 2px 5px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>📈 LME Copper Price Analysis API</h1>
    <p>Automated analysis of LME copper prices from infilearnai.com</p>
    
    <h2>API Endpoints</h2>
    
    <div class="endpoint">
        <h3><code>GET /analysis.json</code></h3>
        <p>Complete analysis results with trends, seasonality, and weekly patterns.</p>
        <a href="analysis.json">View JSON →</a>
    </div>
    
    <div class="endpoint">
        <h3><code>GET /last_updated.json</code></h3>
        <p>Timestamp of last analysis update.</p>
        <a href="last_updated.json">View JSON →</a>
    </div>
    
    <p><strong>Analysis runs automatically every 6 hours via GitHub Actions.</strong></p>
    
    <h3>Usage Example</h3>
    <p>Use this API endpoint in your dashboard:</p>
    <code>https://your-username.github.io/copper-analysis/analysis.json</code>
</body>
</html>'''
        
        with open('docs/index.html', 'w') as f:
            f.write(index_html)
        
        print(f"💾 Results saved to docs/analysis.json")
        print(f"📄 GitHub Pages index created")

def main():
    print("🚀 Starting Copper Price Analysis for GitHub Actions")
    analyzer = CopperPriceAnalyzer()
    
    try:
        results = analyzer.run_comprehensive_analysis()
        if results:
            analyzer.save_results()
            print("🎉 Analysis completed and saved!")
        else:
            print("💥 Analysis failed!")
            exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        exit(1)

if __name__ == "__main__":
    main()

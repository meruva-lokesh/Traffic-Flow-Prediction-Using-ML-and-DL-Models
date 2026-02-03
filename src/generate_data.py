import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Define junctions with different characteristics
junctions = {
    'Junction A': {'capacity': 180, 'busy_multiplier': 1.2},  # High capacity, busier
    'Junction B': {'capacity': 150, 'busy_multiplier': 1.0},  # Medium capacity
    'Junction C': {'capacity': 160, 'busy_multiplier': 0.9}   # Medium-high capacity
}

def get_time_factor(hour):
    """Returns traffic multiplier based on time of day"""
    if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
        return 1.8
    elif 10 <= hour <= 16:  # Business hours
        return 1.3
    elif 20 <= hour <= 22:  # Evening
        return 1.0
    else:  # Night
        return 0.4

def get_weather_impact(weather):
    """Returns vehicle count multiplier and speed reduction"""
    impacts = {
        'Sunny': (1.0, 1.0),
        'Cloudy': (0.95, 0.95),
        'Rainy': (0.85, 0.75),
        'Foggy': (0.80, 0.65),
        'Stormy': (0.70, 0.50)
    }
    return impacts[weather]

def get_day_factor(day_name):
    """Returns multiplier based on day of week"""
    if day_name in ['Monday', 'Friday']:
        return 1.3
    elif day_name in ['Tuesday', 'Wednesday', 'Thursday']:
        return 1.2
    elif day_name == 'Saturday':
        return 0.8
    else:  # Sunday
        return 0.6

def classify_traffic(total, capacity, weather_speed, congestion_ratio):
    """More realistic traffic classification"""
    # Calculate congestion level (0-100+%)
    congestion = (total / capacity) * 100 * (1 / weather_speed)
    
    # Adjusted classification with realistic thresholds
    if congestion < 40:
        return 'Low'
    elif congestion < 65:
        return 'Medium'
    elif congestion < 85:
        return 'High'
    else:
        return 'Severe'

def generate_enhanced_dataset(row_count=5000):
    """Generate realistic traffic dataset"""
    data = []
    start_date = datetime(2024, 1, 1)
    
    for i in range(row_count):
        # Generate time-based features
        current_date = start_date + timedelta(hours=i % 8760)  # Cycle through year
        hour = current_date.hour
        day_name = current_date.strftime('%A')
        month = current_date.month
        
        # Select junction
        junc_name = np.random.choice(list(junctions.keys()))
        junc_info = junctions[junc_name]
        
        # Select weather (seasonal variation)
        if month in [6, 7, 8]:  # Summer
            weather = np.random.choice(['Sunny', 'Cloudy'], p=[0.7, 0.3])
            temp = np.random.uniform(25, 40)
        elif month in [12, 1, 2]:  # Winter
            weather = np.random.choice(['Cloudy', 'Rainy', 'Foggy', 'Stormy'], p=[0.4, 0.3, 0.2, 0.1])
            temp = np.random.uniform(10, 20)
        elif month in [3, 4, 5, 9, 10, 11]:  # Spring/Fall
            weather = np.random.choice(['Sunny', 'Cloudy', 'Rainy'], p=[0.5, 0.3, 0.2])
            temp = np.random.uniform(15, 30)
        
        # Calculate traffic factors
        time_factor = get_time_factor(hour)
        day_factor = get_day_factor(day_name)
        weather_mult, weather_speed = get_weather_impact(weather)
        
        # Base vehicle counts with realistic distributions
        base_multiplier = time_factor * day_factor * weather_mult * junc_info['busy_multiplier']
        
        # Generate vehicle counts with realistic proportions
        car = int(np.random.poisson(35 * base_multiplier))
        bus = int(np.random.poisson(8 * base_multiplier))
        bike = int(np.random.poisson(15 * base_multiplier))
        truck = int(np.random.poisson(5 * base_multiplier))
        
        # Add some randomness
        car += np.random.randint(-5, 15)
        bus += np.random.randint(-2, 5)
        bike += np.random.randint(-3, 10)
        truck += np.random.randint(-1, 4)
        
        # Ensure non-negative
        car = max(0, car)
        bus = max(0, bus)
        bike = max(0, bike)
        truck = max(0, truck)
        
        total = car + bus + bike + truck
        
        # Calculate congestion ratio
        congestion_ratio = total / junc_info['capacity']
        
        # Classify traffic situation
        situation = classify_traffic(total, junc_info['capacity'], weather_speed, congestion_ratio)
        
        # Add features
        is_rush_hour = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
        is_weekend = 1 if day_name in ['Saturday', 'Sunday'] else 0
        
        data.append([
            junc_name, car, bus, bike, truck, total,
            weather, temp, hour, day_name, is_rush_hour, is_weekend,
            situation
        ])
    
    df = pd.DataFrame(data, columns=[
        'Junction', 'CarCount', 'BusCount', 'BikeCount', 'TruckCount',
        'TotalVehicles', 'Weather', 'Temperature', 'Hour', 'DayOfWeek',
        'IsRushHour', 'IsWeekend', 'TrafficSituation'
    ])
    
    return df

# Generate enhanced dataset
print("Generating enhanced traffic dataset...")
df = generate_enhanced_dataset(5000)

# Display statistics
print("\n" + "="*60)
print("DATASET STATISTICS")
print("="*60)
print(f"\nTotal Records: {len(df)}")
print("\nTraffic Situation Distribution:")
print(df['TrafficSituation'].value_counts())
print("\nPercentage:")
print(df['TrafficSituation'].value_counts(normalize=True) * 100)
print("\nWeather Distribution:")
print(df['Weather'].value_counts())
print("\nJunction Distribution:")
print(df['Junction'].value_counts())
print("\nAverage Vehicles by Situation:")
print(df.groupby('TrafficSituation')['TotalVehicles'].mean())

# Save to CSV
df.to_csv('traffic_data.csv', index=False)
print("\n✅ Enhanced dataset saved to 'traffic_data.csv'")
print(f"   Shape: {df.shape}")

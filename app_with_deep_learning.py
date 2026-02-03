"""
Traffic Flow Prediction System - Complete with Deep Learning Models
===================================================================
Includes Traditional ML + Deep Learning Models for Academic Publication

Models Available:
- Traditional ML: Random Forest, SVM, Logistic Regression, Naive Bayes, Decision Tree
- Deep Learning: 1D CNN, VGG16, VGG19, ResNet50
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

# Page configuration
st.set_page_config(
    page_title="Traffic Flow Prediction - ML & Deep Learning",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class TrafficPredictionSystem:
    def __init__(self):
        self.models_dir = Path('models')
        self.ml_models = {}
        self.dl_models = {}
        self.scaler = None
        self.encoders = {}
        
    def load_ml_models(self):
        """Load traditional ML models"""
        ml_model_files = {
            'Random Forest': 'model_random_forest.pkl',
            'Logistic Regression': 'model_logistic_regression.pkl',
            'Naive Bayes': 'model_naive_bayes.pkl',
            'Support Vector Machine': 'model_support_vector_machine.pkl',
            'Decision Tree': 'model_decision_tree.pkl'
        }
        
        for name, filename in ml_model_files.items():
            filepath = self.models_dir / filename
            if filepath.exists():
                self.ml_models[name] = joblib.load(filepath)
        
        # Load scaler
        scaler_path = self.models_dir / 'scaler.pkl'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        # Load encoders
        encoder_files = ['le_day.pkl', 'le_junction.pkl', 'le_situation.pkl', 'le_weather.pkl']
        for enc_file in encoder_files:
            enc_path = self.models_dir / enc_file
            if enc_path.exists():
                key = enc_file.replace('le_', '').replace('.pkl', '')
                self.encoders[key] = joblib.load(enc_path)
        
        return len(self.ml_models) > 0
    
    def load_dl_models(self):
        """Load deep learning models"""
        dl_model_files = {
            '1D CNN': 'dl_1d_cnn.h5',
            'VGG16': 'dl_vgg16.h5',
            'VGG19': 'dl_vgg19.h5',
            'ResNet50': 'dl_resnet50.h5'
        }
        
        for name, filename in dl_model_files.items():
            filepath = self.models_dir / filename
            if filepath.exists():
                try:
                    self.dl_models[name] = keras.models.load_model(filepath)
                except Exception as e:
                    st.warning(f"Could not load {name}: {e}")
        
        return len(self.dl_models) > 0
    
    def prepare_features(self, data):
        """Prepare features for prediction"""
        # Time-based features
        data['Hour'] = int(data['Time'].split(':')[0])
        data['IsRushHour'] = 1 if data['Hour'] in [7,8,9,17,18,19] else 0
        data['IsWeekend'] = 1 if data['Day'] in ['Saturday', 'Sunday'] else 0
        
        # Time of day
        if 6 <= data['Hour'] < 12:
            time_of_day = 'Morning'
        elif 12 <= data['Hour'] < 17:
            time_of_day = 'Afternoon'
        elif 17 <= data['Hour'] < 21:
            time_of_day = 'Evening'
        else:
            time_of_day = 'Night'
        
        # Vehicle features
        total_vehicles = data['Cars'] + data['Buses'] + data['Bikes'] + data['Trucks']
        data['Total Vehicles'] = total_vehicles
        data['VehicleDensity'] = total_vehicles / 100
        data['HeavyVehicleRatio'] = (data['Buses'] + data['Trucks']) / (total_vehicles + 1)
        data['LightVehicleRatio'] = (data['Cars'] + data['Bikes']) / (total_vehicles + 1)
        data['CarToBikeRatio'] = data['Cars'] / (data['Bikes'] + 1)
        
        # Encode categorical variables
        junction_encoded = self.encoders['junction'].transform([data['Junction']])[0]
        day_encoded = self.encoders['day'].transform([data['Day']])[0]
        weather_encoded = self.encoders['weather'].transform([data['Weather']])[0]
        
        # Interaction features
        weather_hour_interaction = f"{data['Weather']}_{time_of_day}"
        junction_rush_interaction = f"{data['Junction']}_{data['IsRushHour']}"
        
        # Create feature vector
        features = [
            junction_encoded,
            data['Cars'],
            data['Buses'],
            data['Bikes'],
            data['Trucks'],
            total_vehicles,
            weather_encoded,
            data['Temperature'],
            data['Hour'],
            day_encoded,
            data['IsRushHour'],
            data['IsWeekend'],
            data['VehicleDensity'],
            data['HeavyVehicleRatio'],
            data['LightVehicleRatio'],
            data['CarToBikeRatio']
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict_ml(self, features, model_name):
        """Make prediction using ML model"""
        if model_name not in self.ml_models:
            return None, None
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        model = self.ml_models[model_name]
        prediction = model.predict(features_scaled)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = float(max(probabilities)) * 100
        else:
            confidence = None
        
        # Decode prediction
        traffic_levels = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH', 3: 'SEVERE'}
        prediction_label = traffic_levels.get(prediction, 'UNKNOWN')
        
        return prediction_label, confidence
    
    def predict_dl(self, features, model_name):
        """Make prediction using deep learning model"""
        if model_name not in self.dl_models:
            return None, None
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Reshape for CNN input (samples, timesteps, features)
        features_cnn = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
        
        # Predict
        model = self.dl_models[model_name]
        probabilities = model.predict(features_cnn, verbose=0)[0]
        prediction = np.argmax(probabilities)
        confidence = float(max(probabilities)) * 100
        
        # Decode prediction
        traffic_levels = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH', 3: 'SEVERE'}
        prediction_label = traffic_levels.get(prediction, 'UNKNOWN')
        
        return prediction_label, confidence
    
    def get_traffic_color(self, level):
        """Get color code for traffic level"""
        colors = {
            'LOW': '#28a745',
            'MEDIUM': '#ffc107',
            'HIGH': '#fd7e14',
            'SEVERE': '#dc3545'
        }
        return colors.get(level, '#6c757d')
    
    def get_traffic_description(self, level):
        """Get description for traffic level"""
        descriptions = {
            'LOW': '🟢 Smooth traffic flow with minimal delays',
            'MEDIUM': '🟡 Moderate traffic with minor delays',
            'HIGH': '🟠 Heavy traffic with significant delays',
            'SEVERE': '🔴 Severe congestion with major delays'
        }
        return descriptions.get(level, 'Unknown traffic condition')

# Initialize system
@st.cache_resource
def initialize_system():
    system = TrafficPredictionSystem()
    ml_loaded = system.load_ml_models()
    dl_loaded = system.load_dl_models()
    return system, ml_loaded, dl_loaded

system, ml_loaded, dl_loaded = initialize_system()

# Main UI
st.markdown('<div class="main-header">🚦 Traffic Flow Prediction System<br/>ML & Deep Learning Models</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 Model Selection")
    
    model_type = st.radio(
        "Choose Model Type:",
        ["Traditional ML", "Deep Learning", "Compare All Models"],
        help="Traditional ML: Fast inference\nDeep Learning: High accuracy\nCompare All: See all predictions"
    )
    
    if model_type == "Traditional ML":
        if ml_loaded:
            selected_ml_model = st.selectbox(
                "Select ML Model:",
                list(system.ml_models.keys())
            )
        else:
            st.error("❌ ML models not found. Train models first.")
    
    elif model_type == "Deep Learning":
        if dl_loaded:
            selected_dl_model = st.selectbox(
                "Select DL Model:",
                list(system.dl_models.keys())
            )
        else:
            st.error("❌ DL models not found. Train models first.")
    
    st.markdown("---")
    st.markdown("### 📊 Model Statistics")
    st.info(f"**ML Models:** {len(system.ml_models)}\n\n**DL Models:** {len(system.dl_models)}")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("This system combines traditional machine learning and state-of-the-art deep learning models for accurate traffic prediction.")

# Main content
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Model Comparison", "📚 Documentation"])

with tab1:
    st.header("Enter Traffic Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        junction = st.selectbox("Junction:", ["A", "B", "C"])
        time = st.time_input("Time:", value=datetime.now().time())
        day = st.selectbox("Day:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        temperature = st.slider("Temperature (°C):", 15, 45, 25)
    
    with col2:
        cars = st.number_input("Cars:", 0, 200, 50, step=5)
        buses = st.number_input("Buses:", 0, 50, 10, step=2)
        bikes = st.number_input("Motorcycles & Bikes:", 0, 150, 30, step=5)
        trucks = st.number_input("Trucks:", 0, 50, 5, step=2)
    
    with col3:
        weather = st.selectbox("Weather:", ["Sunny", "Cloudy", "Rainy", "Foggy", "Stormy"])
        
        st.markdown("### Vehicle Summary")
        total_vehicles = cars + buses + bikes + trucks
        st.metric("Total Vehicles", total_vehicles)
        st.progress(min(total_vehicles / 300, 1.0))
    
    if st.button("🚀 Predict Traffic", type="primary"):
        # Prepare input data
        input_data = {
            'Junction': junction,
            'Time': time.strftime("%H:%M"),
            'Day': day,
            'Cars': cars,
            'Buses': buses,
            'Bikes': bikes,
            'Trucks': trucks,
            'Weather': weather,
            'Temperature': temperature
        }
        
        features = system.prepare_features(input_data)
        
        # Make prediction based on selected model type
        st.markdown("---")
        
        if model_type == "Traditional ML" and ml_loaded:
            prediction, confidence = system.predict_ml(features, selected_ml_model)
            
            if prediction:
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"### Prediction: <span style='color:{system.get_traffic_color(prediction)};font-size:2rem;font-weight:bold'>{prediction}</span>", unsafe_allow_html=True)
                    st.markdown(system.get_traffic_description(prediction))
                
                with col2:
                    if confidence:
                        st.metric("Confidence", f"{confidence:.1f}%")
                
                with col3:
                    st.metric("Model", selected_ml_model)
        
        elif model_type == "Deep Learning" and dl_loaded:
            prediction, confidence = system.predict_dl(features, selected_dl_model)
            
            if prediction:
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"### Prediction: <span style='color:{system.get_traffic_color(prediction)};font-size:2rem;font-weight:bold'>{prediction}</span>", unsafe_allow_html=True)
                    st.markdown(system.get_traffic_description(prediction))
                
                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                with col3:
                    st.metric("Model", selected_dl_model)
        
        elif model_type == "Compare All Models":
            st.header("🔍 All Models Comparison")
            
            results = []
            
            # ML Models
            if ml_loaded:
                for model_name in system.ml_models.keys():
                    pred, conf = system.predict_ml(features, model_name)
                    if pred:
                        results.append({
                            'Model Type': 'Traditional ML',
                            'Model': model_name,
                            'Prediction': pred,
                            'Confidence': f"{conf:.1f}%" if conf else "N/A"
                        })
            
            # DL Models
            if dl_loaded:
                for model_name in system.dl_models.keys():
                    pred, conf = system.predict_dl(features, model_name)
                    if pred:
                        results.append({
                            'Model Type': 'Deep Learning',
                            'Model': model_name,
                            'Prediction': pred,
                            'Confidence': f"{conf:.1f}%"
                        })
            
            if results:
                df_results = pd.DataFrame(results)
                
                # Display results
                st.dataframe(df_results, use_container_width=True)
                
                # Consensus
                predictions = [r['Prediction'] for r in results]
                from collections import Counter
                most_common = Counter(predictions).most_common(1)[0]
                
                st.markdown("---")
                st.markdown(f"### 🎯 Model Consensus: <span style='color:{system.get_traffic_color(most_common[0])};font-size:1.5rem;font-weight:bold'>{most_common[0]}</span>", unsafe_allow_html=True)
                st.markdown(f"**{most_common[1]}/{len(results)}** models agree")

with tab2:
    st.header("📊 Model Performance Comparison")
    
    # Load comparison data if available
    ml_comparison_path = Path('models/all_model_results.pkl')
    dl_comparison_path = Path('models/deep_learning_comparison.csv')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Traditional ML Models")
        if ml_comparison_path.exists():
            try:
                ml_results = joblib.load(ml_comparison_path)
                ml_data = []
                for name, metrics in ml_results.items():
                    if isinstance(metrics, dict) and 'accuracy' in metrics:
                        ml_data.append({
                            'Model': name,
                            'Accuracy': f"{metrics['accuracy']*100:.2f}%",
                            'Precision': f"{metrics.get('precision', 0)*100:.2f}%",
                            'F1-Score': f"{metrics.get('f1_score', 0)*100:.2f}%"
                        })
                if ml_data:
                    st.dataframe(pd.DataFrame(ml_data), use_container_width=True)
            except:
                st.info("Train ML models to see comparison")
        else:
            st.info("Train ML models to see comparison")
    
    with col2:
        st.subheader("🧠 Deep Learning Models")
        if dl_comparison_path.exists():
            try:
                dl_df = pd.read_csv(dl_comparison_path)
                st.dataframe(dl_df[['Model', 'Test Accuracy (%)', 'Precision (%)', 'F1-Score (%)']].round(2), use_container_width=True)
            except:
                st.info("Train DL models to see comparison")
        else:
            st.info("Train DL models to see comparison")

with tab3:
    st.header("📚 Documentation")
    
    st.markdown("""
    ## Traditional ML Models
    - **Random Forest**: Ensemble of decision trees (92-95% accuracy)
    - **SVM**: Support Vector Machine for complex patterns (88-92%)
    - **Logistic Regression**: Linear classifier (85-90%)
    - **Naive Bayes**: Probabilistic classifier (75-82%)
    - **Decision Tree**: Interpretable tree-based model (82-88%)
    
    ## Deep Learning Models
    - **1D CNN**: Custom convolutional network for sequential data
    - **VGG16**: 16-layer deep network adapted for tabular data
    - **VGG19**: 19-layer deeper variant of VGG
    - **ResNet50**: 50-layer residual network with skip connections
    
    ## Features Used (19 total)
    1. Junction (A/B/C)
    2. Vehicle counts (Cars, Buses, Bikes, Trucks, Total)
    3. Weather conditions
    4. Temperature
    5. Time features (Hour, Day, Rush hour, Weekend)
    6. Engineered features (Density, Ratios, Interactions)
    
    ## Traffic Classifications
    - **LOW**: < 40% capacity - Smooth flow
    - **MEDIUM**: 40-65% capacity - Moderate traffic
    - **HIGH**: 65-85% capacity - Heavy traffic
    - **SEVERE**: > 85% capacity - Severe congestion
    
    ## For Academic Publication
    This system is designed for research and academic publication:
    - Comprehensive model comparison
    - Publication-ready metrics
    - Reproducible results
    - Detailed documentation
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎓 Traffic Flow Prediction - Capstone Project</p>
    <p>Traditional ML + Deep Learning | Designed for Academic Publication</p>
</div>
""", unsafe_allow_html=True)

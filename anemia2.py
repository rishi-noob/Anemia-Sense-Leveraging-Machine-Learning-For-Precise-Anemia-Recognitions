import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objs as go

class AnemiaSenseApp:
    def __init__(self):
        # Set page configuration
        st.set_page_config(
            page_title="Anemiasense",
            page_icon="🩸",
            layout="wide"
        )
        
        # Load and prepare dataset
        self.load_dataset()
        
    def load_dataset(self):
        """
        Load and preprocess the anemia dataset
        """
        # Simulated dataset generation
        np.random.seed(42)
        data = {
            'hemoglobin_level': np.random.uniform(8, 16, 1000),
            'red_blood_cell_count': np.random.uniform(3.5, 5.5, 1000),
            'mean_corpuscular_volume': np.random.uniform(80, 100, 1000),
            'age': np.random.uniform(18, 80, 1000),
            'gender': np.random.choice(['male', 'female'], 1000),
            'body_mass_index': np.random.uniform(18, 35, 1000),
            'nutritional_status': np.random.choice(['normal', 'mild_deficiency', 'moderate_deficiency'], 1000),
            'chronic_conditions': np.random.choice(['none', 'diabetes', 'kidney_disease'], 1000)
        }
        
        self.df = pd.DataFrame(data)
        
        # Add anemia status
        def determine_anemia_status(row):
            if row['gender'] == 'female':
                return 1 if row['hemoglobin_level'] < 12 else 0
            else:
                return 1 if row['hemoglobin_level'] < 13 else 0
        
        self.df['anemia_status'] = self.df.apply(determine_anemia_status, axis=1)
        
        # Prepare model
        self.prepare_model()
    
    def prepare_model(self):
        """
        Prepare machine learning model for anemia prediction
        """
        # Prepare features
        features = [
            'hemoglobin_level', 
            'red_blood_cell_count', 
            'mean_corpuscular_volume', 
            'age', 
            'body_mass_index'
        ]
        
        # One-hot encode categorical variables
        df_encoded = pd.get_dummies(
            self.df, 
            columns=['gender', 'nutritional_status', 'chronic_conditions']
        )
        
        # Separate features and target
        X = df_encoded[features]
        y = df_encoded['anemia_status']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Random Forest Classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
    
    def run_app(self):
        """
        Main Streamlit application
        """
        # Title and introduction
        st.title("🩸 Anemiasense: Comprehensive Health Profile Analyzer")
        st.markdown("""
        ### Personalized Health Insights through Advanced Analytics
        Complete the detailed health profile below for a comprehensive analysis.
        """)
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            # Personal Details
            full_name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            
            # Contact Information
            email = st.text_input("Email Address")
            phone = st.text_input("Phone Number")
        
        with col2:
            st.subheader("Medical Measurements")
            # Detailed Medical Measurements
            hemoglobin_level = st.number_input(
                "Hemoglobin Level (g/dL)", 
                min_value=0.0, 
                max_value=20.0, 
                value=12.0,
                step=0.1,
                help="Normal range: 12-15.5 g/dL for women, 13.5-17.5 g/dL for men"
            )
            
            red_blood_cell_count = st.number_input(
                "Red Blood Cell Count (millions/mcL)", 
                min_value=0.0, 
                max_value=10.0, 
                value=4.5,
                step=0.1,
                help="Normal range: 4.2-5.4 for women, 4.7-6.1 for men"
            )
            
            mean_corpuscular_volume = st.number_input(
                "Mean Corpuscular Volume (fL)", 
                min_value=50.0, 
                max_value=150.0, 
                value=90.0,
                step=0.1,
                help="Normal range: 80-96 fL"
            )
        
        # Additional Health Information
        st.subheader("Comprehensive Health Profile")
        
        col3, col4 = st.columns(2)
        
        with col3:
            body_mass_index = st.number_input(
                "Body Mass Index (BMI)", 
                min_value=10.0, 
                max_value=50.0, 
                value=25.0,
                step=0.1
            )
            
            nutritional_status = st.selectbox(
                "Nutritional Status", 
                ["Normal", "Mild Nutritional Deficiency", "Moderate Nutritional Deficiency"]
            )
        
        with col4:
            chronic_conditions = st.multiselect(
                "Chronic Conditions", 
                ["None", "Diabetes", "Kidney Disease", "Heart Condition", "Thyroid Disorder"]
            )
            
            medication = st.text_input("Current Medications (if any)")
        
        # Medical History
        st.subheader("Medical History")
        
        col5, col6 = st.columns(2)
        
        with col5:
            family_anemia_history = st.checkbox("Family History of Anemia")
            recent_surgeries = st.text_input("Recent Surgeries (if any)")
        
        with col6:
            menstrual_history = st.selectbox(
                "Menstrual History (for females)", 
                ["Not Applicable", "Regular", "Irregular", "Heavy Flow"]
            ) if gender.lower() == 'female' else None
        
        # Analyze Button
        if st.button("Analyze Health Profile"):
            # Prepare patient data for prediction
            patient_data = {
                'hemoglobin_level': hemoglobin_level,
                'red_blood_cell_count': red_blood_cell_count,
                'mean_corpuscular_volume': mean_corpuscular_volume,
                'age': age,
                'body_mass_index': body_mass_index,
                'gender': gender.lower(),
                'nutritional_status': nutritional_status.lower().replace(' ', '_'),
                'chronic_conditions': chronic_conditions[0].lower() if chronic_conditions else 'none'
            }
            
            # Prepare patient data for prediction
            patient_df = pd.DataFrame([patient_data])
            patient_encoded = pd.get_dummies(patient_df)
            
            # Ensure all one-hot encoded columns exist
            for col in self.scaler.feature_names_in_:
                if col not in patient_encoded.columns:
                    patient_encoded[col] = 0
            
            # Select and order columns
            patient_encoded = patient_encoded[self.scaler.feature_names_in_]
            
            # Scale and predict
            patient_scaled = self.scaler.transform(patient_encoded)
            prediction = self.model.predict(patient_scaled)
            prediction_proba = self.model.predict_proba(patient_scaled)
            
            # Results Section
            st.subheader(f"Health Analysis for {full_name}")
            
            if prediction[0] == 1:
                st.error("🚨 Potential Anemia Risk Detected")
                risk_level = "High"
                risk_color = "red"
                risk_advice = """
                ### Recommended Next Steps:
                - Consult with a healthcare professional immediately
                - Schedule a comprehensive blood panel
                - Consider iron supplementation under medical supervision
                - Review diet and nutritional intake
                """
            else:
                st.success("✅ Low Anemia Risk")
                risk_level = "Low"
                risk_color = "green"
                risk_advice = """
                ### Preventive Recommendations:
                - Maintain current healthy lifestyle
                - Regular health check-ups
                - Balanced diet rich in iron
                - Stay hydrated
                """
            
            # Risk Probability Gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction_proba[0][1] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Anemia Risk Probability"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': risk_color},
                    'steps' : [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))
            
            st.plotly_chart(fig)
            
            # Detailed Advice
            st.markdown(risk_advice)
            
            # Additional Insights
            with st.expander("Detailed Health Insights"):
                st.write("### Comprehensive Health Overview")
                insights = {
                    "Hemoglobin Level": f"{hemoglobin_level} g/dL",
                    "Red Blood Cell Count": f"{red_blood_cell_count} millions/mcL",
                    "Mean Corpuscular Volume": f"{mean_corpuscular_volume} fL",
                    "Risk Assessment": risk_level,
                    "Probability of Anemia": f"{prediction_proba[0][1]*100:.2f}%"
                }
                
                for key, value in insights.items():
                    st.metric(key, value)

# Run the Streamlit app
def main():
    app = AnemiaSenseApp()
    app.run_app()

if __name__ == "__main__":
    main()
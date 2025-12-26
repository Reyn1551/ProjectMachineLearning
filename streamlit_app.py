
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import joblib
import os
import io

# Config
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Tidy" look
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# --- Logic Class (Modified for Streamlit State) ---
class ChurnManager:
    def __init__(self):
        # Initialize session state for data persistence
        if 'df' not in st.session_state:
            st.session_state['df'] = None
        if 'df_processed' not in st.session_state:
            st.session_state['df_processed'] = None
        if 'models' not in st.session_state:
            st.session_state['models'] = {}
        if 'results' not in st.session_state:
            st.session_state['results'] = {}
        if 'status' not in st.session_state:
            st.session_state['status'] = "Ready"

    def load_data(self, file):
        try:
            df = pd.read_csv(file)
            st.session_state['df'] = df
            st.session_state['df_raw'] = df.copy() # Keep a raw copy
            return True, "Data loaded successfully!"
        except Exception as e:
            return False, str(e)

    def preprocess(self):
        if st.session_state['df'] is None:
            return False, "No data to process."
        
        try:
            df = st.session_state['df'].copy()
            
            # Cleaning Steps
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
            
            categorical_cols = df.select_dtypes(include='object').columns.tolist()
            categorical_cols = [c for c in categorical_cols if c not in ['customerID', 'Churn']]
            
            # SeniorCitizen is categorical
            if 'SeniorCitizen' not in categorical_cols:
                categorical_cols.append('SeniorCitizen')

            # Encoding
            df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
            
            # Target
            if 'Churn' in df_encoded.columns:
                y = df_encoded['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
                X = df_encoded.drop(columns=['customerID', 'Churn'])
            else:
                return False, "Churn column not found"

            # Scaling
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            scaler = StandardScaler()
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
            
            # Save to state
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.session_state['scaler'] = scaler
            st.session_state['feature_names'] = X.columns.tolist()
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.session_state['split'] = (X_train, X_test, y_train, y_test)
            st.session_state['df_processed'] = X # For display
            
            return True, "Preprocessing Complete (Cleaned -> Encoded -> Scaled -> Split)"
        except Exception as e:
            return False, f"Error: {e}"

    def train_models(self, use_grid=False):
        if 'split' not in st.session_state:
            return False, "Process data first."
        
        X_train, X_test, y_train, y_test = st.session_state['split']
        
        status_text = st.empty()
        progress = st.progress(0)
        
        # RF
        status_text.text("Training Random Forest...")
        if use_grid:
             rf = GridSearchCV(RandomForestClassifier(random_state=42), 
                               {'n_estimators': [50, 100], 'max_depth': [10, None]}, cv=3)
             rf.fit(X_train, y_train)
             rf_model = rf.best_estimator_
        else:
             rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
             rf_model.fit(X_train, y_train)
        
        st.session_state['models']['RF'] = rf_model
        progress.progress(50)
        
        # KNN
        status_text.text("Training KNN...")
        if use_grid:
            knn = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}, cv=3)
            knn.fit(X_train, y_train)
            knn_model = knn.best_estimator_
        else:
            knn_model = KNeighborsClassifier(n_neighbors=5)
            knn_model.fit(X_train, y_train)
            
        st.session_state['models']['KNN'] = knn_model
        progress.progress(90)
        
        # Evaluate
        self.evaluate(X_test, y_test)
        progress.progress(100)
        status_text.text("Training Complete!")
        return True, "Models Trained"

    def evaluate(self, X_test, y_test):
        results = {}
        for name, model in st.session_state['models'].items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]
            results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred),
                'Confusion Matrix': confusion_matrix(y_test, y_pred),
                'y_test': y_test,
                'y_prob': y_prob
            }
        st.session_state['results'] = results

manager = ChurnManager()

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/cloud/100/000000/combo-chart.png", width=50) # Placeholder icon
    st.title("Navigation")
    page = st.radio("Go to:", ["1. Data Exploration", "2. Preprocessing", "3. Model Training", "4. Results & Comparison", "5. Business Insights"])
    
    st.info("💡 **Tip**: Follow the steps in order.")
    
    st.markdown("---")
    st.write("**Dataset Status:**")
    if st.session_state['df'] is not None:
        st.success("Loaded")
    else:
        st.error("Not Loaded")

# --- Pages ---

if page == "1. Data Exploration":
    st.title("📊 Data Exploration")
    
    uploaded_file = st.file_uploader("Upload CSV", type='csv')
    
    # Auto-load default if exists and nothing uploaded yet
    default_path = "Telco-Customer-Churn.csv"
    if uploaded_file:
         manager.load_data(uploaded_file)
    elif st.session_state['df'] is None and os.path.exists(default_path):
         if st.button("Load Default Dataset (Telco-Customer-Churn.csv)"):
             manager.load_data(default_path)
             st.rerun()

    if st.session_state['df'] is not None:
        df = st.session_state['df']
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())
        
        # Data Preview
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Distribution Plots
        st.subheader("Key Feature Distributions")
        c1, c2 = st.columns(2)
        
        with c1:
            fig = px.pie(df, names='Churn', title='Target Distribution (Churn)', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            if 'PaymentMethod' in df.columns:
                fig2 = px.histogram(df, x='PaymentMethod', color='Churn', title="Churn by Payment Method", barmode='group')
                st.plotly_chart(fig2, use_container_width=True)
                
        # Numeric Stats
        st.write("Numerical Statistics:")
        st.dataframe(df.describe())

elif page == "2. Preprocessing":
    st.title("⚙️ Data Preprocessing")
    
    if st.session_state['df'] is None:
        st.warning("Please load data in step 1.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Preprocessing Pipeline")
            st.code("""
1. Convert 'TotalCharges' to Numeric (handle errors)
2. Handle Missing Values (Mean Imputation)
3. Encode Categorical Variables (One-Hot)
4. Scale Numerical Features (StandardScaler)
5. Split Train/Test (80/20)
            """)
        
        with c2:
            if st.button("🚀 Run Preprocessing Pipeline", type="primary"):
                success, msg = manager.preprocess()
                if success:
                    st.success(msg)
                    st.session_state['processed'] = True
                    st.rerun()
                else:
                    st.error(msg)
                    
        if st.session_state.get('df_processed') is not None:
            st.divider()
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("Before")
                st.dataframe(st.session_state['df'].iloc[:5, :5], use_container_width=True)
            with col_r:
                st.subheader("After (Transformed)")
                st.write(f"New Shape: {st.session_state['df_processed'].shape}")
                st.dataframe(st.session_state['df_processed'].iloc[:5, :5], use_container_width=True)

elif page == "3. Model Training":
    st.title("🧠 Model Training")
    
    if 'split' not in st.session_state:
        st.warning("Please run Preprocessing first.")
    else:
        st.write("Configure and train Random Forest (RF) and K-Nearest Neighbors (KNN).")
        
        use_grid = st.checkbox("Enable GridSearchCV (Hyperparameter Tuning)", help="Will take longer but may improve results.")
        
        if st.button("Start Training", type="primary"):
            success, msg = manager.train_models(use_grid)
            if success:
                st.success(msg)
            else:
                st.error(msg)

        if st.session_state['models']:
            st.success("Models are trained and ready for evaluation!")

elif page == "4. Results & Comparison":
    st.title("🏆 Results & Comparison")
    
    if not st.session_state['results']:
        st.warning("Train models first!")
    else:
        results = st.session_state['results']
        
        # 1. Metrics Table
        st.subheader("Model Performance Metrics")
        metrics_data = []
        for model, res in results.items():
            metrics_data.append({
                'Model': model,
                'Accuracy': res['Accuracy'],
                'Precision': res['Precision'],
                'Recall': res['Recall'],
                'F1-Score': res['F1']
            })
        metrics_df = pd.DataFrame(metrics_data).set_index('Model')
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen').format("{:.4f}"), use_container_width=True)
        
        # 2. Side-by-side Charts
        st.subheader("Visual Comparison")
        met_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        # Reformat for plotting
        plot_data = pd.melt(metrics_df.reset_index(), id_vars='Model', var_name='Metric', value_name='Score')
        
        fig_bar = px.bar(plot_data, x='Metric', y='Score', color='Model', barmode='group', title='Metric Comparison')
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # 3. Deep Dive Tabs
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curves", "Feature Importance"])
        
        with tab1:
            c1, c2 = st.columns(2)
            for i, (name, res) in enumerate(results.items()):
                with (c1 if i==0 else c2):
                    st.write(f"**{name} Confusion Matrix**")
                    cm = res['Confusion Matrix']
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                       labels=dict(x="Predicted", y="Actual"))
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
        with tab2:
            fig_roc = go.Figure()
            for name, res in results.items():
                fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
                roc_auc = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={roc_auc:.2f})'))
            
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random Chance'))
            fig_roc.update_layout(title="ROC Curve Comparison", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)
            
        with tab3:
            rf = st.session_state['models'].get('RF')
            if rf:
                importances = rf.feature_importances_
                feats = st.session_state['feature_names']
                feat_df = pd.DataFrame({'Feature': feats, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(10)
                
                fig_imp = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Top 10 Feature Importance (Random Forest)")
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Train Random Forest to see Feature Importance.")

elif page == "5. Business Insights":
    st.title("💼 Business Insights & Prediction")
    
    c_left, c_right = st.columns([1, 2])
    
    with c_left:
        st.subheader("Predict New Customer")
        with st.form("predict_form"):
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
            monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
            total = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            senior = st.selectbox("Senior Citizen", [0, 1])
            
            submitted = st.form_submit_button("Predict Churn Risk")
            
    with c_right:
        st.subheader("Strategic Recommendations")
        
        if submitted and 'models' in st.session_state and 'RF' in st.session_state['models']:
            # Predict Logic (Simplified reconstruction of input)
            # Create a dummy DF with all zeros matching structure
            ft_names = st.session_state['feature_names']
            input_row = pd.DataFrame(0, index=[0], columns=ft_names)
            
            # Fill Numeric (Need to Scale!)
            scaler = st.session_state['scaler']
            num_vals = pd.DataFrame([[tenure, monthly, total]], columns=['tenure', 'MonthlyCharges', 'TotalCharges'])
            num_scaled = scaler.transform(num_vals)
            
            input_row['tenure'] = num_scaled[0][0]
            input_row['MonthlyCharges'] = num_scaled[0][1]
            input_row['TotalCharges'] = num_scaled[0][2]
            
            # Fill Categorical via Dummy logic
            # E.g., Contract_One year
            if f"Contract_{contract}" in ft_names: input_row[f"Contract_{contract}"] = 1
            if f"PaymentMethod_{payment}" in ft_names: input_row[f"PaymentMethod_{payment}"] = 1
            # SeniorCitizen might be handled as numeric or cat depending on preproc
            # Our preproc kept it categorical.
            if f"SeniorCitizen_{senior}" in ft_names: input_row[f"SeniorCitizen_{senior}"] = 1
            
            # Predict
            model = st.session_state['models']['RF']
            prob = model.predict_proba(input_row)[0][1]
            
            # Display Result
            if prob > 0.5:
                st.error(f"🔴 High Churn Risk! (Probability: {prob:.1%})")
                
                st.markdown("### ⚠️ Recommended Actions:")
                if "Month" in contract:
                    st.write("- **Contract**: User is on Month-to-month. Offer a 10% discount for switching to a 1-year contract.")
                if "Electronic" in payment:
                    st.write("- **Payment**: Electronic check users have higher churn. Suggest setting up Auto-Pay with a $5 credit.")
                if monthly > 80:
                    st.write("- **Pricing**: High monthly bill detected. Offer a loyalty bundle check.")
            else:
                st.success(f"🟢 Low Churn Risk (Probability: {prob:.1%})")
                st.write("Customer appears stable. Maintain standard engagement.")
                
        elif submitted:
            st.warning("Please train the model first in the 'Model Training' tab.")
        else:
            st.info("👈 Fill out the customer details and click Predict to see insights.")
            
        # General Insights from Feature Importance
        if 'models' in st.session_state and 'RF' in st.session_state['models']:
             st.divider()
             st.markdown("#### General Drivers of Churn (from Model):")
             rf = st.session_state['models']['RF']
             imps = rf.feature_importances_
             indices = np.argsort(imps)[::-1][:3]
             top_feats = [st.session_state['feature_names'][i] for i in indices]
             st.write(f"The top factors influencing churn in this dataset are: **{', '.join(top_feats)}**.")


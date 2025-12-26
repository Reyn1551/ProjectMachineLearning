
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import threading
import time
import os
import joblib
import io


# Set style for plots
plt.style.use('ggplot')

class ChurnModel:
    def __init__(self):
        self.df = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rf_model = None
        self.knn_model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.rf_metrics = {}
        self.knn_metrics = {}
        self.top_features = []
        self.original_df = None # For display before processing

    def load_data(self, filepath):
        try:
            self.df = pd.read_csv(filepath)
            self.original_df = self.df.copy()
            return True, "Data loaded successfully."
        except Exception as e:
            return False, str(e)

    def preprocess(self):
        if self.df is None:
            return False, "No data loaded."
        
        try:
            df = self.df.copy()
            
            # 1. Handle TotalCharges
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
            
            # 2. Identify Numeric and Categorical
            categorical_cols = df.select_dtypes(include='object').columns.tolist()
            categorical_cols = [col for col in categorical_cols if col not in ['customerID', 'Churn']]
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numerical_cols = [col for col in numerical_cols if col not in ['customerID', 'Churn', 'SeniorCitizen']]
            
            # SeniorCitizen is categorical
            if 'SeniorCitizen' in numerical_cols:
                numerical_cols.remove('SeniorCitizen')
            if 'SeniorCitizen' not in categorical_cols:
                categorical_cols.append('SeniorCitizen')
                
            # 3. One-Hot Encoding
            df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
            
            # 4. Target Variable
            if 'Churn' in df_encoded.columns:
                self.y = df_encoded['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
                self.X = df_encoded.drop(columns=['customerID', 'Churn'])
            else:
                return False, "Target column 'Churn' not found."

            # 5. Scaling Numerical Features
            # Note: We need to scale only numerical cols being used. 
            # In the encoded dataframe, the original numerical cols are still there.
            self.X[numerical_cols] = self.scaler.fit_transform(self.X[numerical_cols])
            
            self.df_processed = self.X.copy()
            self.df_processed['Churn'] = self.y
            
            self.feature_names = self.X.columns.tolist()
            
            # Split Data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            
            return True, "Preprocessing completed. Data split into Train/Test."
        except Exception as e:
            return False, str(e)

    def train_rf(self, use_gridsearch=False):
        try:
            if use_gridsearch:
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
                grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)
                grid.fit(self.X_train, self.y_train)
                self.rf_model = grid.best_estimator_
            else:
                self.rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
                self.rf_model.fit(self.X_train, self.y_train)
            
            y_pred = self.rf_model.predict(self.X_test)
            self.rf_metrics = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1': f1_score(self.y_test, y_pred),
                'y_pred': y_pred,
                'y_prob': self.rf_model.predict_proba(self.X_test)[:, 1]
            }
            
            # Feature Importance
            importances = self.rf_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            self.top_features = [(self.feature_names[i], importances[i]) for i in indices[:5]]
            
            return True, "Random Forest Trained."
        except Exception as e:
            return False, f"RF Error: {str(e)}"

    def train_knn(self, k=5, use_gridsearch=False):
        try:
            if use_gridsearch:
                param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
                grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, n_jobs=-1)
                grid.fit(self.X_train, self.y_train)
                self.knn_model = grid.best_estimator_
            else:
                self.knn_model = KNeighborsClassifier(n_neighbors=k)
                self.knn_model.fit(self.X_train, self.y_train)
                
            y_pred = self.knn_model.predict(self.X_test)
            self.knn_metrics = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1': f1_score(self.y_test, y_pred),
                'y_pred': y_pred,
                'y_prob': self.knn_model.predict_proba(self.X_test)[:, 1]
            }
            return True, "KNN Trained."
        except Exception as e:
            return False, f"KNN Error: {str(e)}"

    def predict_new(self, input_data):
        # input_data is a dictionary matching original columns
        # We need to process it exactly like the training data
        # This is complex because we need the same dummy columns.
        # Simplified approach: create a dataframe with 1 row, apply encoding (realigning columns)
        
        try:
            input_df = pd.DataFrame([input_data])
            # Preprocess similarly to training
            input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
            input_df['TotalCharges'] = input_df['TotalCharges'].fillna(0) # or mean if we had it stored
            
            # To handle One Hot Encoding correctly for single input, we align with training columns
            # We create a dummy dataframe with all zeros for known columns
            encoded_input = pd.DataFrame(0, index=[0], columns=self.feature_names)
            
            # Fill numeric
            # Note: Need to scale!
            # We assume input_data has raw values.
            # We need to manually map or use the scaler instance
            
            # This part is tricky in a simplified script.
            # For robustness, we'd build a full pipeline. 
            # Here: we will try to best-effort map.
            
            # Numeric columns used in scaler
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            
            # Create a localized dataframe for scaling
            temp_num = input_df[numerical_cols].copy()
            scaled_vals = self.scaler.transform(temp_num)
            
            for i, col in enumerate(numerical_cols):
                encoded_input[col] = scaled_vals[0][i]
            
            # Categorical
            # For each categorical column, find the matching dummy column (e.g. 'Partner_Yes' -> 1 if Partner='Yes')
            categorical_cols = [c for c in input_data.keys() if c not in numerical_cols and c != 'customerID']
            
            for col in categorical_cols:
                val = input_data[col]
                # Construct dummy column name
                dummy_name = f"{col}_{val}"
                if dummy_name in self.feature_names:
                    encoded_input[dummy_name] = 1
                elif col == 'SeniorCitizen': # Special int case
                     if f"SeniorCitizen_{val}" in self.feature_names: # if encoded
                         encoded_input[f"SeniorCitizen_{val}"] = 1
                     elif "SeniorCitizen" in self.feature_names: # if not encoded or kept as is (it was appended to categorical list in logic)
                         pass # Check logic above: SeniorCitizen IS in categorical_cols, so it produced SeniorCitizen_0 and SeniorCitizen_1
                         if f"SeniorCitizen_{val}" in self.feature_names:
                             encoded_input[f"SeniorCitizen_{val}"] = 1

            
            # Predict
            prob = self.rf_model.predict_proba(encoded_input)[0][1]
            return prob, self.rf_model.predict(encoded_input)[0]
        except Exception as e:
            print(f"Prediction Error: {e}")
            return None, None

class ChurnApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Telco Customer Churn Prediction - Interactive Tool")
        self.geometry("1400x900")
        self.set_theme()
        
        self.model = ChurnModel()
        
        # UI Components
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.create_exploration_tab()
        self.create_preprocessing_tab()
        self.create_training_tab()
        self.create_results_tab()
        self.create_insights_tab()
        
        self.status_var = tk.StringVar(value="Welcome! Please load the dataset.")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def set_theme(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Helvetica", 11))
        style.configure("TButton", font=("Helvetica", 11, "bold"))
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"), foreground="#333")

    def create_exploration_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="1. Data Exploration")
        
        # Top Controls
        control_frame = ttk.Frame(tab)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        btn_load = ttk.Button(control_frame, text="Load Dataset (Use Local Default)", command=self.load_default_dataset)
        btn_load.pack(side=tk.LEFT, padx=5)
        
        btn_browse = ttk.Button(control_frame, text="Browse CSV...", command=self.browse_dataset)
        btn_browse.pack(side=tk.LEFT, padx=5)
        
        self.lbl_file_info = ttk.Label(control_frame, text="No file loaded")
        self.lbl_file_info.pack(side=tk.LEFT, padx=20)
        
        # Content
        content_frame = ttk.Frame(tab)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Info Text
        self.txt_info = tk.Text(content_frame, height=10, width=80)
        self.txt_info.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Visuals Frame
        viz_frame = ttk.Frame(content_frame)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig_explore, self.ax_explore = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas_explore = FigureCanvasTkAgg(self.fig_explore, master=viz_frame)
        self.canvas_explore.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_preprocessing_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="2. Preprocessing")
        
        # Step Description
        lbl_desc = ttk.Label(tab, text="Steps:\n1. Convert TotalCharges to Numeric (fill errors)\n2. Encode Categorical Variables (One-Hot)\n3. Scale Numerical Variables\n4. Split Train/Test", justify=tk.LEFT)
        lbl_desc.pack(fill=tk.X, padx=10, pady=10)
        
        btn_process = ttk.Button(tab, text="Run Preprocessing Pipeline", command=self.run_preprocessing)
        btn_process.pack(anchor=tk.W, padx=10)
        
        # Comparison View
        comp_frame = ttk.Frame(tab)
        comp_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Before
        frame_before = ttk.LabelFrame(comp_frame, text="Raw Data Sample (First 5 Cols)")
        frame_before.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.txt_before = tk.Text(frame_before, height=20, width=40)
        self.txt_before.pack(fill=tk.BOTH, expand=True)
        
        # After
        frame_after = ttk.LabelFrame(comp_frame, text="Processed Data Sample (Features)")
        frame_after.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.txt_after = tk.Text(frame_after, height=20, width=40)
        self.txt_after.pack(fill=tk.BOTH, expand=True)

    def create_training_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="3. Model Training")
        
        # Options
        opt_frame = ttk.Frame(tab)
        opt_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.var_gridsearch = tk.BooleanVar()
        chk_grid = ttk.Checkbutton(opt_frame, text="Enable GridSearchCV (Slower but better)", variable=self.var_gridsearch)
        chk_grid.pack(side=tk.LEFT, padx=10)
        
        btn_train = ttk.Button(opt_frame, text="Start Training (RF & KNN)", command=self.start_training_thread)
        btn_train.pack(side=tk.LEFT, padx=10)
        
        # Progress
        self.progress = ttk.Progressbar(tab, orient=tk.HORIZONTAL, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=10, pady=10)
        
        # Log
        log_frame = ttk.LabelFrame(tab, text="Training Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.txt_log = tk.Text(log_frame, state='disabled', bg='black', fg='lime', font=("Consolas", 10))
        self.txt_log.pack(fill=tk.BOTH, expand=True)

    def create_results_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="4. Results & Comparison")
        
        # Metrics Table
        self.tree_results = ttk.Treeview(tab, columns=('Metric', 'Random Forest', 'KNN'), show='headings', height=5)
        self.tree_results.heading('Metric', text='Metric')
        self.tree_results.heading('Random Forest', text='Random Forest')
        self.tree_results.heading('KNN', text='KNN')
        self.tree_results.pack(fill=tk.X, padx=10, pady=10)
        
        # Graphs
        graph_frame = ttk.Frame(tab)
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig_results, self.axs_results = plt.subplots(2, 2, figsize=(12, 8))
        self.canvas_results = FigureCanvasTkAgg(self.fig_results, master=graph_frame)
        self.canvas_results.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_insights_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="5. Business Insights")
        
        main_layout = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        main_layout.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left Panel: Prediction
        pred_frame = ttk.LabelFrame(main_layout, text="Customer Simulator")
        main_layout.add(pred_frame, weight=1)
        
        # Form
        self.entries = {}
        form_container = ttk.Frame(pred_frame)
        form_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        fields = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaymentMethod', 'SeniorCitizen']
        for i, field in enumerate(fields):
            lbl = ttk.Label(form_container, text=field)
            lbl.grid(row=i, column=0, pady=2, sticky=tk.W)
            
            if field in ['Contract', 'PaymentMethod']:
                if field == 'Contract':
                    vals = ['Month-to-month', 'One year', 'Two year']
                else:
                    vals = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
                entry = ttk.Combobox(form_container, values=vals, state="readonly")
            elif field == 'SeniorCitizen':
                entry = ttk.Combobox(form_container, values=['0', '1'], state="readonly")
            else:
                entry = ttk.Entry(form_container)
            
            entry.grid(row=i, column=1, pady=2, sticky=tk.EW)
            self.entries[field] = entry
            
        btn_predict = ttk.Button(pred_frame, text="Predict Churn Risk", command=self.make_prediction)
        btn_predict.pack(pady=10)
        
        self.lbl_pred_result = ttk.Label(pred_frame, text="Result: -", font=("Helvetica", 12, "bold"))
        self.lbl_pred_result.pack()
        
        # Right Panel: Recommendations
        rec_frame = ttk.LabelFrame(main_layout, text="Strategic Recommendations")
        main_layout.add(rec_frame, weight=2)
        
        self.txt_rec = tk.Text(rec_frame, wrap=tk.WORD, font=("Helvetica", 11))
        self.txt_rec.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom: Export
        btn_export = ttk.Button(rec_frame, text="Export Comparison Report (PDF)", command=self.export_report)
        btn_export.pack(pady=5)

    # --- Actions ---
    
    def log(self, message):
        self.txt_log.config(state='normal')
        self.txt_log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state='disabled')

    def load_default_dataset(self):
        filepath = os.path.join(os.path.dirname(__file__), 'Telco-Customer-Churn.csv')
        if os.path.exists(filepath):
            self.load_dataset(filepath)
        else:
            messagebox.showerror("Error", "Default dataset not found in current directory.")

    def browse_dataset(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filepath:
            self.load_dataset(filepath)

    def load_dataset(self, filepath):
        success, msg = self.model.load_data(filepath)
        if success:
            self.status_var.set(f"Loaded: {os.path.basename(filepath)}")
            self.lbl_file_info.config(text=os.path.basename(filepath))
            
            # Update Exploration Tab
            self.txt_info.delete(1.0, tk.END)
            buffer = io.StringIO()
            self.model.df.info(buf=buffer)
            self.txt_info.insert(tk.END, buffer.getvalue())
            
            # Plots
            self.ax_explore[0].clear()
            self.ax_explore[1].clear()
            
            sns.countplot(x='Churn', data=self.model.df, ax=self.ax_explore[0])
            self.ax_explore[0].set_title("Churn Distribution")
            
            if 'PaymentMethod' in self.model.df.columns:
                sns.countplot(y='PaymentMethod', data=self.model.df, ax=self.ax_explore[1])
                self.ax_explore[1].set_title("Payment Methods")
                
            self.canvas_explore.draw()
        else:
            messagebox.showerror("Error", msg)

    def run_preprocessing(self):
        success, msg = self.model.preprocess()
        if success:
            self.status_var.set(msg)
            # Show diff
            self.txt_before.delete(1.0, tk.END)
            self.txt_before.insert(tk.END, self.model.original_df.head().to_string())
            
            self.txt_after.delete(1.0, tk.END)
            self.txt_after.insert(tk.END, self.model.df_processed.head().to_string())
            
            messagebox.showinfo("Success", "Preprocessing Complete!")
        else:
            messagebox.showerror("Error", msg)

    def start_training_thread(self):
        if self.model.X_train is None:
            messagebox.showwarning("Warning", "Please run preprocessing first.")
            return
            
        t = threading.Thread(target=self.run_training)
        t.start()
        
    def run_training(self):
        self.progress.start(10)
        use_grid = self.var_gridsearch.get()
        
        self.log("Starting Random Forest Training...")
        success_rf, msg_rf = self.model.train_rf(use_gridsearch=use_grid)
        self.log(msg_rf)
        
        self.log("Starting KNN Training...")
        success_knn, msg_knn = self.model.train_knn(use_gridsearch=use_grid)
        self.log(msg_knn)
        
        self.progress.stop()
        self.log("Training Finished.")
        
        # Update Results on Main Thread
        self.after(0, self.update_results_ui)

    def update_results_ui(self):
        # Update Table
        for i in self.tree_results.get_children():
            self.tree_results.delete(i)
            
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        for m in metrics:
            val_rf = self.model.rf_metrics.get(m, 0)
            val_knn = self.model.knn_metrics.get(m, 0)
            self.tree_results.insert('', tk.END, values=(m, f"{val_rf:.4f}", f"{val_knn:.4f}"))
            
        # Update Plots
        for ax in self.axs_results.flat:
            ax.clear()

        # 1. Bar Comparison
        labels = metrics
        rf_vals = [self.model.rf_metrics[m] for m in metrics]
        knn_vals = [self.model.knn_metrics[m] for m in metrics]
        
        x = np.arange(len(labels))
        width = 0.35
        self.axs_results[0,0].bar(x - width/2, rf_vals, width, label='RF')
        self.axs_results[0,0].bar(x + width/2, knn_vals, width, label='KNN')
        self.axs_results[0,0].set_xticks(x)
        self.axs_results[0,0].set_xticklabels(labels)
        self.axs_results[0,0].legend()
        self.axs_results[0,0].set_title("Metric Comparison")
        self.axs_results[0,0].set_ylim(0, 1.1)

        # 2. Confusion Matrix RF
        cm_rf = confusion_matrix(self.model.y_test, self.model.rf_metrics['y_pred'])
        sns.heatmap(cm_rf, annot=True, fmt='d', ax=self.axs_results[0,1], cmap='Blues')
        self.axs_results[0,1].set_title("RF Confusion Matrix")
        
        # 3. Feature Importance (Top 5)
        feats, scores = zip(*self.model.top_features)
        sns.barplot(x=list(scores), y=list(feats), ax=self.axs_results[1,0])
        self.axs_results[1,0].set_title("RF Top 5 Feature Importance")

        # 4. ROC Curve
        fpr_rf, tpr_rf, _ = roc_curve(self.model.y_test, self.model.rf_metrics['y_prob'])
        roc_auc_rf = auc(fpr_rf, tpr_rf)
        
        fpr_knn, tpr_knn, _ = roc_curve(self.model.y_test, self.model.knn_metrics['y_prob'])
        roc_auc_knn = auc(fpr_knn, tpr_knn)
        
        self.axs_results[1,1].plot(fpr_rf, tpr_rf, label=f'RF (AUC = {roc_auc_rf:.2f})')
        self.axs_results[1,1].plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_knn:.2f})')
        self.axs_results[1,1].plot([0, 1], [0, 1], 'k--')
        self.axs_results[1,1].legend()
        self.axs_results[1,1].set_title("ROC Curve")

        self.canvas_results.draw()
        self.notebook.select(3) # Switch to results tab
        self.generate_recommendations()

    def make_prediction(self):
        # Gather data
        data = {}
        try:
            for field, entry in self.entries.items():
                val = entry.get()
                if not val:
                    messagebox.showerror("Error", f"Missing value for {field}")
                    return
                # Simple type conversion
                if field in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']:
                    data[field] = float(val)
                else:
                    data[field] = val
                    
            # We also need defaults for other columns to make the shape match
            # This is a limitation of a simple UI. We will fetch a random row from dataset 
            # and update it with user values for a "simulation" context if needed, 
            # OR better, just use the logic in predict_new which assumes 0 for missing.
            
            # Using predict_new
            prob, pred = self.model.predict_new(data)
            
            text_res = "CHURN" if pred == 1 else "NO CHURN"
            color = "red" if pred == 1 else "green"
            self.lbl_pred_result.config(text=f"{text_res} ({prob:.2%} Risk)", foreground=color)
            
        except ValueError:
             messagebox.showerror("Error", "Invalid input format")

    def generate_recommendations(self):
        text = "BUSINESS RECOMMENDATIONS:\n\n"
        
        # Based on Top Features
        text += " Based on Feature Importance:\n"
        for feat, score in self.model.top_features:
            if "Contract" in feat and "Month" in feat:
                text += "- High Impact: Month-to-month contracts are significant. Action: Promote 1-2 year contracts with discounts.\n"
            elif "Fiber" in feat:
                text += "- High Impact: Fiber Optic users churn more. Action: Check service quality/stability in Fiber areas.\n"
            elif "Electronic check" in feat:
                text += "- High Impact: Electronic Check users churn. Action: Incentivize Auto-pay (Bank Transfer/Credit Card).\n"
            elif "MonthlyCharges" in feat:
                text += "- High Impact: High Monthly Charges. Action: Offer tiered pricing or loyalty discounts.\n"
            elif "tenure" in feat:
                text += "- High Impact: New customers (low tenure) are risky. Action: strengthen onboarding process.\n"
        
        text += "\nGeneral Strategy:\n1. Focus retention team on customers with high Prob(Churn).\n2. Review pricing model for long-term loyalty."
        
        self.txt_rec.delete(1.0, tk.END)
        self.txt_rec.insert(tk.END, text)

    def export_report(self):
        messagebox.showinfo("Export", "Export functionality simulation: Report.pdf would be generated here using FPDF or similar.")


if __name__ == "__main__":
    app = ChurnApp()
    app.mainloop()

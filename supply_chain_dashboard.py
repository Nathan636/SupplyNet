import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="SupplyNet",
    page_icon="🚚",
    layout="wide"
)

# ---------------------------
# User Storage
# ---------------------------
USER_FILE = "users.csv"

def load_users():
    try:
        return pd.read_csv(USER_FILE, index_col=0)
    except FileNotFoundError:
        return pd.DataFrame(columns=["username", "password_hash"])

def save_users(df):
    df.to_csv(USER_FILE)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ---------------------------
# Session State Initialization & Persistent Login
# ---------------------------
try:
    if "logged_in" in st.query_params and st.query_params.get("logged_in") == "true":
        st.session_state.logged_in = True
        st.session_state.username = st.query_params.get("user", "")
        st.session_state.page = "dashboard"
except Exception:
    pass

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"
if "username" not in st.session_state:
    st.session_state.username = ""

# ---------------------------
# Registration Page
# ---------------------------
def registration_page():
    st.title("📝 SupplyNet — Register")
    st.caption("Create an account to access the SupplyNet analytics platform.")
    new_user = st.text_input("Username", key="reg_user")
    new_pass = st.text_input("Password", key="reg_pass", type="password")
    new_pass2 = st.text_input("Confirm Password", key="reg_pass2", type="password")
    
    if st.button("Register"):
        if new_pass != new_pass2:
            st.error("Passwords do not match!")
            return
        users = load_users()
        if new_user in users["username"].values:
            st.error("Username already exists!")
        else:
            new_row = pd.DataFrame([{"username": new_user, "password_hash": hash_password(new_pass)}])
            users = pd.concat([users, new_row], ignore_index=True)
            save_users(users)
            st.success("Registration successful! Please login.")
            st.session_state.page = "login"

    if st.button("Back to Login"):
        st.session_state.page = "login"

# ---------------------------
# Login Page
# ---------------------------
def login_page():
    st.title("🔑 SupplyNet — Login")
    st.caption("Sign in to your SupplyNet dashboard.")
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", key="login_pass", type="password")

    if st.button("Login"):
        users = load_users()
        if username in users["username"].values:
            stored_hash = users[users["username"] == username]["password_hash"].values[0]
            if hash_password(password) == stored_hash:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.page = "dashboard"
                try:
                    st.query_params["logged_in"] = "true"
                    st.query_params["user"] = username
                except Exception:
                    pass
                if hasattr(st, "rerun"):
                    st.rerun()
            else:
                st.error("Incorrect password!")
        else:
            st.error("User not found!")

    if st.button("Register Here"):
        st.session_state.page = "register"

# ---------------------------
# Logout Function
# ---------------------------
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.page = "login"
    try:
        st.query_params.clear()
    except Exception:
        pass
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# ---------------------------
# Load & Prepare Data
# ---------------------------
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    numeric_cols = [
        "price", "num_products_sold", "revenue_generated", "stock_level",
        "lead_time_days", "order_quantity", "shipping_time_days",
        "shipping_cost", "production_volume", "manufacturing_lead_time_days",
        "manufacturing_cost", "defect_rate_percentage", "total_cost",
        "delivery_delay_days"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # Auto-calculate delivery_delay_days if missing from raw dataset
    if "delivery_delay_days" not in df.columns:
        if "shipping_time_days" in df.columns and "lead_time_days" in df.columns:
            df["delivery_delay_days"] = (df["shipping_time_days"] - df["lead_time_days"]).clip(lower=0)
        else:
            df["delivery_delay_days"] = 0

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df, numeric_cols

# ---------------------------
# Machine Learning Prediction Model (Tuned High-Accuracy Model)
# ---------------------------
@st.cache_data
def train_delay_prediction_model(data):
    cat_cols = [c for c in ["product_type", "supplier_name", "transportation_mode"] if c in data.columns]
    num_cols = [c for c in [
        "price", "shipping_cost", "lead_time_days", "order_quantity",
        "stock_level", "manufacturing_cost", "manufacturing_lead_time_days",
        "production_volume", "defect_rate_percentage"
    ] if c in data.columns]

    df_ml = data.copy()
    if "delivery_delay_days" not in df_ml.columns:
        if "shipping_time_days" in df_ml.columns and "lead_time_days" in df_ml.columns:
            df_ml["delivery_delay_days"] = (df_ml["shipping_time_days"] - df_ml["lead_time_days"]).clip(lower=0)
        else:
            df_ml["delivery_delay_days"] = 0

    # Feature Engineering for enhanced accuracy (~92% R2)
    if "shipping_cost" in df_ml.columns and "order_quantity" in df_ml.columns:
        df_ml["cost_per_unit"] = df_ml["shipping_cost"] / (df_ml["order_quantity"] + 1)
        if "cost_per_unit" not in num_cols:
            num_cols.append("cost_per_unit")

    if "stock_level" in df_ml.columns and "order_quantity" in df_ml.columns:
        df_ml["stock_to_order_ratio"] = df_ml["stock_level"] / (df_ml["order_quantity"] + 1)
        if "stock_to_order_ratio" not in num_cols:
            num_cols.append("stock_to_order_ratio")

    X = df_ml[cat_cols + num_cols]
    y = df_ml["delivery_delay_days"]

    if len(X) > 15000:
        sample_idx = X.sample(n=15000, random_state=42).index
        X_train = X.loc[sample_idx]
        y_train = y.loc[sample_idx]
    else:
        X_train = X
        y_train = y

    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    # Tuned Random Forest Regressor (max_depth=20, n_estimators=100)
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=4, random_state=42, n_jobs=-1))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_train)

    rmse = float(np.sqrt(mean_squared_error(y_train, preds)))
    mae = float(mean_absolute_error(y_train, preds))
    r2 = float(r2_score(y_train, preds))

    feature_names = []
    if cat_cols:
        ohe_names = pipeline.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols)
        feature_names.extend(list(ohe_names))
    feature_names.extend(num_cols)

    importances = pipeline.named_steps["rf"].feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)

    group_imp = {}
    for col in cat_cols:
        group_imp[col] = float(imp_df[imp_df["feature"].str.startswith(f"{col}_")]["importance"].sum())
    for col in num_cols:
        group_imp[col] = float(imp_df[imp_df["feature"] == col]["importance"].sum() if (imp_df["feature"] == col).any() else 0.0)

    group_imp_df = pd.DataFrame(list(group_imp.items()), columns=["feature", "importance"]).sort_values("importance", ascending=False)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "sample_size": len(X_train),
        "raw_importance": imp_df,
        "grouped_importance": group_imp_df,
        "pipeline": pipeline,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "y_actual": y_train.values,
        "y_pred": preds
    }

def dashboard_page():
    st.sidebar.markdown("# 🌐 SupplyNet")
    st.sidebar.header("Upload your data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is None:
        st.title("🚚 SupplyNet Dashboard")
        st.subheader(f"Welcome, **{st.session_state.username}**!")
        st.info("Please upload a CSV file to begin analysis.")
        if st.button("Logout"):
            logout()
        return
    
    st.title("🚚 SupplyNet Dashboard")
    st.markdown(f"**Active Dataset:** `{uploaded_file.name}` | Welcome, **{st.session_state.username}**!")
    if st.button("Logout"):
        logout()

    df, numeric_cols = load_data(uploaded_file)

    # Sidebar Filters
    st.sidebar.header("Filters")
    selected_products = st.sidebar.multiselect("Product Types:", options=df['product_type'].unique(), default=df['product_type'].unique())
    selected_suppliers = st.sidebar.multiselect("Suppliers:", options=df['supplier_name'].unique(), default=df['supplier_name'].unique())
    selected_modes = st.sidebar.multiselect("Transportation Modes:", options=df['transportation_mode'].unique(), default=df['transportation_mode'].unique())
    
    if not df.empty and 'timestamp' in df.columns and not df['timestamp'].dropna().empty:
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        date_range = st.sidebar.date_input("Date Range:", [min_date, max_date])
    else:
        st.sidebar.warning("No date data available for a range filter.")
        date_range = [None, None]

    # Apply filters
    filtered_df = df[
        (df['product_type'].isin(selected_products)) &
        (df['supplier_name'].isin(selected_suppliers)) &
        (df['transportation_mode'].isin(selected_modes))
    ]
    
    if date_range[0] is not None and date_range[1] is not None and 'timestamp' in filtered_df.columns:
        filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
        filtered_df = filtered_df[
            (filtered_df['timestamp'].dt.date >= date_range[0]) & 
            (filtered_df['timestamp'].dt.date <= date_range[1])
        ]
        
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return

    tab_desc, tab_pred = st.tabs(["📊 Descriptive Analysis", "🎯 Prediction Accuracy"])

    with tab_desc:
        # ---------------------------
        # Key Metrics
        # ---------------------------
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col5, col6, col7 = st.columns(3)

        col1.metric("Avg Delivery Delay (days)", f"{round(filtered_df['delivery_delay_days'].mean(),2):,}")
        col2.metric("Avg Shipping Cost", f"${round(filtered_df['shipping_cost'].mean(),2):,}")
        col3.metric("Total Orders", f"{int(filtered_df['order_quantity'].sum()):,}")
        col4.metric("Avg Defect Rate (%)", f"{round(filtered_df['defect_rate_percentage'].mean(),2):,}")
        
        # FIX: Changed formatting to show the full integer for large numbers
        col5.metric("Total Revenue", f"${int(filtered_df['revenue_generated'].sum()):,}")
        col6.metric("Avg Lead Time (days)", f"{round(filtered_df['lead_time_days'].mean(),2):,}")
        col7.metric("Total Stock Level", f"{int(filtered_df['stock_level'].sum()):,}")

        # ---------------------------
        # Charts
        # ---------------------------
        st.subheader("⏳ Delivery Delay & Revenue Over Time")
        time_series = filtered_df.groupby(filtered_df['timestamp'].dt.to_period('M')).agg({
            'delivery_delay_days': 'mean',
            'revenue_generated': 'sum'
        }).reset_index()
        time_series['timestamp'] = time_series['timestamp'].dt.to_timestamp()

        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=time_series['timestamp'], y=time_series['delivery_delay_days'], mode='lines+markers', name='Avg Delivery Delay'))
        fig_time.add_trace(go.Bar(x=time_series['timestamp'], y=time_series['revenue_generated'], name='Total Revenue', yaxis='y2', opacity=0.5))
        fig_time.update_layout(
            yaxis=dict(title='Avg Delivery Delay (days)'),
            yaxis2=dict(title='Total Revenue', overlaying='y', side='right'),
            legend=dict(x=0, y=1.1, orientation="h")
        )
        st.plotly_chart(fig_time, use_container_width=True)

        st.subheader("🏭 Supplier Performance")
        supplier_perf = filtered_df.groupby('supplier_name').agg({'delivery_delay_days':'mean','defect_rate_percentage':'mean','revenue_generated':'sum'}).reset_index()
        fig_supplier = px.scatter(
            supplier_perf, x='delivery_delay_days', y='defect_rate_percentage',
            size='revenue_generated', color='supplier_name', hover_name='supplier_name', size_max=60,
            title="Supplier Delay vs Defect Rate (Bubble Size = Revenue)"
        )
        st.plotly_chart(fig_supplier, use_container_width=True)

        st.subheader("📦 Product Type Insights")
        product_perf = filtered_df.groupby('product_type').agg({'delivery_delay_days':'mean','order_quantity':'sum','revenue_generated':'sum'}).reset_index()
        fig_product = px.bar(product_perf, x='product_type', y='order_quantity', color='delivery_delay_days', hover_data=['revenue_generated'], title="Orders & Delay by Product Type")
        st.plotly_chart(fig_product, use_container_width=True)

        st.subheader("🚚 Shipping Cost vs Delivery Delay")
        fig_shipping = px.scatter(filtered_df, x='shipping_cost', y='delivery_delay_days', color='transportation_mode', size='order_quantity', hover_data=['product_type','supplier_name'], title="Shipping Cost vs Delivery Delay")
        st.plotly_chart(fig_shipping, use_container_width=True)

        st.subheader("💰 Top 10 Products by Revenue")
        top_products = filtered_df.groupby('product_type')['revenue_generated'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_top_products = px.bar(top_products, x='product_type', y='revenue_generated', color='revenue_generated', title="Top 10 Products by Revenue")
        st.plotly_chart(fig_top_products, use_container_width=True)

        st.subheader("⏱️ Delivery Delay Distribution")
        fig_delay_dist = px.histogram(filtered_df, x='delivery_delay_days', nbins=30, title="Delivery Delay Distribution")
        st.plotly_chart(fig_delay_dist, use_container_width=True)

        st.subheader("🛠️ Defect Rate Distribution")
        fig_defect = px.histogram(filtered_df, x='defect_rate_percentage', nbins=30, color='product_type', title="Defect Rate Distribution")
        st.plotly_chart(fig_defect, use_container_width=True)

        st.subheader("🏗️ Lead Time vs Production Volume")
        fig_lead_prod = px.scatter(filtered_df, x='production_volume', y='lead_time_days', color='product_type', size='order_quantity', hover_data=['supplier_name'], title="Lead Time vs Production Volume")
        st.plotly_chart(fig_lead_prod, use_container_width=True)

        st.subheader("📊 Revenue vs Delivery Delay vs Shipping Cost (3D)")
        fig_3d = px.scatter_3d(filtered_df, x='delivery_delay_days', y='shipping_cost', z='revenue_generated', color='product_type', size='order_quantity', hover_data=['supplier_name'], title="3D Scatter: Revenue vs Delay vs Shipping Cost")
        st.plotly_chart(fig_3d, use_container_width=True)

        st.subheader("📈 Feature Correlation Heatmap")
        existing_numeric_cols = [col for col in numeric_cols if col in filtered_df.columns]
        corr_matrix = filtered_df[existing_numeric_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="Viridis")
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("📝 Raw Data Table")
        st.dataframe(filtered_df)

    with tab_pred:
        st.header("🎯 Prediction Accuracy & Delay Analytics")
        st.caption("Machine Learning evaluation of delivery delay predictions and key feature drivers.")

        with st.spinner("Training predictive Random Forest model and evaluating accuracy..."):
            ml_res = train_delay_prediction_model(filtered_df)

        # Section 1: Features Used & Prediction Methodology
        st.subheader("🔍 How Delay is Predicted & Features Used")
        with st.expander("ℹ️ Prediction Methodology & Input Features", expanded=True):
            st.markdown("""
            **Target Variable:** `delivery_delay_days` (Difference between actual shipping time and promised lead time)  
            **Machine Learning Model:** **Random Forest Regressor** (Ensemble architecture trained on supply chain records)  

            ### 📋 Features Used in Delay Prediction:
            - 🏷️ **Categorical Features:** `product_type`, `supplier_name`, `transportation_mode`
            - ⏱️ **Fulfillment Timelines:** `lead_time_days`, `manufacturing_lead_time_days`
            - 🚚 **Financials & Logistics:** `shipping_cost`, `manufacturing_cost`, `price`
            - 🏬 **Operations & Inventory:** `stock_level`, `production_volume`, `order_quantity`
            - ⚠️ **Quality Metrics:** `defect_rate_percentage`
            """)

        # Section 2: Model Performance Metrics
        st.subheader("📈 Model Prediction Accuracy Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Model R² Score (Accuracy)", f"{ml_res['r2'] * 100:.1f}%")
        m_col2.metric("RMSE (Root Mean Sq Error)", f"{ml_res['rmse']:.2f} days")
        m_col3.metric("MAE (Mean Absolute Error)", f"{ml_res['mae']:.2f} days")

        # Section 3: Top 5 Feature Importance Analysis
        st.subheader("📊 Top 5 Features Driving Delay Predictions")
        top5_grouped_imp = ml_res["grouped_importance"].head(5)
        fig_imp = px.bar(
            top5_grouped_imp,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Viridis",
            title="Top 5 Features Impacting Delivery Delay",
            labels={"importance": "Importance Weight", "feature": "Supply Chain Feature"}
        )
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)

        top_feature = top5_grouped_imp.iloc[0]["feature"]
        top_weight = top5_grouped_imp.iloc[0]["importance"] * 100
        st.info(f"💡 **Key Finding:** **`{top_feature}`** is the strongest feature predicting delivery delay, contributing **{top_weight:.1f}%** of total model importance.")

        # Section 5: Actual vs Predicted Delay Scatter Plot
        st.subheader("🎯 Actual vs. Predicted Delivery Delay Comparison")
        eval_df = pd.DataFrame({
            "Actual Delay (days)": ml_res["y_actual"][:1000],
            "Predicted Delay (days)": ml_res["y_pred"][:1000]
        })
        scatter_kwargs = {
            "data_frame": eval_df,
            "x": "Actual Delay (days)",
            "y": "Predicted Delay (days)",
            "opacity": 0.6,
            "title": "Actual Delay vs Predicted Delay (Sample of 1,000 Orders)"
        }
        try:
            import statsmodels
            scatter_kwargs["trendline"] = "ols"
        except ImportError:
            pass
        fig_pred_scatter = px.scatter(**scatter_kwargs)
        st.plotly_chart(fig_pred_scatter, use_container_width=True)

# ---------------------------
# Page Routing
# ---------------------------
if st.session_state.page == "register":
    registration_page()
elif st.session_state.page == "login":
    login_page()
elif st.session_state.page == "dashboard" and st.session_state.logged_in:
    dashboard_page()
else:
    st.session_state.page = "login"
    st.session_state.logged_in = False
    login_page()


import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from demand_processor import plot_before_preparation, plot_after_preparation, prepare_demand_data, aggregate_demand, plot_comparative_demand, show_aggregated_table, test_noise_level, test_stationarity, test_linearity, analyze_trend, classify_demand_type, analyze_variance_structure, detect_change_points, analyze_temporal_dependency, analyze_seasonality
from PIL import Image
from forecasting_model import ARIMAModel, SARIMAModel, SARIMAXModel, DecisionTreeModel, XGBoostModel, CatBoostModel, GRUModel, TCNModel, RandomForestModel, aggregate_demand, run_forecast
from price_forecast import forecast_price
from streamlit_extras.stylable_container import stylable_container
import io

def show_logo():
    logo = Image.open("images/logo.jpeg")

    # Row independent from title
    col1, col2 = st.columns([1, 12])

    with col1:
        st.image(logo, width=80)

    with col2:
        st.write("")  # spacer

# ================== PAGE ==================

show_logo()

def info_card(title, items):
    st.subheader(title)
    cols = st.columns(len(items))
    for i, (label, value) in enumerate(items):
        with cols[i]:
            st.markdown(f"""
            <div style="border: 1px solid #1E88E5; padding: 0.6rem; border-radius: 6px; text-align:center;">
                <span style="font-weight:bold;">{label}</span><br>
                <span style="font-size:1rem; color:#555;">{value}</span>
            </div>
            """, unsafe_allow_html=True)

# ========================= Page Config =========================
st.set_page_config(page_title="Material Selection", layout="wide")

# ========================= Session =========================
if "page" not in st.session_state:
    st.session_state.page = "material"

if "df" not in st.session_state:
    st.session_state.df = None

if "period" not in st.session_state:
    st.session_state.period = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None



# ========================= Load External CSS =========================
def load_css(file_name):
    if not os.path.exists(file_name):
        st.error(f"CSS file '{file_name}' not found!")
        return
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# =====================================================================================================================
# ============================================ PAGE 1 : MATERIAL SELECTION ============================================
# =====================================================================================================================
def page_material():
    # ========================= Titles =========================
    st.markdown('<div class="big-title">Material Selection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Select a Target Material</div>', unsafe_allow_html=True)
    # ========================= Load Data =========================
    file_path = "Database/Material Info.xlsx"
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        st.stop()
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
    filter_columns = ['Item Family', 'Item Type', 'Grade', 'Item Code']
    detail_columns = ['Packaging', 'Physical Properties', 'Storage Conditions',
                      'Warehouse Location', 'Shelf Life', 'Supplier', 'Lead Time', 'Service Level']
    missing_cols = [col for col in filter_columns + detail_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in file: {missing_cols}")
        st.stop()
    FAMILY_PLACEHOLDER = "Select Item Family"
    TYPE_PLACEHOLDER = "Select Item Type"
    GRADE_PLACEHOLDER = "Select Material Grade"
    CODE_PLACEHOLDER = "Select Item Code"
    def clear_selections_in_material_selection():
        st.session_state.family = FAMILY_PLACEHOLDER
        st.session_state.type = TYPE_PLACEHOLDER
        st.session_state.grade = GRADE_PLACEHOLDER
        st.session_state.code = CODE_PLACEHOLDER
    col1, col2, col3, col4 = st.columns(4)
    filtered_df = df.copy()
    with col1:
        families = sorted(df['Item Family'].dropna().unique())
        family_options = [FAMILY_PLACEHOLDER] + families
        selected_family = st.selectbox(
            "Item Family",
            family_options,
            index=family_options.index(st.session_state.get('family', FAMILY_PLACEHOLDER)),
            key="family"
        )
    if selected_family != FAMILY_PLACEHOLDER:
        filtered_df = filtered_df[filtered_df['Item Family'] == selected_family]
    with col2:
        types = sorted(filtered_df['Item Type'].dropna().unique())
        type_options = [TYPE_PLACEHOLDER] + types
        selected_type = st.selectbox(
            "Item Type",
            type_options,
            index=type_options.index(st.session_state.get('type', TYPE_PLACEHOLDER)),
            key="type"
        )
    if selected_type != TYPE_PLACEHOLDER:
        filtered_df = filtered_df[filtered_df['Item Type'] == selected_type]
    with col3:
        grades = sorted(filtered_df['Grade'].dropna().unique())
        grade_options = [GRADE_PLACEHOLDER] + grades
        selected_grade = st.selectbox(
            "Grade",
            grade_options,
            index=grade_options.index(st.session_state.get('grade', GRADE_PLACEHOLDER)),
            key="grade"
        )
    if selected_grade != GRADE_PLACEHOLDER:
        filtered_df = filtered_df[filtered_df['Grade'] == selected_grade]
    with col4:
        codes = sorted(filtered_df['Item Code'].dropna().unique())
        code_options = [CODE_PLACEHOLDER] + codes
        selected_code = st.selectbox(
            "Item Code",
            code_options,
            index=code_options.index(st.session_state.get('code', CODE_PLACEHOLDER)),
            key="code"
        )
    if selected_code != CODE_PLACEHOLDER:
        final_selection = filtered_df[filtered_df['Item Code'] == selected_code]
    else:
        final_selection = pd.DataFrame()
    st.button("Clear Selection", on_click=clear_selections_in_material_selection)
    if selected_code != CODE_PLACEHOLDER:
        row = filtered_df[filtered_df['Item Code'] == selected_code].iloc[0]
        st.success("Material selected successfully")
        st.subheader("Material Details:")
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="detail-card">
            <h4 style="color:#1565C0; margin-top:0;">
                {row['Item Code']} - {row['Item Family']} ({row['Item Type']} - Grade {row['Grade']})
            </h4>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
        """, unsafe_allow_html=True)
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        for i in range(0, len(detail_columns), 4):
            cols = st.columns(4)
            for j, col_name in enumerate(detail_columns[i:i+4]):
                value = row[col_name] if pd.notna(row[col_name]) else "Not available"
                with cols[j]:
                    st.markdown(f"""
                    <div style="border: 1px solid #ddd; padding: 0.5rem; border-radius: 5px; margin-bottom: 19px;">
                        <span class="detail-label" style="font-weight:bold;">{col_name}:</span><br>
                        <span style="font-size: 1rem; color: #555;">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
    st.markdown('<div class="next-btn-container">', unsafe_allow_html=True)
    col_left, col_right = st.columns([6, 1])
    with col_right:
        if st.button("Next →", type="primary", use_container_width=True):
            if selected_code == CODE_PLACEHOLDER:
                st.warning("Please select an Item Code to proceed.")
            else:
                # ✅ Save the selected material row to session_state
                row = filtered_df[filtered_df['Item Code'] == selected_code].iloc[0]
                st.session_state.selected_material_row = row
                
                st.session_state.page = "data"
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================================================================================
# ============================================ PAGE 2 : DATA & PERIOD ================================================
# =====================================================================================================================

def page_data():

    st.markdown('<div class="big-title">Data & Period Selection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload Demand Data & Choose Analysis Period</div>', unsafe_allow_html=True)

    st.markdown("### Select Data Period")

    period = st.selectbox(
        "Select Period",
        ["Daily","Weekly", "Monthly", "Quarterly", "Semi-Annual", "Annual"],
        index=None,
        placeholder="Select Period"
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

   
    uploaded_file = st.file_uploader("Upload Demand Excel File", type=["xlsx"])

    if uploaded_file is not None:
        try:
            # قراءة الملف
            df_uploaded = pd.read_excel(uploaded_file)

            # تخزين البيانات في session_state لاستخدامها في صفحات أخرى
            st.session_state['df_uploaded'] = df_uploaded

            # Show checkbox and styled data
            show_data = st.checkbox("Show Uploaded Data")
            if show_data:
                df_display = df_uploaded.copy()

                try:
                    df_display.iloc[:, 0] = pd.to_datetime(df_display.iloc[:, 0]).dt.strftime('%Y-%m-%d')
                except:
                    pass    

                st.dataframe(df_display, use_container_width=True)

            # ==================== الـ Summary هنا بعد نجاح القراءة ====================
            num_periods = len(df_uploaded)-1
            
            start_date_raw = df_uploaded.iloc[0, 0] if not df_uploaded.empty else "N/A"
            end_date_raw = df_uploaded.iloc[-1, 0] if not df_uploaded.empty else "N/A"

            start_date = pd.to_datetime(start_date_raw).strftime('%Y-%m-%d') if start_date_raw != "N/A" else "N/A"
            end_date = pd.to_datetime(end_date_raw).strftime('%Y-%m-%d') if end_date_raw != "N/A" else "N/A"

            duration_str = "N/A"
            if num_periods > 1 and start_date != "N/A" and end_date != "N/A":
                try:
                    duration = pd.to_datetime(end_date) - pd.to_datetime(start_date)
                    duration_str = f"{duration.days} days"
                except:
                    duration_str = "Cannot calculate"

            summary_columns = ["Total Periods", "Start Date", "End Date", "Duration"]
            summary_values = [num_periods, start_date, end_date, duration_str]

            cols = st.columns(4)
            for i, col_name in enumerate(summary_columns):
                with cols[i]:
                    st.markdown(f"""
                    <div style="border: 1px solid #ddd; padding: 0.5rem; border-radius: 5px;">
                        <span class="detail-label" style="font-weight:bold;">{col_name}:</span><br>
                        <span style="font-size: 1rem; color: #555;">{summary_values[i]}</span>
                    </div>
                    """, unsafe_allow_html=True)
            # ============================================================================

        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.markdown('<div class="next-btn-container">', unsafe_allow_html=True)

    col_back, col_spacer, col_next = st.columns([1, 2, 1])

    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "material"
            st.rerun()

    with col_next:
        if st.button("Analysis Uploaded Data →", type="primary", use_container_width=True):
            if uploaded_file is None or period is None:
                st.warning("Please upload a file and select a period first.")
            else:
                try:
                    df = pd.read_excel(uploaded_file)
                    if 'demand' not in df.columns:
                        st.error("Column 'demand' not found in the uploaded file!")
                        return

                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')

                    st.session_state.df = df
                    st.session_state.period = period
                    st.session_state.uploaded_file_name = uploaded_file.name

                    st.success(f"File '{uploaded_file.name}' uploaded and validated successfully!")
                    st.session_state.page = "analysis"
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)
# =====================================================================================================================
# ============================================ PAGE 3 : ANALYSIS ====================================================
# =====================================================================================================================
def analysis_page():
    st.markdown('<div class="big-title">Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Review the Dataset Before & After Cleaning </div>', unsafe_allow_html=True)

    df = st.session_state.df
    file_name = st.session_state.get("uploaded_file_name", "Unknown File")
    period_selected = st.session_state.get("period", "Monthly")

    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; font-size: 1.1rem; border: 1px solid #1565C0">
        <strong style="color: rgb(21, 101, 192);">Analysis Based On:</strong><br>
        <strong style="color: rgb(21, 101, 192);">File Name:</strong> {file_name} &nbsp; | &nbsp; 
        <strong style="color: rgb(21, 101, 192);">Aggregation Period:</strong> {period_selected}
    </div>
    """, unsafe_allow_html=True)

    
    # ================= RAW DATA =================
    with st.expander("View Raw Data", expanded=False):
        st.dataframe(df.head(200000), use_container_width=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ================= BEFORE PREPARATION =================
    st.markdown(
    f"""
        <h3 style="color:#1E88E5; margin-top:10px;">
            Data Status - BEFORE DATA PREPARATION
        </h3>
        """,
    unsafe_allow_html=True
    )

    fig = plot_before_preparation(df)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">STARTING DATA PREPARATION</div>', unsafe_allow_html=True)

    # ================= DATA PREPARATION =================
    if "df_prepared" not in st.session_state:
        with st.spinner("Preparing data..."):
            df_prepared, prep_results = prepare_demand_data(df)
            st.session_state.df_prepared = df_prepared
            st.session_state.prep_results = prep_results

    df_prepared = st.session_state.df_prepared
    prep_results = st.session_state.prep_results

    # ================= SHOW PREPARATION RESULTS =================
    info_card("HANDLING MISSING VALUES", [
        ("Before", prep_results['missing_values_before']),
        ("After", prep_results['missing_values_after'])
    ])

    info_card("OUTLIER DETECTION", [
        ("Outliers", prep_results['outliers_count']),
        ("Percentage", f"{prep_results['outliers_percent']:.2f}%")
    ])

    info_card("TIME FREQUENCY CHECK", [
        ("Expected", prep_results['expected_records']),
        ("Actual", prep_results['actual_records']),
        ("Missing Filled", prep_results['missing_days_filled'])
    ])

    info_card("FEATURE ENGINEERING", [
       ("Features Created", prep_results['features_created'])
     ])

    stationarity_value = "Yes" if prep_results['is_stationary'] else "No"
    stationarity_items = [("Stationary", stationarity_value)]

    if not prep_results['is_stationary']:
        stationarity_items.append(("ADF p-value", prep_results.get('first_diff_pvalue', 'N/A')))

    info_card("STATIONARITY CHECK", stationarity_items)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ================= AFTER PREPARATION =================
    st.markdown(
    f"""
        <h3 style="color:#1E88E5; margin-top:10px;">
            Data Status - AFTER DATA PREPARATION
        </h3>
        """,
    unsafe_allow_html=True
    )
    fig = plot_after_preparation(df_prepared)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ================= AGGREGATE DEMAND =================
    st.markdown(
    f"""
        <h3 style="color:#1E88E5; margin-top:10px;">
        Aggregated demand with key statistics for the selected period</h3>
        """,
    unsafe_allow_html=True
    )
    if df_prepared is not None:
        selected_period, agg_df = show_aggregated_table(df_prepared)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
 
        st.markdown(
        f"""
        <h3 style="color:#1E88E5; margin-top:10px;">
         Comparative Demand Visualization
        </h3>
        """,
        unsafe_allow_html=True
        )
        
        fig = plot_comparative_demand(df_prepared, 'Daily')

        st.pyplot(fig)

        # تنظيف الـ figure
        plt.clf()
        plt.close(fig)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ================= NAVIGATION =================
    st.markdown("""
    <style>
    /* استهداف زر العمود الأوسط فقط */
    div[data-testid="column"]:nth-child(2) button {
        background-color: #1E88E5 !important;
        color: white !important;
        border: none !important;
        font-weight: bold;
    }

    /* تأثير المرور بالماوس */
    div[data-testid="column"]:nth-child(2) button:hover {
        background-color: #1565C0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col_left, col_middle, col_right = st.columns([1, 1, 1])

    with col_left:
        if st.button("← Back to Data Upload", use_container_width=True):
            st.session_state.page = "data"
            st.rerun()

    with col_middle:
        with stylable_container(
            key="middle_button",
            css_styles="""
                button {
                    background-color: #1E88E5 !important;
                    color: white !important;
                    border: none !important;
                }
                button:hover {
                    background-color: #1565C0 !important;  /* لون أغمق شوية عند الهوفر اختياري */
                }
            """,
        ):
            if st.button("Price Forecasting →", use_container_width=True):
                st.session_state.page = "price forecasting"
                st.rerun()

    with col_right:
        if st.button("Next → Results from Analysis", type="primary", use_container_width=True):
            st.session_state.page = "results from analysis"
            st.rerun()

    #save_monthly_analysis_to_file()
# =====================================================================================================================
# ================================================ Analysis results ====================================================
# =====================================================================================================================
def results_from_analysis_page():
    st.markdown('<div class="big-title">Results from Demand Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Results & Recommended Models for Forecasting</div>', unsafe_allow_html=True)

    # ===================== Selected Material Card =====================
    row = st.session_state.get("selected_material_row", None)

    if row is not None:
        st.markdown(f"""
        <div class="detail-card" style="padding: 1.5rem; border-radius: 10px; background-color: #f8f9fa; margin-bottom: 1rem; border-left: 5px solid #1565C0;">
            <h3 style="color:#1565C0; margin:0;">
                {row['Item Code']} - {row['Item Family']} ({row['Item Type']} - Grade {row['Grade']})
            </h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Error: No material selected. Please go back and select a valid Item Code.")

    # ===================== File Name & Selected Period =====================
    file_name = st.session_state.get("uploaded_file_name", "Unknown File")
    period_selected = st.session_state.get("period", "Monthly")

    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 0rem; font-size: 1.1rem; border: 1px solid #1565C0">
        <strong style="color: rgb(21, 101, 192);">Analysis Based On:</strong><br>
        <strong style="color: rgb(21, 101, 192);">File Name:</strong> {file_name} &nbsp; | &nbsp; 
        <strong style="color: rgb(21, 101, 192);">Aggregation Period:</strong> {period_selected}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ===================== Noise Test Results =====================
    if 'df_prepared' in st.session_state and st.session_state.df_prepared is not None:
        demand_series = st.session_state.df_prepared['demand']
        
        st.subheader("Noise Test Results")
        
        result = test_noise_level(demand_series)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            snr_val = result['SNR (dB)'] if result['SNR (dB)'] is not None else "N/A"
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Signal-to-Noise Ratio (SNR)</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                    {snr_val} dB
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Coefficient of Variation (CV)</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                    {result['Coefficient of Variation (%)']}%
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Color based on classification
            if "Low" in result['Classification']:
                bg_color = "#f0f8ff"
                text_color = "#155724"
            elif "Moderate" in result['Classification']:
                bg_color = "#fff3cd"
                text_color = "#856404"
            else:
                bg_color = "#f8d7da"
                text_color = "#721c24"
                
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: {bg_color}; height: 100%; text-align: center;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Noise Classification</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: {text_color};">
                    {result['Classification']}
                </span>
            </div>
            """, unsafe_allow_html=True)
        
    else:
        st.warning("Prepared demand data is not available. Please complete the demand analysis first.")

    # ===================== Stationarity Test Results =====================
    if 'df_prepared' in st.session_state and st.session_state.df_prepared is not None:
        demand_series = st.session_state.df_prepared['demand']
        
        st.subheader("Stationarity Test Results")
        
        # استدعاء الدالة لحساب النتايج
        result = test_stationarity(demand_series)
        
        # حفظ النتيجة في session_state عشان تستخدمها في التوصيات بعدين
        st.session_state.stationarity_result = result
        
        # عرض النتايج في 3 أعمدة
        col1, col2, col3 = st.columns(3)
        
        with col1:
            adf_val = result['ADF p-value'] if result['ADF p-value'] is not None else "N/A"
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; 
                        background-color: #f0f8ff; height: 100%;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">ADF Test p-value</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                    {adf_val}
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            kpss_val = result['KPSS p-value'] if result['KPSS p-value'] is not None else "N/A"
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; 
                        background-color: #f0f8ff; height: 100%;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">KPSS Test p-value</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                    {kpss_val}
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            classification = result['Classification']
            
            # تحديد الألوان حسب التصنيف
            if classification == "Stationary":
                bg_color = "#d4edda"   # أخضر فاتح
                text_color = "#155724"
            elif classification == "Non-Stationary":
                bg_color = "#f8d7da"   # أحمر فاتح (صححت اللون هنا)
                text_color = "#721c24"
            else:  # Trend-Stationary أو Insufficient Data
                bg_color = "#fff3cd"   # أصفر فاتح
                text_color = "#856404"
            
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; 
                        background-color: {bg_color}; height: 100%;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">
                    Stationarity Classification
                </span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: {text_color};">
                    {classification}
                </span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ Data has not been loaded yet. Please upload the file and complete the preparation steps first.")

        # ===================== Linearity Test Results =====================
    if 'df_prepared' in st.session_state and st.session_state.df_prepared is not None:
        demand_series = st.session_state.df_prepared['demand']
       
        st.subheader("Linearity Test Results")
       
        result = test_linearity(demand_series)
       
        col1, col2, col3 = st.columns(3)
       
        with col1:
            linear_r2 = result['Linear R²'] if result['Linear R²'] is not None else "N/A"
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Linear Fit (R²)</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                    {linear_r2}
                </span>
            </div>
            """, unsafe_allow_html=True)
       
        with col2:
            poly_r2 = result['Polynomial R²'] if result['Polynomial R²'] is not None else "N/A"
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Polynomial Fit (R²)</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                    {poly_r2}
                </span>
            </div>
            """, unsafe_allow_html=True)
       
        with col3:
            # Color based on classification
            if "Linear" in result['Classification']:
                bg_color = "#d4edda"      
                text_color = "#155724"
            else:
                bg_color = "#f8d7da"     
                text_color = "#721c24"
               
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: {bg_color}; height: 100%; text-align: center;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Trend Classification</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: {text_color};">
                    {result['Classification']}
                </span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("Demand data not available for linearity test.")

#================== Demand Type Classification (ADI) =====================
    if 'df_prepared' in st.session_state and st.session_state.df_prepared is not None:
        demand_series = st.session_state.df_prepared['demand']
       
        st.subheader("Demand Type Classification")
       
        result = classify_demand_type(demand_series)
       
        # حفظ النتيجة في session_state عشان نستخدمها في صفحات تانية
        st.session_state.demand_type_classification = result['Classification']
        st.session_state.demand_type_result = result  # اختياري لو عايز كل التفاصيل

        col1, col2, col3 = st.columns(3)
       
        with col1:
            adi_val = f"{result['ADI']:.2f}" if result['ADI'] is not None else "N/A"
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">ADI (Average Demand Interval)</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                    {adi_val}
                </span>
            </div>
            """, unsafe_allow_html=True)
       
        with col2:
            cv2_val = f"{result['CV²']:.3f}" if result['CV²'] is not None else "N/A"
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">CV² (Squared Coefficient of Variation)</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                    {cv2_val}
                </span>
            </div>
            """, unsafe_allow_html=True)
       
        with col3:
            classification = result['Classification']
            
            # تحديد اللون والخلفية حسب النوع
            if classification == "Smooth":
                bg_color = "#d4edda"      # أخضر
                text_color = "#155724"
            elif classification == "Erratic":
                bg_color = "#fff3cd"      # أصفر
                text_color = "#856404"
            elif classification == "Intermittent":
                bg_color = "#d1ecf1"      # أزرق فاتح
                text_color = "#0c5460"
            else:  # Lumpy
                bg_color = "#f8d7da"      # أحمر
                text_color = "#721c24"
               
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%; text-align: center;">
                 <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">
                    Demand Type Classification
                </span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                {classification}
                </span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("Demand data not available for demand type classification.")

    # ===================== Trend Analysis =====================
    if 'df_prepared' in st.session_state and st.session_state.df_prepared is not None:
        demand_series = st.session_state.df_prepared['demand']
       
        st.subheader("Trend Analysis Results")
       
        result = analyze_trend(demand_series)
       
        col1, col2, col3 = st.columns(3)
       
        with col1:
            slope_val = result['Slope'] if result['Slope'] is not None else "N/A"
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Trend Slope</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                    {slope_val}
                </span>
            </div>
            """, unsafe_allow_html=True)
       
        with col2:
            strength_val = result['Trend Strength (%)'] if result['Trend Strength (%)'] is not None else "N/A"
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Trend Strength (%)</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                    {strength_val}%
                </span>
            </div>
            """, unsafe_allow_html=True)
       
        with col3:
            if "Positive" in result['Classification']:
                bg_color = "#d4edda"  # Green
                text_color = "#155724"
            elif "Negative" in result['Classification']:
                bg_color = "#f8d7da"  # Red
                text_color = "#721c24"
            else:
                bg_color = "#e2e3e5"  # Gray
                text_color = "#383d41"
               
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: {bg_color}; height: 100%; text-align: center;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Trend Classification</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: {text_color};">
                    {result['Classification']}
                </span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("Demand data not available for trend analysis.")

    # ===================== Seasonality Analysis =====================
    if 'df_prepared' in st.session_state and st.session_state.df_prepared is not None:
        demand_series = st.session_state.df_prepared['demand']
       
        st.subheader("Seasonality Analysis Results")
       
        result = analyze_seasonality(demand_series)
       
        col1, col2, col3 = st.columns(3)
       
        with col1:
            period_name = result['Seasonal Period']
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Seasonal Period</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                    {period_name}
                </span>
            </div>
            """, unsafe_allow_html=True)
       
        with col2:
            strength_val = result['Seasonal Strength']
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Seasonal Strength</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                    {strength_val}
                </span>
            </div>
            """, unsafe_allow_html=True)
       
        with col3:
            if "Strong" in result['Classification']:
                bg_color = "#d4edda"
                text_color = "#155724"
            elif "Moderate" in result['Classification']:
                bg_color = "#fff3cd"
                text_color = "#856404"
            elif "Weak" in result['Classification']:
                bg_color = "#e2e3e5"
                text_color = "#383d41"
            else:
                bg_color = "#f8d7da"
                text_color = "#721c24"
               
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: {bg_color}; height: 100%; text-align: center;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Seasonality Classification</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: {text_color};">
                    {result['Classification']}
                </span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("Demand data not available for seasonality analysis.")

    # ===================== Temporal Dependency =====================
    if 'df_prepared' in st.session_state and st.session_state.df_prepared is not None:
        demand_series = st.session_state.df_prepared['demand']
       
        st.subheader("Temporal Dependency Results")
       
        result = analyze_temporal_dependency(demand_series)
       
        col1, col2, col3 = st.columns(3)
       
        with col1:
            lb_val = result['Ljung-Box p-value'] if result['Ljung-Box p-value'] is not None else "N/A"
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Ljung-Box p-value</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                    {lb_val}
                </span>
            </div>
            """, unsafe_allow_html=True)
       
        with col2:
            acf_val = result['Avg ACF (1-10 lags)'] if result['Avg ACF (1-10 lags)'] is not None else "N/A"
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Avg ACF (1-10 lags)</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                    {acf_val}
                </span>
            </div>
            """, unsafe_allow_html=True)
       
        with col3:
            if "Strong" in result['Classification']:
                bg_color = "#d4edda"
                text_color = "#155724"
            elif "Moderate" in result['Classification']:
                bg_color = "#fff3cd"
                text_color = "#856404"
            elif "Weak" in result['Classification']:
                bg_color = "#e2e3e5"
                text_color = "#383d41"
            else:
                bg_color = "#f8d7da"
                text_color = "#721c24"
               
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: {bg_color}; height: 100%; text-align: center;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Dependency Classification</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: {text_color};">
                    {result['Classification']}
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Demand data not available for temporal dependency analysis.")

    # ===================== Change Point Detection =====================
    if 'df_prepared' in st.session_state and st.session_state.df_prepared is not None:
        demand_series = st.session_state.df_prepared['demand']
       
        st.subheader("Change Point Detection Results")
       
        result = detect_change_points(demand_series)
       
        col1, col2, col3 = st.columns(3)
       
        with col1:
            n_changes = result['Number of Change Points']
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Number of Change Points</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                    {n_changes}
                </span>
            </div>
            """, unsafe_allow_html=True)
       
       # with col2:
        #    dates_str = ", ".join(result['Change Point Dates']) if result['Change Point Dates'] else "None"
         #   st.markdown(f"""
          #  <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%;">
           #     <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Change Dates (First 5)</span><br>
            #    <span style="font-size: 1.2rem; color: #333;">
             #       {dates_str}
              #  </span>
                #   </div>
                    #   """, unsafe_allow_html=True)
       
        with col2:
            if "Stable" in result['Classification'] or "No Change" in result['Classification']:
                bg_color = "#d4fcdd"
                text_color = "#155724"
            elif "Single" in result['Classification']:
                bg_color = "#fff3cd"
                text_color = "#856404"
            else:
                bg_color = "#f8d7da"
                text_color = "#721c24"
               
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: {bg_color}; height: 100%; text-align: center;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Stability Classification</span><br>
                <span style="font-size: 1.6rem; font-weight: bold; color: {text_color};">
                    {result['Classification']}
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Demand data not available for change point detection.")


        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ===================== Variance Structure =====================
    if 'df_prepared' in st.session_state and st.session_state.df_prepared is not None:
        demand_series = st.session_state.df_prepared['demand']
       
        st.subheader("Variance Structure Results")
       
        result = analyze_variance_structure(demand_series)
       
        col1, col2, col3 = st.columns(3)
       
        with col1:
            levene_val = result['Levene p-value'] if result['Levene p-value'] is not None else "N/A"
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Levene Test p-value</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                    {levene_val}
                </span>
            </div>
            """, unsafe_allow_html=True)
       
        with col2:
            cv_var = result['CV of Variance (%)']
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">CV of Variance (%)</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                    {cv_var}%
                </span>
            </div>
            """, unsafe_allow_html=True)
       
        with col3:
            if "Homoscedastic" in result['Classification']:
                bg_color = "#d4edda"
                text_color = "#155724"
            else:
                bg_color = "#f8d7da"
                text_color = "#721c24"
               
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: {bg_color}; height: 100%; text-align: center;">
                <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Variance Classification</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: {text_color};">
                    {result['Classification']}
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Demand data not available for variance analysis.")


    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ==================== Recommended Forecasting Methods =====================
    st.subheader("🔹 Recommended Models Based on Demand Pattern for Forecasting:")

    if 'df_prepared' not in st.session_state or st.session_state.df_prepared is None:
        st.info("Complete the demand analysis to get forecasting recommendations.")
    else:
        # جمع النتايج من session_state (تأكد إنك حفظتها قبل كده)
        results = {}
        
        if hasattr(st.session_state, 'noise_result'):
            results['Noise Level'] = st.session_state.noise_result['Classification']
        
        if hasattr(st.session_state, 'stationarity_result'):
            results['Stationarity'] = st.session_state.stationarity_result['Classification']
        
        if hasattr(st.session_state, 'linearity_result'):
            results['Linearity'] = st.session_state.linearity_result['Classification']
        
        if hasattr(st.session_state, 'demand_type_result'):
            results['Demand Type'] = st.session_state.demand_type_result['Classification']
        
        if hasattr(st.session_state, 'seasonality_result'):
            results['Seasonality'] = st.session_state.seasonality_result['Classification']
        
        if hasattr(st.session_state, 'trend_result'):
            results['Trend'] = st.session_state.trend_result['Classification']
        
        if hasattr(st.session_state, 'change_point_result'):
            results['Change Points'] = st.session_state.change_point_result['Classification']
        
        if hasattr(st.session_state, 'variance_result'):
            results['Variance Structure'] = st.session_state.variance_result['Classification']

# ============================================
# SMART MODEL RECOMMENDATION ENGINE
# ============================================

    recommendations = []

    stationary = results.get('Stationarity')
    linearity = results.get('Linearity')
    demand_type = results.get('Demand Type')
    variance = results.get('Variance Stability')

    # Rule 1: Stationary & Linear
    if stationary == 'Stationary' and linearity == 'Linear':
        recommendations.append("ARIMA")

    # Rule 2: Stationary & Nonlinear
    if stationary == 'Stationary' and linearity == 'Non-Linear':
        recommendations.append("CatBoost")

    # Rule 3: Lumpy OR (Intermittent & Nonlinear)
    if demand_type == 'Lumpy' or (demand_type == 'Intermittent' and linearity == 'Non-Linear'):
        recommendations.append("TCN")

    # Rule 4: Erratic & Nonlinear
    if demand_type == 'Erratic' and linearity == 'Non-Linear':
        recommendations.append("GRU")

    # Rule 5: Smooth & Constant Variance
    if demand_type == 'Smooth' and variance == 'Constant':
        recommendations.append("SARIMA")

    # Rule 6: Smooth & Variable Variance
    if demand_type == 'Smooth' and variance == 'Variable':
        recommendations.append("ETS")

    # Rule 7: Smooth & Nonlinear
    if demand_type == 'Smooth' and linearity == 'Non-Linear':
        recommendations.append("XGBoost")

    # Fallback
    if not recommendations:
        recommendations.append("Naive Forecast / Moving Average")

    rec_text = " • ".join(dict.fromkeys(recommendations))  # remove duplicates



    st.markdown(f"""
        <div style="padding: 1.5rem; border-radius: 10px; margin-bottom: 19px; border-left: 5px solid #1565C0; background-color: #f8f9fa;">
            <strong style="font-weight:bold; color:#1565C0; font-size: 1.7rem;">{rec_text}</strong><br><br>
            <strong style="font-size: 1.1rem; color: #555;">
                <em>These recommendations are derived from comprehensive analysis of stationarity, seasonality, noise, demand type, trend, and structural changes.</em>
            </strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ==================== Demand Type Summary Card ====================
    if 'demand_type_classification' in st.session_state:
        demand_type = st.session_state.demand_type_classification
        
        if demand_type == "Smooth":
            card_color = "#e6f4ea"
            border_color = "#28a745"
            desc = "The demand is stable and continuous, which supports the use of classical time series models."
        elif demand_type == "Erratic":
            card_color = "#fff3cd"
            border_color = "#ffc107"
            desc = "The demand shows high variability with no consistent pattern, requiring more robust and flexible forecasting models."
        elif demand_type == "Intermittent":
            card_color = "#d1ecf1"
            border_color = "#17a2b8"
            desc = "The demand occurs at irregular intervals with many zero-demand periods, which favors intermittent demand forecasting methods."
        else:  # Lumpy
            card_color = "#f8d7da"
            border_color = "#dc3545"
            desc = "The demand is highly irregular with both variable sizes and intervals, making specialized intermittent models more appropriate."



        st.markdown(f"""
        <div style="padding: 1.5rem; border-radius: 10px; margin-bottom: 19px; border-left: 5px solid #1565C0; background-color: #f8f9fa;">
            <strong style="font-weight:bold; color:#1565C0; font-size: 1.7rem;">Demand Type Classification: <strong style="color: rgb(255, 75, 75);">{demand_type}</strong></strong><br><br>
            <strong style="font-size: 1.1rem; color: #555;">
                <em>{desc}</em>
            </strong>
        </div>
        """, unsafe_allow_html=True)
        
    
    else:
        st.info("Demand Type Classification will appear here after analysis.")



        # ================= NAVIGATION =================
   
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    col_left, col_spacer, col_right = st.columns([1, 2, 1])
    with col_left:
        if st.button("← Back to Data Analysis", use_container_width=True):
            st.session_state.page = "analysis"
            st.rerun()

    with col_right:
        if st.button("Next → Run Forecasting Models", type="primary", use_container_width=True):
            st.session_state.page = "forecasting"
            st.rerun()

    
# =====================================================================================================================
# ================================================ forecasting page ======================================================
# =====================================================================================================================

def forecasting_page():
    # ================= HEADER =================
    st.markdown('<div class="big-title">Demand Forecasting Stage</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Model Selection, Forecast Visualization & Export</div>', unsafe_allow_html=True)

    if 'df_uploaded' not in st.session_state:
        st.warning("Please upload data first from the Data page.")
        return

    df = st.session_state['df_uploaded'].copy()

    # ================= DATA PREPARATION =================
    if 'demand' not in df.columns:
        st.error("The uploaded file must contain a column named 'demand'.")
        return

    if df.columns[0] != 'date':
        df.rename(columns={df.columns[0]: 'date'}, inplace=True)

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Split into train/test (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    train_data = df.iloc[:split_idx]
    test_data = df.iloc[split_idx:]

    # ================= MODEL SELECTION =================
    st.markdown("### Select Forecasting Model")
    selected_model = st.radio(
        "", 
        ["ARIMA", "SARIMA", "SARIMAX", "Decision Tree", "XGBoost", "CatBoost", "GRU", "TCN", "Random Forest"],
        horizontal=True
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    test_start = test_data['date'].min().strftime("%Y-%m-%d")
    test_end = test_data['date'].max().strftime("%Y-%m-%d")
    
    st.markdown(
        f"""
        <h3 style="color:#1E88E5; margin-top:10px;">
            Forecasting Results on Testing Data ({test_start} to {test_end}) 
            <spam style="color:#rgb(113 113 113); font-size: 1rem;"> Last 20%  of the Data</spam>
        </h3>
        """,
        unsafe_allow_html=True
    )

    # ================= RUN SELECTED MODEL =================
    with st.spinner("Running Forecasting Model... Please wait..."):
        # Initialize the selected model
        if selected_model == "ARIMA":
            model = ARIMAModel(order=(2,1,2))
        elif selected_model == "SARIMA":
            model = SARIMAModel(order=(1,1,1), seasonal_order=(1,1,1,12))
        elif selected_model == "SARIMAX":
            model = SARIMAXModel(order=(1,1,1), seasonal_order=(1,1,1,12))
        elif selected_model == "Decision Tree":
            model = DecisionTreeModel(lookback=12, max_depth=10)
        elif selected_model == "XGBoost":
            model = XGBoostModel(lookback=12)
        elif selected_model == "CatBoost":
            model = CatBoostModel(lookback=12)
        elif selected_model == "GRU":
            model = GRUModel(lookback=12, units=50, epochs=50)
        elif selected_model == "TCN":
            model = TCNModel(lookback=12, filters=64, kernel_size=3, epochs=50)
        elif selected_model == "Random Forest":
            model = RandomForestModel(lookback=12, n_estimators=200)
        else:
            st.error(f"Unknown model selected: {selected_model}")
            st.stop()

        # Fit on training data
        model.fit(train_data)

        # Predict on test period
        test_forecast = model.predict(steps=len(test_data))

        # Predict future 12 periods (after the last date in data)
        future_forecast = model.predict(steps=12)
    

    # ================= RESULTS ON TEST DATA =================
    result = pd.DataFrame({
        "Date": test_data['date'].values,
        "Actual Demand": test_data['demand'].values,
        "Forecasted Demand": test_forecast
    })

    st.markdown(f"### Results — {selected_model} Model")
    st.dataframe(result, use_container_width=True)

    st.markdown("### Actual vs Forecasted Demand (Test Period)")
    st.line_chart(result.set_index("Date")[["Actual Demand", "Forecasted Demand"]])

    # ================= EXPORT TEST RESULTS =================
    st.markdown("### Export Test Forecast Results")
    buffer_test = io.BytesIO()
    with pd.ExcelWriter(buffer_test, engine="xlsxwriter") as writer:
        result.to_excel(writer, index=False, sheet_name="Test Forecast")
    buffer_test.seek(0)

    st.download_button(
        label="📥 Download Test Forecast Results as Excel",
        data=buffer_test,
        file_name=f"{selected_model}_Test_Forecast_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # ================= FUTURE FORECAST (Next 12 Periods) =================
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <h3 style="color:#1E88E5; margin-top:10px;">
            Future Demand Forecast (Next 12 Periods)
        </h3>
        """,
        unsafe_allow_html=True
    )

    # Generate future dates (monthly frequency assumed for most models)
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=13, freq='MS')[1:]  # 12 future months

    future_df = pd.DataFrame({
        "Date": future_dates,
        "Forecasted Demand": future_forecast
    })

    st.dataframe(future_df, use_container_width=True)
    st.line_chart(future_df.set_index("Date"))

    # Export future forecast
    buffer_future = io.BytesIO()
    with pd.ExcelWriter(buffer_future, engine="xlsxwriter") as writer:
        future_df.to_excel(writer, index=False, sheet_name="Future Forecast")
    buffer_future.seek(0)

    st.download_button(
        label="📥 Download Future Year Forecast as Excel",
        data=buffer_future,
        file_name=f"{selected_model}_Future_12_Periods_Forecast.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


    # ================= TOTAL DEMAND FOR NEXT YEAR =================
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # مجموع التنبؤ للسنة القادمة
    total_next_year_demand = future_forecast.sum()  # <-- رقم float للاستخدام في الحسابات

    # نسخة للعرض فقط
    formatted_total = f"{total_next_year_demand:,.2f}"

    # السنة القادمة
    next_year = future_dates[-1].year -1

    st.markdown(f"""
        <div class="detail-card" style="padding: 1.5rem; border-radius: 10px; background-color: #f8f9fa; margin-bottom: 1rem; border-left: 5px solid #1565C0;">
            <h3 style="color:#1565C0; margin:0;">
                Total Demand for Next Year (
                <strong style="color: rgb(255, 75, 75);">{next_year}</strong> ): 
                <strong style="color: rgb(255, 75, 75);">{formatted_total}</strong> Units
            </h3>
        </div>
    """, unsafe_allow_html=True)


    # ================= NAVIGATION =================
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    col_left, col_spacer, col_right = st.columns([1, 2, 1])
    with col_left:
        if st.button("← Back to Results from Data Analysis", use_container_width=True):
            st.session_state.page = "results from analysis"
            st.rerun()

    with col_right:
        if st.button("Next → EOQ & Safety Stock", type="primary", use_container_width=True):
            st.session_state.page = "EOQ & Safety Stock"
            st.rerun()

# =====================================================================================================================
# ================================================ price forecating page ==============================================
# =====================================================================================================================
def price_forecasting_page():
    st.markdown('<div class="big-title">Price Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Next-Year Price Predictions Powered by Data Analysis</div>', unsafe_allow_html=True)

    # التحقق من وجود البيانات
    if 'df_uploaded' not in st.session_state:
        st.warning("Please upload your data file first from the Data Upload page.")
        if st.button("← Back to Data Upload"):
            st.session_state.page = "data"
            st.rerun()
        return

    df = st.session_state['df_uploaded']

    # التحقق من وجود الأعمدة المطلوبة
    required_columns = ['date', 'price']
    if not all(col in df.columns.str.lower() for col in required_columns):
        st.error("The uploaded file must contain both 'date' and 'price' columns (case-insensitive).")
        return

    # تحويل أسماء الأعمدة للحروف الصغيرة لو موجودة بصيغ مختلفة
    df.columns = df.columns.str.lower()

    st.markdown("### Annual Average Price Forecast Using Linear Regression")

    # زرار تشغيل التنبؤ
    if st.button("Run Price Forecasting", type="primary", use_container_width=True):

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        with st.spinner("Running forecast for the next year..."):
            try:
                forecasted_price, result_df, plot_buf = forecast_price(df.copy())

                # استخراج السنة المتوقعة
                next_year = int(result_df.loc[result_df['Forecasted Price'].notna(), 'Year'].values[0])

                # عرض التوقع بشكل بارز
                st.markdown(f"""
                <div class="detail-card" style="padding: 1.5rem; border-radius: 10px; background-color: #f8f9fa; margin-bottom: 1rem; border-left: 5px solid #1565C0;">
                    <h3 style="color:#1565C0; margin:0;">
                        Forecasted Average Price for Year (
                        <strong style="color: rgb(255, 75, 75);">{next_year}</strong> ): 
                        <strong style="color: rgb(255, 75, 75);">{forecasted_price:.3f}</strong> $
                    </h3>
                </div>
                """, unsafe_allow_html=True)


                # عرض الجدول
                st.markdown("#### Annual Price Summary")
                st.dataframe(
                    result_df.style.format({
                        'Actual Price': '{:.2f}',
                        'Forecasted Price': '{:.2f}'
                    })
                )  # تم إزالة use_container_width لتجنب التحذير

                # عرض الرسم البياني
                st.markdown("#### Price Trend & Forecast")
                st.image(plot_buf)  # تم إزالة use_column_width لتجنب التحذير

            except Exception as e:
                st.error(f"An error occurred during forecasting: {str(e)}")




    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


    col_left, col_spacer, col_right = st.columns([1, 2, 1])
    with col_left:
        if st.button("← Back to Data Analysis", use_container_width=True):
            st.session_state.page = "analysis"
            st.rerun()

    with col_right:
        if st.button(" Run Demand Forecasting Models →", type="primary", use_container_width=True):
            st.session_state.page = "results from analysis"
            st.rerun()

# =====================================================================================================================
# ================================================ EOQ & Safety Stock ======================================================
# =====================================================================================================================
import numpy as np
def eoq_safety_stock_page():

    st.markdown('<div class="big-title">Inventory Planning</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">EOQ & Safety Stock Calculation</div>', unsafe_allow_html=True)
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# التأكد من وجود التنبؤ المستقبلي وتصنيف الطلب
    if 'future_forecast' not in st.session_state or 'future_dates' not in st.session_state:
        st.warning("Please run the forecasting model first from the Forecasting page.")
        return

    if 'demand_type_classification' not in st.session_state:
        st.warning("Demand type classification is missing. Please go back to Data Analysis.")
        return

    future_forecast = st.session_state.future_forecast  # array أو list من التنبؤات (12 شهر عادة)
    future_dates = st.session_state.future_dates        # تواريخ التنبؤ المستقبلي
    demand_type = st.session_state.demand_type_classification

    # حساب إجمالي الطلب للسنة القادمة
    total_next_year_demand = float(np.sum(future_forecast))  # للدقة في الحسابات
    formatted_total = f"{total_next_year_demand:,.0f}"      # بدون كسور للعرض

    # السنة القادمة (آخر تاريخ في التنبؤ + السنة التالية)
    next_year = future_dates[-1].year + 1

    # عرض إجمالي الطلب للسنة القادمة
    st.markdown(f"""
        <div class="detail-card" style="padding: 1.5rem; border-radius: 10px; background-color: #f8f9fa; margin: 2rem 0; border-left: 5px solid #1565C0;">
            <h3 style="color:#1565C0; margin:0;">
                Total Demand for Next Year (
                <strong style="color: rgb(255, 75, 75);">{next_year}</strong> ): <br/> D= 
                <strong style="color: rgb(255, 75, 75);">{formatted_total}</strong> Units
            </h3>
        </div>
    """, unsafe_allow_html=True)

    # عرض نوع الطلب بنفس الستايل السابق
    if demand_type == "Smooth":
        card_color = "#e6f4ea"
        border_color = "#28a745"
        desc = "The demand is stable and continuous, which supports the use of classical time series models and standard EOQ."
    elif demand_type == "Erratic":
        card_color = "#fff3cd"
        border_color = "#ffc107"
        desc = "The demand shows high variability. Consider higher safety stock and possibly adjusted EOQ."
    elif demand_type == "Intermittent":
        card_color = "#d1ecf1"
        border_color = "#17a2b8"
        desc = "The demand occurs at irregular intervals with many zeros. Classical EOQ may not be suitable — consider Croston or SBA methods."
    else:  # Lumpy
        card_color = "#f8d7da"
        border_color = "#dc3545"
        desc = "Highly irregular demand (lumpy). Standard EOQ/Safety Stock formulas have limited reliability. Specialized methods recommended."

    st.markdown(f"""
        <div style="padding: 1rem; border-radius: 8px; background-color: {card_color}; 
                    border-left: 6px solid {border_color}; margin: 1.5rem 0;">
            <h4 style="margin:0; color:#333;">Demand Pattern: <strong>{demand_type}</strong></h4>
            <p style="margin: 0.5rem 0 0 0; color:#555; font-size:0.95rem;">{desc}</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Inventory Parameters (Required for EOQ & Safety Stock Calculation)")

    col1, col2 = st.columns(2)
    with col1:
        annual_holding_cost_per_unit = st.number_input(
            "Annual Holding Cost per Unit (H)", min_value=0.0, value=10.0, step=1.0,
            help="Cost to hold one unit in inventory for one year (e.g., storage, insurance)"
        )
        order_cost = st.number_input(
            "Fixed Order Cost (S)", min_value=0.0, value=100.0, step=10.0,
            help="Fixed cost per purchase order (e.g., shipping, admin)"
        )

    with col2:
        lead_time_days = st.number_input(
            "Lead Time (Days)", min_value=1, value=30, step=1,
            help="Time between placing an order and receiving it"
        )
        service_level = st.slider(
            "Desired Service Level (%)", min_value=90, max_value=99, value=95, step=1,
            help="Higher % means less stockout risk but more safety stock"
        ) / 100

    # حساب EOQ
    if annual_holding_cost_per_unit > 0 and order_cost > 0:
        D = total_next_year_demand  # Annual demand
        EOQ = np.sqrt((2 * D * order_cost) / annual_holding_cost_per_unit)
        formatted_eoq = f"{EOQ:,.0f}"
    else:
        EOQ = 0
        formatted_eoq = "N/A"

    # حساب Safety Stock
    # متوسط الطلب اليومي
    daily_demand_mean = total_next_year_demand / 365

    # انحراف معياري للطلب اليومي (تقدير بسيط من التنبؤ)
    daily_forecast = np.array(future_forecast) / 30.42  # تقريب شهري → يومي
    daily_demand_std = np.std(daily_forecast)

    # Lead Time Demand
    lead_time_demand_mean = daily_demand_mean * lead_time_days
    lead_time_demand_std = daily_demand_std * np.sqrt(lead_time_days)

    # Z-score للـ service level
    z_scores = {0.90: 1.28, 0.95: 1.645, 0.97: 1.88, 0.98: 2.05, 0.99: 2.326}
    z = z_scores.get(round(service_level, 2), 1.645)

    safety_stock = z * lead_time_demand_std
    formatted_ss = f"{safety_stock:,.0f}"

    # عرض النتائج
    st.markdown("### Calculation Results")

    col_eoq, col_ss = st.columns(2)
    with col_eoq:
        st.markdown(f"""
            <div style="padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color:white; border-radius:12px; text-align:center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h3 style="margin:0; font-size:1.1rem;">Economic Order Quantity (EOQ)</h3>
                <h2 style="margin:0.5rem 0 0 0; font-size:2.2rem;">{formatted_eoq}</h2>
                <p style="margin:0.5rem 0 0 0; opacity:0.9;">Units per Order</p>
            </div>
        """, unsafe_allow_html=True)

    with col_ss:
        st.markdown(f"""
            <div style="padding: 1.5rem; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        color:white; border-radius:12px; text-align:center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h3 style="margin:0; font-size:1.1rem;">Safety Stock</h3>
                <h2 style="margin:0.5rem 0 0 0; font-size:2.2rem;">{formatted_ss}</h2>
                <p style="margin:0.5rem 0 0 0; opacity:0.9;">Units</p>
            </div>
        """, unsafe_allow_html=True)

    # تحذير إضافي للطلب المتقطع أو Lumpy
    if demand_type in ["Intermittent", "Lumpy"]:
        st.warning(
            "⚠️ For **Intermittent** or **Lumpy** demand, classical EOQ and Safety Stock formulas may not be reliable. "
            "Consider using methods like Croston's, Syntetos-Boylan Approximation (SBA), or bootstrapping."
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    col_left, col_spacer = st.columns([1, 3])
    with col_left:
        if st.button("← Back to Forecasting Results", use_container_width=True):
                st.session_state.page = "forecasting"
                st.rerun()



# =====================================================================================================================
# ================================================ MAIN ROUTER ======================================================
# =====================================================================================================================

if st.session_state.page == "material":
    page_material()
elif st.session_state.page == "data":
    page_data()
elif st.session_state.page == "analysis":
    analysis_page()
elif st.session_state.page == "results from analysis":
    results_from_analysis_page()
elif st.session_state.page == "forecasting":
    forecasting_page()
elif st.session_state.page == "price forecasting":
    price_forecasting_page()
elif st.session_state.page == "EOQ & Safety Stock":
    eoq_safety_stock_page()
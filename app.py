import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from demand_processor import plot_before_preparation_interactive, plot_after_preparation, prepare_demand_data, aggregate_demand, plot_comparative_demand, show_aggregated_table, test_noise_level, test_stationarity, test_linearity, analyze_trend, classify_demand_type, analyze_variance_structure, detect_change_points, analyze_temporal_dependency, analyze_seasonality
from PIL import Image
from forecasting_model import (
    ARIMAModel, SARIMAModel, SARIMAXModel, DecisionTreeModel, XGBoostModel, CatBoostModel, GRUModel, TCNModel, 
    RandomForestModel, aggregate_demand, run_forecast, calculate_sigma_demand, calculate_sigma_error
)
from price_forecast import forecast_price
from streamlit_extras.stylable_container import stylable_container
import io
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(layout="wide")
#==================================================help sign
st.markdown("""
<style>
.help-icon {
    display: inline-block;
    margin-left: 6px;
    color: #1565C0;
    font-weight: bold;
    font-size: 30px;
    cursor: pointer;
    position: relative;
}

.help-icon:hover::after {
    content: attr(data-help);
    position: absolute;
    left: -120px;
    top: -5px;
    background: #1f2933;
    color: white;
    padding: 8px 10px;
    border-radius: 6px;
    font-size: 1.05rem;
    width: 320px;
    z-index: 999;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    
}
</style>
""", unsafe_allow_html=True)
#============================================================= شكل الصفحة تبقى اوسع واطول
css = """
<style>
    /* الـ selector الأكثر فعالية في الإصدارات الجديدة (2025+) */
    div.block-container {
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-top: 1.7rem !important;
        padding-bottom: 3rem !important;  /* غيّرها زي ما تحب */
    }

    /* بديل قوي تاني لو اللي فوق مش كفاية */
    section.main .block-container {
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-top: 1.7rem !important;
        padding-bottom: 3rem !important;
    }

    /* لو عايز تقلل الـ padding الأصلي أكتر (اختياري) */
    .stMainBlockContainer {
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-top: 1.7rem !important;
    }
</style>
"""

st.markdown(css, unsafe_allow_html=True)

#def show_logo():
#    logo = Image.open("images/logo.jpeg")
#    col1, col2 = st.columns([1, 12])
#    with col1:
#        st.image(logo, width=80)
#    with col2:
#        st.write("")  # spacer
#show_logo()

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

import os
import pandas as pd
import streamlit as st

# =====================================================================================================================
# ============================================ PAGE 1 : MATERIAL SELECTION ============================================
# =====================================================================================================================
def page_material():
    # ========================= Titles =========================
    st.markdown('<div class="big-title">Material Selection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Select a Target Material</div>', unsafe_allow_html=True)
    # ========================= Load Data =========================
    file_path = "Database/Material Information.xlsx"

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
                    'Warehouse Location', 'Shelf Life', 'Supplier', 'Purchasing Unit', 'Service Level']

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
        st.session_state.pop('selected_material', None)

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
        st.session_state['selected_material'] = final_selection.iloc[0].to_dict()
    else:
        final_selection = pd.DataFrame()

    col1, col_spacer, col2 = st.columns([1, 2.2 ,1]) 

    with col1:
        st.button("Clear Selection", on_click=clear_selections_in_material_selection)

    with col2:
        if selected_code != CODE_PLACEHOLDER:
            st.markdown(
                '<p style="color: #28a745; font-size: 1.1rem; margin: 0; padding: 0.5rem 0;">Material selected successfully</p>',
                unsafe_allow_html=True
        )

    if selected_code != CODE_PLACEHOLDER:
        row = filtered_df[filtered_df['Item Code'] == selected_code].iloc[0]
        col1, col2, col_spacer = st.columns([1.5, 5, 2])
        with col1:
            st.subheader("Material Details:")
        with col2:
          st.markdown(f"""
            <div style="
                padding: 1rem; border-radius: 10px; margin-bottom: 19px; border-left: 5px solid #1E88E5; background-color: rgb(240 242 253 / 31%); font-size: 1.2rem; font-weight: bold; color: #1E88E5;
            ">
                {row['Item Code']} - {row['Item Family']} ({row['Item Type']} - Grade {row['Grade']})
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        st.markdown('<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">', unsafe_allow_html=True)
        
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
        if st.button("Next → Data Uploading", type="primary", use_container_width=True):
            if selected_code == CODE_PLACEHOLDER:
                st.warning("Please select an Item Code to proceed.")
            else:
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
    st.markdown('<div class="subtitle" style="margin-bottom: 0rem;">Upload Demand Data & Choose Analysis Period</div>', unsafe_allow_html=True)

    st.markdown("""
        <style>
        .tight-label {
            margin-bottom: 0px !important;
            display: block;
        }
        </style>
    """, unsafe_allow_html=True)



    col1,col2,col3 = st.columns([1,2,2])
    with col1:
        st.markdown("""
        <span class="tight-label" style="font-size: 1.5rem; font-weight:bold; color: rgb(21, 101, 192); margin-bottom:10px;">
            Demand Data Source
        </span>
        """, unsafe_allow_html=True)
    with col2: 
    # Radio Button: Database vs Upload
        data_source_option = st.radio(
            "",
            ["Use demand history in data base", "Upload another file"],
            label_visibility="collapsed",
            horizontal=True
        )

    uploaded_file = None
    df_display = pd.DataFrame()

    # Case 1: Upload another file
    if data_source_option == "Upload another file":
        uploaded_file = st.file_uploader(
            "",
            type=["xlsx"],
            key="demand_uploader"
        )
        if uploaded_file is not None:
            try:
                df_display = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")

    # Case 2: Use database (Reads automatically based on Item Code from page 1)
    else:
        # Check if we have a selected material
        if 'selected_material_row' in st.session_state:
            item_code = st.session_state.selected_material_row['Item Code']
            db_path = "Database/demand_history.xlsx"
            
            if os.path.exists(db_path):
                try:
                    # Read the sheet corresponding to the item code
                    df_display = pd.read_excel(db_path, sheet_name=item_code)
                except ValueError:
                    st.error(f"Sheet '{item_code}' not found in 'demand_history.xlsx'")
                except Exception as e:
                    st.error(f"Error reading database file: {e}")
            else:
                st.error(f"Database file not found: {db_path}")
        else:
            st.warning("Please go back and select a material first to load its history.")

    # ---------------- DISPLAY TABLE AND SUMMARY ----------------
    # Only display if we have data to show (either from DB or Upload)
    if not df_display.empty:
        st.markdown("<br>", unsafe_allow_html=True) # Spacing
        left_col, right_col = st.columns([2, 1])

        # ===== LEFT: Table =====
        with left_col:
            df_copy = df_display.copy()
            try:
                # Try to format the first column as date if possible
                df_copy.iloc[:, 0] = pd.to_datetime(df_copy.iloc[:, 0]).dt.strftime('%Y-%m-%d')
            except:
                pass

            styled_df = df_copy.style.set_properties(**{
                'font-size': '14pt',     
                'font-weight': '400',     
                'text-align': 'center'    
            }).set_table_styles([
                {'selector': 'th', 'props': [('font-size', '16px'), ('font-weight', 'bold')]}
            ])

            st.dataframe(styled_df, height=300, use_container_width=True)

        # ===== RIGHT: Summary =====
        with right_col:
            num_periods = len(df_display)
            start_date_raw = df_display.iloc[0, 0] if not df_display.empty else "N/A"
            end_date_raw = df_display.iloc[-1, 0] if not df_display.empty else "N/A"

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

            for i, col_name in enumerate(summary_columns):
                st.markdown(f"""
                <div style="
                    padding: 0.9rem;
                    border-radius: 10px;
                    margin-bottom: 10px;
                    border-left: 5px solid #1565C0;
                    background-color: #f8f9fa;
                ">
                    <span style="font-weight:bold; color:#1565C0; font-size:1.5rem; ">{col_name}: </span>
                    <span style="font-weight:bold;font-size:1.5rem; color:#555;">{summary_values[i]}</span>
                </div>
                """, unsafe_allow_html=True)

    # ---------------- FOOTER BUTTONS ----------------
    st.markdown("""
        <style>
            div[data-testid="column"]:nth-of-type(1) .stButton {margin-top: -20px;}
            div[data-testid="column"]:nth-of-type(3) .stButton {margin-top: -20px;}
        </style>
    """, unsafe_allow_html=True)

    col_back, col_spacer, col_next = st.columns([1, 2, 1])

    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "material"
            st.rerun()

    with col_next:
        if st.button("Next → Data Analysis", type="primary", use_container_width=True):
            # Validation
            is_data_valid = False
            final_df = pd.DataFrame()

            if data_source_option == "Upload another file":
                if uploaded_file is not None and not df_display.empty:
                    is_data_valid = True
                    final_df = df_display
                else:
                    st.warning("Please upload a file first.")
            else:
                # Database mode
                if 'selected_material_row' in st.session_state and not df_display.empty:
                    is_data_valid = True
                    final_df = df_display
                else:
                    st.warning("No demand history data found in database for this item.")


            if is_data_valid:
                
                # 1. Save the dataframe
                st.session_state.df = final_df
                st.session_state['df_uploaded'] = final_df 
                
                # 3. Save source info
                st.session_state.data_source = data_source_option
                if data_source_option == "Upload another file":
                    st.session_state.uploaded_file_name = uploaded_file.name
                else:
                    st.session_state.uploaded_file_name = f"DB_{st.session_state.selected_material_row['Item Code']}"

                st.session_state.page = "analysis"
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    
# =====================================================================================================================
# ============================================ PAGE 3 : ANALYSIS ====================================================
# =====================================================================================================================
def analysis_page():
    st.markdown('<div class="big-title">Data Analysis Stage</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Review the Dataset Before & After Cleaning </div>', unsafe_allow_html=True)

    df = st.session_state.df
    file_name = st.session_state.get("uploaded_file_name", "Unknown File")
    period_selected = st.session_state.get("period", "Monthly")
    material = st.session_state.get("selected_material", None)
    if material:
        material_info = f"{material['Item Code']} - {material['Item Family']} ({material['Item Type']} - Grade {material['Grade']})"
    else:
        material_info = "No material selected"

    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; margin-top: -5px; font-size: 1.1rem; border: 1px solid #1565C0">
        <strong >Analysis Based On   →   </strong>
        <strong style="color: rgb(21, 101, 192);">Selected Material:</strong> {material_info}
    </div>
    """, unsafe_allow_html=True)

    # ================= BEFORE PREPARATION =================
    st.markdown(
    f"""
        <h3 style="color:#1E88E5; margin-top:-15px; margin-bottom: -20px;">
            Data Status - BEFORE DATA PREPARATION
            <span class="help-icon" data-help="Representing raw data prior to cleaning, since AI forecasting models perform poorly on unprocessed datasets.">
                ?
            </span>
        </h3>
        """,
    unsafe_allow_html=True
    )

    plots = plot_before_preparation_interactive(df)
    plot_options = list(plots.keys())  # ['Raw Demand', 'Distribution', 'Box Plot', 'Missing Values']

    col1, col2 = st.columns([1, 4])  

    with col1:
        st.markdown("<div style='height:180px; display:flex; align-items:center; justify-content:center;'>", unsafe_allow_html=True)
        selected_plot = st.selectbox("Choose a plot:", plot_options)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:  
        st.plotly_chart(plots[selected_plot], use_container_width=True)

    #========next /back button
    col_left, col_spacer, col_right = st.columns([1, 2, 1])

    with col_left:
        if st.button("← Back to Data Upload", use_container_width=True):
            st.session_state.page = "data"
            st.rerun()


    with col_right:
        if st.button("Next → Data After Cleaning", type="primary", use_container_width=True):
            st.session_state.page = "analysis2"
            st.rerun()

    #======================================================================////////////////////////////////////////////////first
    
def analysis2_page():
    st.markdown('<div class="big-title">Data Analysis Stage (Cont.)</div>', unsafe_allow_html=True)
    #st.markdown('<div class="subtitle">Review the Dataset After Cleaning</div>', unsafe_allow_html=True)
    
    # ================= الجلب من session_state =================
    df = st.session_state.df
    file_name = st.session_state.get("uploaded_file_name", "Unknown File")
    period_selected = st.session_state.get("period", "Monthly")
    material = st.session_state.get("selected_material", None)

    if material:
        material_info = f"{material['Item Code']} - {material['Item Family']} ({material['Item Type']} - Grade {material['Grade']})"
    else:
        material_info = "No material selected"

    # ================= DATA PREPARATION =================
    if "df_prepared" not in st.session_state or "prep_results" not in st.session_state:
        with st.spinner("Preparing data..."):
            df_prepared, prep_results = prepare_demand_data(df)
            st.session_state.df_prepared = df_prepared
            st.session_state.prep_results = prep_results
    else:
        df_prepared = st.session_state.df_prepared
        prep_results = st.session_state.prep_results

    # ================= SHOW PREPARATION RESULTS =================
    c1, c2, c3, c4 = st.columns(4)
    
    # -------- Card 1: Missing Values --------
    with c1:
        st.markdown("""
        <div style="border:1px solid #1E88E5; border-radius:10px; padding:12px; text-align:center; min-height:115px; margin-top: -5px;">
            <h5 style="color:#1E88E5; margin-bottom:-5px;">HANDLING MISSING VALUES</h5>
            <div style="font-size:1rem;">
                <b>Before:</b> {} &nbsp; | &nbsp; 
                <b>After:</b> {}
            </div>
        </div>
        """.format(prep_results['missing_values_before'],
                prep_results['missing_values_after']), unsafe_allow_html=True)

    # -------- Card 2: Outlier Detection --------
    with c2:
        st.markdown("""
        <div style="border:1px solid #1E88E5; border-radius:10px; padding:12px; text-align:center;min-height:115px; margin-top: -5px;
">
            <h5 style="color:#1E88E5; margin-bottom:-5px;">OUTLIER DETECTION</h5>
            <div style="font-size:1rem;">
                <b>Outliers:</b> {} &nbsp; | &nbsp; 
                <b>Percentage:</b> {:.2f}%
            </div>
        </div>
        """.format(prep_results['outliers_count'],
                prep_results['outliers_percent']), unsafe_allow_html=True)

    # -------- Card 3: Time Frequency --------
    with c3:
        st.markdown("""
        <div style="border:1px solid #1E88E5; border-radius:10px; padding:12px; text-align:center;min-height:115px; margin-top: -5px;
">
            <h5 style="color:#1E88E5; margin-bottom:-5px;">TIME FREQUENCY CHECK</h5>
            <div style="font-size:.95rem;">
                <b>Expected:</b> {} &nbsp; | &nbsp; 
                <b>Actual:</b> {} </br>
                <b>Missing Filled:</b> {}
            </div>
        </div>
        """.format(prep_results['expected_records'],
                prep_results['actual_records'],
                prep_results['missing_days_filled']), unsafe_allow_html=True)

    # -------- Card 4: Feature Engineering --------
    with c4:
        st.markdown("""
        <div style="border:1px solid #1E88E5; border-radius:10px; padding:12px; text-align:center;min-height:115px; margin-top: -5px;
">
            <h5 style="color:#1E88E5; margin-bottom:-5px;">FEATURE ENGINEERING</h5>
            <div style="font-size:1rem;">
                <b>Features Created:</b> {}
            </div>
        </div>
        """.format(prep_results['features_created']), unsafe_allow_html=True)
 
    # ================= AFTER PREPARATION =================
    df_clean = st.session_state.df_prepared

    st.markdown(
    f"""
        <h3 style="color:#1E88E5; margin-top:10px; margin-bottom: -21px; z-index: 5000; position: relative;">
            Data Status - AFTER DATA PREPARATION
        <span class="help-icon" data-help="Cleaned raw data (zeros and outliers removed)">
            ?
        </span>
        </h3>
        
        """,
    unsafe_allow_html=True
    )
    plots_after = plot_after_preparation(df_clean)
    plot_options_after = list(plots_after.keys())

    

    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown(
            "<div style='height:180px; display:flex; align-items:center; justify-content:center;'>",
            unsafe_allow_html=True
        )
        selected_plot_after = st.selectbox(
            "Choose a plot:",
            plot_options_after,
            key="after_plot_select"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.plotly_chart(plots_after[selected_plot_after], use_container_width=True)

    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] button {
        margin-bottom: -10px !important;
    }
    </style>
    """, unsafe_allow_html=True)
        #========next /back button
    col_left, col_spacer, col_right = st.columns([1, 2, 1])

    with col_left:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "analysis"
            st.rerun()


    with col_right:
        if st.button("Next → Aggregated Visualizations", type="primary", use_container_width=True):
            st.session_state.page = "analysis3"
            st.rerun()

    #======================================================================/////////////////////////////////////////////////////second page

def analysis3_page():
    st.markdown('<div class="big-title">Data Analysis Stage (Cont.)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Aggregated Demand Plots & Statistics After Preparation</div>', unsafe_allow_html=True)
      
        # ================= الجلب من session_state =================
    df = st.session_state.df
    file_name = st.session_state.get("uploaded_file_name", "Unknown File")
    period_selected = st.session_state.get("period", "Monthly")
    material = st.session_state.get("selected_material", None)

    if material:
        material_info = f"{material['Item Code']} - {material['Item Family']} ({material['Item Type']} - Grade {material['Grade']})"
    else:
        material_info = "No material selected"

    # ================= DATA PREPARATION =================
    if "df_prepared" not in st.session_state or "prep_results" not in st.session_state:
        with st.spinner("Preparing data..."):
            df_prepared, prep_results = prepare_demand_data(df)
            st.session_state.df_prepared = df_prepared
            st.session_state.prep_results = prep_results
    else:
        df_prepared = st.session_state.df_prepared
        prep_results = st.session_state.prep_results



    # ================= AGGREGATE DEMAND =================
    if df_prepared is not None:
    # ── Visuals first ────────────────────────────────
       with st.container():
            st.markdown(
                """
                <h3 style="color:#1E88E5; margin-bottom:0.5rem;">
                Comparative Demand Visualization
                <span class="help-icon" data-help="Represent data segmented by time periods">
                ?
                </span>
                </h3>
                """,
                unsafe_allow_html=True
            )

            # Define available plot options (periods)
            plot_options = ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Semi-Annual', 'Annual']

            # Two-column layout
            col1, col2 = st.columns([1, 4])

            with col1:
                st.markdown(
                    "<div style='height:180px; display:flex; align-items:center; justify-content:center;'>",
                    unsafe_allow_html=True
                )
                selected_period = st.selectbox(
                    "Choose as period:",
                    plot_options,
                    key="comparative_plot_select"
                )


            with col2:
                
                fig = plot_comparative_demand(df_prepared, selected_period)
                st.plotly_chart(fig, use_container_width=True)


    col_left, col_spacer, col_right = st.columns([1, 1, 1])

    with col_left:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "analysis2"
            st.rerun()
    with col_spacer:
        if st.button("More Detailed Statistics", type="primary", use_container_width=True):
            st.session_state.page = "analysis4"
            st.rerun()
    with col_right:
        if st.button("Next → Results from Analysis", type="primary", use_container_width=True):
            st.session_state.page = "results from analysis"
            st.rerun()

    #======================================================================/////////////////////////////////////////////////////////////////////third page

def analysis4_page():
    st.markdown('<div class="big-title">Data Analysis Stage (Cont.)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Aggregated Demand Plots & Statistics After Preparation</div>', unsafe_allow_html=True)
      
            # ================= الجلب من session_state =================
    df = st.session_state.df
    file_name = st.session_state.get("uploaded_file_name", "Unknown File")
    period_selected = st.session_state.get("period", "Monthly")
    material = st.session_state.get("selected_material", None)

    if material:
        material_info = f"{material['Item Code']} - {material['Item Family']} ({material['Item Type']} - Grade {material['Grade']})"
    else:
        material_info = "No material selected"

    # ================= DATA PREPARATION =================
    if "df_prepared" not in st.session_state or "prep_results" not in st.session_state:
        with st.spinner("Preparing data..."):
            df_prepared, prep_results = prepare_demand_data(df)
            st.session_state.df_prepared = df_prepared
            st.session_state.prep_results = prep_results
    else:
        df_prepared = st.session_state.df_prepared
        prep_results = st.session_state.prep_results


            # ── Table second ─────────────────────────────────
        st.markdown(
            """
            <h3 style="color:#1E88E5; margin-top:1rem; margin-bottom:0.5rem;">
            Aggregated Demand Statistics
            <span class="help-icon" data-help="Aggregated Demand Statistics refers to the summary measures of total demand accross time">
                ?
            </span>
            </h3>
            """,
            unsafe_allow_html=True
        )

        selected_period, agg_df = show_aggregated_table(df_prepared)

        # ================= NAVIGATION =================

    col_left, col_middle, col_right = st.columns([1, 1, 1])

    with col_left:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "analysis3"
            st.rerun()



    with col_right:
        if st.button("Next → Results from Analysis", type="primary", use_container_width=True):
            st.session_state.page = "results from analysis"
            st.rerun()


# =====================================================================================================================
# ================================================ Analysis results ====================================================
# =====================================================================================================================
def results_from_analysis_page():
    st.markdown('<div class="big-title" style=" margin-bottom: 1rem;">Summary from Demand Analysis</div>', unsafe_allow_html=True)

    col11, col22, col33 = st.columns(3)

    with col11:
        # ===================== Noise Test Results =====================
        if 'df_prepared' in st.session_state and st.session_state.df_prepared is not None:
            demand_series = st.session_state.df_prepared['demand']
            result = test_noise_level(demand_series)

            with st.container(border=True):
                st.markdown("""
                    <div style='display: flex; align-items: center; gap: 10px;'>
                        <h3 style='margin: 0; margin-top:-15px; color: #1E88E5;'>📊 Noise Analysis</h3>
                        <span style="margin-top:-15px;" class="help-icon" data-help="Measures useful signal vs. random noise using SNR and CV. High noise makes data unpredictable and reduces forecasting accuracy.">?</span>
                    </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    snr = result['SNR (dB)'] if result['SNR (dB)'] is not None else "N/A"
                    st.metric("SNR", f"{snr} dB")
                with col2:
                    cv = result['Coefficient of Variation (%)']
                    st.metric("Coefficient Var.", f"{cv}%")
                
                cls = result['Classification']
                cls_color = "#4caf50" if "Low" in cls else "#ff9800" if "Moderate" in cls else "#f44336"
                
                st.markdown(f"""
                    <div style='background-color: {cls_color}20; border-left: 5px solid {cls_color}; padding: 10px; border-radius: 4px; margin-top: -20px;'>
                        <span style='font-size: 1.4rem; font-weight: bold; color: {cls_color};'>{cls}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Data has not been loaded yet.")

    with col22:
        # ===================== Stationarity Test Results =====================
        if 'df_prepared' in st.session_state and st.session_state.df_prepared is not None:
            demand_series = st.session_state.df_prepared['demand']
            result = test_stationarity(demand_series)
            st.session_state.stationarity_result = result

            with st.container(border=True):
                st.markdown("""
                    <div style='display: flex; align-items: center; gap: 10px;'>
                        <h3 style='margin: 0; margin-top:-15px;  color: #1E88E5;'>📈 Stationarity Test</h3>
                        <span style="margin-top:-15px;" class="help-icon" data-help="Determines if statistical properties (mean/variance) stay constant over time. Stationary data is easier to model; non-stationary data often needs differencing.">?</span>
                    </div>
                """, unsafe_allow_html=True)

                adf_val = result['ADF p-value'] if result['ADF p-value'] is not None else "N/A"
                kpss_val = result['KPSS p-value'] if result['KPSS p-value'] is not None else "N/A"
                
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("ADF p-value", adf_val)
                with c2:
                    st.metric("KPSS p-value", kpss_val)

                classification = result['Classification']
                text_color = "#2e7d32" if classification == "Stationary" else "#c62828" if classification == "Non-Stationary" else "#ef6c00"
                border_color = "#4caf50" if classification == "Stationary" else "#ef5350" if classification == "Non-Stationary" else "#ffa726"

                st.markdown(f"""
                    <div style='background-color: {border_color}20; border-left: 5px solid {border_color}; padding: 10px; border-radius: 4px; margin-top: -20px;'>
                        <span style='font-size: 1.4rem; font-weight: bold; color: {text_color};'>{classification}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Data has not been loaded yet.")

    with col33:
        # ===================== Linearity Test Results =====================
        if 'df_prepared' in st.session_state and st.session_state.df_prepared is not None:
            demand_series = st.session_state.df_prepared['demand']
            result = test_linearity(demand_series)

            with st.container(border=True):
                st.markdown("""
                    <div style='display: flex; align-items: center; gap: 10px;'>
                        <h3 style='margin: 0; margin-top:-15px;  color: #1E88E5;'>📉 Trend / Linearity</h3>
                        <span style="margin-top:-15px;" class="help-icon" data-help="Checks if the demand follows a straight line (linear) or a curve (polynomial). This helps decide between simple linear models or complex non-linear ones.">?</span>
                    </div>
                """, unsafe_allow_html=True)

                linear_r2 = result['Linear R²'] if result['Linear R²'] is not None else "N/A"
                poly_r2 = result['Polynomial R²'] if result['Polynomial R²'] is not None else "N/A"

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Linear R²", linear_r2)
                with c2:
                    st.metric("Poly R²", poly_r2)

                cls = result['Classification']
                text_color = "#c62828" if "Linear" in cls else "#2e7d32"
                border_color = "#ef5350" if "Linear" in cls else "#4caf50"

                st.markdown(f"""
                    <div style='background-color: {border_color}20; border-left: 5px solid {border_color}; padding: 10px; border-radius: 4px; margin-top: -20px;'>
                        <span style='font-size: 1.4rem; font-weight: bold; color: {text_color};'>{cls}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Data has not been loaded yet.")

    col111, col222, col333 = st.columns(3)
    
    with col111:
        #================== Demand Type Classification (ADI) =====================
        if 'df_prepared' in st.session_state and st.session_state.df_prepared is not None:
            demand_series = st.session_state.df_prepared['demand']
        
            with st.container(border=True):
                st.markdown("""
                    <div style='display: flex; align-items: center; gap: 10px;'>
                        <h3 style='margin: 0; margin-top:-15px;  color: #1E88E5;'>📦 Demand Type</h3>
                        <span style="margin-top:-15px;" class="help-icon" data-help="Uses ADI (Average Demand Interval) and CV² to categorize demand as Smooth, Erratic, Intermittent, or Lumpy. Vital for choosing the right inventory strategy.">?</span>
                    </div>
                """, unsafe_allow_html=True)
        
                result = classify_demand_type(demand_series)
                st.session_state.demand_type_classification = result['Classification']
                st.session_state.demand_type_result = result

                c1, c2 = st.columns(2)
                with c1:
                    adi_val = f"{result['ADI']:.2f}" if result['ADI'] is not None else "N/A"
                    st.metric("ADI", adi_val)
                with c2:
                    cv2_val = f"{result['CV²']:.3f}" if result['CV²'] is not None else "N/A"
                    st.metric("CV²", cv2_val)
        
                classification = result['Classification']
                status_map = {
                    "Smooth": ("#2e7d32", "#4caf50"),
                    "Erratic": ("#ef6c00", "#ffa726"),
                    "Intermittent": ("#0277bd", "#039be5"),
                    "Lumpy": ("#c62828", "#ef5350")
                }
                text_color, border_color = status_map.get(classification, ("#424242", "#9e9e9e"))
                
                st.markdown(f"""
                    <div style='background-color: {border_color}20; border-left: 5px solid {border_color}; padding: 10px; border-radius: 4px; margin-top: -20px;'>
                        <span style='font-size: 1.4rem; font-weight: bold; color: {text_color};'>{classification}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Data has not been loaded yet.")

    with col222:
        # ===================== Trend Analysis =====================
        if 'df_prepared' in st.session_state and st.session_state.df_prepared is not None:
            demand_series = st.session_state.df_prepared['demand']
        
            with st.container(border=True):
                st.markdown("""
                    <div style='display: flex; align-items: center; gap: 10px;'>
                        <h3 style='margin: 0; margin-top:-15px;  color: #1E88E5;'>📈 Trend Analysis</h3>
                        <span style="margin-top:-15px;" class="help-icon" data-help="Identifies the long-term direction of the data. Positive trends show growth, while negative trends indicate decline.">?</span>
                    </div>
                """, unsafe_allow_html=True)
        
                result = analyze_trend(demand_series)
                c1, c2 = st.columns(2)
                with c1:
                    slope_val = result['Slope'] if result['Slope'] is not None else "N/A"
                    st.metric("Slope", slope_val)
                with c2:
                    strength_val = result['Trend Strength (%)'] if result['Trend Strength (%)'] is not None else "N/A"
                    st.metric("Strength (%)", f"{strength_val}%")
        
                cls = result['Classification']
                text_color = "#2e7d32" if "Positive" in cls else "#c62828" if "Negative" in cls else "#424242"
                border_color = "#4caf50" if "Positive" in cls else "#ef5350" if "Negative" in cls else "#9e9e9e"
                
                st.markdown(f"""
                    <div style='background-color: {border_color}20; border-left: 5px solid {border_color}; padding: 10px; border-radius: 4px; margin-top: -20px;'>
                        <span style='font-size: 1.4rem; font-weight: bold; color: {text_color};'>{cls}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Data has not been loaded yet.")

    with col333:
        # ===================== Seasonality Analysis =====================
        if 'df_prepared' in st.session_state and st.session_state.df_prepared is not None:
            demand_series = st.session_state.df_prepared['demand']
        
            with st.container(border=True):
                st.markdown("""
                    <div style='display: flex; align-items: center; gap: 10px;'>
                        <h3 style='margin: 0;margin-top:-15px; color: #1E88E5;'>🔄 Seasonality</h3>
                        <span style="margin-top:-15px;" class="help-icon" data-help="Detects repeating patterns at fixed intervals (e.g., weekly or monthly). Strong seasonality suggests the need for seasonal-aware models like SARIMA.">?</span>
                    </div>
                """, unsafe_allow_html=True)
        
                result = analyze_seasonality(demand_series)
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Period", result['Seasonal Period'])
                with c2:
                    st.metric("Strength", result['Seasonal Strength'])
        
                cls = result['Classification']
                colors = {"Strong": ("#2e7d32", "#4caf50"), "Moderate": ("#ef6c00", "#ffa726"), "Weak": ("#424242", "#9e9e9e")}
                text_color, border_color = colors.get(next((k for k in colors if k in cls), ""), ("#c62828", "#ef5350"))
                
                st.markdown(f"""
                    <div style='background-color: {border_color}20; border-left: 5px solid {border_color}; padding: 10px; border-radius: 4px; margin-top: -20px;'>
                        <span style='font-size: 1.4rem; font-weight: bold; color: {text_color};'>{cls}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Data has not been loaded yet.")

    col1111, col2222, col3333 = st.columns(3)

    with col1111:
        # ===================== Temporal Dependency =====================
        if 'df_prepared' in st.session_state and st.session_state.df_prepared is not None:
            demand_series = st.session_state.df_prepared['demand']
        
            with st.container(border=True):
                st.markdown("""
                    <div style='display: flex; align-items: center; gap: 10px;'>
                        <h3 style='margin: 0;margin-top:-15px; color: #1E88E5;'>🔗 Temporal Dependency</h3>
                        <span style="margin-top:-15px;" class="help-icon" data-help="Tests if past values significantly influence future values using the Ljung-Box test. If dependency is high, time-series models are very effective.">?</span>
                    </div>
                """, unsafe_allow_html=True)
        
                result = analyze_temporal_dependency(demand_series)
                c1, c2 = st.columns(2)
                with c1:
                    lb_val = result['Ljung-Box p-value'] if result['Ljung-Box p-value'] is not None else "N/A"
                    st.metric("Ljung-Box p-val", lb_val)
                with c2:
                    acf_val = result['Avg ACF (1-10 lags)'] if result['Avg ACF (1-10 lags)'] is not None else "N/A"
                    st.metric("Avg ACF", acf_val)
        
                cls = result['Classification']
                colors = {"Strong": ("#2e7d32", "#4caf50"), "Moderate": ("#ef6c00", "#ffa726"), "Weak": ("#424242", "#9e9e9e")}
                text_color, border_color = colors.get(next((k for k in colors if k in cls), ""), ("#c62828", "#ef5350"))
                
                st.markdown(f"""
                    <div style='background-color: {border_color}20; border-left: 5px solid {border_color}; padding: 10px; border-radius: 4px; margin-top: -20px;'>
                        <span style='font-size: 1.4rem; font-weight: bold; color: {text_color};'>{cls}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Data has not been loaded yet.")

    with col2222:
        # ===================== Change Point Detection =====================
        if 'df_prepared' in st.session_state and st.session_state.df_prepared is not None:
            demand_series = st.session_state.df_prepared['demand']
        
            with st.container(border=True):
                st.markdown("""
                    <div style='display: flex; align-items: center; gap: 10px;'>
                        <h3 style='margin: 0;margin-top:-15px; color: #1E88E5;'>⚠️ Change Point</h3>
                        <span style="margin-top:-15px;" class="help-icon" data-help="Detects sudden shifts in the mean or variance of the data. Multiple change points suggest structural changes in demand behavior.">?</span>
                    </div>
                """, unsafe_allow_html=True)
        
                result = detect_change_points(demand_series)
                st.metric("Change Points", result['Number of Change Points'])
        
                cls = result['Classification']
                text_color = "#2e7d32" if any(x in cls for x in ["Stable", "No Change"]) else "#ef6c00" if "Single" in cls else "#c62828"
                border_color = "#4caf50" if any(x in cls for x in ["Stable", "No Change"]) else "#ffa726" if "Single" in cls else "#ef5350"
                
                st.markdown(f"""
                    <div style='background-color: {border_color}20; border-left: 5px solid {border_color}; padding: 10px; border-radius: 4px; margin-top: -20px;'>
                        <span style='font-size: 1.4rem; font-weight: bold; color: {text_color};'>{cls}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Data has not been loaded yet.")

    with col3333:
        # ===================== Variance Structure =====================
        if 'df_prepared' in st.session_state and st.session_state.df_prepared is not None:
            demand_series = st.session_state.df_prepared['demand']
        
            with st.container(border=True):
                st.markdown("""
                    <div style='display: flex; align-items: center; gap: 10px;'>
                        <h3 style='margin: 0; margin-top:-15px; color: #1E88E5;'>📉 Variance Structure</h3>
                        <span style="margin-top:-15px;" class="help-icon" data-help="Determines if the volatility (variance) is constant (Homoscedastic) or changing (Heteroscedastic). Changing variance may require data transformation.">?</span>
                    </div>
                """, unsafe_allow_html=True)
        
                result = analyze_variance_structure(demand_series)
                c1, c2 = st.columns(2)
                with c1:
                    levene_val = result['Levene p-value'] if result['Levene p-value'] is not None else "N/A"
                    st.metric("Levene p-val", levene_val)
                with c2:
                    st.metric("CV Variance", f"{result['CV of Variance (%)']}%")
        
                cls = result['Classification']
                text_color, border_color = ("#2e7d32", "#4caf50") if "Homoscedastic" in cls else ("#c62828", "#ef5350")
                
                st.markdown(f"""
                    <div style='background-color: {border_color}20; border-left: 5px solid {border_color}; padding: 10px; border-radius: 4px; margin-top: -20px;'>
                        <span style='font-size: 1.4rem; font-weight: bold; color: {text_color};'>{cls}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Demand data not available for variance analysis.")

    # Navigation buttons
    col_left, col_spacer, col_right = st.columns([1, 2, 1])
    with col_left:
        if st.button("← Back to Data Analysis", use_container_width=True):
            st.session_state.page = "analysis4"
            st.rerun()
            
    with col_spacer:
        
        if st.button("Unit Price Forecasting", use_container_width=True):
            st.session_state.page = "price forecasting"
            st.rerun()
    with col_right:
        if st.button("Next → Run Forecasting Models", type="primary", use_container_width=True):
            if 'recommendation' in st.session_state and st.session_state['recommendation']:
                st.session_state.selected_model = st.session_state['recommendation'].split(" • ")[0]
            else:
                st.session_state.selected_model = "ARIMA"
            st.session_state.page = "recommendation"
            st.rerun()

            
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


def recommendation_page(): 
    st.markdown('<div class="big-title" style="margin-bottom: 1.5rem;">Sammary & Recommendation Forecasting Model</div>', unsafe_allow_html=True)

    # ==================== AUTO-RUN PRICE FORECAST BLOCK ====================
    # We check if the result exists. If not, we calculate it immediately.
    if 'price_forecast_df' not in st.session_state or st.session_state.price_forecast_df is None:
        
        # 1. Validate Data Availability
        if 'df_uploaded' not in st.session_state:
            st.error("Data not found. Please upload a file first.")
            if st.button("Go to Upload"):
                st.session_state.page = "data"
                st.rerun()
            return

        df = st.session_state['df_uploaded'].copy()

        # 2. Validate Columns (Standardize names)
        df.columns = df.columns.str.lower()
        if not all(col in df.columns for col in ['date', 'price']):
            st.error("The uploaded file must contain 'date' and 'price' columns.")
            return

        # 3. Run Calculation Automatically
        with st.spinner("Automatically running Price Forecast..."):
            try:
                # We use a progress bar to visualize the auto-run
                progress_bar = st.progress(0, text="Initializing Price Model...")
                
                # Call your existing forecast function (Assuming it returns: price, df, plot_buf)
                forecasted_price, result_df, plot_buf = forecast_price(df)
                
                progress_bar.progress(50, text="Saving results...")
                
                # SAVE TO SESSION STATE so it persists
                st.session_state.forecasted_price = forecasted_price
                st.session_state.price_forecast_df = result_df
                st.session_state.price_forecast_plot = plot_buf
                
                progress_bar.progress(100, text="Done!")
                
                # --- REMOVED time.sleep(0.5) HERE TO FIX THE ERROR ---
                progress_bar.empty()

                #st.success("Price Forecast calculated successfully.")

            except Exception as e:
                st.error(f"Auto-forecast failed: {str(e)}")
                return
    # =======================================================================

    # ==================== DISPLAY PRICE FORECAST RESULTS ====================
    # Now that we are sure the data exists (from above or previous visit), we display it.
    
    if 'price_forecast_df' in st.session_state:

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div style='display: flex; align-items: center; gap: 10px;'>
                    <h3>Price Forecasting Results</h3>
                    <span class="help-icon" data-help="Price forecasting is conducted using linear regression on the uploaded dataset, and the forecasted values are utilized in EOQ and safety stock inventory management models.">?</span>
                </div>
            """, unsafe_allow_html=True)
             
            # Display the Forecasted Price Card
            forecast_val = st.session_state.forecasted_price
            res_df = st.session_state.price_forecast_df
            
            # Extract the year (Safe method)
            next_year = res_df.loc[res_df['Forecasted Price'].notna(), 'Year'].values[0] if not res_df.empty else "N/A"

            st.markdown(f"""
            <div class="detail-card" style="padding: 0.5rem; border-radius: 10px; background-color: #f8f9fa; margin-bottom: 1rem; border-left: 5px solid #1565C0;">
                <h3 style="color:#1565C0; margin:0;">
                    Forecasted Average Price for Year (
                    <strong style="color: rgb(255, 75, 75);">{next_year}</strong> ): 
                    <strong style="color: rgb(255, 75, 75);">{forecast_val:.3f}</strong> $             
                </h3>
            </div>
            """, unsafe_allow_html=True)

            # Table with scroll enabled (height ~4 rows) and formatting
            st.dataframe(
                res_df.style.format({
                    'Actual Price': '{:.2f}',
                    'Forecasted Price': '{:.2f}'
                }),
                height=280  # Sets a fixed height to enable vertical scrolling
            ) 

        with col2:
            if 'price_forecast_plot' in st.session_state:
              st.image(st.session_state.price_forecast_plot)
        
        #st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


    # ==================== Recommended Forecasting Methods =====================

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


    col1, col2 , col3 = st.columns(3)
    with col2:
        recommendations = []

        stationary = results.get('Stationarity')
        linearity = results.get('Linearity')
        demand_type = results.get('Demand Type')
        variance = results.get('Variance Structure') 

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
            recommendations.append("Random Forest")

        # Rule 7: Smooth & Nonlinear
        if demand_type == 'Smooth' and linearity == 'Non-Linear':
            recommendations.append("XGBoost")

        # Fallback
        if not recommendations:
            recommendations.append("Decision Tree")

        rec_text = " • ".join(dict.fromkeys(recommendations))  # remove duplicates

        # ========================= SAVE TO SESSION STATE =========================
        st.session_state['recommendation'] = rec_text
        # =========================================================================

        # ==================== Demand Type Summary Card ====================
        if 'demand_type_classification' in st.session_state:
            demand_type = st.session_state.demand_type_classification
            
            # Define colors
            if demand_type == "Smooth":
                card_color = "#e6f4ea"
                border_color = "#28a745"
            elif demand_type == "Erratic":
                card_color = "#fff3cd"
                border_color = "#ffc107"
            elif demand_type == "Intermittent":
                card_color = "#d1ecf1"
                border_color = "#17a2b8"
            else:  # Lumpy
                card_color = "#f8d7da"
                border_color = "#dc3545"

            # Display Card (Title + Value Only, No Desc)
            st.markdown(f"""
            <div style="padding: 1rem; border-radius: 10px; margin-bottom: 19px; border-left: 5px solid #1565C0; background-color: #f8f9fa; display: flex; align-items: center; justify-content: center;">
                <div style="text-align: center;">
                    <strong style="font-weight:bold; color:#1565C0; font-size: 1.2rem;">Demand Type Classification:</strong><br>
                    <strong style="color: {border_color}; font-size: 2.5rem;">{demand_type}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        
        else:
            st.info("Demand Type Classification will appear here after analysis.")
    with col3:
        st.markdown(f"""
            <div style="padding: 1rem; border-radius: 10px; margin-bottom: 19px; border-left: 5px solid #1565C0; background-color: #f8f9fa; display: flex; align-items: center; justify-content: center;">
                <div style="text-align: center;">
                    <strong style="font-weight:bold; color:#1565C0; font-size: 1.2rem;">Recommended Model:</strong><br>
                    <strong style="color: rgb(255, 75, 75); font-size: 2.5rem;">{rec_text}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
    with col1:
            st.markdown("""
                <div style='display: flex; align-items: center; gap: 10px;'>
                    <h3>Recommended Models Based on Demand Pattern for Forecasting:</h3>
                    <span style="margin-left:-30px; margin-top:40px; " class="help-icon" data-help="These recommendations are derived from comprehensive analysis of stationarity, seasonality, noise, demand type, trend, and structural changes">?</span>
                </div>
            """, unsafe_allow_html=True)


        

        
    col_left, col_spacer, col_right = st.columns([1, 2, 1])
    with col_left:
        if st.button("← Back to Data Analysis", use_container_width=True):
            st.session_state.page = "results from analysis"
            st.rerun()




    with col_right:
        if st.button("Next → Run Forecasting Models", type="primary", use_container_width=True):
            # Check if a recommendation was generated in the previous step
            if 'recommendation' in st.session_state and st.session_state['recommendation']:
                # Use the saved recommendation
                st.session_state.selected_model = st.session_state['recommendation'].split(" • ")[0]
            else:
                # Fallback to default if no recommendation exists
                st.session_state.selected_model = "ARIMA"
            
            st.session_state.page = "forecasting"
            st.rerun()


    
# =====================================================================================================================
# ================================================ forecasting page ======================================================
# =====================================================================================================================

def forecasting_page():
    # ================= HEADER =================
    st.markdown('<div class="big-title" style="margin-bottom: 2rem;">Model Selection Stage</div>', unsafe_allow_html=True)

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

    # Save training data to session so the next page can use it for re-training/fitting
    st.session_state['train_data'] = train_data

    # ================= MODEL SELECTION =================

    # Adjusted ratios to [3, 3, 8] to accommodate the dropdown in older Streamlit versions
    
    col1, col2, col3 = st.columns([3.5, 4, 7])

    with col1:
        st.markdown(
            '<div style="margin-top: 0.6rem; font-size: 30px; font-weight: bold; white-space: nowrap;">Select Forecasting Model</div>', 
            unsafe_allow_html=True
        )

    with col2: 
        model_options = ["ARIMA", "SARIMA", "SARIMAX", "Decision Tree", "Random Forest", "XGBoost", "CatBoost", "GRU", "TCN"]
        
        default_index = 0
        if 'selected_model' in st.session_state:
            if st.session_state.selected_model in model_options:
                default_index = model_options.index(st.session_state.selected_model)

        selected_model = st.selectbox(
            "",  
            model_options,
            index=default_index,
            key="model_selector_radio", 
            label_visibility="collapsed"
        )
    
    with col3:
        test_start = test_data['date'].min().strftime("%Y-%m-%d")
        test_end = test_data['date'].max().strftime("%Y-%m-%d")
        
        st.markdown(
            f"""
            <h3 style="color:#1E88E5; margin-top: -13px;">
                Results on Testing Data ({test_start} to {test_end})  <br/>
                <span style="color:#717171;">( Last 20%  of the Data )</span>
            </h3>
            """,
            unsafe_allow_html=True
        )
    #st.markdown("---")

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
        
        # Save the fitted model object to session state so we can reuse it in the next page
        st.session_state['fitted_model'] = model
        st.session_state.selected_model = selected_model

        # ===========================================
        # ADDED: CALCULATE SUMS
        # ===========================================
        
        # Calculate Sum of Actual Demand
        total_actual_demand = test_data['demand'].sum()

        # Calculate Sum of Forecasted Demand
        # Use np.sum to handle potential numpy arrays/lists safely
        import numpy as np
        total_forecasted_demand = np.sum(test_forecast)
        
        # Calculate Difference
        demand_variance = total_actual_demand - total_forecasted_demand
        
        # Calculate Percentage Error (Accuracy)
        if total_actual_demand != 0:
            variance_pct = (abs(demand_variance) / total_actual_demand) * 100
        else:
            variance_pct = 0.0


    # ================= RESULTS ON TEST DATA =================
    col1, col2 = st.columns([1, 2])

    with col1:
        result = pd.DataFrame({
            "Date": test_data['date'].values,
            "Actual Demand": test_data['demand'].values,
            "Forecasted Demand": test_forecast
        })


    with col2:
        import altair as alt
        #st.markdown("### Actual vs Forecasted Demand (Test Period)")

        chart_data = result.melt("Date", var_name="Type", value_name="Demand")

        chart = alt.Chart(chart_data).mark_line(strokeWidth=3).encode(
            x="Date:T",
            y="Demand:Q",
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    domain=["Actual Demand", "Forecasted Demand"],
                    range=["#1f77b4", "#d62728"]   # Blue / Red
                ),
                legend=alt.Legend(title="")
            ),
            tooltip=["Date:T", "Type:N", "Demand:Q"]
        ).interactive().properties(height=370)


    with st.spinner("Calculating Linear Regression Forecast..."):
        from sklearn.linear_model import LinearRegression
        import numpy as np

        # 1. Prepare Data for Linear Regression (Daily)
        X_train = train_data['date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        y_train = train_data['demand'].values

        X_test = test_data['date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)

        # 2. Fit Linear Regression on Training Data (First 80%)
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        # 3. Forecast on Test Period (Last 20%)
        lr_forecast = lr_model.predict(X_test)

        # 4. Calculate Sum of Linear Regression Forecast (Daily total for metrics)
        sum_lr_forecast = np.sum(lr_forecast)

    # ================= NEW THREE-COLUMN LAYOUT =================
    main_col1, main_col2, main_col3 = st.columns([3, 3, 8])

    # --- COLUMN 1: Volume Comparison ---

    with main_col1:
        st.markdown("### 📊 Total Demand")
                
        st.markdown(f"""
            <div style="margin-bottom: 20px;">
                <div style="font-size: 25px; font-weight: bold; color: #1E88E5;">Actual Demand</div>
                <div style="font-size: 32px; font-weight: 800; color: #555;">{total_actual_demand:,.2f}</div>
            </div>
            
            <div style="margin-bottom: 20px;">
                <div style="font-size: 25px; font-weight: bold; color: #1E88E5;">{selected_model} Forecast</div>
                <div style="font-size: 32px; font-weight: 800; color: #555;">{total_forecasted_demand:,.2f}</div>
            </div>
            
            <div style="margin-bottom: 20px;">
                <div style="font-size: 25px; font-weight: bold; color: #1E88E5;">Linear Regression</div>
                <div style="font-size: 32px; font-weight: 800; color: #555;">{sum_lr_forecast:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)

    # --- COLUMN 2: Model Performance (Error Metrics) ---
    with main_col2:
        st.markdown("### 📉 Error Metrics")
        # Metric calculation logic
        steps_test = len(test_data)
        test_forecast_vals = model.predict(steps_test)
        actual_vals = test_data['demand'].values
        
        non_zero_mask = actual_vals != 0
        actual_non_zero = actual_vals[non_zero_mask]
        forecast_non_zero = test_forecast_vals[non_zero_mask]
        
        mape_val = (np.mean(np.abs((actual_non_zero - forecast_non_zero) / actual_non_zero)) * 100) if len(actual_non_zero) > 0 else 0.0
        mae_val = np.mean(np.abs(actual_vals - test_forecast_vals))
        rmse_val = np.sqrt(np.mean((actual_vals - test_forecast_vals)**2))
        
        
        st.markdown(f"""
            <div style="margin-bottom: 20px;">
                <div style="font-size: 25px; font-weight: bold; color: #1E88E5;">MAE</div>
                <div style="font-size: 32px; font-weight: 800; color: #555;">{mae_val:.2f}</div>
            </div>
            
            <div style="margin-bottom: 20px;">
                <div style="font-size: 25px; font-weight: bold; color: #1E88E5;">RMSE</div>
                <div style="font-size: 32px; font-weight: 800; color: #555;">{rmse_val:.2f}</div>
            </div>
            
            <div style="margin-bottom: 20px;">
                <div style="font-size: 25px; font-weight: bold; color: #1E88E5;">MAPE</div>
                <div style="font-size: 32px; font-weight: 800; color: #555;">{mape_val:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)

    # --- COLUMN 3: Detailed Comparison Chart ---
    with main_col3:
        st.markdown("### 📈 Segmented Comparison")
        # ── NEW: AGGREGATE DATA INTO 12 EQUAL CHUNKS ────────────────────────────────────
        daily_comparison = pd.DataFrame({
            "Date": test_data['date'],
            "Actual": test_data['demand'].values,
            f"{selected_model}": test_forecast,
            "Linear Regression": lr_forecast
        })
        total_rows = len(daily_comparison)
        # Create a 'Segment' column (0 to 11)
        labels = [f"Segment {i+1}" for i in range(12)]
        daily_comparison['Segment'] = pd.cut(daily_comparison.index, bins=12, labels=labels, right=False)
        # ── FIX: Exclude 'Date' from the sum operation ─────────────────────────────────────
        # We only sum the numerical columns, keeping 'Segment' as the grouping key
        cols_to_sum = [col for col in daily_comparison.columns if col not in ['Date', 'Segment']]
        aggregated_comparison = daily_comparison.groupby('Segment', as_index=False)[cols_to_sum].sum()
        # We find the date in the middle of each chunk to use for plotting
        chunk_size = total_rows // 12
        segment_mid_indices = [int((i * chunk_size) + (chunk_size / 2)) for i in range(12)]
        segment_mid_indices = [min(idx, total_rows - 1) for idx in segment_mid_indices] # Safety check
        # Assign the mid-point dates to the aggregated dataframe
        aggregated_comparison['Date'] = daily_comparison.iloc[segment_mid_indices]['Date'].values
        # ── NEW: Create simplified Linear Regression with only first and last points ─────
        lr_simple = pd.DataFrame({
            "Date": [aggregated_comparison['Date'].iloc[0], aggregated_comparison['Date'].iloc[-1]],
            "Linear Regression": [aggregated_comparison['Linear Regression'].iloc[0], 
                                aggregated_comparison['Linear Regression'].iloc[-1]]
        })
        # Prepare data for Actual and Selected Model (12 points each)
        actual_selected_data = aggregated_comparison[['Date', 'Actual', f"{selected_model}"]].melt(
            "Date", var_name="Forecast Type", value_name="Demand"
        )
        # Prepare data for Linear Regression (2 points only)
        lr_data = lr_simple.melt("Date", var_name="Forecast Type", value_name="Demand")
        # Combine both datasets
        comp_chart_data = pd.concat([actual_selected_data, lr_data], ignore_index=True)
        # Color mapping: Actual=Blue, Selected Model=Red, Regression=Green
        color_scale = alt.Scale(
            domain=["Actual", f"{selected_model}", "Linear Regression"],
            range=["#1f77b4", "#d62728", "#2ca02c"]
        )
        # Create Chart
        comp_chart = alt.Chart(comp_chart_data).mark_line(strokeWidth=3, point=True).encode(
            x=alt.X("Date:T", title="Timeline (Segment Mid-Points)"),
            y=alt.Y("Demand:Q", title="Total Demand per Segment"),
            color=alt.Color("Forecast Type:N", scale=color_scale, legend=alt.Legend(title="Legend")),
            tooltip=["Date:T", "Forecast Type:N", "Demand:Q"]
        ).interactive().properties(height=300)
        st.altair_chart(comp_chart, use_container_width=True)


    col1,col2,colspacer = st.columns([2,3,2])

    with col1:
        # Button to show aggregated data in a dialog/popup
        if st.button("📊 View Aggregated Segment Data", key="view_segment_data_btn"):
            st.session_state.show_segment_dialog = True

        # Display dialog if button was clicked
        @st.dialog("Aggregated Segment Data (12 Points)")
        def show_segment_data():
            st.dataframe(aggregated_comparison, use_container_width=True)
            if st.button("Close"):
                st.session_state.show_segment_dialog = False
                st.rerun()

        if st.session_state.get('show_segment_dialog', False):
            show_segment_data()
    with col2:
        st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2px;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
                color: white;
                margin-bottom:10px;
            ">
                <h4 style="margin: 5px; font-size: 26px; font-weight: bold;">Selected Forecasting Model: {selected_model} </h4>
                <h2 style=">hellow</h2>
            </div>
        """, unsafe_allow_html=True)

    # ================= NAVIGATION =================
    col_left, col_middle, col_right = st.columns([1, 2, 1])
    
    with col_left:
        if st.button("← Back to Data Analysis", use_container_width=True):
            st.session_state.page = "recommendation"
            st.rerun()

    # with col_middle:
    #    # Export Test Results
    #    buffer_test = io.BytesIO()
    #    with pd.ExcelWriter(buffer_test, engine="xlsxwriter") as writer:
    #        result.to_excel(writer, index=False, sheet_name="Test Forecast")
    #    buffer_test.seek(0)

    #    st.download_button(
    #        label="⬇ Download Test Forecast Results as Excel File", 
    #        data=buffer_test,
    #        file_name=f"{selected_model}_Test_Forecast_Results.xlsx",
    #        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    #        use_container_width=True
    #    )

    with col_right:
        if st.button("Next → Run Forecasting Models", type="primary", use_container_width=True):
            st.session_state.page = "future forecasting"
            st.rerun()


# =====================================================================================================================
# ================================================ future forecasting page ======================================================
# =====================================================================================================================


def future_forecasting_page():
    st.markdown('<div class="big-title" style="margin-bottom: 2rem;">Future Forecast Stage</div>', unsafe_allow_html=True)

    # Retrieve the fitted model and training data from the previous page
    if 'fitted_model' not in st.session_state or 'train_data' not in st.session_state:
        st.warning("Please run the model selection step first.")
        return

    model = st.session_state['fitted_model']
    train_data = st.session_state['train_data']
    df = st.session_state['df_uploaded'].copy()

    # ================= FUTURE FORECAST HORIZON =================
    if 'demand_type_classification' in st.session_state:
        classification = st.session_state.demand_type_classification
        
        if classification == "Smooth":
            min_val = 6
            max_val = 24
            default_val = 12
        elif classification == "Erratic":
            min_val = 2
            max_val = 6
            default_val = 3
        else:
            min_val = 1
            max_val = 12
            default_val = 6

        col1, col2, col3 = st.columns([2, 3, 4])
        with col1:
            st.markdown(
                f"""
                <h3 style="color:#1E88E5; margin-top: 40px;">Future Forecast Horizon:</h3>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                "<div style='margin-top:50px;'></div>",
                unsafe_allow_html=True
            )

            # --- CUSTOM CSS TO STYLE THE INPUT FIELD ---
            st.markdown("""
                <style>
                    /* Target the specific input using its key */
                    div[data-testid="stNumberInput"] [role="spinbutton"] {
                        font-size: 24px !important;
                        font-weight: bold !important;
                        color: #1E88E5 !important; /* Blue color for the value */
                        margin-top: -50px;  
                    }

                    /* Target the label text */
                    div[data-testid="stNumberInput"] label > div {
                        font-size: 20px !important;
                        font-weight: bold !important;
                        color: #333 !important;
                        margin-top: -60px;  
                    }
                </style>
            """, unsafe_allow_html=True)

            future_steps = st.number_input(
                "How many months do you want to be forecasted?",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=1,
                key="future_steps_input", # This key links to the CSS above
                help="Choose the number of future months to predict (e.g., 12 for one year)"
            )

            st.session_state.future_steps = future_steps

        with col3:
            # 1. Ensure we have test_data. If not, reconstruct it from the original df and train_data.
            if 'test_data' not in st.session_state:
                split_idx = int(len(df) * 0.8)
                test_data = df.iloc[split_idx:].copy()
            else:
                test_data = st.session_state['test_data']

            # 2. Calculate Metrics
            with st.spinner("Calculating Error Metrics..."):
                steps_test = len(test_data)
                test_forecast = model.predict(steps_test)
                
                actual_demand = test_data['demand'].values
                
                # --- CALCULATE MAPE SAFELY ---
                # 1. Filter out rows where actual_demand is 0 to avoid division by zero
                non_zero_mask = actual_demand != 0
                actual_demand_non_zero = actual_demand[non_zero_mask]
                test_forecast_non_zero = test_forecast[non_zero_mask]
                
                # 2. Calculate MAPE only on the non-zero values
                if len(actual_demand_non_zero) > 0:
                    mape = np.mean(np.abs((actual_demand_non_zero - test_forecast_non_zero) / actual_demand_non_zero)) * 100
                else:
                    mape = 0.0 # If all demand is zero, error is theoretically zero
                
                # --- OTHER METRICS (Standard) ---
                mae = np.mean(np.abs(actual_demand - test_forecast))
                rmse = np.sqrt(np.mean((actual_demand - test_forecast)**2))
                
                ss_res = np.sum((actual_demand - test_forecast) ** 2)
                ss_tot = np.sum((actual_demand - np.mean(actual_demand)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # 3. Display Metrics in a row
            st.markdown("### Model Performance (Error Metrics)")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)

            with col_m1:
                st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 5px;
                        border-radius: 15px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        text-align: center;
                        color: white;
                    ">
                        <h4 style="margin: 0; font-size: 20px;">MAE</h4>
                        <h2 style="margin: -16px; font-size: 26px; font-weight: bold;">{:.2f}</h2>
                    </div>
                """.format(mae), unsafe_allow_html=True)

            with col_m2:
                st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                        padding: 5px;
                        border-radius: 15px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        text-align: center;
                        color: white;
                    ">
                        <h4 style="margin: 0; font-size: 20px;">RMSE</h4>
                        <h2 style="margin: -16px; font-size: 26px; font-weight: bold;">{:.2f}</h2>
                    </div>
                """.format(rmse), unsafe_allow_html=True)

            with col_m3:
                st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                        padding: 5px;
                        border-radius: 15px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        text-align: center;
                        color: white;
                    ">
                        <h4 style="margin: 0; font-size: 20px; opacity: 0.9;">MAPE</h4>
                        <h2 style="margin: -16px; font-size: 26px; font-weight: bold;">{:.2f}</h2>
                    </div>
                """.format(mape), unsafe_allow_html=True)

            with col_m4:
                st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
                        padding: 5px;
                        border-radius: 15px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        text-align: center;
                        color: white;
                    ">
                        <h4 style="margin: 0; font-size: 20px; opacity: 0.9;">R2 Score</h4>
                        <h2 style="margin: -16px; font-size: 26px; font-weight: bold;">{:.2f}</h2>
                    </div>
                """.format(r2), unsafe_allow_html=True)
                            


    else:
        st.warning("Demand type not available yet. Please run classification first.")
        # Fallback if classification is missing
        future_steps = 12


    # ================= FUTURE FORECAST GENERATION =================
    # Predict future periods based on user input
    with st.spinner(f"Generating forecast for the next {future_steps} months..."):
        future_forecast = model.predict(steps=future_steps)
        st.session_state['future_forecast'] = future_forecast

    # Generate future dates
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_steps, freq='MS')
    st.session_state['future_dates'] = future_dates
    
    future_df = pd.DataFrame({
        "Date": future_dates,
        "Forecasted Demand": future_forecast
    })

    # ================= LAYOUT: TABLE | PLOT =================
    #st.markdown("---")

    col_table, col_plot = st.columns([1.5, 2])

    # Left Column: Data Table
    with col_table:
        st.markdown("#### 📊 Forecast Data Table")
        st.dataframe(future_df, use_container_width=True, height=250)

    # Right Column: Line Chart
    with col_plot:
        st.markdown("#### 📈 Forecast Trend")
        st.line_chart(future_df.set_index("Date"), height=250)

    # ================= NAVIGATION =================
    total_next_period_demand = future_forecast.sum()
    formatted_total = f"{total_next_period_demand:,.2f}"
    forecast_end_year = future_dates[-1].year

    st.markdown(f"""
        <div class="detail-card" style="padding-left: 2.5rem; border-radius: 10px; background-color: #f8f9fa; margin-bottom: 1rem; border-left: 5px solid #1565C0;">
            <h3 style="color:#1565C0; margin:0;">
                Total Demand for the Forecasted Period ( 
                <strong style="color: rgb(255, 75, 75);">{future_steps} Months</strong> ): ( <strong style="color: rgb(255, 75, 75);">{formatted_total}</strong> )
            </h3>
        </div>
    """, unsafe_allow_html=True)

   
   
   
    col_left, col_middle, col_right = st.columns([1, 1, 1])
    
    with col_left:
        if st.button("← Back to Model Selection", use_container_width=True):
            st.session_state.page = "forecasting"
            st.rerun()

    with col_middle:
        # Export future forecast
        buffer_future = io.BytesIO()
        with pd.ExcelWriter(buffer_future, engine="xlsxwriter") as writer:
            future_df.to_excel(writer, index=False, sheet_name="Future Forecast")
        buffer_future.seek(0)

        # FIX: Added use_container_width=True to force equal width
        st.download_button(
            label="⬇ Download Excel", 
            data=buffer_future,
            file_name=f"{st.session_state.get('selected_model', 'Model')}_Future_{future_steps}_Periods_Forecast.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True  # <--- This ensures it respects the column width
        )

    with col_right:
        # Check if classification exists to determine button behavior
        # We default to None if not found yet
        demand_type = st.session_state.get('demand_type_classification', None)

        if st.button("Next → EOQ", type="primary", use_container_width=True):
            
            # 1. RUN PRICE FORECASTING AUTOMATICALLY IN BACKGROUND
            if 'df_uploaded' in st.session_state:
                df_price = st.session_state['df_uploaded'].copy()
                df_price.columns = df_price.columns.str.lower()
                
                if 'price' in df_price.columns and 'date' in df_price.columns:
                    with st.spinner("Calculating future price for EOQ..."):
                        try:
                            # Ensure forecast_price function is imported or defined in your script
                            forecasted_price, _, _ = forecast_price(df_price)
                            st.session_state.forecasted_price = forecasted_price
                            # We keep the success message, but it will appear briefly before rerun
                        except Exception as e:
                            st.error(f"Could not forecast price: {e}")
                            st.session_state.forecasted_price = None
            
            # 2. NAVIGATION LOGIC BASED ON DEMAND TYPE
            if demand_type == "Smooth":
                st.session_state.page = "eoq_smooth1"
            
            elif demand_type == "Erratic":
                st.session_state.page = "eoq_erratic1"
            
            elif demand_type in ["Intermittent", "Lumpy"] or demand_type is None:
                # Show warning for Intermittent, Lumpy, or unknown types
                st.warning(f"⚠️ EOQ Calculation skipped for demand type: **{demand_type}**.")
                # Do not rerun, so the user stays on the page to see the warning
                return 
            
            else:
                # Fallback for any other case
                st.warning(f"⚠️ Unknown demand type: **{demand_type}**.")
                return

            # 3. RERUN TO NAVIGATE
            st.rerun()


# =====================================================================================================================
# ================================================ price forecating page ==============================================
# =====================================================================================================================
def price_forecasting_page(): 
    st.markdown('<div class="big-title">Price Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Average Price Forecasting for Next-Year</div>', unsafe_allow_html=True)

    # التحقق من وجود البيانات
    if 'df_uploaded' not in st.session_state:
        st.warning("Please upload your data file first from the Data Upload page.")
        if st.button("← Back to Data Upload"):
            st.session_state.page = "data"
            st.rerun()
        return

    df = st.session_state['df_uploaded']
    df.columns = df.columns.str.lower()

    if not all(col in df.columns for col in ['date', 'price']):
        st.error("The uploaded file must contain both 'date' and 'price' columns.")
        return


    # ================= AUTO-RUN FORECAST =================
    with st.spinner("Calculating forecast..."):
        try:
            forecasted_price, result_df, plot_buf = forecast_price(df.copy())
            
            st.session_state.forecasted_price = forecasted_price
            st.session_state.price_forecast_df = result_df 

            next_year = int(result_df.loc[result_df['Forecasted Price'].notna(), 'Year'].values[0])


            # 2. Create 3 Columns for the data details
            col1, col2, col3 = st.columns([1,1,2]) # Adjust ratios as needed

            with col1:
                st.markdown("#### Price Forecasting Using Linear Regression Method")
                # 1. Main Forecast Result (Full Width)
                st.markdown(f"""
                <div class="detail-card" style="padding: 1.5rem; border-radius: 10px; background-color: #f8f9fa; margin-bottom: 4rem; border-left: 5px solid #1565C0;">
                    <h3 style="color:#1565C0; margin:0;">
                        Forecasted Average Price for Year (
                        <strong style="color: rgb(255, 75, 75);">{next_year}</strong> ): <br/>
                        <strong style="color: rgb(255, 75, 75);">{forecasted_price:.3f}</strong> $
                    </h3>
                </div>
                """, unsafe_allow_html=True)


            with col2:
                st.markdown("#### Annual Price Summary")
                st.dataframe(
                    result_df.style.format({
                        'Actual Price': '{:.2f}',
                        'Forecasted Price': '{:.2f}'
                    }),
                    height=400 
                )  




            with col3:
                st.markdown("#### Price Trend & Forecast")
                st.image(plot_buf, use_container_width=True) 

        except Exception as e:
            st.error(f"An error occurred during forecasting: {str(e)}")

    # Navigation Buttons
    col_left, col_spacer, col_right = st.columns([1, 2, 1])
    with col_left:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "results from analysis"
            st.rerun()

    with col_right:
        if st.button("Next → Recommendated Model", type="primary", use_container_width=True):
            st.session_state.page = "recommendation"
            st.rerun()

# =====================================================================================================================
# ================================================ EOQ & Safety Stock ======================================================
# =====================================================================================================================
import numpy as np
import streamlit as st

def eoq_smooth1_page():
    st.markdown('<div class="big-title">Inventory Planning</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Unit Price, Holding Cost & Supplier Selection</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        # ================= RETRIEVE PRICE =================
        unit_price = 0.0
        
        # Check if we have a forecasted price available
        has_forecast = 'forecasted_price' in st.session_state and st.session_state.forecasted_price is not None

        if has_forecast:
            # Display the available forecasted price
            st.markdown("""
            <style>
            /* Target the text inside st.info */
            .stAlert p {
                font-size: 22px !important;
            }

            /* Make the bold price value even larger and bolder */
            .stAlert strong {
                font-size: 28px !important;
                font-weight: 800 !important;
            }
            </style>
            """, unsafe_allow_html=True)

            st.info(f"Forecasted Price Available: **${st.session_state.forecasted_price:.2f}**")            
            
        # Create the radio selection for the user
        st.markdown("""
        <style>
        /* Target the radio button labels */
        div[data-testid="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p,
        div[role="radiogroup"] > label > div[data-testid="stMarkdownContainer"] > p {
            font-size: 24px !important;
            font-weight: bold !important;
            line-height: 1.5;

        }
        </style>
        """, unsafe_allow_html=True)

        # --- Radio selection ---
        price_selection = st.radio(
            "Please select a unit price in order to proceed with the calculations.",
            ["Use Forecasted Price", "Enter Price Manually"],
            horizontal=True
        )

        if price_selection == "Use Forecasted Price":
            unit_price = st.session_state.forecasted_price
        else:
            unit_price = st.number_input(
                "Enter Unit Price ($)",
                min_value=0.0,
                value=float(st.session_state.forecasted_price),
                step=0.1
            )
        
        # ================= SAVE UNIT PRICE =================
        st.session_state['final_unit_price'] = unit_price

        if 'future_steps' in st.session_state:
            months_to_forecast = st.session_state.future_steps
        else:
            months_to_forecast = 12  

    with col2:

        # ================= REST OF YOUR CODE =================
        if 'df_uploaded' not in st.session_state:
            st.warning("You Should Upload File Firstly")
            return

        df = st.session_state['df_uploaded']

        aggregated_data = aggregate_demand(df)


        selected_model = "ARIMA"       
        selected_time_domain = "daily" 

        forecast_periods =  months_to_forecast

        lookback = 12

        test_data, test_forecast, forecast_df, metrics = run_forecast(
            aggregated_data=aggregated_data,
            time_domain=selected_time_domain,
            model_name=selected_model,
            forecast_periods=forecast_periods,
            lookback=lookback
        )

        sigma_demand = calculate_sigma_demand(test_data["demand"].values)
        sigma_error  = calculate_sigma_error(test_data["demand"].values, test_forecast)

        # ================= SAVE SIGMA VALUES =================
        st.session_state['sigma_demand'] = sigma_demand
        st.session_state['sigma_error'] = sigma_error

        future_forecast = st.session_state.get('future_forecast', None)
        future_dates = st.session_state.get('future_dates', None)       
        demand_type = st.session_state.get('demand_type_classification', None)

        if future_forecast is None or future_dates is None:
            st.warning("Please run the forecasting model first from the Forecasting page.")
            return

        if demand_type is None:
            st.warning("Demand type classification is missing. Please go back to Data Analysis.")
            return

        # ================= CALCULATE AVERAGE LEAD TIME & SERVICE LEVEL FROM DATA =================
        # Standardize column names
        df_copy = df.copy()
        df_copy.columns = df_copy.columns.str.lower().str.strip()
        
        # Initialize variables
        avg_lead_time = None
        avg_service_level = None
        
        # Check if 'lead time' column exists and calculate average (ignoring zeros and NaN)
        if 'lead time' in df_copy.columns:
            lead_time_values = df_copy['lead time'].replace(0, np.nan).dropna()
            if len(lead_time_values) > 0:
                avg_lead_time = lead_time_values.mean()
                st.session_state['avg_lead_time_from_data'] = avg_lead_time
        
        # Check if 'service level' column exists and calculate average (ignoring zeros and NaN)
        if 'service level' in df_copy.columns:
            service_level_values = df_copy['service level'].replace(0, np.nan).dropna()
            if len(service_level_values) > 0:
                avg_service_level = service_level_values.mean()
                st.session_state['avg_service_level_from_data'] = avg_service_level


        total_next_year_demand = float(np.sum(future_forecast))
        formatted_total = f"{total_next_year_demand:,.2f}"
        next_year = future_dates[-1].year - 1

        st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 27px; border-radius: 10px; text-align: center; color: white; margin-bottom: 15px;'>
                    <h3 style='margin: 0;'>📊 Total Demand for the Forecasted Period:<br/> {formatted_total} Units</h3>
                </div>
            """, unsafe_allow_html=True)
        
        # ================= DISPLAY METRICS IN 2x2 GRID =================
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
                <div style='background-color: #e8f5e9; padding: 15px; border-radius: 8px; text-align: center;   margin-bottom: 20px; '>
                    <strong style="font-size:20px ;" >Sigma Demand = {sigma_demand:.2f}</strong>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style='background-color: #fff3e0; padding: 15px; border-radius: 8px; text-align: center;    margin-bottom: 20px;'>
                    <strong style="font-size:20px ;">Sigma Error = {sigma_error:.2f}</strong>
                </div>
            """, unsafe_allow_html=True)
        
        # Second row for Lead Time and Service Level
        col3, col4 = st.columns([1, 1])
        
        with col3:
            if avg_lead_time is not None:
                st.markdown(f"""
                    <div style='background-color: #e3f2fd; padding: 15px; border-radius: 8px; text-align: center;'>
                        <strong style="font-size:20px ;">Avg Lead Time = {avg_lead_time:.2f}</strong>
                    <span style = "font-size: 22px;" class="help-icon" data-help="The Avgerage of the Lead Time & Service Level have been calculated based on the previously uploaded data.">?</span>
                    </div>
                """, unsafe_allow_html=True)
        
        with col4:
            if avg_service_level is not None:
                st.markdown(f"""
                    <div style='background-color: #f3e5f5; padding: 15px; border-radius: 8px; text-align: center;'>
                        <strong style="font-size:20px ;">Avg Service Level = {avg_service_level:.2f}</strong>
                    </div>
                """, unsafe_allow_html=True)


    is_smooth_demand = False
    if isinstance(demand_type, str) and "smooth" in demand_type.lower():
        is_smooth_demand = True

    # ================= Layout ============================================================
    if is_smooth_demand:
                # ================= SUPPLIER & SAFETY STOCK =================
        st.markdown("### Supplier Selection & Safety Stock")

        # 1. Load Supplier Database
        supplier_db_path = os.path.join("Database", "Suppliers Info.xlsx")
        
        # Initialize variables
        lead_time_input = 0
        service_level_input = 0
        selected_supplier_name = "None"

        if not os.path.exists(supplier_db_path):
            st.warning(f"Supplier database not found at: {supplier_db_path}")
            # Allow user to enter manually if file missing
            col1, col2 = st.columns(2)
            with col1:
                lead_time_input = st.number_input("Enter Lead Time (Days)", min_value=1, value=30)
                st.session_state['supplier_lead_time'] = lead_time_input
            with col2:
                service_level_input = st.number_input("Enter Service Level (e.g., 0.95)", min_value=0.0, max_value=1.0, value=0.95)
                st.session_state['supplier_service_level'] = service_level_input
                
        else:
            try:
                df_suppliers = pd.read_excel(supplier_db_path)
                
                # Clean column names
                df_suppliers.columns = df_suppliers.columns.str.strip()
                
                # Ensure required columns exist
                needed_cols = ['Supplier', 'Lead Time (L)', 'Service Level', 'Defect rate %', 'On-time delivery %']
                if not all(col in df_suppliers.columns for col in needed_cols):
                    st.error("Suppliers Excel is missing columns: 'Supplier', 'Lead Time (L)', 'Service Level', 'Defect rate %', 'On-time delivery %'")
                else:
                    # ================= MAIN LAYOUT =================
                    # Split screen into two equal columns: [1, 1]
                    col_left, col_right = st.columns([1, 1])
                    
                    # ================= LEFT COLUMN =================
                    with col_left:
                        # Row 1: Selectbox
                        selected_supplier_name = st.selectbox(
                            "Select Supplier", 
                            options=df_suppliers['Supplier'].tolist(),
                            label_visibility="visible",
                            key="supplier_selector" # Added key
                        )
                        
                        # Row 2: "Extracted from database" message
                        # We calculate this immediately to show the message
                        supplier_info = df_suppliers[df_suppliers['Supplier'] == selected_supplier_name].iloc[0]
                        
                        # Extract values
                        lead_time_input = supplier_info['Lead Time (L)']
                        sl_val = supplier_info['Service Level']
                        service_level_input = sl_val

                        # ================= SAVE SUPPLIER DATA =================
                        st.session_state['selected_supplier_name'] = selected_supplier_name
                        st.session_state['supplier_lead_time'] = lead_time_input
                        st.session_state['supplier_service_level'] = service_level_input
                        st.session_state['supplier_defect_rate'] = supplier_info['Defect rate %']
                        st.session_state['supplier_ontime_delivery'] = supplier_info['On-time delivery %']


                        # Columns for the inner layout
                        c_r1_c1, c_r1_c2, c_r2_c3, c_r2_c24 = st.columns(4)

                        # Helper function for individual items inside the box
                        def kpi_box(title, value, label):
                            return f"""
                                <div style="text-align: center;">
                                    <div style="font-size: 1rem; color: #666; margin-bottom: 5px;">{title}</div>
                                    <div style="font-size: 1.8rem; font-weight: bold; color: #1565C0; margin-bottom: 8px;">{value}</div>
                                </div>
                            """

                        with c_r1_c1:
                            st.markdown(kpi_box(
                                "Lead Time (L)", 
                                f"{supplier_info['Lead Time (L)']}", 
                                "Days"
                            ), unsafe_allow_html=True)
                        
                        with c_r1_c2:
                            # Handling display if value is 0.95 vs 95 based on assumption. Assuming direct display.
                            st.markdown(kpi_box(
                                "Service Level", 
                                f"{supplier_info['Service Level']}", 
                                "Target"
                            ), unsafe_allow_html=True)

                        with c_r2_c3:
                            st.markdown(kpi_box(
                                "Defect Rate", 
                                f"{supplier_info['Defect rate %']}%", 
                                "Quality"
                            ), unsafe_allow_html=True)

                        with c_r2_c24:
                            st.markdown(kpi_box(
                                "On-Time Delivery", 
                                f"{supplier_info['On-time delivery %']*100}%", 
                                "Performance"
                            ), unsafe_allow_html=True)

                        # Close the outer container div
                        st.markdown("</div>", unsafe_allow_html=True)

                    # ================= RIGHT COLUMN =================
                    with col_right:
                        st.markdown("### Holding Cost Parameters")
                        
                        # Create columns: [1, 1] splits the screen into two equal halves
                        col_input, col_result = st.columns([1, 1])

                        with col_input:
                            # Input Field (Left Column)
                            holding_rate = st.number_input(
                                "Holding Rate (%)", 
                                min_value=0.0, 
                                max_value=1.0, 
                                value=0.2, 
                                step=0.01, 
                                format="%.2f",
                                help="Enter the holding cost as a decimal (e.g., 0.20 for 20%)"
                            )

                        with col_result:
                            # Result Display (Right Column)
                            # FIX: Use 'final_unit_price' instead of 'forecasted_price'
                            if 'final_unit_price' in st.session_state:
                                unit_price = st.session_state.final_unit_price
                                holding_cost_per_unit = unit_price * holding_rate
                                
                                st.markdown(f"""
                                    <div style="padding: 1rem; background-color: #e3f2fd; border-left: 5px solid #2196f3; border-radius: 4px; font-size:20px;">
                                        <b>Holding Cost per Unit:</b> $
                                            <span style="color: red; font-size:80x;">{holding_cost_per_unit:.2f} <span> 
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Save to session state for EOQ calculation
                                st.session_state.holding_cost_per_unit = holding_cost_per_unit
                            else:
                                st.warning("Unit price not found. Cannot calculate holding cost per unit.")
                                st.session_state.holding_cost_per_unit = 0.0

            except Exception as e:
                st.error(f"Error reading Supplier database: {e}")
        
        # 4. Convert Service Level to Z-Index (Normal Distribution)
        from scipy.stats import norm
        
        # ppf = Percent Point Function (Inverse of CDF)
        # Ensure service_level_input is valid for norm.ppf (between 0 and 1 exclusive)
        if service_level_input <= 0: service_level_input = 0.0001
        if service_level_input >= 1: service_level_input = 0.9999
        
        z_index = norm.ppf(service_level_input)

        # 5. Calculate Safety Stock
        # SS = Z * sqrt(L) * sigma_error
        # (sigma_error was calculated earlier in your script)
        safety_stock = z_index * np.sqrt(lead_time_input) * sigma_error
        
        # 6. Calculate Reorder Point (ROP)
        # ROP = (Avg Demand / Periods) * Lead Time + Safety Stock
        # We calculate average demand per day (or period) based on your 'months_to_forecast'
        annual_demand = float(np.sum(st.session_state.get('future_forecast', [0])))
        avg_demand_per_period = annual_demand / months_to_forecast
        
        # If lead time is in days, and demand is monthly, we adjust:
        # Assuming 'lead_time_input' is in the same unit as your forecast frequency (e.g., months)
        # If lead time is in DAYS and you forecast MONTHLY, you need to divide by 30 (approx).
        # For this code, I assume Lead Time matches the forecast period unit.
        
        reorder_point = (avg_demand_per_period * lead_time_input) + safety_stock

        # ================= SAVE FINAL CALCULATIONS =================
        st.session_state['z_index'] = z_index
        st.session_state['safety_stock'] = safety_stock
        st.session_state['reorder_point'] = reorder_point
        st.session_state['avg_demand_per_period'] = avg_demand_per_period
        st.session_state['holding_rate'] = holding_rate



        # ================= NAVIGATION =================
        col_left, col_middle, col_right = st.columns([1, 1, 1])
        
        with col_left:
            if st.button("← Back", use_container_width=True):
                st.session_state.page = "recommendation"
                st.rerun()

        with col_right:
            if st.button("Next → Sefety Stick & Re-Ordering Point", type="primary", use_container_width=True):
                st.session_state.page = "eoq_smooth2"
                st.rerun()


#============================================================================================================================
#================================================================================================================================================================
#============================================================================================================================

def eoq_smooth2_page():
    st.markdown('<div class="big-title">Inventory Planning</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Safety Stock & Re-Ordering Point Calculation</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # 1. CORRECTED RETRIEVE DATA FROM SESSION STATE
    # ─────────────────────────────────────────────────────────────
    future_forecast = st.session_state.get('future_forecast', None)
    total_next_year_demand = float(np.sum(future_forecast))
    formatted_total = f"{total_next_year_demand:,.2f}"

    # Basic Inputs
    unit_price = st.session_state.get('final_unit_price', 0.0)
    holding_cost = st.session_state.get('holding_cost_per_unit', 0.0)
    
    # Supplier Info
    supplier_name = st.session_state.get('selected_supplier_name', "Not Selected")
    
    # Variable name used here is 'lead_time'
    lead_time = st.session_state.get('supplier_lead_time', 0)
    
    service_level = st.session_state.get('supplier_service_level', 0)
    
    # Metrics & Stats
    sigma_error = st.session_state.get('sigma_error', 0.0)
    sigma_demand = st.session_state.get('sigma_demand', 0.0)
    
    # Calculated Results
    z_index = st.session_state.get('z_index', 0.0)
    safety_stock = st.session_state.get('safety_stock', 0.0)
    reorder_point = st.session_state.get('reorder_point', 0.0)
    avg_demand = st.session_state.get('avg_demand_per_period', 0.0)

    # Check if data exists
    if unit_price == 0.0:
        st.warning("No inventory data found. Please ensure you ran the calculations on the previous page.")
        return

    # ─────────────────────────────────────────────────────────────
    # 2. DISPLAY DATA IN A 2x2 GRID LAYOUT
    # ─────────────────────────────────────────────────────────────


    st.markdown("### Cost & Demand Inputs")
    st.markdown("""
    <style>
    .info-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 14px 16px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 2px #1565C0;
        margin-bottom: 12px;
    }
    .info-title {
        font-size: 20px;
        margin-bottom: 4px;
    }
    .info-value {
        font-size: 25px;
        font-weight: 600;
        color: #1565C0;
    }
    </style>
    """, unsafe_allow_html=True)


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">💲 Unit Price</div>
            <div class="info-value">${unit_price:.2f}</div>
        </div>
        <div class="info-card">
            <div class="info-title">📦 Total Demand</div>
            <div class="info-value">{formatted_total}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">📊 Avg Demand / Period</div>
            <div class="info-value">{avg_demand:.2f}</div>
        </div>
        <div class="info-card">
            <div class="info-title">🏷 Holding Cost / Unit</div>
            <div class="info-value">${holding_cost:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">📉 Sigma Error</div>
            <div class="info-value">{sigma_error:.2f}</div>
        </div>
        <div class="info-card">
            <div class="info-title">📐 Z-Index</div>
            <div class="info-value">{z_index:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">⏱ Lead Time</div>
            <div class="info-value">{lead_time:.2f}</div>
        </div>
        <div class="info-card">
            <div class="info-title">🏭 Supplier</div>
            <div class="info-value">{supplier_name}</div>
        </div>
        """, unsafe_allow_html=True)



    def cost_box(title, value, formula, is_total=False):
        bg_color = "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)" 
        border_color = "#1565C0" if is_total else "#1565C0"
        font_color = "#0D47A1" if is_total else "#1565C0"
        
        return f"""
            <div style="
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 20px;
                padding: 15px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
                margin-bottom: 20px;
            ">
                <div style="font-size: 1.5rem; color: #666; margin-bottom: 5px;">{title}</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: {font_color}; margin-bottom: 8px;">{value}</div>
                <div style="font-size: 1rem; background: rgba(0,0,0,0.05); padding: 4px 8px; border-radius: 4px;">{formula}</div>
            </div>
        """

    st.markdown("### Safety Stock & Reorder Point Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        # FIX: Use 'lead_time' instead of 'lead_time_input'
        formula_text = f"Z: {z_index:.2f} × root(LT): {np.sqrt(lead_time):.2f} × σ error: {sigma_error:.2f}"
        
        st.markdown(cost_box(
            "Safety Stock (SS)", 
            f"{safety_stock:.2f}", 
            formula_text
        ), unsafe_allow_html=True)

    with col2:
        formula_text = f"Lead Time: {lead_time:.2f} × Avg Demand: {avg_demand:.2f} + SS: {safety_stock:.2f}"

        
        st.markdown(cost_box(
            "Reorder Point (ROP)", 
            f"{reorder_point:.2f}", 
            formula_text
        ), unsafe_allow_html=True)    
            
    
    # Save results to session state (Already saved from previous page, but good to ensure)
    st.session_state.safety_stock = safety_stock
    st.session_state.reorder_point = reorder_point
        
        
    
    col_left, col_middle, col_right = st.columns([1, 1, 1])
    
    with col_left:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "eoq_smooth1"
            st.rerun()

    with col_right:
        if st.button("Next → Ordering Cost Determination", type="primary", use_container_width=True):
            st.session_state.page = "eoq_smooth3"
            st.rerun()



#============================================================================================================================
#================================================================================================================================================================
#============================================================================================================================

def eoq_smooth3_page():
    st.markdown('<div class="big-title">Inventory Planning</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Ordering Cost Determination</div>', unsafe_allow_html=True)
     

    # 1. Ask User how they want to provide Ordering Cost
    cost_method = st.selectbox(
        "Do you know the Total Ordering Cost?",
        options=[
            "Yes, I know the total cost",
            "Yes, I know approximate cost of each main category",
            "No, calculate from Incoterms (detailed)"
        ],
        key="cost_method_selector"
    )

    # Initialize total_ordering_cost
    total_ordering_cost = 0.0

    # ──────────────────────────────────────────────────────────────
    if cost_method == "Yes, I know the total cost":
        # ── Option 1: Direct total input ──
        known_cost = st.session_state.get('total_ordering_cost', 0.0)
        
        input_val = st.number_input(
            "Enter Total Ordering Cost (per Order)",
            min_value=0.0,
            value=known_cost,
            step=1.0,
            key="input_total_ordering_cost",
            help="Enter the estimated total cost per order"
        )
        total_ordering_cost = input_val


    elif cost_method == "Yes, I know approximate cost of each main category":
        # ── Option 2: Enter cost per main category ──
        st.info("Enter approximate values for each major cost category (per order)")


        col_1 ,col_2 = st.columns(2)

        with col_1:
            col1, col2 = st.columns(2)

            with col1:
                admin_cost = st.number_input(
                    "Administrative Costs",
                    min_value=0.0,
                    value=st.session_state.get('cat_admin', 0.0),
                    step=1.0,
                    key="cat_admin_input",
                    help="All administrative & paperwork related costs"
                )

                internal_cost = st.number_input(
                    "Internal Setup / Processing",
                    min_value=0.0,
                    value=st.session_state.get('cat_internal', 0.0),
                    step=1.0,
                    key="cat_internal_input"
                )

            with col2:
                receiving_cost = st.number_input(
                    "Receiving / Inspection",
                    min_value=0.0,
                    value=st.session_state.get('cat_receiving', 0.0),
                    step=1.0,
                    key="cat_receiving_input"
                )

                transport_cost = st.number_input(
                    "Transportation / Freight",
                    min_value=0.0,
                    value=st.session_state.get('cat_transport', 0.0),
                    step=1.0,
                    key="cat_transport_input",
                    help="Total transportation cost depending on Incoterms"
                )
        with col_2:
            purchasing_type = st.radio(
                "Select Purchasing Type",
                options=["Routine", "Blanket", "Contract"],
                horizontal=True,
                key="purchasing_type_select"
            )


        # Calculate total
        total_ordering_cost = admin_cost + internal_cost + receiving_cost + transport_cost

        # Optional: remember values
        st.session_state.cat_admin = admin_cost
        st.session_state.cat_internal = internal_cost
        st.session_state.cat_receiving = receiving_cost
        st.session_state.cat_transport = transport_cost


    else:


        # ================= HELPER FUNCTION TO SAVE STATE =================
        def save_input(widget_key):
            """
            Callback triggered when a number input changes.
            It saves the value from the temporary widget key to a permanent storage key.
            """
            # Get the new value directly from the session state using the widget's key
            new_value = st.session_state[widget_key]
            
            # Save it to a permanent storage key (prefixed with 'storage_')
            # This ensures the value stays even when the widget disappears (switching tabs)
            st.session_state[f"storage_{widget_key}"] = new_value


        # ================= SETUP & DATA LOADING =================

        col_select1, col_select2, col_select3 = st.columns([1, 4, 3])

        # 1. Select Origin & Load Data
        with col_select1:
            purchasing_origin = st.radio(
                "Type of Purchase",
                options=["Local", "Foreign"],
                horizontal=True,
                key="purchasing_origin_select"
            )

            sheet_name = "local" if purchasing_origin == "Local" else "foreign"

            # Load Excel
            db_path = os.path.join("Database", "Ordering Cost Components.xlsx")
            
            if not os.path.exists(db_path):
                st.error(f"Database file not found at: {db_path}")
                st.stop()

            try:
                df = pd.read_excel(db_path, sheet_name=sheet_name)
                
                # Clean & map columns
                clean_cols = {col: col.strip().replace("_", " ") for col in df.columns}
                df = df.rename(columns=clean_cols)
                
                needed_headers = ['cost type', 'components', 'Routine', 'Blanket', 'Contract']
                found_mapping = {}
                for needed in needed_headers:
                    match = next((col for col in df.columns if col.lower() == needed.lower()), None)
                    if match: found_mapping[needed] = match
                
                if len(found_mapping) < 5:
                    st.error(f"Missing required columns in sheet '{sheet_name}'")
                    st.stop()
                    
                df = df.rename(columns=found_mapping)
                
            except Exception as e:
                st.error(f"Error reading sheet '{sheet_name}': {e}")
                st.stop()

        # 2. Select Transport Method
        with col_select2:
            transport_options = ["EXW", "FOB", "CIF"]
            
            # Handle initialization safely
            if 'prev_transport' not in st.session_state:
                st.session_state.prev_transport = transport_options[0]
                
            prev = st.session_state.prev_transport
            
            selected_transport = st.selectbox(
                "Select Transportation Method",
                options=transport_options,
                key="transport_select",
                index=transport_options.index(prev) if prev in transport_options else 0
            )
            
            if selected_transport != prev:
                st.session_state.prev_transport = selected_transport

        # 3. Select Purchasing Type
        with col_select3:
            purchasing_type = st.radio(
                "Select Purchasing Type",
                options=["Routine", "Blanket", "Contract"],
                horizontal=True,
                key="purchasing_type_select"
            )


        # ================= LOGIC: DROPDOWN & DYNAMIC FIELDS =================

        # Define the list of available cost types
        fixed_cost_types = ["Administrative", "Internal Setup", "receiving", selected_transport]

        # Create 2 Columns: Left for Selection, Right for Inputs
        col_selector, col_inputs = st.columns([1, 4])

        with col_selector:
            st.markdown("### Select Cost Type")
            
            # Dropdown to select which cost type to edit
            selected_cost_type = st.selectbox(
                "Choose a category to edit:",
                options=fixed_cost_types,
                key="cost_type_selector",
                label_visibility="collapsed"
            )
            
            st.info(f"Editing: **{selected_cost_type}** components.")

        with col_inputs:
            st.markdown(f"### {selected_cost_type} Components")
            
            # Filter dataframe for the selected type
            group_df = df[df['cost type'] == selected_cost_type].copy()

            if group_df.empty:
                st.warning("No components found for this selection.")
            else:
                components_list = group_df.to_dict('records')
                
                # Loop through the components in steps of 4
                for i in range(0, len(components_list), 4):
                    # Create a row of 4 columns
                    cols = st.columns(4)
                    
                    # Iterate through the current batch of up to 4 items
                    for j in range(4):
                        # Check if there is a component for this column
                        if i + j < len(components_list):
                            item = components_list[i + j]
                            
                            with cols[j]:
                                comp_name = item['components']
                                # Get multiplier based on selected purchasing type
                                multiplier = item[purchasing_type]

                                # Create unique key for the widget
                                widget_key = f"cost_{selected_transport}_{comp_name}_{purchasing_type}_{purchasing_origin}_widget"
                                storage_key = f"storage_{widget_key}"

                                # Retrieve saved value
                                current_value = st.session_state.get(storage_key, 0.0)

                                # --- INPUT FIELD ---
                                input_val = st.number_input(
                                    f"{comp_name} (×{multiplier})",
                                    min_value=0.0,
                                    value=float(current_value),
                                    step=1.0,
                                    key=widget_key,
                                    on_change=save_input,
                                    args=(widget_key,),
                                    label_visibility="visible"
                                )

                                # Calculate line item total
                                calculated_line = input_val * multiplier
                                
                                # Display the calculated subtotal
                                st.markdown(f"""
                                    <div style="text-align:right; color:#1565C0; font-weight:bold; font-size:0.85rem; margin-top:5px; margin-bottom:15px;">
                                        Subtotal: ${calculated_line:.2f}
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                
                                        # ================= CALCULATE GRAND TOTAL =================
        # We need to sum up ALL cost types, not just the one selected.
        total_ordering_cost = 0.0

        # Iterate through ALL available cost types defined
        for cost_type in fixed_cost_types:
            # Filter the main dataframe for this type
            type_df = df[df['cost type'] == cost_type]
            
            if not type_df.empty:
                components_list = type_df.to_dict('records')
                for item in components_list:
                    comp_name = item['components']
                    multiplier = item[purchasing_type]
                    
                    # Construct the SAME key used in the input generation
                    widget_key = f"cost_{selected_transport}_{comp_name}_{purchasing_type}_{purchasing_origin}_widget"
                    storage_key = f"storage_{widget_key}"
                    
                    # Get the saved value
                    saved_val = st.session_state.get(storage_key, 0.0)
                    
                    # Add to total
                    total_ordering_cost += (saved_val * multiplier)

        # ================= DISPLAY TOTAL =================

   # st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


    # Always update session state at the end
    st.session_state.total_ordering_cost = total_ordering_cost    



    col_left, col_middle, col_right = st.columns([1, 1, 1])
    
    with col_left:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "eoq_smooth2"
            st.rerun()

    with col_middle:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: -50px; border-radius: 10px; text-align: center; color: white; margin-bottom: 10px; font-weight:bold;'>
                <h2 style='margin: 0; font-size: 2rem;'>Total Ordering Cost: ${total_ordering_cost:,.2f}</h2>
            </div>
        """, unsafe_allow_html=True)

    with col_right:
        if st.button("Next → EOQ & Total Cost Calculation", type="primary", use_container_width=True):
            st.session_state.page = "eoq_smooth4"
            st.rerun()



#============================================================================================================================
#============================================================================================================================
#============================================================================================================================



def eoq_smooth4_page():
    st.markdown('<div class="big-title">Inventory Planning</div>', unsafe_allow_html=True)
    #st.markdown('<div class="subtitle">Ordering Cost Determination</div>', unsafe_allow_html=True)

    future_forecast = st.session_state.get('future_forecast', None)
    total_next_year_demand = float(np.sum(future_forecast))
    formatted_total = f"{total_next_year_demand:,.2f}"

    # Basic Inputs
    unit_price = st.session_state.get('final_unit_price', 0.0)
    holding_cost = st.session_state.get('holding_cost_per_unit', 0.0)
    
    # Supplier Info
    supplier_name = st.session_state.get('selected_supplier_name', "Not Selected")
    
    # Variable name used here is 'lead_time'
    lead_time = st.session_state.get('supplier_lead_time', 0)
    
    service_level = st.session_state.get('supplier_service_level', 0)
    
    # Metrics & Stats
    sigma_error = st.session_state.get('sigma_error', 0.0)
    sigma_demand = st.session_state.get('sigma_demand', 0.0)
    
    # Calculated Results
    z_index = st.session_state.get('z_index', 0.0)
    safety_stock = st.session_state.get('safety_stock', 0.0)
    reorder_point = st.session_state.get('reorder_point', 0.0)
    avg_demand = st.session_state.get('avg_demand_per_period', 0.0)

    total_ordering_cost  = st.session_state.get('total_ordering_cost ', 0.0)

    holding_rate = st.session_state.get('holding_rate' , 0.0)


    # ================= EOQ & TOTAL COST CALCULATION =================
    st.markdown("### EOQ & Total Cost Calculation")

    # Retrieve required values
    # D (Annual Demand)
    try:
        annual_demand = float(np.sum(st.session_state.get('future_forecast', [0])))
    except:
        annual_demand = 0.0

    # S (Ordering Cost)
    ordering_cost = st.session_state.get('total_ordering_cost', 0.0)

    # H (Holding Cost per Unit)
    holding_cost_per_unit = st.session_state.get('holding_cost_per_unit', 0.0)

    # Unit Price (For Purchase Cost)
    # FIX: Retrieve the user's choice (forecast or manual) saved in session state
    unit_price = st.session_state.get('final_unit_price', 0.0)

    # --- CALCULATE EOQ ---
    if ordering_cost > 0 and annual_demand > 0 and holding_cost_per_unit > 0:
        import math
        
        # EOQ = sqrt( (2 * S * D) / H )
        eoq = math.sqrt((2 * ordering_cost * annual_demand) / holding_cost_per_unit)
        

        # --- CALCULATE TOTAL COST COMPONENTS ---
        
        # 1. Purchase Cost (Cost per unit × Annual Demand)
        purchase_cost_batch = unit_price * annual_demand
        
        # 2. Number of Orders per Year
        num_orders = annual_demand / eoq
        
        # 3. Ordering Cost
        total_ordering_cost_calc = num_orders * ordering_cost
        
        # 4. Holding Cost
        total_holding_cost = holding_cost_per_unit *( (eoq / 2) + safety_stock )
        
        # 5. Total Cost (Sum of all)
        total_cost = purchase_cost_batch + total_ordering_cost_calc + total_holding_cost

        # --- DISPLAY RESULTS ---
        
        # 1. EOQ Result
        col_eoq, col_num = st.columns([1, 1])

        with col_eoq:
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 12px;
                    border-radius: 15px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    text-align: center;
                    color: white;
                ">
                    <h4 style="margin: 0; font-size: 24px;">
                        Economical Order Quantity (EOQ): {eoq:.2f} Units
                    </h4>

                </div>
            """, unsafe_allow_html=True)

        with col_num:
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
                    padding: 12px;
                    border-radius: 15px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    text-align: center;
                    color: white;
                ">
                    <h4 style="margin: 0; font-size: 24px;">
                        Number of Orders per Year: {num_orders:.2f}
                    </h4>
                </div>
            """, unsafe_allow_html=True)


        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # 2. Cost Breakdown Display (3 Columns, 2 Rows)
        
        # Define a reusable style for the boxes (Redefining to ensure local scope is clean)
        def cost_box(title, value, formula, is_total=False):
            bg_color = "#E3F2FD" if is_total else "#F8F9FA"  # Blue for Total, Grey for others
            font_color = "#0D47A1" if is_total else "#1565C0"
            
            return f"""
                <div style="
                    background-color: {bg_color};
                    border-radius: 10px;
                    padding: 15px;
                    text-align: center;
                    border: 1px solid #e0e0e0;
                    box-shadow: 0 2px 2px #1565C0;
                    height: 100%;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                ">
                    <div style="font-size: 1.2rem; color: #666; margin-bottom: 5px;">{title}</div>
                    <div style="font-size: 1.8rem; font-weight: bold; color: {font_color}; margin-bottom: 8px;">{value}</div>
                    <div style="font-size: 1rem; background: rgba(0,0,0,0.05); padding: 4px 8px; border-radius: 4px;">{formula}</div>
                </div>
            """

        # --- Row 1 ---
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown(cost_box(
                "Purchase Cost", 
                f"${purchase_cost_batch:,.2f}", 
                f"Unit Price × Annual Demand = ${unit_price:.2f} × {annual_demand:.2f}"
            ), unsafe_allow_html=True)
            
        with col2:
            st.markdown(cost_box(
                "Ordering Cost", 
                f"${total_ordering_cost_calc:,.2f}", 
                f"No. of Orders × Ordering Cost = {num_orders:.2f} × ${ordering_cost:.2f}"
            ), unsafe_allow_html=True)

        with col3:
            st.markdown(cost_box(
                "Holding Cost", 
                f"${total_holding_cost:,.2f}", 
                f"Holding Cost × ((EOQ/2) + SS) = ${holding_cost_per_unit:.2f} × ({eoq:.2f}/2)"
            ), unsafe_allow_html=True)

        st.markdown("") # Small spacer

        # --- Row 2 ---
        col4, col5, col6 = st.columns([1, 5, 1])

        with col5:
            st.markdown(cost_box(
                "Total Cost", 
                f"${total_cost:,.2f}", 
                "Sum of All Costs",
            ), unsafe_allow_html=True)

        # The other two columns in Row 2 are empty
        with col4:
            # Placeholder for future use or Safety Stock
            pass 
            
        with col6:
            # Placeholder for future use or Reorder Point
            pass

    else:
        st.warning("Cannot calculate EOQ. Please ensure Demand, Ordering Cost, Holding Rate, and Price are all greater than 0.")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


    col_left, col_middle, col_right = st.columns([1, 1, 1])
    
    with col_left:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "eoq_smooth3"
            st.rerun()

    with col_right:
        if st.button("Next → Supplier Discount Comparison", type="primary", use_container_width=True):
            st.session_state.page = "eoq_smooth5"
            st.rerun()






def eoq_smooth5_page():

    st.markdown('<div class="big-title">Inventory Planning</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Supplier Discount Comparison</div>', unsafe_allow_html=True)


    future_forecast = st.session_state.get('future_forecast', None)
    total_next_year_demand = float(np.sum(future_forecast))
    formatted_total = f"{total_next_year_demand:,.2f}"

    # Basic Inputs
    unit_price = st.session_state.get('final_unit_price', 0.0)
    holding_cost = st.session_state.get('holding_cost_per_unit', 0.0)
    
    # Supplier Info
    supplier_name = st.session_state.get('selected_supplier_name', "Not Selected")
    
    # Variable name used here is 'lead_time'
    lead_time = st.session_state.get('supplier_lead_time', 0)
    
    service_level = st.session_state.get('supplier_service_level', 0)
    
    # Metrics & Stats
    sigma_error = st.session_state.get('sigma_error', 0.0)
    sigma_demand = st.session_state.get('sigma_demand', 0.0)
    
    # Calculated Results
    z_index = st.session_state.get('z_index', 0.0)
    safety_stock = st.session_state.get('safety_stock', 0.0)
    reorder_point = st.session_state.get('reorder_point', 0.0)
    avg_demand = st.session_state.get('avg_demand_per_period', 0.0)

    total_ordering_cost  = st.session_state.get('total_ordering_cost ', 0.0)

    holding_rate = st.session_state.get('holding_rate' , 0.0)


    # ================= EOQ & TOTAL COST CALCULATION =================
    # Retrieve required values
    # D (Annual Demand)
    try:
        annual_demand = float(np.sum(st.session_state.get('future_forecast', [0])))
    except:
        annual_demand = 0.0

    # S (Ordering Cost)
    ordering_cost = st.session_state.get('total_ordering_cost', 0.0)

    # H (Holding Cost per Unit)
    holding_cost_per_unit = st.session_state.get('holding_cost_per_unit', 0.0)

    # Unit Price (For Purchase Cost)
    # FIX: Retrieve the user's choice (forecast or manual) saved in session state
    unit_price = st.session_state.get('final_unit_price', 0.0)

    # --- CALCULATE EOQ ---
    if ordering_cost > 0 and annual_demand > 0 and holding_cost_per_unit > 0:
        import math
        
        # EOQ = sqrt( (2 * S * D) / H )
        eoq = math.sqrt((2 * ordering_cost * annual_demand) / holding_cost_per_unit)
        

        # --- CALCULATE TOTAL COST COMPONENTS ---
        
        # 1. Purchase Cost (Cost per unit × Annual Demand)
        purchase_cost_batch = unit_price * annual_demand
        
        # 2. Number of Orders per Year
        num_orders = annual_demand / eoq
        
        # 3. Ordering Cost
        total_ordering_cost_calc = num_orders * ordering_cost
        
        # 4. Holding Cost
        total_holding_cost = holding_cost_per_unit *( (eoq / 2) + safety_stock )
        
        # 5. Total Cost (Sum of all)
        total_cost = purchase_cost_batch + total_ordering_cost_calc + total_holding_cost

   

    import math
    import pandas as pd

    # ================= SUPPLIER DISCOUNT COMPARISON =================
    #st.markdown("### Supplier Discount Analysis")

    # --- SESSION STATE INITIALIZATION ---
    if 'supplier_offers' not in st.session_state:
        st.session_state.supplier_offers = []
    
    # --- FIX: CLEAN UP OLD DATA ---
    if st.session_state.supplier_offers:
        if 'Total Savings' not in st.session_state.supplier_offers[0]:
            st.session_state.supplier_offers = []

    baseline_cost = total_cost 

    #st.markdown("Enter a discount offer from a supplier to compare it against your baseline cost.")

    # 1. INPUT AREA
    col_m, col_d = st.columns(2)
    
    with col_m:
        min_order_qty = st.number_input(
            "Minimum Order Quantity (MOQ)", 
            min_value=1, 
            value=100, 
            step=1,
            key="input_mox",
            help="Enter a discount offer from a supplier to compare it against your baseline cost."
        )
        
    with col_d:
        discount_rate = st.number_input(
            "Discount Rate (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=5.0, 
            step=0.5, 
            help="e.g., 5 for 5%",
            key="input_discount"
        )

    # 2. BUTTONS
    col_add, col_clear = st.columns(2)
    
    with col_add:
        if st.button("Add Offer & Calculate", type="primary", use_container_width=True):
            # --- CALCULATION LOGIC ---
            new_unit_price = unit_price * (1 - (discount_rate / 100.0))
            new_holding_cost = new_unit_price * holding_rate
            new_eoq = math.sqrt((2 * ordering_cost * annual_demand) / new_holding_cost)
            
            if new_eoq >= min_order_qty:
                final_order_qty = new_eoq
                reason = "EOQ Accepted"
            else:
                final_order_qty = min_order_qty
                reason = "Adjusted to MOQ"

            new_purchase_cost = new_unit_price * annual_demand
            new_num_orders = annual_demand / final_order_qty
            new_ordering_cost = new_num_orders * ordering_cost
            new_holding_total_cost = new_holding_cost * (final_order_qty / 2 + safety_stock)
            new_total_cost = new_purchase_cost + new_ordering_cost + new_holding_total_cost
            savings = baseline_cost - new_total_cost

            offer_data = {
                "MOQ": min_order_qty,
                "Discount %": discount_rate,
                "New Price": new_unit_price,
                "Order Qty": final_order_qty,
                "Total Cost": new_total_cost,
                "Total Savings": savings,
                "Status": reason
            }
            st.session_state.supplier_offers.append(offer_data)

    with col_clear:
        if st.button("Reset Comparison", use_container_width=True):
            st.session_state.supplier_offers = []
            st.rerun()

    #st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # 3. DISPLAY RESULTS
    if st.session_state.supplier_offers:
        st.markdown("#### Comparison Table")
        
        # Create DataFrame from session state (This contains NUMERIC data)
        df = pd.DataFrame(st.session_state.supplier_offers)
        
        # --- STEP 1: FORMAT NUMBERS TO STRINGS ---
        # We use .map() to convert numbers to currency strings safely
        df_styled = df.copy()
        
        df_styled["New Price"] = df_styled["New Price"].map("${:,.2f}".format)
        df_styled["Order Qty"] = df_styled["Order Qty"].map("{:.0f}".format)
        df_styled["Discount %"] = df_styled["Discount %"].map("{:.2f}".format)
        df_styled["Total Cost"] = df_styled["Total Cost"].map("${:,.2f}".format)
        df_styled["Total Savings"] = df_styled["Total Savings"].map("${:,.2f}".format)

        # --- STEP 2: APPLY COLORS ---
        # We apply the style using the ORIGINAL 'df' (which has numbers), 
        # but we pass 'df_styled' as the data to display.
        
        def color_savings_negative_red(val):
            """
            Takes a scalar value and returns a string with
            the css property 'color: red' for negative
            strings, black otherwise.
            """
            color = 'red' if val < 0 else 'green'
            return f'color: {color}'

        if "Total Savings" in df.columns:
            styles = df[["Total Savings"]].applymap(color_savings_negative_red)
            
            styler = df_styled.style
            
            def colorize_string_money(s):
                try:
                    val = float(s.replace('$', '').replace(',', ''))
                    return 'color: red' if val < 0 else 'color: green'
                except:
                    return 'color: black'

            styler = styler.applymap(colorize_string_money, subset=["Total Savings"])
        else:
            styler = df_styled.style

        st.dataframe(styler, use_container_width=True, height=150)

        # --- OLD VS BEST COMPARISON ---
        st.markdown("#### Baseline vs. Best Discount")
        col_best, col_old, col_new, col_diff = st.columns([2,1,1,1])
        
        with col_best:
                    # --- HIGHLIGHT THE BEST OPTION ---
            best_offer = min(st.session_state.supplier_offers, key=lambda x: x['Total Cost'])
            
            st.info(f"""
            💡 **Best Option:** MOQ of **{best_offer['MOQ']}** with a **{best_offer['Discount %']}%** discount.
            \n- **Final Unit Price:** ${best_offer['New Price']:.2f}
            \n- **Recommended Order Quantity:** {best_offer['Order Qty']:.2f} units
            """)
#\n- **Total Annual Cost:** ${best_offer['Total Cost']:,.2f}
#\n- **Total Savings:** ${best_offer['Total Savings']:,.2f}
            

        with col_old:
            st.markdown(f"""
                <div style="background-color:#F8F9FA; border:1px solid #E0E0E0; border-radius:10px; padding:15px; text-align:center;">
                    <div style="font-size:0.9rem; color:#666;">Baseline Total Cost</div>
                    <div style="font-size:1.8rem; font-weight:bold; color:#1565C0;">${baseline_cost:,.2f}</div>
                    <div style="font-size:0.85rem; color:#777;">Standard EOQ</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_new:
            st.markdown(f"""
                <div style="background-color:#E8F5E9; border:1px solid #4CAF50; border-radius:10px; padding:15px; text-align:center;">
                    <div style="font-size:0.9rem; color:#666;">Best Offer Cost</div>
                    <div style="font-size:1.8rem; font-weight:bold; color:#2E7D32;">${best_offer['Total Cost']:,.2f}</div>
                    <div style="font-size:0.85rem; color:#777;">Qty: {best_offer['Order Qty']:.0f} • Price: ${best_offer['New Price']:.2f}</div>
                </div>
            """, unsafe_allow_html=True)

        with col_diff:
            savings = baseline_cost - best_offer['Total Cost']
            color_savings = "#2E7D32" if savings > 0 else "#C62828"
            label_savings = "Total Savings" if savings > 0 else "Extra Cost"
            
            st.markdown(f"""
                <div style="background-color:#FFF3E0; border:1px solid #FF9800; border-radius:10px; padding:15px; text-align:center;">
                    <div style="font-size:0.9rem; color:#666;">{label_savings}</div>
                    <div style="font-size:1.8rem; font-weight:bold; color:{color_savings};">${savings:,.2f}</div>
                    <div style="font-size:0.85rem; color:#777;">Difference</div>
                </div>
            """, unsafe_allow_html=True)

    else:
        st.info("👆 Enter an offer above and click 'Add Offer' to start the comparison.")

                                                                                            #//////////////////////////////////////////////////////////////
    
    col_left, col_middle, col_right = st.columns([1, 2,1])
    with col_left:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "eoq_smooth4"
            st.rerun()


    with col_right:
        if st.button("Suppliers Dashboard", type="primary", use_container_width=True):
            st.session_state.page = "supplier report"
            st.rerun()




#============================================================================================================================
#================================================================================================================================================================
#============================================================================================================================




# =====================================================================================================================
# ================================================ supplier report======================================================
# =====================================================================================================================
from reports import show_supplier_dashboard, get_material_report_section,get_data_report_section, ReportGenerator

def supplier_report_page():
    st.markdown('<div class="big-title">Supplier Performance Dashboard</div>', unsafe_allow_html=True)

    # Path to the database
    db_path = os.path.join("Database", "Suppliers Info.xlsx")

    # Check if file exists before proceeding
    if not os.path.exists(db_path):
        st.error(f"Database file not found at: {db_path}")
        return

    # Load data to be used for the Data Table section
    try:
        df = pd.read_excel(db_path)
        df.columns = df.columns.str.strip()
        
        # Ensure numeric types for sorting/display
        df['On-time delivery %'] = pd.to_numeric(df['On-time delivery %'], errors='coerce')
        df['Defect rate %'] = pd.to_numeric(df['Defect rate %'], errors='coerce')
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # This function now contains the logic for rows and descending order
    show_supplier_dashboard()

    col_left, col_spacer, col_right = st.columns([1, 2, 1])
    with col_left:
        if st.button("← Back to Data Analysis", use_container_width=True):
            st.session_state.page = "eoq_smooth5"
            st.rerun()



# =====================================================================================================================
# ================================================ MAIN ROUTER ======================================================
# =====================================================================================================================
# Define the page names
PAGES = [
    "material",
    "data",
    "analysis",
    "analysis2",
    "analysis3",
    "analysis4",
    "results from analysis",
    "recommendation",
    "forecasting",
    "future forecasting",
    "price forecasting",
    "eoq_smooth1",
    "eoq_smooth2",
    "eoq_smooth3",
    "eoq_smooth4",
    "eoq_smooth5",
    "eoq_erratic1",
    "eoq_erratic2",
    "eoq_erratic3",
    "supplier report"
]

# 1. Initialize session state for the current page
if 'page' not in st.session_state:
    st.session_state.page = "material" 

st.sidebar.title("Navigation")

# 2. Handle the "EOQ & Safety Stock" Logic specifically

# Helper to get the current index for the radio button
def get_page_index():
    try:
        return PAGES.index(st.session_state.page)
    except ValueError:
        return 0

# We capture the user's selection from the sidebar
selection = st.sidebar.radio(
    "Go to", 
    PAGES, 
    index=get_page_index(),
    label_visibility="collapsed"
)

# 3. Logic: Did the user just click "EOQ & Safety Stock"?
if selection == "EOQ & Safety Stock" and st.session_state.page != "EOQ & Safety Stock":
    
    # --- YOUR FORECASTING CODE START ---
    # 1. RUN PRICE FORECASTING AUTOMATICALLY IN BACKGROUND
    # We need the data (assuming 'df_uploaded' has a 'price' column)
    if 'df_uploaded' in st.session_state:
        df = st.session_state['df_uploaded'].copy()
        
        # Ensure column names match what forecast_price expects
        df.columns = df.columns.str.lower()
        
        # Check if 'price' exists
        if 'price' in df.columns and 'date' in df.columns:
            with st.spinner("Calculating future price for EOQ..."):
                try:
                    # CALL YOUR FORECAST FUNCTION (Ensure it's imported/defined above)
                    # from price_forecast import forecast_price  # <--- Make sure this import exists if in separate file
                    forecasted_price, _, _ = forecast_price(df)
                    
                    # 2. SAVE THE PRICE TO SESSION STATE
                    st.session_state.forecasted_price = forecasted_price
                    st.success("Future price calculated successfully!")
                except Exception as e:
                    st.error(f"Could not forecast price: {e}")
                    st.session_state.forecasted_price = None
    else:
        st.warning("No data found to forecast price.")
    # --- YOUR FORECASTING CODE END ---

    # 3. NAVIGATE TO EOQ PAGE (Now that forecasting is done)
    st.session_state.page = "EOQ & Safety Stock"
    st.rerun()

# 4. Standard Navigation for all other pages
elif selection != st.session_state.page:
    st.session_state.page = selection
    st.rerun()



if st.session_state.page == "material":
    page_material() 

elif st.session_state.page == "data":
    page_data()

elif st.session_state.page == "analysis":
    analysis_page()

elif st.session_state.page == "analysis2":
    analysis2_page()

elif st.session_state.page == "analysis3":
    analysis3_page()

elif st.session_state.page == "analysis4":
    analysis4_page()

elif st.session_state.page == "results from analysis":
    results_from_analysis_page() 

elif st.session_state.page == "recommendation":
    recommendation_page()

elif st.session_state.page == "forecasting":
    forecasting_page()

elif st.session_state.page == "future forecasting":
    future_forecasting_page()

elif st.session_state.page == "price forecasting":
    price_forecasting_page()

elif st.session_state.page == "eoq_smooth1":
    eoq_smooth1_page()
    
elif st.session_state.page == "eoq_smooth2":
    eoq_smooth2_page()

elif st.session_state.page == "eoq_smooth3":
    eoq_smooth3_page()

elif st.session_state.page == "eoq_smooth4":
    eoq_smooth4_page()

elif st.session_state.page == "eoq_smooth5":
    eoq_smooth5_page()

elif st.session_state.page == "supplier report":
    supplier_report_page()


    

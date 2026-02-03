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




# ========================= Page Config =========================
st.set_page_config(page_title="Material Selection", layout="wide")

# ========================= Session State Initialization =========================
# Initialize state variables
if "page" not in st.session_state:
    st.session_state.page = "landing"  # Initial page is landing

if "df" not in st.session_state:
    st.session_state.df = None

if "period" not in st.session_state:
    st.session_state.period = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "role" not in st.session_state:
    st.session_state.role = None

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ========================= AUTHENTICATION LOGIC =========================
def check_credentials(username, password, selected_role):
    """Checks if user exists in Database folder and matches role."""
    
    # 1. Define possible file names
    db_folder = "Database"
    possible_filenames = ["users.csv", "users.xlsx", "users.xls"] # Added .xls just in case
    
    db_path = None
    file_type = None
    
    # Check if Database folder exists
    if not os.path.exists(db_folder):
        st.error(f"Database folder not found at {os.path.abspath(db_folder)}")
        return False

    # Find the file
    for filename in possible_filenames:
        temp_path = os.path.join(db_folder, filename)
        if os.path.exists(temp_path):
            db_path = temp_path
            file_type = "csv" if filename.endswith(".csv") else "excel"
            break
            
    if db_path is None:
        st.error(f"User file not found in {db_folder}. Looking for 'users.csv', 'users.xlsx', or 'users.xls'.")
        return False

    try:
        # 2. READ THE FILE
        if file_type == "csv":
            df_users = pd.read_csv(db_path)
        else:
            df_users = pd.read_excel(db_path)
        
        # ---------------------------------------------------------
        # DEBUGGING: Print the columns found in your file to Terminal
        # ---------------------------------------------------------
        print(f"DEBUG: Columns found in file: {list(df_users.columns)}")
        # ---------------------------------------------------------

        # 3. CLEAN THE DATA
        # Remove all spaces from headers and convert to lowercase
        df_users.columns = [c.strip().lower().replace(" ", "") for c in df_users.columns]
        
        # Clean the actual data: remove spaces, convert to lowercase
        for col in df_users.columns:
            df_users[col] = df_users[col].astype(str).str.strip().str.lower()

        # 4. FIND THE CORRECT COLUMNS (Flexible Mapping)
        # We need to find columns that contain 'username', 'password', 'role'
        col_user = None
        col_pass = None
        col_role = None

        for col in df_users.columns:
            if "username" in col or "user" in col:
                col_user = col
            if "password" in col or "pass" in col:
                col_pass = col
            if "role" in col:
                col_role = col
        
        # Check if we found all columns
        if not all([col_user, col_pass, col_role]):
            st.error(f"Could not find user columns. Found: {list(df_users.columns)}")
            return False

        # 5. VERIFY CREDENTIALS
        # Clean the input
        username_clean = str(username).strip().lower()
        password_clean = str(password).strip().lower()
        role_clean = str(selected_role).strip().lower()
        
        # Note: Ensure 'middle' maps to the correct role in your database.
        # If your DB says 'Middle Level', role_clean is 'middle', so partial match is safer:
        
        # Filter for matching user
        user_match = df_users[
            (df_users[col_user] == username_clean) & 
            (df_users[col_pass] == password_clean) & 
            # Check if the role in DB contains the input role (e.g. "middle" inside "middle level")
            (df_users[col_role].str.contains(role_clean))
        ]
        
        if not user_match.empty:
            print(f"DEBUG: Login successful for {username_clean}")
            return True
        else:
            print(f"DEBUG: Login failed. Input: {username_clean}/{role_clean}")
            return False

    except Exception as e:
        st.error(f"Error reading database: {e}")
        import traceback
        print(traceback.format_exc()) # Print full error to terminal
        return False


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


# ========================= Load External CSS =========================
def load_css(file_name):
    if not os.path.exists(file_name):
        st.error(f"CSS file '{file_name}' not found!")
        return
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")



# ========================= PAGE: LANDING (Role Selection) =========================
if st.session_state.page == "landing":
    # CSS Styling for the buttons to make them attractive
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            width: 60em !important;
            white-space: normal;
            background-color: #0099ff;
            color: white;
            font-size: 15px !important;
            margin-bottom:20px;
            height: 5em;
            border-radius: 10px;
            border: none;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            display: block;
        }
        div.stButton > button:hover {
            background-color: #007acc;
            transform: translateY(-2px);
            box-shadow: 0px 6px 8px rgba(0,0,0,0.3);
        }
        /* Specifically target the button text */
        div.stButton > button:first-child p {
            font-size: 25px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Display Welcome Message
    st.markdown('<div class="big-title">Welcome To Inventory Management System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Please enter your role to continue</div>', unsafe_allow_html=True)
    
    # Create 3 columns for the buttons (Center aligned)
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        # Button shows "Top Management" but sets role to "admin" (matches database)
        if st.button("Top Management", use_container_width=False):
            st.session_state.role = "admin"
            st.success("Welcome, Top Management!")
            st.session_state.page = "login"
            st.rerun()
            
        # Button shows "Middle" and sets role to "middle" (matches database)
        if st.button("Middle Level", use_container_width=False):
            st.session_state.role = "middle"
            st.success("Welcome, Middle Management!")
            st.session_state.page = "login"
            st.rerun()
            
        # Button shows "Specialist" but sets role to "user" (matches database)
        if st.button("Specialist", use_container_width=False):
            st.session_state.role = "user"
            st.success("Welcome, Specialist!")
            st.session_state.page = "login"
            st.rerun()
    st.stop() 


# ========================= PAGE: LOGIN =========================
if st.session_state.page == "login":
    # Custom CSS for attractive login styling
    st.markdown("""
    <style>
    /* Login form container with blue border and shadow */
    div[data-testid="stForm"] {
        background: #ffffff;
        padding: 3rem 2.5rem;
        border-radius: 20px;
        border: 3px solid #1E88E5;
        box-shadow: 0 15px 35px rgba(30, 136, 229, 0.3), 
                    0 5px 15px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    div[data-testid="stForm"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 45px rgba(30, 136, 229, 0.4), 
                    0 10px 20px rgba(0, 0, 0, 0.15);
    }
    
    /* Input labels */
    div[data-testid="stForm"] label {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    
    /* Input fields */
    div[data-testid="stForm"] input {
        border: 2px solid #e3f2fd;
        border-radius: 12px;
        padding: 0.85rem 1.2rem;
        font-size: 1.5rem;
        transition: all 0.3s ease;
        background: #f8fbff;
    }
    
    div[data-testid="stForm"] input:focus {
        border-color: #1E88E5;
        box-shadow: 0 0 0 4px rgba(30, 136, 229, 0.15);
        background: #ffffff;
        outline: none;
    }
    
    /* Login button */
    div[data-testid="stForm"] button[kind="primary"] {
        background: linear-gradient(135deg, #1E88E5 0%, #1565c0 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.95rem 2rem;
        border-radius: 12px;
        border: none;
        margin-top: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(30, 136, 229, 0.4);
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    div[data-testid="stForm"] button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 25px rgba(30, 136, 229, 0.5);
        background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
    }
    
    /* Back button styling */
    div.stButton > button:not([kind="primary"]) {
        background: transparent;
        color: #1E88E5;
        font-weight: 500;
        font-size: 1rem;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border: 2px solid #1E88E5;
        margin-top: 2rem;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:not([kind="primary"]):hover {
        background: #1E88E5;
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 5px 15px rgba(30, 136, 229, 0.3);
    }
    
    /* Alert messages */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
        font-weight: 500;
    }
    
    /* Success message */
    div[data-baseweb="notification"][kind="success"] {
        background: #e8f5e9;
        border-left-color: #4caf50;
        color: #2e7d32;
    }
    
    /* Error message */
    div[data-baseweb="notification"][kind="error"] {
        background: #ffebee;
        border-left-color: #f44336;
        color: #c62828;
    }
    
    /* Add spacing between form elements */
    div[data-testid="stForm"] > div {
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="big-title">Secure Access Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Please authenticate to continue</div>', unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("Login", use_container_width=True, type="primary")
            
            if submit_button:
                if check_credentials(username, password, st.session_state.role):
                    st.session_state.logged_in = True
                    user_role = st.session_state.get("role")
                    if user_role == "user":
                        # ================= FIXED: EVERYONE GOES TO "material" PAGE =================
                        st.session_state.page = "Material Selection"
                        st.rerun()
                    elif user_role == "middle":
                        # ================= FIXED: EVERYONE GOES TO "material" PAGE =================
                        st.session_state.page = "Material Selection"
                        st.rerun()
                    else:
                        # ================= FIXED: EVERYONE GOES TO "material" PAGE =================
                        st.session_state.page = "top manager"
                        st.rerun()
                else:
                    st.error(f"Error reading uploaded file")
                    st.rerun()

        # Back button centered
        if st.button("← Back to Role Selection", use_container_width=True):
            st.session_state.role = None
            st.session_state.page = "landing"
            st.rerun()
        
    st.stop()


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
                padding: 0.5rem; border-radius: 10px; margin-bottom: 5px; border-left: 5px solid #1E88E5; background-color: rgb(240 242 253 / 31%); font-size: 1.6rem; font-weight: bold; color: #1E88E5;
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
                    <div style="border: 1px solid #1E88E5; padding: 0.5rem; border-radius: 5px; margin-bottom: 19px;">
                        <span class="detail-label" style="font-weight:bold;">{col_name}:</span><br>
                        <span style="font-size: 1.4rem; color: #555;">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown('<div class="next-btn-container">', unsafe_allow_html=True)


    user_role = st.session_state.get("role", "admin")

    col_left, col_middle , col_right = st.columns([1, 4, 1])

    with col_left:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()


    with col_right:
        if st.button("Next → Data Uploading", type="primary", use_container_width=True):
            if selected_code == CODE_PLACEHOLDER:
                st.warning("Please select an Item Code to proceed.")
            else:
                user_role = st.session_state.get("role")
                
                if user_role == "user":
                    # ================= SAVE SECTION =================
                    # 1. Get the row as a Dictionary
                    material_data_dict = filtered_df[filtered_df['Item Code'] == selected_code].iloc[0].to_dict()
                    
                    # 2. Save to session state
                    # 'selected_material_info' is the key we will use in the other file
                    st.session_state.selected_material_info = material_data_dict
                    st.session_state.selected_material_row = material_data_dict  # <-- ADD THIS LINE
                    
                    # 3. Explicitly save specific values if you prefer (optional but handy)
                    st.session_state.selected_purchasing_unit = material_data_dict['Purchasing Unit']
                    
                    # 4. Change page
                    st.session_state.page = "data"
                    st.rerun()

                elif user_role == "middle":
                    # ================= FIX: Save material_data_dict before navigating =================
                    material_data_dict = filtered_df[filtered_df['Item Code'] == selected_code].iloc[0].to_dict()
                    st.session_state.selected_material_info = material_data_dict
                    st.session_state.selected_material_row = material_data_dict  # <-- ADD THIS LINE
                    # ===============================================================================
                    st.session_state.page = "material_excel"
                    st.rerun()
                    
                else:
                    # Default for admin/others
                    material_data_dict = filtered_df[filtered_df['Item Code'] == selected_code].iloc[0].to_dict()
                    st.session_state.selected_material_info = material_data_dict
                    st.session_state.selected_material_row = material_data_dict  # <-- ADD THIS LINE
                    st.session_state.page = "data"
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)



# =====================================================================================================================
# ============================================ PAGE 2 : DATA & PERIOD ================================================
# =====================================================================================================================
def page_data():
    st.markdown('<div class="big-title">Data & Period Selection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle" style="margin-bottom: 0rem;">Upload Demand Data & Choose Analysis Period</div>', unsafe_allow_html=True)

    # ===== Inject the SAME table style CSS =====
    st.markdown("""
    <style>
        /* General Font Setup */
        .custom-table-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin-bottom: 30px;
            max-height: 400px;          /* ≈ 10 rows */
            overflow-y: auto;           /* vertical scroll */
            overflow-x: auto;
        }

        /* The Table */
        .modern-table {
            width: 100%;
            border-collapse: collapse;
            border: 2px solid #1E88E5; /* Outer border using the requested color */
            border-radius: 8px;
            overflow: hidden; /* Ensures border radius works */
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* Table Headers */
        .modern-table th {
            background-color: #1E88E5; /* Blue Header */
            color: white;
            padding: 10px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 1em;
            letter-spacing: 0.5px;
        }

        /* Table Rows */
        .modern-table tr.data-row {
            border-bottom: 1px solid #e0e0e0;
            transition: background-color 0.2s ease;
        }

        /* Hover Effect */
        .modern-table tr.data-row:hover {
            background-color: #f1f8ff; /* Very light blue hover */
        }

        /* Table Cells (Default) */
        .modern-table td {
            padding: 10px;
            font-size: 0.8em;
            color: #333;
            border: 1px solid #e0e0e0; /* Light grey borders for info */
            border-top: none;
            border-bottom: none;
        }

        /* Fix First and Last Cell Borders relative to Table Border */
        .modern-table td:first-child {
            border-left: none;
        }
        .modern-table td:last-child {
            border-right: none;
        }

        /* Tight label style (your original) */
        .tight-label {
            margin-bottom: 0px !important;
            display: block;
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 2])
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
            db_path = "Database/aaai demand history.xlsx"

            if os.path.exists(db_path):
                try:
                    # Read the sheet corresponding to the item code
                    df_display = pd.read_excel(db_path, sheet_name=item_code)
                except ValueError:
                    st.error(f"Sheet '{item_code}' not found in 'aaai demand history.xlsx'")
                except Exception as e:
                    st.error(f"Error reading database file: {e}")
            else:
                st.error(f"Database file not found: {db_path}")
        else:
            st.warning("Please go back and select a material first to load its history.")

    # ---------------- DISPLAY TABLE AND SUMMARY ----------------
    # Only display if we have data to show (either from DB or Upload)
    if not df_display.empty:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        left_col, right_col = st.columns([2, 1])

        # ===== LEFT: Table =====
        with left_col:
            df_copy = df_display.copy()
            try:
                # Try to format the first column as date if possible
                df_copy.iloc[:, 0] = pd.to_datetime(df_copy.iloc[:, 0]).dt.strftime('%Y-%m-%d')
            except:
                pass

            # Build HTML table using the same "modern-table" style
            table_html = '<div class="custom-table-container"><table class="modern-table">'

            # Header
            table_html += '<thead><tr>'
            for col in df_copy.columns:
                table_html += f'<th>{col}</th>'
            table_html += '</tr></thead><tbody>'

            # Rows
            for _, row in df_copy.iterrows():
                table_html += '<tr class="data-row">'
                for col in df_copy.columns:
                    val = row[col]
                    table_html += f'<td>{val}</td>'
                table_html += '</tr>'

            table_html += '</tbody></table></div>'

            st.markdown(table_html, unsafe_allow_html=True)

        # ===== RIGHT: Summary =====
        with right_col:
            num_periods = len(df_display) - 1
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
            st.session_state.page = "Material Selection"
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
# ============================================ middle first page  ================================================
# =====================================================================================================================

def page_material_excel():
    st.markdown('<div class="big-title">Material Cost History</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Average Unit Price by Supplier and Year</div>', unsafe_allow_html=True)

    if "selected_material_row" not in st.session_state:
        st.warning("No material selected. Please go back.")
        return

    item_code = st.session_state.selected_material_row["Item Code"]
    db_path = "Database/aaai demand history.xlsx"

    if not os.path.exists(db_path):
        st.error("Demand history database not found.")
        return

    try:
        df = pd.read_excel(db_path, sheet_name=item_code)
    except ValueError:
        st.error(f"No sheet found for Item Code: {item_code}")
        return

    # ================= VALIDATION =================
    required_cols = ["date", "supplier", "quantity received", "price received"]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        st.error(f"Missing columns in Excel file: {missing_cols}")
        return

    # ================= DATA PREPARATION =================
    df["date"] = pd.to_datetime(df["date"])
    df["Year"] = df["date"].dt.year

    df["Total Price"] = df["quantity received"] * df["price received"]

    # ================= AGGREGATION =================
    agg_df = (
        df
        .groupby(["Year", "supplier"], as_index=False)
        .agg(
            total_price=("Total Price", "sum"),
            total_quantity=("quantity received", "sum"),
            avg_service_level=("service level", "mean")
        )
    )

    agg_df["Avg Unit Price"] = agg_df["total_price"] / agg_df["total_quantity"]


    # ================= DISPLAY RAW DATA =================
    st.subheader(f"Item Code: {item_code}")
    st.dataframe(df, use_container_width=True)

    # ================= GROUPED BAR CHART =================
    st.subheader("Average Unit Price per Supplier (Grouped by Year)")

    import matplotlib.pyplot as plt
    import numpy as np

    years = sorted(agg_df["Year"].unique())
    suppliers = sorted(agg_df["supplier"].unique())

    bar_width = 0.8 / len(suppliers)
    x = np.arange(len(years))

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, supplier in enumerate(suppliers):
        supplier_data = agg_df[agg_df["supplier"] == supplier]

        values = [
            supplier_data[supplier_data["Year"] == year]["Avg Unit Price"].values[0]
            if year in supplier_data["Year"].values else 0
            for year in years
        ]

        ax.bar(
            x + i * bar_width,
            values,
            width=bar_width,
            label=supplier
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Average Unit Price")
    ax.set_title("Supplier Price Comparison by Year")

    ax.set_xticks(x + bar_width * (len(suppliers) - 1) / 2)
    ax.set_xticklabels(years)

    ax.legend(title="supplier", bbox_to_anchor=(1.02, 1), loc="upper left")

    st.pyplot(fig)

#-----------------------------------
    st.subheader("Total Quantity Received per Supplier (Grouped by Year)")

    import matplotlib.pyplot as plt
    import numpy as np

    years = sorted(agg_df["Year"].unique())
    suppliers = sorted(agg_df["supplier"].unique())

    bar_width = 0.8 / len(suppliers)
    x = np.arange(len(years))

    fig_qty, ax_qty = plt.subplots(figsize=(12, 5))

    for i, supplier in enumerate(suppliers):
        supplier_data = agg_df[agg_df["supplier"] == supplier]

        values = [
            supplier_data[supplier_data["Year"] == year]["total_quantity"].values[0]
            if year in supplier_data["Year"].values else 0
            for year in years
        ]

        ax_qty.bar(
            x + i * bar_width,
            values,
            width=bar_width,
            label=supplier
        )

    ax_qty.set_xlabel("Year")
    ax_qty.set_ylabel("Total Quantity Received")
    ax_qty.set_title("Supplier Quantity Comparison by Year")

    ax_qty.set_xticks(x + bar_width * (len(suppliers) - 1) / 2)
    ax_qty.set_xticklabels(years)

    ax_qty.legend(title="supplier", bbox_to_anchor=(1.02, 1), loc="upper left")

    st.pyplot(fig_qty)


#------------------------
    # ================= GROUPED BAR CHART (SERVICE LEVEL) =================
    st.subheader("Average Service Level per Supplier (Grouped by Year)")

    import matplotlib.pyplot as plt
    import numpy as np

    years = sorted(agg_df["Year"].unique())
    suppliers = sorted(agg_df["supplier"].unique())

    bar_width = 0.8 / len(suppliers)
    x = np.arange(len(years))

    fig_sl, ax_sl = plt.subplots(figsize=(12, 5))

    for i, supplier in enumerate(suppliers):
        supplier_data = agg_df[agg_df["supplier"] == supplier]

        values = [
            supplier_data[supplier_data["Year"] == year]["avg_service_level"].values[0]
            if year in supplier_data["Year"].values else 0
            for year in years
        ]

        ax_sl.bar(
            x + i * bar_width,
            values,
            width=bar_width,
            label=supplier
        )

    ax_sl.set_xlabel("Year")
    ax_sl.set_ylabel("Average Service Level")
    ax_sl.set_title("Supplier Service Level Comparison by Year")

    ax_sl.set_xticks(x + bar_width * (len(suppliers) - 1) / 2)
    ax_sl.set_xticklabels(years)

    ax_sl.legend(title="Supplier", bbox_to_anchor=(1.02, 1), loc="upper left")

    st.pyplot(fig_sl)

#----------------------------------------
    # ================= GROUPED BAR CHART – AVERAGE LEAD TIME =================
    st.subheader("Average Lead Time per Supplier (Grouped by Year)")

    # Check if the column exists
    lead_time_col = None
    possible_names = ["lead time", "Lead Time", "lead_time", "LeadTime", "delivery days", "leadtime"]

    for col in df.columns:
        if col.lower().replace(" ", "") in [n.lower().replace(" ", "") for n in possible_names]:
            lead_time_col = col
            break

    if not lead_time_col:
        st.warning("Column 'lead time' (or similar) not found in the data → skipping lead time chart.")
    else:
        # Add average lead time to aggregation
        agg_df["avg_lead_time"] = df.groupby(["Year", "supplier"])[lead_time_col].mean().values

        import matplotlib.pyplot as plt
        import numpy as np

        years = sorted(agg_df["Year"].unique())
        suppliers = sorted(agg_df["supplier"].unique())

        bar_width = 0.8 / len(suppliers)
        x = np.arange(len(years))

        fig_lt, ax_lt = plt.subplots(figsize=(12, 5))

        for i, supplier in enumerate(suppliers):
            supplier_data = agg_df[agg_df["supplier"] == supplier]

            values = []
            for year in years:
                match = supplier_data[supplier_data["Year"] == year]
                if not match.empty:
                    values.append(match["avg_lead_time"].iloc[0])
                else:
                    values.append(0)   # or np.nan if you prefer gaps

            ax_lt.bar(
                x + i * bar_width,
                values,
                width=bar_width,
                label=supplier
            )

        ax_lt.set_xlabel("Year")
        ax_lt.set_ylabel("Average Lead Time (days)")
        ax_lt.set_title("Supplier Lead Time Comparison by Year")

        ax_lt.set_xticks(x + bar_width * (len(suppliers) - 1) / 2)
        ax_lt.set_xticklabels(years)

        ax_lt.legend(title="Supplier", bbox_to_anchor=(1.02, 1), loc="upper left")

        # Optional: better grid & formatting
        ax_lt.grid(axis='y', linestyle='--', alpha=0.5)
        ax_lt.set_axisbelow(True)

        st.pyplot(fig_lt)

#-------------------------------
# ================= TOTAL QUANTITY RECEIVED PER YEAR (SIMPLE BAR CHART) =================
# ================= TOTAL QUANTITY RECEIVED PER YEAR =================
    st.subheader("Total Quantity Received per Year")

    yearly_total = (
        df
        .groupby("Year", as_index=False)
        .agg(total_quantity=("quantity received", "sum"))
        .sort_values("Year")
    )

    if yearly_total.empty:
        st.info("No quantity data available.")
    else:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        # Real historical years
        real_years = yearly_total["Year"].astype(int).tolist()
        real_quantities = yearly_total["total_quantity"].tolist()

        # Add "This Year" as extra bar
        all_labels = [str(y) for y in real_years] + ["This Year"]
        all_values  = real_quantities + [40000]

        # Positions: years consecutive, then small gap before "This Year"
        x_positions = list(range(len(real_years))) + [len(real_years) + 0.7]

        fig, ax = plt.subplots(figsize=(10, 5.5))

        bars = ax.bar(
            x_positions,
            all_values,
            color = ["#4CAF50"] * len(real_years) + ["#FF7043"],   # different color for This Year
            width = 0.68,
            edgecolor = "black",
            linewidth = 0.8
        )

        # Value labels on top
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                h,
                f"{int(h):,}",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(all_labels)

        ax.set_xlabel("Year")
        ax.set_ylabel("Total Quantity Received")
        ax.set_title("Annual Total Quantity Received")

        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format(int(x), ",")))

        plt.tight_layout()
        st.pyplot(fig)
#-----------------------------------------------------
# ================= AVERAGE (QUANTITY × PRICE) PER YEAR =================
# ================= AVERAGE (QUANTITY × PRICE) PER YEAR =================
    st.subheader("Average Receipt Value per Year (Qty × Price per record)")

    yearly_avg_value = (
        df
        .groupby("Year", as_index=False)
        .agg(
            avg_receipt_value=("Total Price", "mean"),
            count_records=("Total Price", "count")
        )
        .sort_values("Year")
    )

    if yearly_avg_value.empty:
        st.info("No price/quantity data available for this calculation.")
    else:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        # Real historical years
        real_years = yearly_avg_value["Year"].astype(int).tolist()
        real_avg_values = yearly_avg_value["avg_receipt_value"].tolist()
        real_counts = yearly_avg_value["count_records"].tolist()

        # Add "This Year" bar
        all_labels = [str(y) for y in real_years] + ["This Year"]
        all_values = real_avg_values + [150000]
        
        # Positions: consecutive for history, small gap before "This Year"
        x_positions = list(range(len(real_years))) + [len(real_years) + 0.7]

        fig_avg, ax_avg = plt.subplots(figsize=(10, 5.5))

        bars = ax_avg.bar(
            x_positions,
            all_values,
            color=["#FF9800"] * len(real_years) + ["#E91E63"],   # orange historical → pink/magenta for This Year
            width=0.68,
            edgecolor="black",
            linewidth=0.9
        )

        # Value labels on top of each bar
        for i, bar in enumerate(bars):
            height = bar.get_height()
            count_text = f"n={real_counts[i]}" if i < len(real_counts) else ""
            
            # Main value
            ax_avg.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f"{height:,.0f}" if height >= 100 else f"{height:,.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#111"
            )
            
            # Small record count note (only for real years)
            if count_text:
                ax_avg.text(
                    bar.get_x() + bar.get_width()/2,
                    height * 0.92 if height > 0 else 0,
                    count_text,
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="#555"
                )

        ax_avg.set_xticks(x_positions)
        ax_avg.set_xticklabels(all_labels)

        ax_avg.set_xlabel("Year", fontsize=12)
        ax_avg.set_ylabel("Average Receipt Value\n(Qty × Unit Price)", fontsize=12)
        ax_avg.set_title("Average Value per Receipt / Delivery Event per Year", fontsize=14, pad=15)

        ax_avg.grid(axis="y", linestyle="--", alpha=0.35)
        ax_avg.set_axisbelow(True)

        # Thousands separator
        ax_avg.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ",")))

        plt.tight_layout()
        st.pyplot(fig_avg)

        st.caption("• \"This Year\" shows a fixed reference value of 2000 (not calculated from data)")
        
    # ================= NAVIGATION =================
    st.markdown("<hr>", unsafe_allow_html=True)

    col_back, col_next = st.columns([1, 1])

    with col_back:
        if st.button("← Back"):
            st.session_state.page = "Material Selection"
            st.rerun()

    with col_next:
        if st.button("Continue → Data Uploading", type="primary"):
            st.session_state.page = "data"
            st.rerun()




# =====================================================================================================================
# ============================================ PAGE 3 : ANALYSIS ====================================================
# =====================================================================================================================
import streamlit as st
import pandas as pd
import os

st.markdown("""
<style>
    /* General Font Setup */
    .custom-table-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin-bottom: 30px;
        overflow-x: auto;
    }

    /* The Table */
    .modern-table {
        width: 100%;
        border-collapse: collapse;
        border: 2px solid #1E88E5; /* Outer border using the requested color */
        border-radius: 8px;
        overflow: hidden; /* Ensures border radius works */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Table Headers */
    .modern-table th {
        background-color: #1E88E5; /* Blue Header */
        color: white;
        padding: 12px 15px;
        text-align: left;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 1.2em;
        letter-spacing: 0.5px;
    }

    /* Specific Logic for Header Borders (Removing divider between Mat Name and SKU) */
    .modern-table th.mat-name-header {
        border-right: 1px solid #1E88E5; /* Blend with background or remove */
    }
    .modern-table th.sku-header {
        border-left: none; /* No border between Mat Name and SKU */
    }

    /* Table Rows */
    .modern-table tr.data-row {
        border-bottom: 1px solid #e0e0e0;
        transition: background-color 0.2s ease;
    }

    /* Hover Effect */
    .modern-table tr.data-row:hover {
        background-color: #f1f8ff; /* Very light blue hover */
    }

    /* Table Cells (Default) */
    .modern-table td {
        padding: 10px 15px;
        font-size:20px;
        color: #333;
        border: 1px solid #e0e0e0; /* Light grey borders for info */
        border-top: none;
        border-bottom: none;
    }

    /* 
       REQUIREMENT: 
       "First columns (Material Name) with outer border, below it info in rows.
       All information in same row with outer border EXCEPT Material Name.
       All material name as column be in one border."
    */

    /* 1. Material Name Cell Styling */
    .modern-table td.mat-name-cell {
        border: none !important; /* Remove individual cell border */
        background-color: #f9f9f9; /* Slight tint to distinguish the "column" area */
        font-weight: bold;
        color: #1565C0; /* Darker blue for text */
    }

    /* 2. SKU Cell Styling (Neighbor to Material Name) */
    .modern-table td.sku-cell {
        border-left: none !important; /* Remove vertical divider between Mat Name and SKU */
    }

    /* 3. Fix First and Last Cell Borders relative to Table Border */
    .modern-table td:first-child {
        border-left: none;
    }
    .modern-table td:last-child {
        border-right: none;
    }

    /* Type Title Styling */
    .section-title {
        color: #1E88E5;
        font-size: 34px;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 15px;
        border-left: 5px solid #1E88E5;
        padding-left: 10px;
    }
    


</style>
""", unsafe_allow_html=True)


def top_page():
    st.markdown('<div class="big-title">Top Management Reveiw</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Review the Dataset Before & After Cleaning </div>', unsafe_allow_html=True)


    # Configuration
    folder_path = 'Database'
    file_name = 'master.xlsx'
    file_path = os.path.join(folder_path, file_name)

    # Check if file exists
    if os.path.exists(file_path):
        try:
            # Read the Excel file
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip()

            # Check if 'Type' column exists
            if 'Type' in df.columns:
                types = df['Type'].unique()

                for t in types:
                    # Display attractive Section Title
                    st.markdown(f'<div class="section-title">Type: {t}</div>', unsafe_allow_html=True)
                    
                    df_type = df[df['Type'] == t]

                    # Custom HTML Table Generation
                    table_html = f'<div class="custom-table-container"><table class="modern-table">'
                    
                    # --- Header Row ---
                    table_html += '<thead><tr>'
                    cols_to_display = ['Material Name', 'SKU', 'Quantity', 'Average Price', 'Total Value', 'Contract Type', 'Incoterm']
                    
                    for i, col in enumerate(cols_to_display):
                        if col in df_type.columns:
                            css_class = ""
                            if col == 'Material Name':
                                css_class = "mat-name-header"
                            elif col == 'SKU':
                                css_class = "sku-header"
                            table_html += f'<th class="{css_class}">{col}</th>'
                    table_html += '</tr></thead><tbody>'

                    # --- Data Rows ---
                    for index, row in df_type.iterrows():
                        table_html += '<tr class="data-row">'
                        
                        for i, col in enumerate(cols_to_display):
                            if col in df_type.columns:
                                cell_value = row[col]
                                cell_class = ""
                                
                                if col == 'Material Name':
                                    cell_class = "mat-name-cell"
                                elif col == 'SKU':
                                    cell_class = "sku-cell"
                                
                                table_html += f'<td class="{cell_class}">{cell_value}</td>'
                        
                        table_html += '</tr>'

                    table_html += '</tbody></table></div>'
                    st.markdown(table_html, unsafe_allow_html=True)
                    

            else:
                st.error("The Excel file does not contain a 'Type' column.")

        except Exception as e:
            st.error(f"Error reading Excel file: {e}")

    else:
        st.warning(f"File not found: {file_path}. Please ensure 'master.xlsx' is inside the 'Database' folder.")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 10px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); text-align: center; color: white;">
            <div style='font-size: 2.9rem; opacity: 1;'>Total 2026 Materials Budget: 123456 $</div>
        </div>
    """, unsafe_allow_html=True)


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
            st.session_state.page = "analysis3"
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
    st.markdown('<div class="big-title">Summary & Recommendation</div>', unsafe_allow_html=True)
    purchasing_unit = st.session_state.get('selected_purchasing_unit', 'Unit')
    # 1. Check if the result exists
    if 'price_forecast_df' not in st.session_state or st.session_state.price_forecast_df is None:
        
        # 2. VALIDATE that the uploaded file exists
        if 'df_uploaded' not in st.session_state:
            st.error("Please upload data first.")
            return

        # FIXED INDENTATION STARTS HERE
        df = st.session_state['df_uploaded'].copy()
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()

        # --- DEBUGGING STEP ---
        expected = ['date', 'price received', 'quantity received']
        
        if not all(col in df.columns for col in expected):
            st.error(f"❌ Column mismatch!")
            st.write("Your file has these columns:", list(df.columns))
            st.info("Required: 'date', 'price received', and 'quantity received'")
            return

        # If check passes, run the forecast
        try:
            forecasted_price, result_df, plot_buf = forecast_price(df)
            
            # Save results to session state so they persist
            st.session_state.forecasted_price = forecasted_price
            st.session_state.price_forecast_df = result_df
            st.session_state.price_forecast_plot = plot_buf
            
        except Exception as e:
            st.error(f"Error in calculation: {e}")
            
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
                    <strong style="color: rgb(255, 75, 75);">{forecast_val:.3f}</strong> $ per {purchasing_unit}             
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
            recommendations.append("ARIMA")

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
                    <span style="margin-left:-40px; margin-top:40px; " class="help-icon" data-help="These recommendations are derived from comprehensive analysis of stationarity, seasonality, noise, demand type, trend, and structural changes">?</span>
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

        st.session_state['test_mape'] = mape_val
        st.session_state['test_mae'] = mae_val
        st.session_state['test_rmse'] = rmse_val
        st.session_state['test_total_actual'] = total_actual_demand
        st.session_state['test_total_forecast'] = total_forecasted_demand
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
                help="Smooth Demand: Forecast a horizon of 6 to 24 months. Scientific stationarity and low variance in smooth data allow for high long-term statistical confidence, enabling more stable strategic procurement and capacity planning." 
                "Erratic Demand: Limit the forecast horizon to 2 to 6 months. High volatility and a high Coefficient of Variation ($CV^2$) lead to rapid error accumulation (forecast drift). Shorter horizons are required to maintain agility and prevent massive over-stocking or stockouts."
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
                <strong style="color: rgb(255, 75, 75);">{future_steps} Months</strong> ): ( <strong style="color: rgb(255, 75, 75);">{formatted_total}</strong> ) Units
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


            @st.dialog("⚠️ Model Incompatibility Warning", width="large")
            def show_demand_incompatibility_warning(demand_type):
                st.markdown(f"### 🧬 Scientific Rationale: {demand_type} Demand")
                
                st.error(f"Economic Order Quantity (EOQ) and Continuous/Periodic Review models cannot be applied to **{demand_type}** demand patterns.")
                
                st.markdown("""
                The mathematical foundations of traditional inventory theory rely on **Stationary Demand**. Your data violates these core assumptions:
                
                * **Violation of Normality:** Intermittent and Lumpy demand patterns contain frequent 'zero-demand' periods. Standard models assume a Gaussian (Normal) distribution, which results in mathematically impossible safety stock levels when zeros are present.
                * **Coefficient of Variation ($CV^2$):** In Lumpy demand, the variance-to-mean ratio ($CV^2 > 0.49$) is too high. Using EOQ here would cause the **Bullwhip Effect**, leading to massive over-stocking or catastrophic stockouts.
                * **Discrete Pulse Nature:** These demand types are 'Discrete Events' rather than 'Continuous Flows.' The Square Root Law ($Q = \sqrt{2DS/H}$) fails because demand does not 'drain' at a constant rate.
                """)

                

                st.info("💡 **Recommendation:** Consider using **Croston’s Method** or **Syntetos-Babai (SBA)** logic for this specific item to ensure replenishment matches actual consumption spikes.")
                
                if st.button("Acknowledge & Return", use_container_width=True):
                    st.session_state.show_warning = False
                    st.rerun()

            # ================= CALCULATION LOGIC =================

            # Check demand type suitability
            if demand_type in ["Intermittent", "Lumpy"] or demand_type is None:
                # Trigger the dialog state
                if "show_warning" not in st.session_state:
                    st.session_state.show_warning = True
                
                if st.session_state.show_warning:
                    show_demand_incompatibility_warning(demand_type)
                    st.stop()


            elif demand_type == "Smooth":
                st.session_state.page = "eoq_smooth1"
            
            elif demand_type == "Erratic":
                st.session_state.page = "eoq_erratic1"
            
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
    st.markdown('<div class="big-title">Price Forecasting (Weighted)</div>', unsafe_allow_html=True)

    if 'df_uploaded' not in st.session_state:
        st.warning("Please upload your data file first.")
        return

    df = st.session_state['df_uploaded'].copy()
    # Normalize column names to lowercase for easier checking
    df.columns = df.columns.str.lower()

    # Updated Required Columns Check
    required_cols = ['date', 'price received', 'quantity received']
            
    if not all(col in df.columns for col in required_cols):
        st.error(f"The uploaded file must contain the following columns: {', '.join(required_cols)}")
        return

    with st.spinner("Calculating weighted forecast..."):
        try:
            forecasted_price, result_df, plot_buf = forecast_price(df)
            
            # Identify next year for display
            next_year = int(result_df['Year'].max())

            col1, col2, col3 = st.columns([1,1,2])

            with col1:
                st.markdown("#### Weighted Average Forecast")
                st.markdown(f"""
                <div style="padding: 1.5rem; border-radius: 10px; background-color: #f8f9fa; border-left: 5px solid #1565C0;">
                    <h3 style="color:#1565C0; margin:0;">
                        Forecast for {next_year}: <br/>
                        <span style="color: #FF4B4B;">{forecasted_price:.3f} $</span>
                    </h3>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("#### Annual Summary")
                st.dataframe(result_df.style.format({'Actual Price': '{:.2f}', 'Forecasted Price': '{:.2f}'}))

            with col3:
                st.markdown("#### Price Trend")
                st.image(plot_buf, use_container_width=True) 

        except Exception as e:
            st.error(f"Error: {str(e)}")


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
            if st.button("Next → Sefety Stock & Re-Ordering Point", type="primary", use_container_width=True):
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
            
            if st.session_state.supplier_offers:
                best_offer = min(st.session_state.supplier_offers, key=lambda x: x['Total Cost'])
                st.session_state['best_offer_data'] = best_offer

        with col_old:
            st.markdown(f"""
                <div style="background-color:#F8F9FA; border:1px solid #E0E0E0; border-radius:10px; padding:15px; text-align:center;">
                    <div style="font-size:0.9rem; color:#111;">Baseline Total Cost</div>
                    <div style="font-size:1.8rem; font-weight:bold; color:#1565C0;">${baseline_cost:,.2f}</div>
                    <div style="font-size:0.85rem; color:#222;">Standard EOQ</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_new:
            st.markdown(f"""
                <div style="background-color:#E8F5E9; border:1px solid #4CAF50; border-radius:10px; padding:15px; text-align:center;">
                    <div style="font-size:0.9rem; color:#111;">Best Offer Cost</div>
                    <div style="font-size:1.8rem; font-weight:bold; color:#2E7D32;">${best_offer['Total Cost']:,.2f}</div>
                    <div style="font-size:0.85rem; color:#222;">Qty: {best_offer['Order Qty']:.0f} • Price: ${best_offer['New Price']:.2f}</div>
                </div>
            """, unsafe_allow_html=True)

        with col_diff:
            savings = baseline_cost - best_offer['Total Cost']
            color_savings = "#2E7D32" if savings > 0 else "#C62828"
            label_savings = "Total Savings" if savings > 0 else "Extra Cost"
            
            st.markdown(f"""
                <div style="background-color:#FFF3E0; border:1px solid #FF9800; border-radius:10px; padding:15px; text-align:center;">
                    <div style="font-size:0.9rem; color:#111;">{label_savings}</div>
                    <div style="font-size:1.8rem; font-weight:bold; color:{color_savings};">${savings:,.2f}</div>
                    <div style="font-size:0.85rem; color:#222;">Difference</div>
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

    with col_middle:
        if st.button("Final Summary", type="primary", use_container_width=True):
            st.session_state.page = "smooth_final_summary"
            st.rerun()



def smooth_final_summary_page():
    st.markdown('<div class="big-title">Executive Inventory Summary</div>', unsafe_allow_html=True)
    #st.markdown('<div class="subtitle">Forecasting Accuracy, Costs, and Inventory Policy</div>', unsafe_allow_html=True)

    # ================= RETRIEVE DATA FROM SESSION STATE =================
    
    # 1. Forecasting & Model Selection
    selected_model = st.session_state.get('selected_model', 'N/A')
    # ... (Error metrics retrieval remains same) ...
    test_mape = st.session_state.get('test_mape', 0)
    test_rmse = st.session_state.get('test_rmse', 0)
    
    # 2. Demand & Future Forecast
    demand_type = st.session_state.get('demand_type_classification', 'N/A')
    future_forecast_arr = st.session_state.get('future_forecast', [0])
    total_future_demand = float(np.sum(future_forecast_arr))
    future_dates = st.session_state.get('future_dates', None)
    future_steps = len(future_forecast_arr)
    
    # 3. Price & Cost
    # Initially retrieve standard values
    unit_price = st.session_state.get('final_unit_price', 0)
    purchasing_unit = st.session_state.get('selected_purchasing_unit', 'Unit')
    ordering_cost = st.session_state.get('total_ordering_cost', 0)
    holding_cost_unit = st.session_state.get('holding_cost_per_unit', 0)
    holding_rate = st.session_state.get('holding_rate', 0)
    
    # ================= CHECK FOR DISCOUNTS =================
    # Check if user visited the discount page and selected a best offer
    best_offer = st.session_state.get('best_offer_data', None)
    
    if best_offer:
        # OVERRIDE VALUES WITH DISCOUNTED ONES
        unit_price = best_offer.get('New Price', unit_price)
        total_cost = best_offer.get('Total Cost', 0) # The discounted total cost
        discount_rate = best_offer.get('Discount %', 0)
        min_order_qty = best_offer.get('MOQ', 0)
        best_status = best_offer.get('Status', 'Standard EOQ')
        
        # Recalculate holding cost based on NEW price
        holding_cost_unit = unit_price * holding_rate
        st.session_state['holding_cost_per_unit'] = holding_cost_unit # Update session state with new holding cost
    else:
        # No discount, calculate standard totals
        discount_rate = 0
        min_order_qty = 0
        best_status = "Standard EOQ"
        if holding_cost_unit > 0 and total_future_demand > 0 and ordering_cost > 0:
            import math
            eoq = math.sqrt((2 * ordering_cost * total_future_demand) / holding_cost_unit)
            num_orders = total_future_demand / eoq
            annual_holding_cost = holding_cost_unit * ((eoq / 2) + st.session_state.get('safety_stock', 0))
            annual_ordering_cost = num_orders * ordering_cost
            material_cost = unit_price * total_future_demand
            total_cost = material_cost + annual_holding_cost + annual_ordering_cost
        else:
            eoq = 0

    # 4. Supplier & Logistics
    supplier_name = st.session_state.get('selected_supplier_name', 'N/A')
    lead_time = st.session_state.get('supplier_lead_time', 0)
    service_level_input = st.session_state.get('supplier_service_level', 0)
    sigma_error = st.session_state.get('sigma_error', 0)
    
    # 5. Calculated Results
    z_index = st.session_state.get('z_index', 0)
    safety_stock = st.session_state.get('safety_stock', 0)
    reorder_point = st.session_state.get('reorder_point', 0)
    target_level = st.session_state.get('target_level', 0)
    
    # Recalculate EOQ specifically for display (in case it changed due to price change)
    if holding_cost_unit > 0 and total_future_demand > 0 and ordering_cost > 0:
        import math
        eoq = math.sqrt((2 * ordering_cost * total_future_demand) / holding_cost_unit)
        material_cost = unit_price * total_future_demand # Recalc material cost
    else:
        eoq = 0


    # ================= LAYOUT =================

    # --- ROW 1: FORECASTING PERFORMANCE ---
    st.markdown("### 🤖 Forecasting Model Results")
    
    # (Using same style as before)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 10px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                <div style='font-size: 0.9rem;'>Total Future Demand in {purchasing_unit}s</div>
                <div style='font-size: 1.8rem; font-weight: bold;'>{total_future_demand:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); padding: 10px; border-radius: 15px; text-align: center; color: #0d47a1; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                <div style='font-size: 0.9rem;'>selected model</div>
                <div style='font-size: 1.8rem; font-weight: bold;'>{selected_model}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%); padding: 10px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                <div style='font-size: 0.9rem;'>Safety Stock in {purchasing_unit}s</div>
                <div style='font-size: 1.8rem; font-weight: bold;'>{safety_stock:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        # Highlight Total Cost if Discount is applied
        cost_color = "#f6d365" # Default Orange
        if best_offer:
            cost_color = "#2ecc71" # Green if discounted
            cost_label = "Total Cost (Discounted)"
        else:
            cost_label = "Total Annual Cost"

        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); padding: 10px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                <div style='font-size: 1rem;'>{cost_label}</div>
                <div style='font-size: 1.8rem; font-weight: bold;'>${total_cost:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    
    # --- ROW 3: COST BREAKDOWN & SUPPLIER & POLICY (3 Cols) ---
    col_left, col_middle, col_right = st.columns(3)

    with col_left:
        st.subheader("💰 Cost Breakdown")
        cost_style = """
            <div style="background-color: #f8f9fa; border-left: 4px solid #28a745; padding: 15px; margin-bottom: 10px; border-radius: 5px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 1.2rem; color: #555;">{title}</div>
                </div>
                <div style="font-size: 1.2rem; font-weight: bold; color: #155724;">{val}</div>
            </div>
        """
        # Show the FINAL unit price (discounted or original)
        st.markdown(cost_style.format(title="Unit Price", val=f"${unit_price:.2f} / {purchasing_unit}"), unsafe_allow_html=True)
        st.markdown(cost_style.format(title="Holding Cost", val=f"${holding_cost_unit:.2f} / {purchasing_unit}"), unsafe_allow_html=True)
        st.markdown(cost_style.format(title="Ordering Cost", val=f"${ordering_cost:.2f} / order"), unsafe_allow_html=True)
        
        # Material cost calculation
        mat_cost_display = unit_price * total_future_demand
        st.markdown(cost_style.format(title="Total Material Cost", val=f"${mat_cost_display:,.0f}"), unsafe_allow_html=True)

    with col_middle:
        st.subheader("🏭 Supplier & Logistics")
        supplier_style = """
            <div style="background-color: #ffffff; border: 1px solid #e0e0e0; padding: 15px; margin-bottom: 20px; border-radius: 5px; display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: 1.4rem; color: #555;">{title}</div>
                <div style="font-size: 1.4rem; font-weight: bold; color: #1565C0;">{val}</div>
            </div>
        """
        st.markdown(supplier_style.format(title="Supplier", val=supplier_name), unsafe_allow_html=True)
        st.markdown(supplier_style.format(title="Demand Type", val=demand_type), unsafe_allow_html=True)
        st.markdown(supplier_style.format(title="Lead Time", val=f"{lead_time:.1f}"), unsafe_allow_html=True)
        st.markdown(supplier_style.format(title="Service Level", val=f"{service_level_input:.0%}"), unsafe_allow_html=True)

    with col_right:
        st.subheader("📦 Inventory Policy Recommendation")
        
        # DETERMINE POLICY TYPE
        if target_level > 0:
            policy_text = "Periodic Review (P, Q)"
            display_level = target_level
            level_name = "Order Up-to Level (S)"
        else:
            policy_text = "Continuous Review (Q, R)"
            display_level = reorder_point
            level_name = "Reorder Point (ROP)"
        
        st.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: #e3f2fd; border-radius: 10px; border-top: 5px solid #2196f3; margin-bottom:15px;">
                <div style="color: #1976d2; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px;">Policy Type</div>
                <div style="color: #333; font-size: 1.1rem;">{policy_text}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: #fff3e0; border-radius: 10px; border-top: 5px solid #ff9800;margin-bottom:15px;">
                <div style="color: #f57c00; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px;">{level_name}</div>
                <div style="color: #333; font-size: 2rem; font-weight: bold;">{display_level:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: #e8f5e9; border-radius: 10px; border-top: 5px solid #4caf50;margin-bottom:15px;">
                <div style="color: #388e3c; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px;">Order Quantity (Q)</div>
                <div style="color: #333; font-size: 2rem; font-weight: bold;">{eoq:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)

    # ================= FOOTER =================
    col_left, col_middle, col_right = st.columns([1, 1, 1])

    with col_left:
        if st.button("← Back to Calculations", use_container_width=True):
            st.session_state.page = "eoq_smooth4"
            st.rerun()

    with col_middle:
        if st.button("📋 View Action Plan", use_container_width=True, type="primary"):
            st.session_state.show_action_plan = True


    # ================= ACTION PLAN POPUP =================
    if st.session_state.get('show_action_plan', False):
        @st.dialog("📋 Inventory Action Plan", width="large")
        def show_action_plan_popup():
            # Dynamic Text based on Discount Status
            if best_offer:
                discount_status = f"YES - Applied discount of {discount_rate}%"
                price_text = f"Discounted Price: ${unit_price:.2f} / {purchasing_unit}"
                moq_note = f"*(Adjusted for MOQ: {min_order_qty} units)" if best_status == "Adjusted to MOQ" else ""
            else:
                discount_status = "**NO** - Using standard pricing"
                price_text = f"**Standard Price:** ${unit_price:.2f} / {purchasing_unit}"
                moq_note = ""
            
            st.markdown(f"""
            <style>
                .big-title {{
                    font-size: 1.5rem;
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                .section-label {{
                    font-size: 1.2rem;
                    font-weight: bold;
                }}
                .section-value {{
                    font-size: 1.3rem;
                }}
                .number-highlight {{
                    font-size: 1.5rem;
                    font-weight: bold;
                    color: #1565C0;
                }}
            </style>

            <div class="big-title">Supplier Discount Applied: {discount_status}</div>

            <hr>

            <p><span class="section-label">🛡️ Policy Type:<br/></span> <span class="section-value">{policy_text}</span></p>

            <p><span class="section-label">💲 Unit Price:<br/></span> <span class="section-value">{price_text}</span> <span style="font-size: 0.9rem;">{moq_note}</span></p>

            <p><span class="section-label">🔔 Trigger Point (ROP):</span><br>
            <span class="section-value">Place an order when <strong>Inventory Position</strong> drops to or below:</span> <span class="number-highlight">{display_level:,.0f} {purchasing_unit}s</span></p>

            <p><span class="section-label">🚚 Action to Take:</span><br>
            <span class="section-value">Order exactly:</span>
            <span class="number-highlight">{eoq:,.0f} {purchasing_unit}s</span></p>

            <p><span class="section-label">📦 Safety Stock:</span><br>
            <span class="section-value">Always maintain:</span>
            <span class="number-highlight">{safety_stock:,.0f} {purchasing_unit}s</span></p>


            """, unsafe_allow_html=True)            

            
     
            if st.button("Close", use_container_width=True, type="primary"):
                st.session_state.show_action_plan = False
                st.rerun()

        show_action_plan_popup()


#============================================================================================================================
#================================================================================================================================================================
#============================================================================================================================

def eoq_erratic1_page():
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
            months_to_forecast = 3  


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
    if isinstance(demand_type, str) and "erratic" in demand_type.lower():
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
            if st.button("Next → Sefety Stock & Re-Ordering Point", type="primary", use_container_width=True):
                st.session_state.page = "eoq_erratic2"
                st.rerun()




def eoq_erratic2_page():
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

    col1, col2, col3 = st.columns([1,2,2])

    with col1:
        if 'review_period_p' not in st.session_state:
            st.session_state['review_period_p'] = 1.0  

        # Number input: Review Period
        review_period_p = st.number_input(
            "Review Period (P) in Months",
            min_value=0.0,
            max_value=12.0,
            value=float(st.session_state['review_period_p']),
            step=0.5,
            key="p_input_widget"
        )
        st.session_state['review_period_p'] = review_period_p

        # --- THE MISSING STEP: RECALCULATION ---
        # We recalculate SS and ROP based on the new P input
        new_safety_stock = z_index * sigma_error * np.sqrt(lead_time + review_period_p)
        new_reorder_point = (avg_demand * lead_time) + (avg_demand * review_period_p) + new_safety_stock
        
        # Update variables for display
        safety_stock = new_safety_stock
        reorder_point_erratic = new_reorder_point

        with col2:
            # Now this will update live when P changes!
            formula_text_ss = f"Z: {z_index:.2f} × √({lead_time} + {review_period_p}): {np.sqrt(lead_time + review_period_p):.2f} × σ: {sigma_error:.2f}"
            
            st.markdown(cost_box(
                "Safety Stock (SS)", 
                f"{safety_stock:.2f}", 
                formula_text_ss
            ), unsafe_allow_html=True)

        with col3:
            # For Periodic Review, ROP usually covers Lead Time + Review Period
            formula_text_rop = f"(LT+P)×Avg Demand + SS"

            st.markdown(cost_box(
                "Order up-to level", 
                f"{reorder_point_erratic:.2f}", 
                formula_text_rop
            ), unsafe_allow_html=True) 

        # Save the NEW calculated values back to session state for the next pages
        st.session_state.safety_stock = safety_stock
        st.session_state.reorder_point_erratic = reorder_point_erratic       
            
    
    col_left, col_middle, col_right = st.columns([1, 1, 1])
    
    with col_left:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "eoq_erratic1"
            st.rerun()

    with col_right:
        if st.button("Next → Ordering Cost Determination", type="primary", use_container_width=True):
            st.session_state.page = "eoq_erratic3"
            st.rerun()




def eoq_erratic3_page():
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
            st.info("Enter approximate values for each major cost category (per order)")

            col_1, col_2 = st.columns(2)

            with col_1:
                # Sub-columns to keep the inputs compact
                sub_col1, sub_col2 = st.columns(2)

                with sub_col1:
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

                with sub_col2:
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
                # Logic: Demand is erratic, so type is fixed to Blanket
                purchasing_type = "Blanket"
                st.warning(f"Demand Type: **Erratic**")
                st.info(f"Purchasing Type fixed to: **{purchasing_type}**")

            # Calculate total
            total_ordering_cost = admin_cost + internal_cost + receiving_cost + transport_cost

            # Save values to session state
            st.session_state.cat_admin = admin_cost
            st.session_state.cat_internal = internal_cost
            st.session_state.cat_receiving = receiving_cost
            st.session_state.cat_transport = transport_cost

    else:


# ================= HELPER FUNCTION TO SAVE STATE =================
        def save_input(widget_key):
            new_value = st.session_state[widget_key]
            st.session_state[f"storage_{widget_key}"] = new_value

        # ================= SETUP & DATA LOADING =================
        # Reduced to 2 columns since we no longer need the Purchasing Type radio
        col_select1, col_select2 , col_select3 = st.columns(3)

        # 1. Select Origin & Load Data
        with col_select1:
            purchasing_origin = st.radio(
                "Type of Purchase",
                options=["Local", "Foreign"],
                horizontal=True,
                key="purchasing_origin_select"
            )

            sheet_name = "local" if purchasing_origin == "Local" else "foreign"

            db_path = os.path.join("Database", "Ordering Cost Components.xlsx")
            
            if not os.path.exists(db_path):
                st.error(f"Database file not found at: {db_path}")
                st.stop()

            try:
                df = pd.read_excel(db_path, sheet_name=sheet_name)
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

        # 2. Select Transport Method & Fixed Purchasing Type
        with col_select2:
            transport_options = ["EXW", "FOB", "CIF"]
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
        with col_select3:
            # FIXED LOGIC: Purchasing Type is now hardcoded to Blanket
            purchasing_type = "Blanket"
            st.info(f"Demand Type: Erratic | Purchasing Type: **{purchasing_type}**")

        # ================= LOGIC: DROPDOWN & DYNAMIC FIELDS =================

        fixed_cost_types = ["Administrative", "Internal Setup", "receiving", selected_transport]

        col_selector, col_inputs = st.columns([1, 4])

        with col_selector:
            st.markdown("### Select Cost Type")
            selected_cost_type = st.selectbox(
                "Choose a category to edit:",
                options=fixed_cost_types,
                key="cost_type_selector",
                label_visibility="collapsed"
            )
            st.info(f"Editing: **{selected_cost_type}**")

        with col_inputs:
            st.markdown(f"### {selected_cost_type} Components")
            group_df = df[df['cost type'] == selected_cost_type].copy()

            if group_df.empty:
                st.warning("No components found for this selection.")
            else:
                components_list = group_df.to_dict('records')
                
                for i in range(0, len(components_list), 4):
                    cols = st.columns(4)
                    for j in range(4):
                        if i + j < len(components_list):
                            item = components_list[i + j]
                            
                            with cols[j]:
                                comp_name = item['components']
                                # Multiplier is now pulled strictly from the 'Blanket' column
                                multiplier = item[purchasing_type]

                                # Key now includes 'Blanket' as part of the unique identifier
                                widget_key = f"cost_{selected_transport}_{comp_name}_{purchasing_type}_{purchasing_origin}_widget"
                                storage_key = f"storage_{widget_key}"

                                current_value = st.session_state.get(storage_key, 0.0)

                                input_val = st.number_input(
                                    f"{comp_name} (×{multiplier})",
                                    min_value=0.0,
                                    value=float(current_value),
                                    step=1.0,
                                    key=widget_key,
                                    on_change=save_input,
                                    args=(widget_key,)
                                )

                                calculated_line = input_val * multiplier
                                
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

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


    # Always update session state at the end
    st.session_state.total_ordering_cost = total_ordering_cost    



    col_left, col_middle, col_right = st.columns([1, 1, 1])
    
    with col_left:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "eoq_erratic2"
            st.rerun()

    with col_middle:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: -50px; border-radius: 10px; text-align: center; color: white; margin-bottom: 10px; font-weight:bold;'>
                <h2 style='margin: 0; font-size: 1rem;'>Total Ordering Cost: ${total_ordering_cost:,.2f}</h2>
            </div>
        """, unsafe_allow_html=True)

    with col_right:
        if st.button("Next → EOQ & Total Cost Calculation", type="primary", use_container_width=True):
            st.session_state.page = "eoq_erratic4"
            st.rerun()


def eoq_erratic4_page():
    st.markdown('<div class="big-title">Inventory Planning</div>', unsafe_allow_html=True)

    future_forecast = st.session_state.get('future_forecast', None)
    total_next_year_demand = float(np.sum(future_forecast)) if future_forecast is not None else 0.0
    formatted_total = f"{total_next_year_demand:,.2f}"


    review_period_p = st.session_state.get('review_period_p', 0.0)   
    reorder_point_erratic = st.session_state.get('reorder_point_erratic', 0.0)
    # Basic Inputs
    unit_price = st.session_state.get('final_unit_price', 0.0)
    holding_cost = st.session_state.get('holding_cost_per_unit', 0.0)
    
    # Supplier Info
    supplier_name = st.session_state.get('selected_supplier_name', "Not Selected")
    lead_time = st.session_state.get('supplier_lead_time', 0)
    service_level = st.session_state.get('supplier_service_level', 0)
    
    # Metrics & Stats
    sigma_error = st.session_state.get('sigma_error', 0.0)
    sigma_demand = st.session_state.get('sigma_demand', 0.0)
    z_index = st.session_state.get('z_index', 0.0)
    
    # Previous Calculations
    safety_stock = st.session_state.get('safety_stock', 0.0)
    # Note: 'reorder_point' from the previous page effectively acts as the 'Order up-to level' for Periodic Review
    order_up_to_level = st.session_state.get('reorder_point', 0.0)
    avg_demand = st.session_state.get('avg_demand_per_period', 0.0)

    total_ordering_cost = st.session_state.get('total_ordering_cost', 0.0)
    holding_rate = st.session_state.get('holding_rate', 0.0)


    # ================= CALCULATIONS SECTION =================
    st.markdown("### EOQ & Total Cost Calculation")

    # 1. Calculate Standard EOQ for Reference
    try:
        annual_demand = total_next_year_demand
        ordering_cost = total_ordering_cost
        holding_cost_per_unit = holding_cost

        if ordering_cost > 0 and annual_demand > 0 and holding_cost_per_unit > 0:
            import math
            
            # EOQ = sqrt( (2 * S * D) / H )
            eoq = math.sqrt((2 * ordering_cost * annual_demand) / holding_cost_per_unit)
            
            # --- TOTAL COST COMPONENTS ---

            # --- DISPLAY EOQ & TOTAL COST RESULTS ---
            
            col1, col2,col3 = st.columns([1,2,2])

            with col3:
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 10px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); text-align: center; color: white;">
                        <h4 style="margin: 0; font-size: 30px;">Economical Order Quantity (EOQ): <br/>{eoq:.2f} Units</h4>

                    </div>
                """, unsafe_allow_html=True)

            with col1:
                st.markdown(
                    """
                    <style>
                    /* Label text */
                    label[data-testid="stWidgetLabel"] > div {
                        font-size: 20px !important;
                        font-weight: 600;
                    }

                    /* Number input box text */
                    input[type="number"] {
                        font-size: 20px !important;
                        height: 45px;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # 2. Input: Inventory Position
                inventory_position = st.number_input(
                    "Current Inventory Position (I)",
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                    help="Current stock + On-order units - Backorders"
                )

            with col2:
                # --- CALCULATION ---
                
                # Recalculate Order Up-to Level (S) based on current P input
                # S = d * (L + P) + Z * sigma * sqrt(L + P)
                
                # Calculate Protection Interval (Lead Time + Review Period)
                protection_interval = lead_time + review_period_p
                
                # Recalculate Safety Stock for this interval
                ss_dynamic = z_index * sigma_error * np.sqrt(protection_interval)
                
                # Recalculate Order Up-to Level (Target Level)
                # Demand during protection interval
                demand_during_interval = avg_demand * protection_interval
                target_level = demand_during_interval + ss_dynamic

                # Calculate Order Quantity (Q)
                order_quantity = target_level - inventory_position
                
                # Handle negative result (no order needed)
                if order_quantity < 0:
                    order_quantity = 0.0
                    order_status = "No Order Needed (I > Target)"
                    box_bg = "#E8F5E9" # Greenish
                else:
                    order_status = "Place Order"
                    box_bg = "#FFF3E0" # Orangeish

                # Display Result
                st.markdown(f"""
                    <div style="
                        background-color: {box_bg};
                        border: 2px solid #1565C0;
                        border-radius: 15px;
                        padding: 10px;
                        text-align: center;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    ">
                        <div style="font-size: 1.1rem; color: #555; margin-bottom: 10px;">
                            Order up-to level({target_level:.2f}) - Inventory Position({inventory_position:.2f})
                        </div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #1565C0; margin-bottom: 5px;">
                            Order Quantity = <span style="font-size: 2rem;">{order_quantity:.2f}</span> Units
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Save these specific calculations to session state in case needed on next page
                st.session_state['final_order_quantity'] = order_quantity
                st.session_state['target_level'] = target_level

            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

            # --- COST BREAKDOWN ---
            def cost_box(title, value, formula, is_total=False):
                bg_color = "#E3F2FD" if is_total else "#F8F9FA"
                font_color = "#0D47A1" if is_total else "#1565C0"
                return f"""
                    <div style="background-color: {bg_color}; border-radius: 10px; padding: 15px; text-align: center; border: 1px solid #e0e0e0; box-shadow: 0 2px 2px #1565C0; height: 100%; display: flex; flex-direction: column; justify-content: center; margin-bottom:20px">
                        <div style="font-size: 1.2rem; color: #666; margin-bottom: 5px;">{title}</div>
                        <div style="font-size: 1.8rem; font-weight: bold; color: {font_color}; margin-bottom: 8px;">{value}</div>
                        <div style="font-size: 1rem; background: rgba(0,0,0,0.05); padding: 4px 8px; border-radius: 4px;">{formula}</div>
                    </div>
                """

            if 'future_steps' in st.session_state:
                months_to_forecast = st.session_state.future_steps
            else:
                months_to_forecast = 3


            purchase_cost_batch = unit_price * order_quantity
            num_orders = annual_demand / eoq
            total_ordering_cost_calc = (months_to_forecast / review_period_p) * ordering_cost
            # Note: For Periodic Review, Cycle stock is often approximated as (Demand during Review Interval)/2
            # but using EOQ/2 is standard for Total Cost comparison in hybrid models.
            total_holding_cost = holding_cost_per_unit * ((order_quantity / 2) + safety_stock)
            total_cost = purchase_cost_batch + total_ordering_cost_calc + total_holding_cost


            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.markdown(cost_box("Purchase Cost", f"${purchase_cost_batch:,.2f}", f"unit price (${unit_price:.2f}) × order quantity({order_quantity:.2f})"), unsafe_allow_html=True)
            with col2:
                st.markdown(cost_box("Ordering Cost", f"${total_ordering_cost_calc:,.2f}", f"no. of period forecasted({months_to_forecast:.2f}) / review period({review_period_p:.2f})<br/> × order cost (${ordering_cost:.2f})"), unsafe_allow_html=True)
            with col3:
                st.markdown(cost_box("Holding Cost", f" ${total_holding_cost:,.2f}", f"holding cost per unit(${holding_cost_per_unit:.2f}) × <br/>(order quantity({order_quantity:.2f})/2 + SS({safety_stock:.2f}))"), unsafe_allow_html=True)

            st.markdown("")
            col4, col5, col6 = st.columns([1, 5, 1])
            with col5:
                st.markdown(cost_box("Total Cost", f"${total_cost:,.2f}", "Sum of All Costs", is_total=True), unsafe_allow_html=True)

        else:
            st.warning("Cannot calculate EOQ. Please ensure Demand, Ordering Cost, and Holding Rate are > 0.")
            # Set defaults to avoid crash later
            eoq = 0
            total_cost = 0

    except Exception as e:
        st.error(f"Error in calculations: {e}")
        eoq = 0
        total_cost = 0


    # ================= NAVIGATION =================

    col_left, col_middle, col_right = st.columns([1, 1, 1])
    
    with col_left:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "eoq_erratic3"
            st.rerun()

    with col_right:
        if st.button("Suppliers Dashboard", type="primary", use_container_width=True):
            st.session_state.page = "supplier report"
            st.rerun()

    with col_middle:
        if st.button("Final Summary", type="primary", use_container_width=True):
            st.session_state.page = "erratic_final_summary"
            st.rerun()



def erratic_final_summary_page():
    st.markdown('<div class="big-title">Executive Inventory Summary</div>', unsafe_allow_html=True)
    #st.markdown('<div class="subtitle">Forecasting Accuracy, Costs, and Inventory Policy</div>', unsafe_allow_html=True)

    # ================= RETRIEVE DATA FROM SESSION STATE =================
    
    # 1. Forecasting & Model Selection
    selected_model = st.session_state.get('selected_model', 'N/A')
    test_mape = st.session_state.get('test_mape', 0)
    test_mae = st.session_state.get('test_mae', 0)
    test_rmse = st.session_state.get('test_rmse', 0)
    test_actual = st.session_state.get('test_total_actual', 0)
    test_forecast = st.session_state.get('test_total_forecast', 0)
    
    # 2. Demand & Future Forecast
    demand_type = st.session_state.get('demand_type_classification', 'N/A')
    future_forecast_arr = st.session_state.get('future_forecast', [0])
    total_future_demand = float(np.sum(future_forecast_arr))
    future_dates = st.session_state.get('future_dates', None)
    future_steps = len(future_forecast_arr)
    
    # 3. Price & Cost
    unit_price = st.session_state.get('final_unit_price', 0)
    purchasing_unit = st.session_state.get('selected_purchasing_unit', 'Unit')
    ordering_cost = st.session_state.get('total_ordering_cost', 0)
    holding_cost_unit = st.session_state.get('holding_cost_per_unit', 0)
    holding_rate = st.session_state.get('holding_rate', 0)
    
    # 4. Supplier & Logistics
    supplier_name = st.session_state.get('selected_supplier_name', 'N/A')
    lead_time = st.session_state.get('supplier_lead_time', 0)
    service_level_input = st.session_state.get('supplier_service_level', 0)
    sigma_error = st.session_state.get('sigma_error', 0)
    
    # 5. Calculated Results
    z_index = st.session_state.get('z_index', 0)
    safety_stock = st.session_state.get('safety_stock', 0)
    reorder_point = st.session_state.get('reorder_point', 0)
    target_level = st.session_state.get('target_level', 0)
    
    # Recalculate Totals for display
    if holding_cost_unit > 0 and total_future_demand > 0 and ordering_cost > 0:
        import math
        eoq = math.sqrt((2 * ordering_cost * total_future_demand) / holding_cost_unit)
        num_orders = total_future_demand / eoq
        annual_holding_cost = holding_cost_unit * ((eoq / 2) + safety_stock)
        annual_ordering_cost = num_orders * ordering_cost
        material_cost = unit_price * total_future_demand
        total_cost = material_cost + annual_holding_cost + annual_ordering_cost
    else:
        eoq = 0
        total_cost = 0

    # ================= LAYOUT =================

    # --- ROW 1: FORECASTING PERFORMANCE (New Addition) ---
    st.markdown("### 🤖 Forecasting Model Results")
    
    

    # --- ROW 2: KEY METRICS (Demand, EOQ, Cost) ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 10px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                <div style='font-size: 0.9rem; opacity: 0.9;'>Total Future Demand in {purchasing_unit}s</div>
                <div style='font-size: 1.8rem; font-weight: bold;'>{total_future_demand:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); padding: 10px; border-radius: 15px; text-align: center; color: #0d47a1; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                <div style='font-size: 0.9rem; opacity: 0.9;'>selected model</div>
                <div style='font-size: 1.8rem; font-weight: bold;'>{selected_model}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%); padding: 10px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                <div style='font-size: 0.9rem; opacity: 0.9;'>Safety Stock in {purchasing_unit}s</div>
                <div style='font-size: 1.8rem; font-weight: bold;'>{safety_stock:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); padding: 10px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                <div style='font-size: 0.9rem; opacity: 0.9;'>Total Annual Cost</div>
                <div style='font-size: 1.8rem; font-weight: bold;'>${total_cost:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)

    
    
    # --- ROW 3: COST BREAKDOWN & SUPPLIER ---
    col_left, col_right , col3= st.columns(3)

    with col_left:
        st.subheader("💰 Cost Breakdown")
        cost_style = """
            <div style="background-color: #f8f9fa; border-left: 4px solid #28a745; padding: 15px; margin-bottom: 10px; border-radius: 5px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 1.2rem; color: #555;">{title}</div>
                </div>
                <div style="font-size: 1.2rem; font-weight: bold; color: #155724;">{val}</div>
            </div>
        """
        st.markdown(cost_style.format(title="Unit Price", val=f"${unit_price:.2f} / {purchasing_unit}"), unsafe_allow_html=True)
        st.markdown(cost_style.format(title="Holding Cost", val=f"${holding_cost_unit:.2f} / {purchasing_unit}"), unsafe_allow_html=True)
        st.markdown(cost_style.format(title="Ordering Cost", val=f"${ordering_cost:.2f} / order"), unsafe_allow_html=True)
        st.markdown(cost_style.format(title="Total Material Cost", val=f"${material_cost:,.0f}"), unsafe_allow_html=True)

    with col_right:
        st.subheader("🏭 Supplier & Logistics")
        supplier_style = """
            <div style="background-color: #ffffff; border: 1px solid #e0e0e0; padding: 15px; margin-bottom: 20px; border-radius: 5px; display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: 1.4rem; color: #555;">{title}</div>
                <div style="font-size: 1.4rem; font-weight: bold; color: #1565C0;">{val}</div>
            </div>
        """
        st.markdown(supplier_style.format(title="Supplier", val=supplier_name), unsafe_allow_html=True)
        st.markdown(supplier_style.format(title="Demand Type", val=demand_type), unsafe_allow_html=True)
        st.markdown(supplier_style.format(title="Lead Time", val=f"{lead_time:.1f}"), unsafe_allow_html=True)
        st.markdown(supplier_style.format(title="Service Level", val=f"{service_level_input:.0%}"), unsafe_allow_html=True)

    with col3:
        # --- ROW 4: INVENTORY POLICY ---
        st.subheader("📦 Inventory Policy Recommendation")
        
        policy_text = "Periodic Review (P, Q)" if target_level > 0 else "Continuous Review (Q, R)"
        display_level = target_level if target_level > 0 else reorder_point
        level_name = "Order Up-to Level (S)" if target_level > 0 else "Reorder Point (ROP)"
        

        st.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: #e3f2fd; border-radius: 10px; border-top: 5px solid #2196f3; margin-bottom:15px;">
                <div style="color: #1976d2; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px;">Policy Type</div>
                <div style="color: #333; font-size: 1.1rem;">{policy_text}</div>
            </div>
        """, unsafe_allow_html=True)


        st.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: #fff3e0; border-radius: 10px; border-top: 5px solid #ff9800;margin-bottom:15px;">
                <div style="color: #f57c00; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px;">{level_name}</div>
                <div style="color: #333; font-size: 2rem; font-weight: bold;">{display_level:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)


        st.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: #e8f5e9; border-radius: 10px; border-top: 5px solid #4caf50;margin-bottom:15px;">
                <div style="color: #388e3c; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px;">Order Quantity (Q)</div>
                <div style="color: #333; font-size: 2rem; font-weight: bold;">{eoq:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)

# ================= FOOTER =================
    col_left, col_middle, col_right = st.columns([1, 1, 1])

    with col_left:
        if st.button("← Back to Calculations", use_container_width=True):
            st.session_state.page = "eoq_erratic4"
            st.rerun()

    with col_middle:
        if st.button("📋 View Action Plan", use_container_width=True, type="primary"):
            st.session_state.show_action_plan = True


    # ================= ACTION PLAN POPUP =================
    if st.session_state.get('show_action_plan', False):
        @st.dialog("📋 Action Plan")
        def show_action_plan_popup():
            st.markdown(f"""
            ### Inventory Management Action Plan
            
            **Policy:** {policy_text}
            
            **Trigger Condition:**  
            When Inventory Position ≤ **{display_level:,.0f}** {purchasing_unit}s
            
            **Action Required:**  
            Order **{eoq:,.0f}** {purchasing_unit}s
            
            **Safety Buffer:**  
            Maintain **{safety_stock:,.0f}** {purchasing_unit}s as Safety Stock
            
            ---
            
            #### Quick Reference:
            - 🔔 **Reorder Point:** {display_level:,.0f} {purchasing_unit}s
            - 📦 **Order Quantity:** {eoq:,.0f} {purchasing_unit}s
            - 🛡️ **Safety Stock:** {safety_stock:,.0f} {purchasing_unit}s
            """)
            
            if st.button("Close", use_container_width=True, type="primary"):
                st.session_state.show_action_plan = False
                st.rerun()
        
        show_action_plan_popup()







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
            st.session_state.page = "future forecasting"
            st.rerun()



# =====================================================================================================================
# ================================================ MAIN ROUTER ======================================================
# =====================================================================================================================


# ========================= MAIN APPLICATION LOGIC =========================
# Only runs if logged in
if st.session_state.logged_in:

    # --- 1. DEFINE PAGE GROUPS ---
    COMMON_PAGES = [
        "Material Selection", "material_excel", "data", "top manager","analysis", "analysis2", "analysis3", "analysis4", 
        "results from analysis", "recommendation", "forecasting", 
        "future forecasting"
    ]

    SMOOTH_PAGES = [
        "eoq_smooth1", "eoq_smooth2", "eoq_smooth3", "eoq_smooth4", "eoq_smooth5", "smooth_final_summary"
    ]

    ERRATIC_PAGES = [
        "eoq_erratic1", "eoq_erratic2", "eoq_erratic3", "eoq_erratic4", "erratic_final_summary"
    ]

    FINAL_PAGES = ["supplier report"]

    # --- 2. BUILD DYNAMIC LIST BASED ON CLASSIFICATION ---
    # Get the classification from session state (defaults to None if not analyzed yet)
    demand_type = st.session_state.get('demand_type_classification', None)

    if demand_type == "Smooth":
        CURRENT_PAGES = COMMON_PAGES + SMOOTH_PAGES + FINAL_PAGES
    elif demand_type == "Erratic":
        CURRENT_PAGES = COMMON_PAGES + ERRATIC_PAGES + FINAL_PAGES
    else:
        # If not yet classified, show only the common workflow
        CURRENT_PAGES = COMMON_PAGES + FINAL_PAGES

    # --- 3. SESSION STATE MANAGEMENT ---
    # Ensure page is in the current list, otherwise reset to 'material'
    if st.session_state.page not in CURRENT_PAGES:
        st.session_state.page = "material"

    # --- 4. SIDEBAR NAVIGATION ---
    st.sidebar.title(f"Navigation ({st.session_state.role})")

    # Logout button in sidebar
    if st.sidebar.button("Logout"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

    def get_page_index():
        try:
            return CURRENT_PAGES.index(st.session_state.page)
        except ValueError:
            return 0

    selection = st.sidebar.radio(
        "Go to", 
        CURRENT_PAGES, 
        index=get_page_index()
    )

    # --- 5. AUTOMATIC PRICE FORECAST TRIGGER ---
    # Trigger calculation when moving into EOQ sections
    if selection in (SMOOTH_PAGES + ERRATIC_PAGES) and 'forecasted_price' not in st.session_state:
        if 'df_uploaded' in st.session_state:
            df = st.session_state['df_uploaded'].copy()
            df.columns = df.columns.str.strip().str.lower()
            
            # Check for your specific spelling "price received"
            if 'price received' in df.columns and 'date' in df.columns:
                with st.spinner("Calculating future price for EOQ..."):
                    try:
                        # Ensure forecast_price is defined in your imports
                        # f_price, _, _ = forecast_price(df) 
                        # st.session_state.forecasted_price = f_price
                        pass # Placeholder logic since imports are commented out
                    except Exception as e:
                        st.error(f"Auto-forecast failed: {e}")
        else:
            st.warning("Please upload data to enable EOQ calculations.")

    # Handle Page Routing
    if selection != st.session_state.page:
        st.session_state.page = selection
        st.rerun()

    # --- 6. PAGE ROUTING LOGIC ---
    # Note: You must ensure these functions (e.g., page_material) are defined 
    # elsewhere in your script or below this block.
    
    if st.session_state.page == "Material Selection":
        page_material() 
    elif st.session_state.page == "material_excel":
        page_material_excel()
    elif st.session_state.page == "top manager":
        top_page()
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

    # Smooth Flow
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
    elif st.session_state.page == "smooth_final_summary":
        smooth_final_summary_page()

    # Erratic Flow
    elif st.session_state.page == "eoq_erratic1":
        eoq_erratic1_page()
    elif st.session_state.page == "eoq_erratic2":
        eoq_erratic2_page()
    elif st.session_state.page == "eoq_erratic3":
        eoq_erratic3_page()
    elif st.session_state.page == "eoq_erratic4":
        eoq_erratic4_page()
    elif st.session_state.page == "erratic_final_summary":
        erratic_final_summary_page()

    elif st.session_state.page == "supplier report":
        supplier_report_page()










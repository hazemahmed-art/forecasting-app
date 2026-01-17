import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. Define a consistent color theme


def show_supplier_dashboard():
    """
    Displays an interactive dashboard where user selects a chart from a dropdown.
    """
    BG_COLOR = "#F8F9FA"
    # Define the path to your database
    db_path = os.path.join("Database", "Suppliers Info.xlsx")

    # Check if file exists
    if not os.path.exists(db_path):
        st.error(f"Database file not found at: {db_path}")
        return

    # Load Data
    try:
        df = pd.read_excel(db_path)
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Ensure required columns exist
        required_cols = ['Supplier', 'Lead Time (L)', 'Service Level', 'On-time delivery %', 'Defect rate %']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing columns in Excel file: {', '.join(missing_cols)}")
            return

    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return

    # 2. Add a KPI Overview
    st.markdown("### Key Performance Indicators Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Convert to float for calculation just in case
    df['On-time delivery %'] = pd.to_numeric(df['On-time delivery %'], errors='coerce')
    df['Defect rate %'] = pd.to_numeric(df['Defect rate %'], errors='coerce')

    # Calculate values
    total_suppliers = len(df)
    avg_lead = df['Lead Time (L)'].mean()
    avg_otd = df['On-time delivery %'].mean()
    avg_defect = df['Defect rate %'].mean()

    # ================= KPI 1: Total Suppliers (Larger Style) =================
    with col1:
        st.markdown(f"""
            <div style="padding: 1rem; background-color: #e3f2fd; border-left: 5px solid #2196f3; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="font-size: 0.9rem; color: #555; font-weight: 600; margin-bottom: 8px;">Total Suppliers</div>
                <div style="font-size: 2.5rem; font-weight: bold; color: #0d47a1; line-height: 1;">
                    {total_suppliers} <span style="font-size: 1.2rem; color: #555; font-weight: normal;">Suppliers</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ================= KPI 2: Avg Lead Time =================
    with col2:
        st.markdown(f"""
            <div style="padding: 1rem; background-color: #e3f2fd; border-left: 5px solid #2196f3; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="font-size: 0.9rem; color: #555; font-weight: 600; margin-bottom: 8px;">Avg Lead Time</div>
                <div style="font-size: 2.5rem; font-weight: bold; color: #0d47a1; line-height: 1;">
                    {avg_lead:.1f} <span style="font-size: 1.2rem; color: #555; font-weight: normal;">Days</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ================= KPI 3: Avg On-Time Delivery =================
    with col3:
        st.markdown(f"""
            <div style="padding: 1rem; background-color: #e3f2fd; border-left: 5px solid #2196f3; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="font-size: 0.9rem; color: #555; font-weight: 600; margin-bottom: 8px;">Avg On-Time Delivery</div>
                <div style="font-size: 2.5rem; font-weight: bold; color: #0d47a1; line-height: 1;">
                    {avg_otd * 100:.2f}<span style="font-size: 1.2rem; color: #555; font-weight: normal;">%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ================= KPI 4: Avg Defect Rate =================
    with col4:
        st.markdown(f"""
            <div style="padding: 1rem; background-color: #e3f2fd; border-left: 5px solid #2196f3; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom:20px;">
                <div style="font-size: 0.9rem; color: #555; font-weight: 600; margin-bottom: 8px;">Avg Defect Rate</div>
                <div style="font-size: 2.5rem; font-weight: bold; color: #0d47a1; line-height: 1;">
                    {avg_defect:.2f}<span style="font-size: 1.2rem; color: #555; font-weight: normal;">%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ================= MAIN LAYOUT: SIDEBAR (DROPDOWN) | CONTENT (CHART) =================
    
    # Create 2 columns: Left narrow for controls, Right wide for chart
    col_left, col_right = st.columns([1, 4])

    with col_left:
        st.markdown("#### Select Chart")
        # Dropdown options
        chart_options = [
            "Lead Times",
            "Service Levels",
            "On-Time Delivery",
            "Defect Rate"
        ]
        
        selected_chart = st.selectbox(
            "Choose a metric to view",
            chart_options,
            label_visibility="collapsed"
        )

    with col_right:
        # Display the selected chart
        if selected_chart == "Lead Times":
            st.markdown("#### 📦 Lead Times (Days)")
            df_sorted = df.sort_values(by='Lead Time (L)', ascending=False)
            
            fig = px.bar(
                df_sorted, 
                x='Supplier', 
                y='Lead Time (L)',
                color='Lead Time (L)',
                color_continuous_scale='Viridis',
                text='Lead Time (L)'
            )
            
        elif selected_chart == "Service Levels":
            st.markdown("#### 🛡️ Service Levels (%)")
            df['Service Level Display'] = df['Service Level'] * 100
            df_sorted = df.sort_values(by='Service Level Display', ascending=False)
            
            fig = px.bar(
                df_sorted,
                x='Supplier',
                y='Service Level Display',
                color='Service Level Display',
                color_continuous_scale='Viridis',
                text='Service Level Display'
            )

        elif selected_chart == "On-Time Delivery":
            st.markdown("#### ⏱️ On-Time Delivery (%)")
            df_sorted = df.sort_values(by='On-time delivery %', ascending=False)
            
            fig = px.bar(
                df_sorted,
                x='Supplier',
                y='On-time delivery %',
                color='On-time delivery %',
                color_continuous_scale='RdYlGn',
                text='On-time delivery %'
            )

        elif selected_chart == "Defect Rate":
            st.markdown("#### ⚠️ Defect Rate (%)")
            df_sorted = df.sort_values(by='Defect rate %', ascending=False)
            
            fig = px.bar(
                df_sorted,
                x='Supplier',
                y='Defect rate %',
                color='Defect rate %',
                color_continuous_scale='Viridis',
                text='Defect rate %'
            )

        # Common layout updates for whichever chart was selected
        fig.update_layout(
            plot_bgcolor=BG_COLOR,
            paper_bgcolor=BG_COLOR,
            font=dict(color="#333333"),
            height=370, 
            margin=dict(l=20, r=20, t=30, b=40)
        )
        
        # Update text format based on metric type
        if selected_chart in ["Service Levels", "On-Time Delivery"]:
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        elif selected_chart == "Defect Rate":
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        else:
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')

        st.plotly_chart(fig, use_container_width=True)


import streamlit as st
import base64
import time
from io import BytesIO

try:
    from xhtml2pdf import pisa  # Import the new library
    XHTML2PDF_AVAILABLE = True
except ImportError:
    XHTML2PDF_AVAILABLE = False

class ReportGenerator:
    def __init__(self):
        self.content = []
        
        # CSS for xhtml2pdf (Standard CSS)
        self.css_style = """
        <style>
            @page { size: A4; margin: 2cm; }
            body { font-family: Helvetica; color: #333; font-size: 12px; }
            .report-header { text-align: center; margin-bottom: 40px; border-bottom: 2px solid #0d47a1; padding-bottom: 20px; }
            .report-title { color: #0d47a1; font-size: 24px; font-weight: bold; margin-bottom: 10px; }
            .report-subtitle { color: #666; font-size: 14px; }
            .section-title { font-size: 18px; font-weight: bold; color: #0d47a1; border-bottom: 1px solid #ddd; margin-top: 30px; margin-bottom: 15px; }
            .info-box { background-color: #f0f4f8; border-left: 5px solid #0d47a1; padding: 15px; margin-bottom: 20px; }
            .grid-container { display: table; width: 100%; margin-bottom: 20px; }
            .grid-item { display: table-row; }
            .grid-col { display: table-cell; width: 50%; padding: 5px; border-bottom: 1px solid #eee; }
            .label { font-size: 0.8rem; color: #666; font-weight: bold; text-transform: uppercase; }
            .value { font-size: 1rem; color: #000; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 10px; border: 1px solid #000; }
            th, td { border: 1px solid #000; padding: 5px; text-align: left; }
            th { background-color: #e9ecef; }
        </style>
        """

    def add_header(self, title, subtitle=""):
        html_content = f"""
        <div class="report-header">
            <div class="report-title">{title}</div>
            <div class="report-subtitle">{subtitle}</div>
            <div class="report-subtitle">Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</div>
        </div>
        """
        self.content.append(html_content)

    def add_section(self, title, body_html):
        section_html = f"""
        <div class="section-title">{title}</div>
        <div>{body_html}</div>
        """
        self.content.append(section_html)

    def generate_pdf_bytes(self):
        if not XHTML2PDF_AVAILABLE:
            st.error("xhtml2pdf library is not installed.")
            return None
            
        full_html = f"<!DOCTYPE html><html><head>{self.css_style}</head><body>{''.join(self.content)}</body></html>"
        
        # Create a file-like buffer to receive PDF data
        buffer = BytesIO()
        
        # Convert HTML to PDF
        pisa.CreatePDF(full_html, dest=buffer)
        
        # Get the value of the buffer
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes

    def download_button(self):
        pdf_bytes = self.generate_pdf_bytes()
        
        if pdf_bytes:
            st.download_button(
                label="Download Report as PDF",
                data=pdf_bytes,
                file_name=f"Report_{time.strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
# --- Wrapper for Material Page ---
def get_material_report_section():
    # Check if material is selected
    if 'selected_material' not in st.session_state or not st.session_state['selected_material']:
        return "<p>No material selected.</p>"
    
    mat = st.session_state['selected_material']

    # 1. Group the details into lists of 4
    group_1 = [
        ("Item Code", mat.get('Item Code', '-')),
        ("Item Family", mat.get('Item Family', '-')),
        ("Item Type", mat.get('Item Type', '-')),
        ("Grade", mat.get('Grade', '-'))
    ]
    
    group_2 = [
        ("Packaging", mat.get('Packaging', '-')),
        ("Physical Properties", mat.get('Physical Properties', '-')),
        ("Storage Conditions", mat.get('Storage Conditions', '-')),
        ("Warehouse Location", mat.get('Warehouse Location', '-'))
    ]
    
    group_3 = [
        ("Shelf Life", mat.get('Shelf Life', '-')),
        ("Supplier", mat.get('Supplier', '-')),
        ("Lead Time", mat.get('Lead Time', '-')),
        ("Service Level", mat.get('Service Level', '-'))
    ]

    # 2. Helper function to build a row with minimal padding
    def build_row(data_list):
        cells = ""
        for label, value in data_list:
            cells += f"""
            <td style="padding: 4px 6px; border: 1px solid #ddd; vertical-align: top;">
                <div style="font-size: 0.7rem; color: #555; font-weight: bold; margin-bottom: 2px;">{label}</div>
                <div style="font-size: 0.85rem; color: #000;">{value}</div>
            </td>
            """
        return f"<tr>{cells}</tr>"

    # 3. Construct HTML with reduced spacing
    html = f"""
    <div class="info-box" style="margin-bottom: 10px; padding: 10px;">
        <strong>Subject Material:</strong> 
        The following analysis is based on material <strong>{mat['Item Code']}</strong> ({mat['Item Family']} - {mat['Item Type']}).
    </div>
    
    <!-- Compact Table -->
    <table style="width: 100%; border-collapse: collapse; margin: 0; table-layout: fixed;">
        <tbody>
            {build_row(group_1)}
            {build_row(group_2)}
            {build_row(group_3)}
        </tbody>
    </table>
    """
    return html

def get_data_report_section():
    # Retrieve data from session state
    df = st.session_state.get('df_uploaded')
    
    # --- FIX: Check both the widget key AND the variable saved by the button ---
    period = st.session_state.get('selected_period') or st.session_state.get('period')
    
    file_name = st.session_state.get('uploaded_file_name', 'Unknown File')    

    
    # Construct the HTML Report Section
    # Fallback for text display
    display_period = period if period else "N/A"

    # --- CORRECTED HERE: Added the closing """ and parenthesis ---
    html = f"""
        <div style="margin-bottom: 15px;">
            <h3 style="color: #0d47a1; margin: 0 0 5px 0; font-size: 16px;">Data Source & Period</h3>
            <div style="background-color: #f8f9fa; padding: 10px; border-left: 4px solid #0d47a1; font-size: 12px;">
                <strong>File Name:</strong> {file_name} <br>
                <strong>Selected Period:</strong> {display_period}
            </div>
        </div>
    """
    
    return html

def get_analysis2_report_section():
    # Retrieve data from session state
    prep_results = st.session_state.get('prep_results')
    
    if not prep_results:
        return "<p>No preparation results found.</p>"

    # 1. Build the Single-Row Table HTML
    # We use inline styles for compactness (padding: 4px)
    table_html = f"""
    <table style="width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 10px; table-layout: fixed;">
        <thead>
            <tr style="background-color: #0d47a1; color: white;">
                <th style="padding: 6px; border: 1px solid #000; width: 25%;">Missing Values</th>
                <th style="padding: 6px; border: 1px solid #000; width: 25%;">Outliers</th>
                <th style="padding: 6px; border: 1px solid #000; width: 25%;">Frequency Check</th>
                <th style="padding: 6px; border: 1px solid #000; width: 25%;">Features</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 6px; border: 1px solid #ddd; vertical-align: top;">
                    <div><strong>Before:</strong> {prep_results.get('missing_values_before', 0)}</div>
                    <div><strong>After:</strong> {prep_results.get('missing_values_after', 0)}</div>
                </td>
                <td style="padding: 6px; border: 1px solid #ddd; vertical-align: top;">
                    <div><strong>Count:</strong> {prep_results.get('outliers_count', 0)}</div>
                    <div><strong>Perc:</strong> {prep_results.get('outliers_percent', 0):.2f}%</div>
                </td>
                <td style="padding: 6px; border: 1px solid #ddd; vertical-align: top;">
                    <div><strong>Expected:</strong> {prep_results.get('expected_records', 0)}</div>
                    <div><strong>Actual:</strong> {prep_results.get('actual_records', 0)}</div>
                    <div><strong>Missing Filled:</strong> {prep_results.get('missing_days_filled', 0)}</div>
                </td>
                <td style="padding: 6px; border: 1px solid #ddd; vertical-align: top;">
                    <div><strong>Created:</strong> {prep_results.get('features_created', 0)}</div>
                </td>
            </tr>
        </tbody>
    </table>
    """

    # 2. Handle Plot Image
    # Note: PDF generators (like xhtml2pdf) need an image file path (src="path/to/img.png").
    # They cannot render interactive HTML/JS plots from Plotly directly.
    # Ideally, you should save the plot image in your main app using: fig.write_image("plot.png")
    # and then reference that image here.
    
    plot_image_path = st.session_state.get('prep_plot_image_path', None)
    
    plot_html = ""
    if plot_image_path:
        # If you saved the image, display it
        plot_html = f'<img src="{plot_image_path}" style="width: 100%; height: auto; border: 1px solid #ddd;">'
    else:
        # Placeholder for the report
        plot_html = """
        <div style="border: 1px dashed #ccc; padding: 20px; text-align: center; color: #888; background-color: #f9f9f9;">
            [Plot visualization placeholder] 
            <br><small>To include the actual plot, save it as an image in the main app (e.g., fig.write_image) and pass the path here.</small>
        </div>
        """

    # 3. Construct the Final HTML
    html = f"""
    <div style="margin-bottom: 20px;">
        <h3 style="color: #0d47a1; margin: 0 0 10px 0; font-size: 16px;">Data Preparation Results</h3>
        {table_html}
        
        <div style="margin-top: 20px;">
            <h4 style="margin: 0 0 5px 0; font-size: 12px; color: #555;">DATA STATUS - AFTER PREPARATION (Selected Plot)</h4>
            {plot_html}
        </div>
    </div>
    """
    
    return html











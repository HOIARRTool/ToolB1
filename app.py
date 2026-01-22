# ==============================================================================
# HOIA-RR Streamlit Application
# Hospital Occurrence/Incident Analysis & Risk Register
#
# Clean & Copyright-Ready Full Code (Re-organized)
# ==============================================================================
from __future__ import annotations

# ==============================================================================
# IMPORTS
# ==============================================================================

# --- Standard library ---
import os
import re
import base64
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

# --- Third-party ---
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from tqdm import tqdm

# --- Optional third-party ---
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from weasyprint import HTML
except Exception:
    HTML = None

# --- Local app modules ---

from analytics import log_visit, log_button_click
from anonymizer import load_ner_model, anonymize_column
from streamlit_modal import Modal
from ai_assistant import get_consultation_response
from risk_register_assistant import get_risk_register_consultation


# ==============================================================================
# PAGE CONFIGURATION (ต้องเรียกก่อนคำสั่ง Streamlit อื่นๆ)
# ==============================================================================
PAGE_TITLE = "HOIA-RR"
PAGE_ICON = "🏥"  # Streamlit แนะนำให้ใช้ emoji หรือไฟล์ local
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")


# ==============================================================================
# SAFE DEFAULT CSV URL
# ==============================================================================
def _safe_get_default_csv_url() -> str:
    """
    โหลด DEFAULT_CSV_URL จาก st.secrets -> ENV -> fallback
    """
    # 1) st.secrets
    try:
        return st.secrets["DEFAULT_CSV_URL"]
    except Exception:
        pass

    # 2) ENV
    env_val = os.environ.get("DEFAULT_CSV_URL")
    if env_val:
        return env_val

    # 3) fallback
    return "https://raw.githubusercontent.com/HOIARRTool/ToolB1/main/Validate.csv"


DEFAULT_CSV_URL = _safe_get_default_csv_url()


# ==============================================================================
# GLOBAL CONFIG / PATHS
# ==============================================================================
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

PERSISTED_DATA_PATH = DATA_DIR / "processed_incident_data.parquet"
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin1234")

LOGO_URL = "https://raw.githubusercontent.com/HOIARRTool/hoiarr/refs/heads/main/logo1.png"

# URLs ของโลโก้ทั้ง 4
LOGO_URLS = [
    "https://github.com/HOIARRTool/appqtbi/blob/main/messageImage_1763018987241.jpg?raw=true",
    "https://github.com/HOIARRTool/appqtbi/blob/main/messageImage_1763018963411.jpg?raw=true",
    "https://mfu.ac.th/fileadmin/_processed_/6/7/csm_logo_mfu_3d_colour_15e5a7a50f.png",
    "https://github.com/HOIARRTool/appqtbi/blob/main/logoSHS.png?raw=true",
]

# ==============================================================================
# UI: HEADER LOGOS + GLOBAL CSS
# ==============================================================================
st.markdown(
    """
    <div style="
        width: 100%;
        display: flex;
        justify-content: flex-end;
        align-items: flex-start;
        gap: 12px;
        padding: 8px 24px 0 0;
    ">
        <img src="https://github.com/HOIARRTool/appqtbi/blob/main/messageImage_1763018987241.jpg?raw=true" style="height:60px;">
        <img src="https://github.com/HOIARRTool/appqtbi/blob/main/messageImage_1763018963411.jpg?raw=true"
            style="height:75px; margin-top:-4px;">
        <img src="https://mfu.ac.th/fileadmin/_processed_/6/7/csm_logo_mfu_3d_colour_15e5a7a50f.png"
             style="height:70px; margin-top:-1px;">
        <img src="https://github.com/HOIARRTool/appqtbi/blob/main/logoSHS.png?raw=true"
             style="height:45px; margin-top:12px;">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&display=swap');

    /* ✅ ฟอนต์หลัก */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        font-family: 'Kanit', sans-serif;
    }

    /* Gradient Text */
    .gradient-text {
        background-image: linear-gradient(45deg, #f09433, #e6683c, #dc2743, #bc1888, #833ab4);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        display: inline-block;
    }

    [data-testid="stChatInput"] textarea {
        min-height: 80px;
        height: 100px;
        resize: vertical;
        background-color: transparent;
        border: none;
    }

    .metric-box {
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-box .label { font-size: 0.9rem; color: #555; }
    .metric-box .value { font-size: 1.8rem; font-weight: bold; color: #262730; }

    .metric-box-1 { background-color: #e6fffa; border-color: #b2f5ea; }
    .metric-box-2 { background-color: #fff3e0; border-color: #ffe0b2; }
    .metric-box-3 { background-color: #fce4ec; border-color: #f8bbd0; }
    .metric-box-4 { background-color: #e3f2fd; border-color: #bbdefb; }
    .metric-box-5 { background-color: #f0f4c3; border-color: #e6ee9c; }
    .metric-box-6 { background-color: #ffecb3; border-color: #ffd54f; }
    .metric-box-7 { background-color: #ffcdd2; border-color: #ef9a9a; }

    .summary-table { width: 100%; border-collapse: collapse; table-layout: fixed; }
    .summary-table th, .summary-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .summary-table th { background-color: #f2f2f2; }

    .summary-table-4-col th:nth-child(1), .summary-table-4-col td:nth-child(1) { width: 20%; }
    .summary-table-4-col th:nth-child(2), .summary-table-4-col td:nth-child(2) { width: 20%; }
    .summary-table-4-col th:nth-child(3), .summary-table-4-col td:nth-child(3) { width: 10%; }
    .summary-table-4-col th:nth-child(4), .summary-table-4-col td:nth-child(4) { width: 50%; }

    .summary-table-5-col th:nth-child(1), .summary-table-5-col td:nth-child(1) { width: 15%; }
    .summary-table-5-col th:nth-child(2), .summary-table-5-col td:nth-child(2) { width: 15%; }
    .summary-table-5-col th:nth-child(3), .summary-table-5-col td:nth-child(3) { width: 20%; }
    .summary-table-5-col th:nth-child(4), .summary-table-5-col td:nth-child(4) { width: 10%; }
    .summary-table-5-col th:nth-child(5), .summary-table-5-col td:nth-child(5) { width: 40%; }

    .summary-table-6-col th:nth-child(1), .summary-table-6-col td:nth-child(1) { width: 12%; }
    .summary-table-6-col th:nth-child(2), .summary-table-6-col td:nth-child(2) { width: 12%; }
    .summary-table-6-col th:nth-child(3), .summary-table-6-col td:nth-child(3) { width: 20%; }
    .summary-table-6-col th:nth-child(4), .summary-table-6-col td:nth-child(4) { width: 16%; }
    .summary-table-6-col th:nth-child(5), .summary-table-6-col td:nth-child(5) { width: 8%; }
    .summary-table-6-col th:nth-child(6), .summary-table-6-col td:nth-child(6) { width: 32%; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@media print {
    div[data-testid="stHorizontalBlock"] {
        display: grid !important;
        grid-template-columns: repeat(5, 1fr) !important;
        gap: 1.2rem !important;
    }
    .stDataFrame, .stTable {
        break-inside: avoid;
        page-break-inside: avoid;
    }
    thead, tr, th, td {
        break-inside: avoid !important;
        page-break-inside: avoid !important;
    }
    h1, h2, h3, h4, h5 {
        page-break-after: avoid;
    }
}
.custom-header {
    font-size: 20px;
    font-weight: bold;
    margin-top: 0px !important;
    padding-top: 0px !important;
}
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] {
    border: 1px solid #ddd;
    padding: 0.75rem;
    border-radius: 0.5rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
div[data-testid="stHorizontalBlock"] > div:nth-child(1) div[data-testid="stMetric"] { background-color: #e6fffa; border-color: #b2f5ea; }
div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stMetric"] { background-color: #fff3e0; border-color: #ffe0b2; }
div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stMetric"] { background-color: #fce4ec; border-color: #f8bbd0; }
div[data-testid="stHorizontalBlock"] > div:nth-child(4) div[data-testid="stMetric"] { background-color: #e3f2fd; border-color: #bbdefb; }

div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] [data-testid="stMetricLabel"] > div,
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #262730 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] > div {
    font-size: 0.8rem !important;
    line-height: 1.2 !important;
    white-space: normal !important;
    overflow-wrap: break-word !important;
    word-break: break-word;
    display: block !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

div[data-testid="stHorizontalBlock"] > div .stExpander { border: none !important; box-shadow: none !important; padding: 0 !important; margin-top: 0.5rem; }
div[data-testid="stHorizontalBlock"] > div .stExpander header { padding: 0.25rem 0.5rem !important; font-size: 0.75rem !important; border-radius: 0.25rem; }
div[data-testid="stHorizontalBlock"] > div .stExpander div[data-testid="stExpanderDetails"] { max-height: 200px; overflow-y: auto; }

.stDataFrame table td, .stDataFrame table th { color: black !important; font-size: 0.9rem !important; }
.stDataFrame table th { font-weight: bold !important; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# STATIC DATA DEFINITIONS / MASTER FILE LOADS
# ==============================================================================
PSG9_FILE_PATH = "PSG9code.xlsx"
SENTINEL_FILE_PATH = "Sentinel2024.xlsx"
ALLCODE_FILE_PATH = "Code2024.xlsx"
RISK_MITIGATION_FILE = "risk_mitigations.xlsx"

psg9_r_codes_for_counting = set()
sentinel_composite_keys = set()
df2 = pd.DataFrame()
df_mitigation = pd.DataFrame()
PSG9code_df_master = pd.DataFrame()
Sentinel2024_df = pd.DataFrame()

try:
    if Path(PSG9_FILE_PATH).is_file():
        PSG9code_df_master = pd.read_excel(PSG9_FILE_PATH)
        if 'รหัส' in PSG9code_df_master.columns:
            psg9_r_codes_for_counting = set(PSG9code_df_master['รหัส'].astype(str).str.strip().unique())

    if Path(SENTINEL_FILE_PATH).is_file():
        Sentinel2024_df = pd.read_excel(SENTINEL_FILE_PATH)
        if 'รหัส' in Sentinel2024_df.columns and 'Impact' in Sentinel2024_df.columns:
            Sentinel2024_df['รหัส'] = Sentinel2024_df['รหัส'].astype(str).str.strip()
            Sentinel2024_df['Impact'] = Sentinel2024_df['Impact'].astype(str).str.strip()
            Sentinel2024_df.dropna(subset=['รหัส', 'Impact'], inplace=True)
            sentinel_composite_keys = set((Sentinel2024_df['รหัส'] + '-' + Sentinel2024_df['Impact']).unique())

    if Path(ALLCODE_FILE_PATH).is_file():
        allcode2024_df = pd.read_excel(ALLCODE_FILE_PATH)
        req_cols = ["ชื่ออุบัติการณ์ความเสี่ยง", "กลุ่ม", "หมวด"]
        if 'รหัส' in allcode2024_df.columns and all(c in allcode2024_df.columns for c in req_cols):
            df2 = allcode2024_df[["รหัส", "ชื่ออุบัติการณ์ความเสี่ยง", "กลุ่ม", "หมวด"]].drop_duplicates().copy()
            df2['รหัส'] = df2['รหัส'].astype(str).str.strip()

    if Path(RISK_MITIGATION_FILE).is_file():
        df_mitigation = pd.read_excel(RISK_MITIGATION_FILE)
    else:
        st.warning(f"ไม่พบไฟล์ '{RISK_MITIGATION_FILE}', Risk Register Assistant อาจให้คำแนะนำได้ไม่สมบูรณ์")

except Exception as e:
    st.error(f"เกิดปัญหาในการโหลดไฟล์นิยาม: {e}")


risk_color_data = {
    'Category Color': [
        "Critical", "Critical", "Critical", "Critical", "Critical",
        "High", "High", "Critical", "Critical", "Critical",
        "Medium", "Medium", "High", "Critical", "Critical",
        "Low", "Medium", "Medium", "High", "High",
        "Low", "Low", "Low", "Medium", "Medium"
    ],
    'Risk Level': [
        "51", "52", "53", "54", "55",
        "41", "42", "43", "44", "45",
        "31", "32", "33", "34", "35",
        "21", "22", "23", "24", "25",
        "11", "12", "13", "14", "15"
    ]
}
risk_color_df = pd.DataFrame(risk_color_data)

display_cols_common = [
    'Occurrence Date', 'รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Impact',
    'Impact Level', 'รายละเอียดการเกิด_Anonymized', 'Resulting Actions'
]

month_label = {
    1: '01 มกราคม', 2: '02 กุมภาพันธ์', 3: '03 มีนาคม', 4: '04 เมษายน',
    5: '05 พฤษภาคม', 6: '06 มิถุนายน', 7: '07 กรกฎาคม', 8: '08 สิงหาคม',
    9: '09 กันยายน', 10: '10 ตุลาคม', 11: '11 พฤศจิกายน', 12: '12 ธันวาคม'
}

PSG9_label_dict = {
    1: '01 ผ่าตัดผิดคน ผิดข้าง ผิดตำแหน่ง ผิดหัตถการ',
    2: '02 บุคลากรติดเชื้อจากการปฏิบัติหน้าที่',
    3: '03 การติดเชื้อสำคัญ (SSI, VAP,CAUTI, CLABSI)',
    4: '04 การเกิด Medication Error และ Adverse Drug Event',
    5: '05 การให้เลือดผิดคน ผิดหมู่ ผิดชนิด',
    6: '06 การระบุตัวผู้ป่วยผิดพลาด',
    7: '07 ความคลาดเคลื่อนในการวินิจฉัยโรค',
    8: '08 การรายงานผลการตรวจทางห้องปฏิบัติการ/พยาธิวิทยา คลาดเคลื่อน',
    9: '09 การคัดกรองที่ห้องฉุกเฉินคลาดเคลื่อน'
}

type_name = {
    'CPS': 'Safe Surgery',
    'CPI': 'Infection Prevention and Control',
    'CPM': 'Medication & Blood Safety',
    'CPP': 'Patient Care Process',
    'CPL': 'Line, Tube & Catheter and Laboratory',
    'CPE': 'Emergency Response',
    'CSG': 'Gynecology & Obstetrics diseases and procedure',
    'CSS': 'Surgical diseases and procedure',
    'CSM': 'Medical diseases and procedure',
    'CSP': 'Pediatric diseases and procedure',
    'CSO': 'Orthopedic diseases and procedure',
    'CSD': 'Dental diseases and procedure',
    'GPS': 'Social Media and Communication',
    'GPI': 'Infection and Exposure',
    'GPM': 'Mental Health and Mediation',
    'GPP': 'Process of work',
    'GPL': 'Lane (Traffic) and Legal Issues',
    'GPE': 'Environment and Working Conditions',
    'GOS': 'Strategy, Structure, Security',
    'GOI': 'Information Technology & Communication, Internal control & Inventory',
    'GOM': 'Manpower, Management',
    'GOP': 'Policy, Process of work & Operation',
    'GOL': 'Licensed & Professional certificate',
    'GOE': 'Economy'
}

# สี risk-matrix (ตามต้นฉบับ)
colors2 = np.array([
    ["#e1f5fe", "#f6c8b6", "#dd191d", "#dd191d", "#dd191d", "#dd191d", "#dd191d"],
    ["#e1f5fe", "#f6c8b6", "#ff8f00", "#ff8f00", "#dd191d", "#dd191d", "#dd191d"],
    ["#e1f5fe", "#f6c8b6", "#ffee58", "#ffee58", "#ff8f00", "#dd191d", "#dd191d"],
    ["#e1f5fe", "#f6c8b6", "#42db41", "#ffee58", "#ffee58", "#ff8f00", "#ff8f00"],
    ["#e1f5fe", "#f6c8b6", "#42db41", "#42db41", "#42db41", "#ffee58", "#ffee58"],
    ["#e1f5fe", "#f6c8b6", "#f6c8b6", "#f6c8b6", "#f6c8b6", "#f6c8b6", "#f6c8b6"],
    ["#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe"],
])


# ==============================================================================
# DATA LOADING HELPERS
# ==============================================================================
def load_csv_from_url_fallback(url: str) -> pd.DataFrame:
    """
    โหลด CSV จาก GitHub RAW และกันกรณีได้ HTML แทน CSV
    """
    try:
        df = pd.read_csv(url, keep_default_na=False, encoding='utf-8-sig')
        if df.shape[1] == 1 and str(df.columns[0]).startswith("<!DOCTYPE"):
            st.error("URL ที่ให้เป็นหน้า HTML (น่าจะเป็นลิงก์แบบ blob/) กรุณาใช้ลิงก์แบบ raw.githubusercontent.com")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"โหลด CSV จาก GitHub ไม่สำเร็จ: {e}")
        return pd.DataFrame()


def save_processed(df: pd.DataFrame, note: str = "") -> None:
    try:
        df.to_parquet(PERSISTED_DATA_PATH, index=False)
        st.success(f"บันทึกข้อมูลสำเร็จ ({len(df):,} แถว) {note}")
    except Exception as e:
        st.error(f"บันทึกข้อมูลล้มเหลว: {e}")

# ==============================================================================
# CORE: PROCESS INCIDENT DATAFRAME (FINAL SINGLE VERSION)
# ==============================================================================
def process_incident_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    แปลง DataFrame ดิบให้อยู่ในฟอร์แมตภายในของระบบ:
    - ตรวจคอลัมน์หลัก
    - แตก Incident / ชื่อ / รหัส
    - แปลงวันที่
    - คำนวณ Impact Level / Frequency Level / Risk Level (+ สี)
    - เติม กลุ่ม/หมวด (จาก df2)
    - ผูก PSG9 (จาก PSG9code_df_master + PSG9_label_dict)
    - Anonymize ข้อความ + เก็บตก HN
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # ---------------- ตรวจคอลัมน์จำเป็น ----------------
    required_source_cols = ["รหัส: เรื่องอุบัติการณ์", "วันที่เกิดอุบัติการณ์", "ความรุนแรง"]
    missing_source_cols = [c for c in required_source_cols if c not in df.columns]
    if missing_source_cols:
        st.error(f"ไม่พบคอลัมน์จำเป็น: {', '.join(missing_source_cols)}")
        return pd.DataFrame()

    # ---------------- คอลัมน์หลัก ----------------
    df.rename(columns={"วันที่เกิดอุบัติการณ์": "Occurrence Date", "ความรุนแรง": "Impact"}, inplace=True)

    # แตก Incident / ชื่อ / รหัส
    df['Incident'] = df['รหัส: เรื่องอุบัติการณ์'].astype(str).str.split(':', n=1).str[0].str.strip()
    df = df[df['Incident'] != ''].copy()
    if df.empty:
        st.error("ไม่พบ Incident code ที่ถูกต้องหลังกรอง")
        return pd.DataFrame()

    df['ชื่ออุบัติการณ์ความเสี่ยง'] = df['รหัส: เรื่องอุบัติการณ์'].astype(str).str.split(':', n=1).str[1].str.strip()
    df['รหัส'] = df['Incident'].astype(str).str.slice(0, 6).str.strip()

    # Resulting Actions
    if 'สถานะ' in df.columns:
        df['Resulting Actions'] = df['สถานะ'].apply(lambda x: 'None' if 'รอแก้ไข' in str(x) else str(x))
    else:
        df['Resulting Actions'] = 'N/A'

    # ทำความสะอาดค่าที่ว่าง
    df.replace('', 'None', inplace=True)
    df = df.fillna('None')
    df['Impact'] = df['Impact'].astype(str).str.strip()

    # ---------------- เติม กลุ่ม/หมวด (จาก df2) ----------------
    if isinstance(df2, pd.DataFrame) and not df2.empty:
        df = pd.merge(df, df2[['รหัส', 'กลุ่ม', 'หมวด']], on='รหัส', how='left')
    for col in ['กลุ่ม', 'หมวด']:
        if col not in df.columns:
            df[col] = 'N/A'
        else:
            df[col].fillna('N/A', inplace=True)

    # ---------------- แปลงวันที่ ----------------
    df['Occurrence Date'] = pd.to_datetime(df['Occurrence Date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['Occurrence Date'], inplace=True)
    if df.empty:
        st.error("ไม่มีแถวที่วันที่ถูกต้อง")
        return pd.DataFrame()

    # ---------------- Impact / Frequency / Risk ----------------
    impact_level_map = {
        ('A', 'B', '1'): '1',
        ('C', 'D', '2'): '2',
        ('E', 'F', '3'): '3',
        ('G', 'H', '4'): '4',
        ('I', '5'): '5'
    }

    def map_impact_level_func(val):
        s = str(val).strip()
        for k, v in impact_level_map.items():
            if s in k:
                return v
        return 'N/A'

    df['Impact Level'] = df['Impact'].apply(map_impact_level_func)

    max_p = df['Occurrence Date'].max().to_period('M')
    min_p = df['Occurrence Date'].min().to_period('M')
    total_month_calc = max(1, (max_p.year - min_p.year) * 12 + (max_p.month - min_p.month) + 1)

    counts = df['Incident'].value_counts()
    df['count'] = df['Incident'].map(counts)
    df['Incident Rate/mth'] = (df['count'] / total_month_calc).round(1)

    cond = [
        (df['Incident Rate/mth'] < 2.0),
        (df['Incident Rate/mth'] < 3.9),
        (df['Incident Rate/mth'] < 6.9),
        (df['Incident Rate/mth'] < 29.9)
    ]
    choice = ['1', '2', '3', '4']
    df['Frequency Level'] = np.select(cond, choice, default='5')

    df['Risk Level'] = df.apply(
        lambda r: f"{r['Impact Level']}{r['Frequency Level']}" if r['Impact Level'] != 'N/A' else 'N/A', axis=1
    )

    df = pd.merge(df, risk_color_df, on='Risk Level', how='left')
    df['Category Color'].fillna('Undefined', inplace=True)

    # ---------------- คอลัมน์ช่วย ----------------
    df['Incident Type'] = df['Incident'].astype(str).str[:3]
    df['Month'] = df['Occurrence Date'].dt.month
    df['เดือน'] = df['Month'].map(month_label)
    df['Year'] = df['Occurrence Date'].dt.year.astype(str)

    # ---------------- PSG9 ----------------
    PSG9_ID_COL = 'PSG_ID'
    if isinstance(PSG9code_df_master, pd.DataFrame) and not PSG9code_df_master.empty and (PSG9_ID_COL in PSG9code_df_master.columns):
        mm = PSG9code_df_master[['รหัส', PSG9_ID_COL]].drop_duplicates(subset=['รหัส']).copy()
        mm['รหัส'] = mm['รหัส'].astype(str).str.strip()
        df = pd.merge(df, mm, on='รหัส', how='left')
        df['หมวดหมู่มาตรฐานสำคัญ'] = df[PSG9_ID_COL].map(PSG9_label_dict).fillna("ไม่จัดอยู่ใน PSG9 Catalog")
    else:
        df['หมวดหมู่มาตรฐานสำคัญ'] = "ไม่สามารถระบุ (PSG9code.xlsx ไม่ได้โหลด)"

    # ---------------- Anonymize + เก็บตก HN ----------------
    try:
        ner_model = load_ner_model()
        df = anonymize_column(
            df, text_col="รายละเอียดการเกิด", ner_model=ner_model, out_col="รายละเอียดการเกิด_Anonymized"
        )
        if 'รายละเอียดการเกิด_Anonymized' in df.columns:
            df['รายละเอียดการเกิด_Anonymized'] = df['รายละเอียดการเกิด_Anonymized'].astype(str).apply(
                lambda x: re.sub(r'HN\s*[:.\-#]?\s*\d+', '[HN_REDACTED]', x, flags=re.IGNORECASE)
            )
    except Exception as e:
        st.warning(f"ไม่สามารถทำการ anonymize ได้ครบถ้วน: {e}")

    return df

# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================
def create_summary_table_by_code(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    สร้างตารางสรุปจำนวนอุบัติการณ์ตาม 'รหัส' และระดับความรุนแรง
    โดยในแถวจะแสดงทั้งรหัสและชื่อของอุบัติการณ์
    """
    required_cols = ['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Impact']
    if not all(col in dataframe.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in dataframe.columns]
        st.warning(f"ไม่สามารถสร้างตารางได้ เนื่องจากขาดคอลัมน์: {', '.join(missing_cols)}")
        return pd.DataFrame()

    df_copy = dataframe.copy()
    df_copy['รหัส | ชื่ออุบัติการณ์'] = df_copy['รหัส'].astype(str) + " | " + df_copy['ชื่ออุบัติการณ์ความเสี่ยง'].fillna('')
    df_valid = df_copy.dropna(subset=['รหัส | ชื่ออุบัติการณ์', 'Impact'])
    if df_valid.empty:
        return pd.DataFrame()

    summary = pd.crosstab(df_valid['รหัส | ชื่ออุบัติการณ์'], df_valid['Impact'])

    severity_levels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    summary = summary.reindex(columns=severity_levels, fill_value=0)

    e_to_i_cols = [col for col in ['E', 'F', 'G', 'H', 'I'] if col in summary.columns]
    summary['รวม E-up'] = summary[e_to_i_cols].sum(axis=1)

    total_e_up_incidents = summary['รวม E-up'].sum()
    if total_e_up_incidents > 0:
        summary['ร้อยละ E-up'] = (summary['รวม E-up'] / total_e_up_incidents * 100).map('{:.2f}%'.format)
    else:
        summary['ร้อยละ E-up'] = '0.00%'

    summary = summary[summary.drop(columns=['ร้อยละ E-up']).sum(axis=1) > 0]
    summary.index.name = "รหัส | ชื่ออุบัติการณ์"
    return summary

def create_summary_table_by_category(dataframe: pd.DataFrame, category_column_name: str) -> pd.DataFrame:
    """
    สร้างตารางสรุปจำนวนอุบัติการณ์ตามหมวดหมู่และระดับความรุนแรง
    """
    if category_column_name not in dataframe.columns or 'Impact' not in dataframe.columns:
        st.error(f"ไม่พบคอลัมน์ '{category_column_name}' หรือ 'Impact' ในข้อมูล")
        return pd.DataFrame()

    df_valid = dataframe.dropna(subset=[category_column_name, 'Impact'])
    if df_valid.empty:
        return pd.DataFrame()

    summary = pd.crosstab(df_valid[category_column_name], df_valid['Impact'])

    severity_levels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    summary = summary.reindex(columns=severity_levels, fill_value=0)

    e_to_i_cols = [col for col in ['E', 'F', 'G', 'H', 'I'] if col in summary.columns]
    summary['รวม E-up'] = summary[e_to_i_cols].sum(axis=1)

    total_e_up_incidents = summary['รวม E-up'].sum()
    if total_e_up_incidents > 0:
        summary['ร้อยละ E-up'] = (summary['รวม E-up'] / total_e_up_incidents * 100).map('{:.2f}%'.format)
    else:
        summary['ร้อยละ E-up'] = '0.00%'

    summary.index.name = "หมวดหมู่"
    return summary


def load_data(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_excel(uploaded_file, engine='openpyxl', keep_default_na=False)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ Excel: {e}")
        return pd.DataFrame()

@st.cache_data
def calculate_persistence_risk_score(_df: pd.DataFrame, total_months: int) -> pd.DataFrame:
    """
    Persistence Risk Score = Frequency Score + Avg Severity Score (normalized)
    """
    risk_level_map_to_score = {
        "51": 21, "52": 22, "53": 23, "54": 24, "55": 25,
        "41": 16, "42": 17, "43": 18, "44": 19, "45": 20,
        "31": 11, "32": 12, "33": 13, "34": 14, "35": 15,
        "21": 6, "22": 7, "23": 8, "24": 9, "25": 10,
        "11": 1, "12": 2, "13": 3, "14": 4, "15": 5
    }

    if _df.empty or 'รหัส' not in _df.columns or 'Risk Level' not in _df.columns:
        return pd.DataFrame()

    analysis_df = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Risk Level']].copy()
    analysis_df['Ordinal_Risk_Score'] = analysis_df['Risk Level'].astype(str).map(risk_level_map_to_score)
    analysis_df.dropna(subset=['Ordinal_Risk_Score'], inplace=True)
    if analysis_df.empty:
        return pd.DataFrame()

    persistence_metrics = analysis_df.groupby('รหัส').agg(
        Average_Ordinal_Risk_Score=('Ordinal_Risk_Score', 'mean'),
        Total_Occurrences=('รหัส', 'size')
    ).reset_index()

    total_months = max(1, int(total_months))
    persistence_metrics['Incident_Rate_Per_Month'] = persistence_metrics['Total_Occurrences'] / total_months

    max_rate = max(1, float(persistence_metrics['Incident_Rate_Per_Month'].max()))
    persistence_metrics['Frequency_Score'] = persistence_metrics['Incident_Rate_Per_Month'] / max_rate
    persistence_metrics['Avg_Severity_Score'] = persistence_metrics['Average_Ordinal_Risk_Score'] / 25.0
    persistence_metrics['Persistence_Risk_Score'] = persistence_metrics['Frequency_Score'] + persistence_metrics['Avg_Severity_Score']

    incident_names = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง']].drop_duplicates()
    final_df = pd.merge(persistence_metrics, incident_names, on='รหัส', how='left')
    return final_df.sort_values(by='Persistence_Risk_Score', ascending=False)

@st.cache_data
def calculate_frequency_trend_poisson(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Poisson slope (แนวโน้มความถี่)
    """
    if _df.empty or 'รหัส' not in _df.columns or 'Occurrence Date' not in _df.columns:
        return pd.DataFrame()

    analysis_df = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Occurrence Date']].copy()
    analysis_df.dropna(subset=['Occurrence Date'], inplace=True)
    if analysis_df.empty:
        return pd.DataFrame()

    analysis_df['YearMonth'] = pd.to_datetime(analysis_df['Occurrence Date']).dt.to_period('M')
    full_date_range = pd.period_range(
        start=analysis_df['YearMonth'].min(),
        end=analysis_df['YearMonth'].max(),
        freq='M'
    )

    results = []
    for code in analysis_df['รหัส'].unique():
        incident_subset = analysis_df[analysis_df['รหัส'] == code]
        if len(incident_subset) < 3 or len(incident_subset.groupby('YearMonth')) < 2:
            continue

        monthly_counts = incident_subset.groupby('YearMonth').size().reindex(full_date_range, fill_value=0)
        y = monthly_counts.values
        X = sm.add_constant(np.arange(len(monthly_counts)))

        try:
            model = sm.Poisson(y, X).fit(disp=0)
            results.append({
                'รหัส': code,
                'Poisson_Trend_Slope': model.params[1],
                'Total_Occurrences': int(y.sum()),
                'Months_Observed': int(len(y))
            })
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    final_df = pd.DataFrame(results)
    incident_names = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง']].drop_duplicates()
    final_df = pd.merge(final_df, incident_names, on='รหัส', how='left')
    return final_df.sort_values(by='Poisson_Trend_Slope', ascending=False)

def create_poisson_trend_plot(df: pd.DataFrame, selected_code_for_plot: str, show_ci: bool = True) -> go.Figure:
    """
    กราฟจำนวนรายเดือน + แนวโน้ม Poisson + 95% CI
    """
    full_date_range_for_plot = pd.period_range(
        start=pd.to_datetime(df['Occurrence Date']).dt.to_period('M').min(),
        end=pd.to_datetime(df['Occurrence Date']).dt.to_period('M').max(),
        freq='M'
    )

    subset = df[df['รหัส'] == selected_code_for_plot].copy()
    subset['YearMonth'] = pd.to_datetime(subset['Occurrence Date']).dt.to_period('M')
    counts = subset.groupby('YearMonth').size().reindex(full_date_range_for_plot, fill_value=0)

    y = counts.values.astype(float)
    t = np.arange(len(counts), dtype=float)
    X = sm.add_constant(t)

    beta1 = None
    mu_hat = None
    mu_lo = None
    mu_hi = None

    if len(y) >= 2 and y.sum() > 0:
        try:
            model = sm.Poisson(y, X).fit(disp=0)
            beta0, beta1 = model.params
            eta = beta0 + beta1 * t
            mu_hat = np.exp(eta)

            if show_ci:
                cov = model.cov_params()
                design = np.column_stack([np.ones_like(t), t])
                se_eta = np.sqrt(np.einsum('ij,jk,ik->i', design, cov, design))
                eta_lo = eta - 1.96 * se_eta
                eta_hi = eta + 1.96 * se_eta
                mu_lo = np.exp(eta_lo)
                mu_hi = np.exp(eta_hi)

        except Exception as e:
            st.warning(f"คำนวณเส้นแนวโน้ม Poisson ไม่สำเร็จ: {e}")

    fig_plot = go.Figure()

    fig_plot.add_trace(go.Bar(
        x=counts.index.strftime('%Y-%m'),
        y=y,
        name='จำนวนครั้งที่เกิดจริง',
        marker=dict(color='#AED6F1')
    ))

    if mu_hat is not None:
        fig_plot.add_trace(go.Scatter(
            x=counts.index.strftime('%Y-%m'),
            y=mu_hat,
            mode='lines',
            name='แนวโน้มคาดหมาย (Poisson)',
            line=dict(width=2)
        ))

        if show_ci and (mu_lo is not None) and (mu_hi is not None):
            fig_plot.add_trace(go.Scatter(
                x=counts.index.strftime('%Y-%m'),
                y=mu_hi,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig_plot.add_trace(go.Scatter(
                x=counts.index.strftime('%Y-%m'),
                y=mu_lo,
                mode='lines',
                fill='tonexty',
                name='95% CI',
                line=dict(width=0),
                fillcolor='rgba(0,0,0,0.08)'
            ))

    fig_plot.update_layout(
        title=f'การกระจายตัวของอุบัติการณ์: {selected_code_for_plot}',
        xaxis_title='เดือน-ปี',
        yaxis_title='จำนวนครั้งที่เกิด',
        barmode='overlay',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)')
    )

    if beta1 is not None:
        factor = float(np.exp(beta1))
        annot_text = (f"<b>Poisson slope: {beta1:.4f}</b><br>"
                      f"อัตราเปลี่ยนแปลง: x{factor:.2f} ต่อเดือน")
    else:
        annot_text = "<b>Poisson slope: N/A</b><br>อัตราเปลี่ยนแปลง: N/A"

    fig_plot.add_annotation(
        x=0.5, y=0.98,
        xref="paper", yref="paper",
        text=annot_text,
        showarrow=False,
        font=dict(size=12, color="black"),
        align="center",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="rgba(255, 255, 224, 0.7)"
    )
    return fig_plot

def create_goal_summary_table(
    data_df_goal: pd.DataFrame,
    goal_category_name_param: str,
    e_up_non_numeric_levels_param: List[str],
    e_up_numeric_levels_param: Optional[List[str]] = None,
    is_org_safety_table: bool = False
) -> pd.DataFrame:
    """
    สรุป Safety Goals (คง logic ตามต้นฉบับ)
    """
    goal_category_name_param = str(goal_category_name_param).strip()
    if 'หมวด' not in data_df_goal.columns:
        return pd.DataFrame()

    df_filtered_by_goal_cat = data_df_goal[
        data_df_goal['หมวด'].astype(str).str.strip() == goal_category_name_param
    ].copy()

    if df_filtered_by_goal_cat.empty:
        return pd.DataFrame()

    if 'Incident Type' not in df_filtered_by_goal_cat.columns or 'Impact' not in df_filtered_by_goal_cat.columns:
        return pd.DataFrame()

    try:
        pvt_table_goal = pd.crosstab(
            df_filtered_by_goal_cat['Incident Type'],
            df_filtered_by_goal_cat['Impact'].astype(str).str.strip(),
            margins=True,
            margins_name='รวมทั้งหมด'
        )
    except Exception:
        return pd.DataFrame()

    if 'รวมทั้งหมด' in pvt_table_goal.index:
        pvt_table_goal = pvt_table_goal.drop(index='รวมทั้งหมด')
    if pvt_table_goal.empty:
        return pd.DataFrame()

    if 'รวมทั้งหมด' not in pvt_table_goal.columns:
        pvt_table_goal['รวมทั้งหมด'] = pvt_table_goal.sum(axis=1)

    all_impact_columns_goal = [str(col).strip() for col in pvt_table_goal.columns if col != 'รวมทั้งหมด']
    e_up_non_numeric_levels_param_stripped = [str(level).strip() for level in e_up_non_numeric_levels_param]
    e_up_numeric_levels_param_stripped = [str(level).strip() for level in e_up_numeric_levels_param] if e_up_numeric_levels_param else []

    e_up_columns_goal = [
        col for col in all_impact_columns_goal
        if col not in e_up_non_numeric_levels_param_stripped
        and (not e_up_numeric_levels_param_stripped or col not in e_up_numeric_levels_param_stripped)
    ]

    report_data_goal = []
    for incident_type_goal, row_data_goal in pvt_table_goal.iterrows():
        total_e_up_count_goal = sum(
            row_data_goal[col] for col in e_up_columns_goal if col in row_data_goal.index and pd.notna(row_data_goal[col])
        )
        total_all_impacts_goal = row_data_goal['รวมทั้งหมด'] if 'รวมทั้งหมด' in row_data_goal else 0
        percent_e_up_goal = (total_e_up_count_goal / total_all_impacts_goal * 100) if total_all_impacts_goal > 0 else 0

        report_data_goal.append({
            'Incident Type': incident_type_goal,
            'รวม E-up': total_e_up_count_goal,
            'ร้อยละ E-up': percent_e_up_goal
        })

    report_df_goal = pd.DataFrame(report_data_goal)

    merged_report_table_goal = pd.merge(
        pvt_table_goal.reset_index(),
        report_df_goal,
        on='Incident Type',
        how='outer'
    )

    merged_report_table_goal['รวม E-up'].fillna(0, inplace=True)
    merged_report_table_goal['ร้อยละ E-up'].fillna(0.0, inplace=True)

    cols_to_drop_from_display_goal = [col for col in e_up_non_numeric_levels_param_stripped if col in merged_report_table_goal.columns]
    if e_up_numeric_levels_param_stripped:
        cols_to_drop_from_display_goal.extend([col for col in e_up_numeric_levels_param_stripped if col in merged_report_table_goal.columns])

    merged_report_table_goal = merged_report_table_goal.drop(columns=cols_to_drop_from_display_goal, errors='ignore')

    total_col_original_name, e_up_col_name, percent_e_up_col_name = 'รวมทั้งหมด', 'รวม E-up', 'ร้อยละ E-up'

    if is_org_safety_table:
        total_col_display_name, e_up_col_display_name, percent_e_up_display_name = 'รวม 1-5', 'รวม 3-5', 'ร้อยละ 3-5'
        merged_report_table_goal.rename(
            columns={
                total_col_original_name: total_col_display_name,
                e_up_col_name: e_up_col_display_name,
                percent_e_up_col_name: percent_e_up_display_name
            },
            inplace=True, errors='ignore'
        )
    else:
        total_col_display_name = 'รวม A-I'
        e_up_col_display_name = e_up_col_name
        percent_e_up_display_name = percent_e_up_col_name
        merged_report_table_goal.rename(columns={total_col_original_name: total_col_display_name}, inplace=True, errors='ignore')

    merged_report_table_goal['Incident Type Name'] = merged_report_table_goal['Incident Type'].map(type_name).fillna(
        merged_report_table_goal['Incident Type']
    )

    final_columns_goal_order = (
        ['Incident Type Name']
        + [col for col in e_up_columns_goal if col in merged_report_table_goal.columns]
        + [e_up_col_display_name, total_col_display_name, percent_e_up_display_name]
    )
    final_columns_present_goal = [col for col in final_columns_goal_order if col in merged_report_table_goal.columns]
    merged_report_table_goal = merged_report_table_goal[final_columns_present_goal]

    if percent_e_up_display_name in merged_report_table_goal.columns and pd.api.types.is_numeric_dtype(merged_report_table_goal[percent_e_up_display_name]):
        merged_report_table_goal[percent_e_up_display_name] = merged_report_table_goal[percent_e_up_display_name].astype(float).map('{:.2f}%'.format)

    return merged_report_table_goal.set_index('Incident Type Name')

def create_psg9_summary_table(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    สรุป PSG9 (คง logic ต้นฉบับ)
    """
    if not isinstance(input_df, pd.DataFrame) or 'หมวดหมู่มาตรฐานสำคัญ' not in input_df.columns or 'Impact' not in input_df.columns:
        return pd.DataFrame()

    psg9_placeholders = [
        "ไม่จัดอยู่ใน PSG9 Catalog",
        "ไม่สามารถระบุ (Merge PSG9 ล้มเหลว)",
        "ไม่สามารถระบุ (เช็คคอลัมน์ใน PSG9code.xlsx)",
        "ไม่สามารถระบุ (PSG9code.xlsx ไม่ได้โหลด/ว่างเปล่า)",
        "ไม่สามารถระบุ (Merge PSG9 ล้มเหลว - rename)",
        "ไม่สามารถระบุ (Merge PSG9 ล้มเหลว - no col)",
        "ไม่สามารถระบุ (PSG9code.xlsx ไม่ได้โหลด/ข้อมูลไม่ครบถ้วน)"
    ]

    df_filtered = input_df[
        ~input_df['หมวดหมู่มาตรฐานสำคัญ'].isin(psg9_placeholders) & input_df['หมวดหมู่มาตรฐานสำคัญ'].notna()
    ].copy()

    if df_filtered.empty:
        return pd.DataFrame()

    try:
        summary_table = pd.crosstab(
            df_filtered['หมวดหมู่มาตรฐานสำคัญ'],
            df_filtered['Impact'],
            margins=True,
            margins_name='รวม A-I'
        )
    except Exception:
        return pd.DataFrame()

    if 'รวม A-I' in summary_table.index:
        summary_table = summary_table.drop(index='รวม A-I')
    if summary_table.empty:
        return pd.DataFrame()

    all_impacts = list('ABCDEFGHI')
    e_up_impacts = list('EFGHI')

    for impact_col in all_impacts:
        if impact_col not in summary_table.columns:
            summary_table[impact_col] = 0

    if 'รวม A-I' not in summary_table.columns:
        summary_table['รวม A-I'] = summary_table[[col for col in all_impacts if col in summary_table.columns]].sum(axis=1)

    summary_table['รวม E-up'] = summary_table[[col for col in e_up_impacts if col in summary_table.columns]].sum(axis=1)
    summary_table['ร้อยละ E-up'] = (summary_table['รวม E-up'] / summary_table['รวม A-I'] * 100).fillna(0)

    psg_order = [PSG9_label_dict[i] for i in sorted(PSG9_label_dict.keys())]
    summary_table = summary_table.reindex(psg_order).fillna(0)

    display_cols_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'รวม E-up', 'รวม A-I', 'ร้อยละ E-up']
    final_table = summary_table[[col for col in display_cols_order if col in summary_table.columns]].copy()

    for col in final_table.columns:
        if col != 'ร้อยละ E-up':
            final_table[col] = final_table[col].astype(int)

    final_table['ร้อยละ E-up'] = final_table['ร้อยละ E-up'].map('{:.2f}%'.format)
    return final_table

def get_text_color_for_bg(hex_color: str) -> str:
    """
    เลือกสีตัวอักษรขาว/ดำ ตาม background
    """
    try:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return '#000000'
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
        return '#FFFFFF' if luminance < 0.5 else '#000000'
    except ValueError:
        return '#000000'

def prioritize_incidents_nb_logit_v2(
    _df: pd.DataFrame,
    horizon: int = 3,
    min_months: int = 4,
    min_total: int = 5,
    w_expected_severe: float = 0.7,
    w_freq_growth: float = 0.2,
    w_sev_growth: float = 0.1,
    alpha_floor: float = 1e-8
) -> pd.DataFrame:
    """
    Early Warning: NB (frequency) + Logit (severity)
    """
    req = ['รหัส', 'Occurrence Date', 'Impact Level']
    if any(c not in _df.columns for c in req):
        return pd.DataFrame()

    d = _df.copy()
    d = d[pd.to_datetime(d['Occurrence Date'], errors='coerce').notna()]
    if d.empty:
        return pd.DataFrame()

    d['YearMonth'] = pd.to_datetime(d['Occurrence Date']).dt.to_period('M')
    full_range = pd.period_range(d['YearMonth'].min(), d['YearMonth'].max(), freq='M')

    rows = []
    for code, sub in d.groupby('รหัส'):
        if len(sub) < min_total:
            continue

        # ===== NB (freq) =====
        counts = sub.groupby('YearMonth').size().reindex(full_range, fill_value=0).astype(float)
        y = counts.values
        n_months = len(counts)
        t = np.arange(n_months, dtype=float)
        X = sm.add_constant(t)

        nb_beta0 = np.nan
        nb_beta1 = np.nan
        nb_p = np.nan
        nb_factor = np.nan
        alpha_hat = np.nan
        mu_future = np.zeros(horizon, dtype=float)

        if n_months >= min_months and y.sum() > 0:
            try:
                pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()
                mu = pois.fittedvalues
                num = float(((y - mu)**2 - y).sum())
                den = float(max((mu**2).sum(), 1e-12))
                alpha_hat = max(num/den, alpha_floor)

                nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha_hat)).fit()
                nb_beta0, nb_beta1 = float(nb.params[0]), float(nb.params[1])
                nb_p = float(nb.pvalues[1])
                nb_factor = float(np.exp(nb_beta1))

                t_future = np.arange(n_months, n_months + horizon, dtype=float)
                eta_future = nb_beta0 + nb_beta1 * t_future
                mu_future = np.exp(eta_future)
            except Exception:
                pass

        # ===== Logit (sev) =====
        sub = sub.copy()
        sub['__sev__'] = sub['Impact Level'].astype(str).isin(['3', '4', '5']).astype(int)

        sev_counts = sub.groupby('YearMonth')['__sev__'].sum().reindex(full_range, fill_value=0).astype(float)
        n_counts = sub.groupby('YearMonth').size().reindex(full_range, fill_value=0).astype(float)
        mask = n_counts > 0

        lg_beta1 = np.nan
        lg_p = np.nan
        sev_or = np.nan
        p_future = np.full(horizon, np.nan, dtype=float)

        if mask.sum() >= min_months and sev_counts[mask].sum() > 0 and (sev_counts[mask] < n_counts[mask]).any():
            try:
                endog = (sev_counts[mask] / n_counts[mask]).values
                Xt = sm.add_constant(np.arange(n_months)[mask].astype(float))
                logit = sm.GLM(endog, Xt, family=sm.families.Binomial(), freq_weights=n_counts[mask].values).fit()

                lg_beta0, lg_beta1 = float(logit.params[0]), float(logit.params[1])
                lg_p = float(logit.pvalues[1])
                sev_or = float(np.exp(lg_beta1))

                t_future_all = np.arange(n_months, n_months + horizon, dtype=float)
                lin = lg_beta0 + lg_beta1 * t_future_all
                p_future = 1/(1 + np.exp(-lin))
                p_future = np.clip(p_future, 1e-6, 1-1e-6)
            except Exception:
                pass
        else:
            base_p = (sev_counts.sum()/n_counts.sum()) if n_counts.sum() > 0 else 0.0
            p_future = np.full(horizon, base_p, dtype=float)

        expected_all_nextH = float(np.nansum(mu_future))
        expected_sev_nextH = float(np.nansum(mu_future * p_future))

        freq_rising = (nb_beta1 > 0) and (pd.notna(nb_p) and nb_p < 0.05)
        sev_rising = (lg_beta1 > 0) and (pd.notna(lg_p) and lg_p < 0.05)

        rows.append({
            'รหัส': code,
            'ชื่ออุบัติการณ์ความเสี่ยง': sub['ชื่ออุบัติการณ์ความเสี่ยง'].iloc[0] if 'ชื่ออุบัติการณ์ความเสี่ยง' in sub else '',
            'Months_Observed': int(n_months),
            'Total_Occurrences': int(y.sum()),
            'NB_alpha_hat': alpha_hat,
            'Freq_NB_Slope': nb_beta1,
            'Freq_p_value': nb_p,
            'Freq_Factor_per_month': nb_factor,
            'Severity_Logit_Slope': lg_beta1,
            'Severity_p_value': lg_p,
            'Severe_OR_per_month': sev_or,
            'Expected_All_nextH': expected_all_nextH,
            'Expected_Severe_nextH': expected_sev_nextH,
            'Freq_Rising': freq_rising,
            'Sev_Rising': sev_rising
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    def _safe_log_pos_arr(x):
        arr = np.asarray(x, dtype=float)
        arr[~np.isfinite(arr)] = 1.0
        arr = np.clip(arr, 1e-12, None)
        return np.log(arr)

    def _norm01_pos_arr(x):
        arr = np.asarray(x, dtype=float)
        arr[~np.isfinite(arr)] = 0.0
        arr = np.clip(arr, 0, None)
        rng = arr.max() - arr.min()
        return (arr - arr.min())/rng if rng > 0 else np.zeros_like(arr)

    out['Freq_Factor_per_month'] = pd.to_numeric(out['Freq_Factor_per_month'], errors='coerce').fillna(1.0)
    out['Severe_OR_per_month'] = pd.to_numeric(out['Severe_OR_per_month'], errors='coerce').fillna(1.0)
    out['Expected_Severe_nextH'] = pd.to_numeric(out['Expected_Severe_nextH'], errors='coerce').fillna(0.0)

    out['Score_expected_severe'] = _norm01_pos_arr(out['Expected_Severe_nextH'].values)
    out['Score_freq_growth'] = _norm01_pos_arr(_safe_log_pos_arr(out['Freq_Factor_per_month'].values))
    out['Score_sev_growth'] = _norm01_pos_arr(_safe_log_pos_arr(out['Severe_OR_per_month'].values))

    bonus = np.where((out['Freq_Rising']) & (out['Sev_Rising']), 0.05, 0.0)

    out['Priority_Score'] = (
        w_expected_severe * out['Score_expected_severe'] +
        w_freq_growth * out['Score_freq_growth'] +
        w_sev_growth * out['Score_sev_growth'] +
        bonus
    )

    cols = [
        'รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Months_Observed', 'Total_Occurrences',
        'Expected_All_nextH', 'Expected_Severe_nextH',
        'Freq_Factor_per_month', 'Freq_p_value',
        'Severe_OR_per_month', 'Severity_p_value',
        'NB_alpha_hat', 'Priority_Score',
        'Freq_Rising', 'Sev_Rising'
    ]

    out = out[cols].sort_values('Priority_Score', ascending=False).reset_index(drop=True)
    return out

# ==============================================================================
# PDF GENERATION (Executive Summary)
# ==============================================================================
def generate_executive_summary_pdf(
    df_filtered: pd.DataFrame,
    metrics_data: Dict,
    total_month: int,
    min_date_str: str,
    max_date_str: str,
    df_freq: pd.DataFrame
) -> Optional[bytes]:
    """
    สร้าง PDF (WeasyPrint) - ย้ายโค้ด HTML ออกมาเป็นฟังก์ชันเฉพาะให้เป็นระบบ
    """
    if HTML is None:
        st.error("ไม่พบไลบรารี weasyprint (HTML). กรุณาติดตั้ง weasyprint ก่อนใช้งานฟีเจอร์ PDF")
        return None

    # --- Risk Matrix & Top10 ---
    impact_level_keys = ['5', '4', '3', '2', '1']
    freq_level_keys = ['1', '2', '3', '4', '5']

    matrix_df = df_filtered[
        df_filtered['Impact Level'].isin(impact_level_keys) &
        df_filtered['Frequency Level'].isin(freq_level_keys)
    ].copy()

    matrix_data_html = "<p>ไม่มีข้อมูล</p>"
    frequency_legend_html = ""

    if not matrix_df.empty:
        matrix_data = pd.crosstab(matrix_df['Impact Level'], matrix_df['Frequency Level'])
        matrix_data = matrix_data.reindex(index=impact_level_keys, columns=freq_level_keys, fill_value=0)

        impact_labels = {
            '5': "5 (Extreme)", '4': "4 (Major)", '3': "3 (Moderate)", '2': "2 (Minor)", '1': "1 (Insignificant)"
        }
        freq_labels = {'1': "F1", '2': "F2", '3': "F3", '4': "F4", '5': "F5"}

        matrix_data_html = matrix_data.rename(index=impact_labels, columns=freq_labels).to_html(
            classes="styled-table",
            table_id="risk-matrix-table"
        )

        frequency_legend_html = """
        <div class="legend">
            <h4>หมายเหตุ คำอธิบายความถี่ (Frequency)</h4>
            <p>
                <b>F1</b> = Remote (น้อยกว่า 2 ครั้ง/เดือน)<br>
                <b>F2</b> = Uncommon (2-3 ครั้ง/เดือน)<br>
                <b>F3</b> = Occasional (4-6 ครั้ง/เดือน)<br>
                <b>F4</b> = Probable (7-29 ครั้ง/เดือน)<br>
                <b>F5</b> = Frequent (มากกว่าหรือเท่ากับ 30 ครั้ง/เดือน)
            </p>
        </div>
        """

    top10_df = df_freq.nlargest(10, 'count').copy()
    incident_names = df_filtered[['Incident', 'ชื่ออุบัติการณ์ความเสี่ยง']].drop_duplicates()
    top10_df = pd.merge(top10_df, incident_names, on='Incident', how='left')

    top10_html = top10_df[['Incident', 'count']].to_html(
        classes="styled-table",
        index=False,
        table_id="top10-table"
    )

    # --- Sentinel Events ---
    sentinel_html = "<p>ไม่พบ Sentinel Events ในช่วงเวลานี้</p>"
    if 'Sentinel code for check' in df_filtered.columns and sentinel_composite_keys:
        sentinel_df = df_filtered[df_filtered['Sentinel code for check'].isin(sentinel_composite_keys)]
        if not sentinel_df.empty:
            sentinel_html = sentinel_df[['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด_Anonymized']].to_html(
                classes="styled-table",
                index=False,
                table_id="sentinel-table"
            )

    # --- PSG9 ---
    psg9_summary_table = create_psg9_summary_table(df_filtered)
    psg9_html = "<p>ไม่พบข้อมูล PSG9 ในช่วงเวลานี้</p>"
    if psg9_summary_table is not None and not psg9_summary_table.empty:
        psg9_html = psg9_summary_table.to_html(classes="styled-table", table_id="psg9-table")

    # --- Safety Goals ---
    goal_definitions = {
        "Patient Safety/ Common Clinical Risk": "P:Patient Safety Goals หรือ Common Clinical Risk Incident",
        "Specific Clinical Risk": "S:Specific Clinical Risk Incident",
        "Personnel Safety": "P:Personnel Safety Goals",
        "Organization Safety": "O:Organization Safety Goals"
    }
    safety_goals_html_parts = []
    for display_name, cat_name in goal_definitions.items():
        is_org_safety = (display_name == "Organization Safety")
        summary_table = create_goal_summary_table(
            df_filtered, cat_name,
            e_up_non_numeric_levels_param=[] if is_org_safety else ['A', 'B', 'C', 'D'],
            e_up_numeric_levels_param=['1', '2'] if is_org_safety else None,
            is_org_safety_table=is_org_safety
        )
        if summary_table is not None and not summary_table.empty:
            safety_goals_html_parts.append(f"<h3>{display_name}</h3>")
            safety_goals_html_parts.append(summary_table.to_html(classes="styled-table auto-width-table"))

    safety_goals_html = "".join(safety_goals_html_parts) if safety_goals_html_parts else "<p>ไม่มีข้อมูลสำหรับสรุปตามเป้าหมาย</p>"

    # --- Unresolved severe ---
    unresolved_severe_df = df_filtered[
        df_filtered['Impact Level'].isin(['3', '4', '5']) &
        df_filtered['Resulting Actions'].astype(str).isin(['None', '', 'nan'])
    ]
    unresolved_severe_html = "<p>ไม่พบอุบัติการณ์รุนแรงที่ยังไม่ถูกแก้ไขในช่วงเวลานี้</p>"
    if not unresolved_severe_df.empty:
        df_for_pdf = unresolved_severe_df[['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด_Anonymized']].copy()
        df_for_pdf['Occurrence Date'] = df_for_pdf['Occurrence Date'].dt.strftime('%d/%m/%Y')
        unresolved_severe_html = df_for_pdf.to_html(classes="styled-table", index=False, table_id="unresolved-table")

    # --- Persistence Top5 ---
    persistence_html = "<p>ไม่มีข้อมูลเพียงพอสำหรับวิเคราะห์ความเสี่ยงเรื้อรัง</p>"
    persistence_df = calculate_persistence_risk_score(df_filtered, total_month)
    if not persistence_df.empty:
        top_persistence = persistence_df.head(5)
        p_list_items = ["<ol>"]
        for _, row in top_persistence.iterrows():
            item_text = (
                f"<li><b>{row['รหัส']}: {row['ชื่ออุบัติการณ์ความเสี่ยง']}</b><br>"
                f"<small><i>ดัชนีความเรื้อรัง: {row['Persistence_Risk_Score']:.2f} "
                f"(เกิดเฉลี่ย: {row['Incident_Rate_Per_Month']:.2f} ครั้ง/เดือน)</i></small></li>"
            )
            p_list_items.append(item_text)
        p_list_items.append("</ol>")
        persistence_html = "".join(p_list_items)

    # --- Early Warning Top5 ---
    early_warning_html = "<p>ไม่มีข้อมูลเพียงพอสำหรับวิเคราะห์ Early Warning</p>"
    ew_df = prioritize_incidents_nb_logit_v2(df_filtered, horizon=3, min_months=4, min_total=5)
    if ew_df is not None and not ew_df.empty:
        top_ew = ew_df.head(5)
        ew_list_items = ["<ol>"]
        for _, row in top_ew.iterrows():
            item_text = (
                f"<li><b>{row['รหัส']}: {row['ชื่ออุบัติการณ์ความเสี่ยง']}</b><br>"
                f"<small><i>คะแนนความสำคัญ: {row['Priority_Score']:.3f}, "
                f"คาดการณ์เหตุรุนแรงใน 3 เดือน: {row['Expected_Severe_nextH']:.2f} ครั้ง</i></small></li>"
            )
            ew_list_items.append(item_text)
        ew_list_items.append("</ol>")
        early_warning_html = "".join(ew_list_items)

    html_string = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm 1.5cm;
                @bottom-center {{
                    content: "หน้า " counter(page) " / " counter(pages);
                    font-family: "TH SarabunPSK", sans-serif;
                    font-size: 9pt;
                    color: #888;
                }}
            }}
            body {{ font-family: "TH SarabunPSK", sans-serif; font-size: 12pt; }}
            h1, h2, h3 {{
                font-family: "TH SarabunPSK", sans-serif;
                color: #001f3f;
                border-bottom: 2px solid #001f3f;
                padding-bottom: 5px;
            }}
            h2 {{ page-break-before: always; }}
            h1 + h2 {{ page-break-before: auto; }}

            .styled-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 1em;
                table-layout: fixed;
            }}
            .styled-table th, .styled-table td {{
                border: 1px solid #ddd;
                padding: 6px;
                text-align: left;
                word-wrap: break-word;
            }}
            .styled-table th {{ background-color: #f2f2f2;  }}

            .metric-container {{
                display: flex;
                justify-content: space-around;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }}
            .metric {{ text-align: top; }}
            .metric-label {{ font-size: 11pt; color: #555; }}
            .metric-value {{ font-size: 16pt; font-weight: bold; }}

            #sentinel-table th:nth-child(1), #sentinel-table td:nth-child(1) {{ width: 24%; }}
            #sentinel-table th:nth-child(2), #sentinel-table td:nth-child(2) {{ width: 24%; }}
            #sentinel-table th:nth-child(3), #sentinel-table td:nth-child(3) {{ width: 10%; }}
            #sentinel-table th:nth-child(4), #sentinel-table td:nth-child(4) {{ width: 38%; }}

            #top10-table th:nth-child(1), #top10-table td:nth-child(1) {{ width: 80%; }}
            #top10-table th:nth-child(2), #top10-table td:nth-child(2) {{ width: 20%; }}

            #risk-matrix-table th:nth-child(1), #risk-matrix-table td:nth-child(1) {{ width: 40%; }}
            #risk-matrix-table th:nth-child(n+2), #risk-matrix-table td:nth-child(n+2) {{ width: 10%; }}

            #psg9-table th:nth-child(1), #psg9-table td:nth-child(1) {{ width: 28%; text-align: left; }}
            #psg9-table th:nth-child(n+2):nth-child(-n+10),
            #psg9-table td:nth-child(n+2):nth-child(-n+10) {{ width: 3.4%; text-align: center; }}
            #psg9-table th:nth-child(n+11):nth-child(-n+12),
            #psg9-table td:nth-child(n+11):nth-child(-n+12) {{ width: 6.5%; text-align: center; }}
            #psg9-table th:nth-child(13), #psg9-table td:nth-child(13) {{ width: 10%; text-align: center; }}

            #unresolved-table th:nth-child(1), #unresolved-table td:nth-child(1) {{ width: 16%; }}
            #unresolved-table th:nth-child(2), #unresolved-table td:nth-child(2) {{ width: 22%; }}
            #unresolved-table th:nth-child(3), #unresolved-table td:nth-child(3) {{ width: 10%; }}
            #unresolved-table th:nth-child(4), #unresolved-table td:nth-child(4) {{ width: 48%; }}

            .auto-width-table {{ table-layout: auto; }}

            ol {{ padding-left: 30px; }}
            li {{ margin-bottom: 10px; }}
            small {{ color: #555; }}
        </style>
    </head>
    <body>
        <h1>บทสรุปสำหรับผู้บริหาร</h1>
        <p><b>ช่วงข้อมูลที่วิเคราะห์:</b> {min_date_str} ถึง {max_date_str} (รวม {total_month} เดือน)</p>
        <p><b>จำนวนอุบัติการณ์ที่พบทั้งหมด:</b> {metrics_data.get('total_processed_incidents', 0):,} รายการ</p>

        <h1>1. แดชบอร์ดสรุปภาพรวม</h1>
        <div class="metric-container">
            <div class="metric"><div class="metric-label">อุบัติการณ์ทั้งหมด</div><div class="metric-value">{metrics_data.get('total_processed_incidents', 0):,}</div></div>
            <div class="metric"><div class="metric-label">Sentinel Events</div><div class="metric-value">{metrics_data.get('total_sentinel_incidents_for_metric1', 0):,}</div></div>
            <div class="metric"><div class="metric-label">PSG9</div><div class="metric-value">{metrics_data.get('total_psg9_incidents_for_metric1', 0):,}</div></div>
            <div class="metric"><div class="metric-label">ความรุนแรงสูง</div><div class="metric-value">{metrics_data.get('total_severe_incidents', 0):,}</div></div>
            <div class="metric"><div class="metric-label">รุนแรง & ยังไม่แก้ไข</div><div class="metric-value">{metrics_data.get('total_severe_unresolved_incidents_val', 'N/A')}</div></div>
        </div>

        <h1>2. Risk Matrix และ Top 10 อุบัติการณ์</h1>
        <h3>Risk Matrix</h3>
        {matrix_data_html}
        {frequency_legend_html}

        <h2>Top 10 อุบัติการณ์ (ตามความถี่)</h2>
        {top10_html}

        <h2>3. รายการ Sentinel Events</h2>
        {sentinel_html}

        <h2>4. วิเคราะห์ตามหมวดหมู่ มาตรฐานสำคัญจำเป็นต่อความปลอดภัย 9 ข้อ</h2>
        {psg9_html}

        <h2>5. สรุปอุบัติการณ์ตามเป้าหมาย (Safety Goals)</h2>
        {safety_goals_html}

        <h2>6. สรุปอุบัติการณ์ที่เป็นปัญหาเรื้อรัง (Persistence Risk - Top 5)</h2>
        {persistence_html}

        <h3>7. Early Warning: อุบัติการณ์ที่มีแนวโน้มสูงขึ้น ใน 3 เดือนข้างหน้า (Top 5)</h3>
        {early_warning_html}

        <h2>8. รายการอุบัติการณ์รุนแรง (E-I & 3-5) ที่ยังไม่ถูกแก้ไข</h2>
        {unresolved_severe_html}
    </body>
    </html>
    """

    return HTML(string=html_string).write_pdf()

# ==============================================================================
# USER GUIDE PAGE (คงเนื้อหาเดิมทั้งหมด)
# ==============================================================================
def display_user_guide():
    st.markdown("<h2 style='color: #001f3f;'>คู่มือการใช้งาน HOIA-RR Application</h2>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    **HOIA-RR (Hospital Occurrence/Incident Analysis & Risk Register)** คือเครื่องมือวิเคราะห์และจัดการอุบัติการณ์ความเสี่ยงในโรงพยาบาล ที่ถูกออกแบบมาเพื่อเปลี่ยนข้อมูลดิบให้กลายเป็นข้อมูลเชิงลึกที่นำไปปฏิบัติได้จริง ช่วยให้ผู้บริหารและผู้รับผิดชอบความเสี่ยง (Risk Owner) สามารถมองเห็นภาพรวม, ติดตามแนวโน้ม และตัดสินใจได้อย่างมีประสิทธิภาพ
    """)

    st.markdown("### **แอปพลิเคชันนี้ทำอะไรได้บ้าง? 💡**")
    st.markdown("""
    - **ประมวลผลข้อมูลอัตโนมัติ**: เพียงอัปโหลดไฟล์รายงานอุบัติการณ์ (`.xlsx` หรือ `.csv`) ระบบจะทำความสะอาดข้อมูล, คำนวณระดับความเสี่ยง (Risk Level), และจัดหมวดหมู่ให้โดยอัตโนมัติ
    - **ปกปิดข้อมูลส่วนบุคคล (Anonymize)**: ปกป้องข้อมูลสำคัญ เช่น `HN` ของผู้ป่วย, ชื่อ-นามสกุล ก่อนนำมาแสดงผล เพื่อความปลอดภัยของข้อมูล
    - **แสดงผลแดชบอร์ดเชิงโต้ตอบ (Interactive Dashboard)**: สรุปภาพรวมอุบัติการณ์ในหลากหลายมิติ ตั้งแต่ภาพรวมจนถึงการเจาะลึกรายหมวดหมู่
    - **วิเคราะห์ความเสี่ยงขั้นสูง**: ชี้เป้า "ความเสี่ยงเรื้อรัง" (Persistence Risk) ที่เกิดขึ้นซ้ำๆ และส่ง "สัญญาณเตือนล่วงหน้า" (Early Warning) สำหรับอุบัติการณ์ที่มีแนวโน้มรุนแรงขึ้น
    
    - **AI ผู้ช่วยให้คำปรึกษา**: มาพร้อมกับ AI สองรูปแบบ คือ `RCA Helpdesk` สำหรับให้คำปรึกษาอุบัติการณ์ ซึ่งจะมีรายละเอียดได้แก่ รหัสที่แนะนำสำหรับลงอุบัติการณ์อ้างอิงตาม NRLS, สาเหตุหรือ contributing factor, คำแนะนำสำหรับการแก้ไขเบื้องต้น และการเรียนรู้ไปกับ 3P Safety & HA Standard อีกทั้ง `Risk Register Assistant` สำหรับช่วย Risk Owner ค้นหาข้อมูลเชิงลึกของอุบัติการณ์ที่ดูแลอยู่ (Review อุบัติการณ์ในช่วงเวลาที่เลือก) พร้อมคำแนะนำสำหรับ Risk Transfer & Prevention และ Risk Monitor  
    """)

    st.markdown("### **เริ่มต้นใช้งาน: สำหรับผู้ดูแลระบบ (Admin) 🛠️**")
    st.markdown("""
    หน้าที่หลักของผู้ดูแลระบบคือการนำข้อมูลเข้าสู่ระบบเพื่อให้ทุกคนสามารถใช้งานได้

    **ขั้นตอน:**
    1.  ไปที่เมนู **"จัดการข้อมูล (Admin)"**
    2.  เตรียมไฟล์รายงานอุบัติการณ์ของคุณให้เป็นไฟล์ **`.xlsx`** (แนะนำ) หรือ `.csv`
    3.  **ตรวจสอบคอลัมน์ที่จำเป็น**: ไฟล์ของคุณต้องมีคอลัมน์หลักดังนี้:
        - `Incident` (รหัสและชื่ออุบัติการณ์)
        - `วดป.ที่เกิด` (วันที่เกิดเหตุ)
        - `ความรุนแรง` (ระดับ A-I)
        - `สถานะ` (เช่น รอแก้ไข, แก้ไขแล้ว)
        - `รายละเอียดการเกิด`
    4.  คลิก **"Browse files"** และเลือกไฟล์ของคุณ
    5.  รอให้ระบบประมวลผลจนขึ้นข้อความ **"ประมวลผลสำเร็จ!"**
    """)

    st.markdown("### **การใช้งานแดชบอร์ด: เพื่อประโยชน์สูงสุดของผู้ใช้งาน 📈**")
    st.info("แต่ละเมนูถูกออกแบบมาเพื่อตอบคำถามที่แตกต่างกัน:")

    st.markdown("""
    | เมนูใน Sidebar | สิ่งที่คุณจะได้รับและวิธีใช้งานให้เกิดประโยชน์สูงสุด |
    | :--- | :--- |
    | **แดชบอร์ดสรุปภาพรวม** | **ภาพรวมสุขภาพขององค์กรในหน้าเดียว**: ใช้เป็นจุดเริ่มต้นในการประชุมประจำวัน/สัปดาห์ เพื่อดูตัวเลขสำคัญทั้งหมด เช่น จำนวนอุบัติการณ์, Sentinel Events, เคสรุนแรงที่ยังไม่แก้ไข |
    | **Incidents Analysis** | **การวิเคราะห์อุบัติการณ์**: จากรายงานอุบัติการณ์ในแต่ละหมวดหมู่ (โดยเฉพาะ PSG9) เกิดขึ้นในระดับความรุนแรงใดบ้าง ถูกแก้ไขไปแล้วกี่เปอร์เซ็นต์ และมีเคสใดค้างอยู่บ้างเพื่อทำการติดตามได้อย่างครอบคลุม  |
    | **Risk Matrix (Interactive)** | **เห็นการกระจายตัวของความเสี่ยง**: แสดง Risk Matrix (I x F) ที่คุณสามารถคลิกที่ตัวเลขในช่องต่างๆ เพื่อเจาะลึกดูรายการอุบัติการณ์ในระดับความเสี่ยงนั้นๆ ได้ทันที |
    | **Risk Register Assistant** | **เครื่องมือสำหรับ Risk Owner**: เพียงป้อนรหัสอุบัติการณ์ (เช่น `CPM201`) เพื่อดูข้อมูลสรุป, สถิติ, และมาตรการที่เกี่ยวข้องทั้งหมดในที่เดียวสำหรับเตรียมการทบทวน Risk Profile ไปจนถึง Risk Register |
    | **Heatmap รายเดือน** | **ค้นหารูปแบบตามช่วงเวลา**: แสดงความถี่ของอุบัติการณ์ในแต่ละเดือน ช่วยให้มองเห็นแนวโน้มหรือปัญหาที่มักเกิดซ้ำในฤดูกาลหรือช่วงเวลาเดิมๆ |
    | **Sentinel Events & Top 10** | **โฟกัสเรื่องสำคัญและเรื่องที่เกิดบ่อย**: แสดงรายการอุบัติการณ์รุนแรง (Sentinel Events) ที่ต้องให้ความสำคัญสูงสุด และ 10 อันดับอุบัติการณ์ที่เกิดขึ้นบ่อยที่สุด |
    | **Sankey: ภาพรวม** | **เห็นเส้นทางการไหลของความเสี่ยง**: แสดงแผนภาพที่เชื่อมโยงจาก "หมวด" ไปสู่ "ระดับความรุนแรง" และ "ระดับความเสี่ยง" ช่วยให้เข้าใจว่าอุบัติการณ์ประเภทไหนมักจะนำไปสู่ความเสี่ยงระดับใด |
    | **Sankey: PSG9** | **เจาะลึกเส้นทางความเสี่ยงของ PSG9**: เหมือน Sankey ภาพรวม แต่เน้นเฉพาะอุบัติการณ์ที่เกี่ยวข้องกับมาตรฐานความปลอดภัย 9 ข้อ เพื่อการวิเคราะห์ที่ตรงจุด |
    | **สรุปอุบัติการณ์ตาม Safety Goals** | **ค้นหา "จุดร้อน" ที่มีความรุนแรงสูง**: เปรียบเทียบ "สัดส่วนอุบัติการณ์รุนแรง (% E-up)" ระหว่างหัวข้อต่างๆ เพื่อมองหาหัวข้อที่มีสัดส่วนเคสรุนแรงสูงผิดปกติ และควรเข้าไปตรวจสอบ รวมถึงนำไป Benchmark กับ NRLS ระดับประเทศ|
    | **Persistence Risk Index** | **ค้นหา "ปัญหาเรื้อรัง" ขององค์กร**: จัดอันดับอุบัติการณ์ที่เกิดขึ้นบ่อยและมีความรุนแรงเฉลี่ยสูง ซึ่งเป็นปัญหาที่ควรได้รับการทบทวนและแก้ไขในเชิงระบบ |
    | **Early Warning** | **สัญญาณเตือนภัยล่วงหน้า**: วิเคราะห์แนวโน้มเพื่อค้นหาอุบัติการณ์ที่ "มีแนวโน้ม" จะเกิดบ่อยขึ้นหรือรุนแรงขึ้นในอนาคต **(เมนูนี้ต้องดูอย่างสม่ำเสมอ!)** |
    | **บทสรุปสำหรับผู้บริหาร** | **รายงานสรุปฉบับพิมพ์ได้**: รวบรวมข้อมูลสำคัญทั้งหมดจากทุกเมนูมาไว้ในหน้าเดียวในรูปแบบที่กระชับและเหมาะสำหรับการนำเสนอหรือพิมพ์เป็นเอกสาร |
    """)

    st.markdown("### **เคล็ดลับเพื่อการใช้งานสูงสุด ✨**")
    st.success("""
    - **ข้อมูลที่มีคุณภาพคือหัวใจ**: ความถูกต้องของข้อมูลที่อัปโหลด มีผลโดยตรงต่อความแม่นยำของการวิเคราะห์
    - **ใช้งานเป็นประจำ**: ควรเข้ามาดูแดชบอร์ดอย่างน้อยสัปดาห์ละครั้ง โดยเฉพาะเมนู **Early Warning** และ **Persistence Risk**
    - **ใช้ฟิลเตอร์ให้เป็นประโยชน์**: ใช้ตัวกรองช่วงเวลา (Filter by Date) เพื่อเปรียบเทียบข้อมูลรายไตรมาส หรือดูแนวโน้มแบบปีต่อปี
    - **ต่อยอดสู่การปฏิบัติ**: เป้าหมายของแอปนี้คือการนำข้อมูลไปสู่ "การลงมือทำ" เพื่อเพิ่มความปลอดภัยให้กับผู้ป่วยและบุคลากร
    """)

# ==============================================================================
# ADMIN PAGE
# ==============================================================================
def display_admin_page():
    st.title("🔑 Admin: Data Upload")
    st.header("อัปโหลดไฟล์รายงานอุบัติการณ์ (.csv)")

    st.markdown("""
    <div style="font-size:16px">
      <ul>
        <li>เข้าสู่ระบบ <b>HRMS</b> ด้วยสิทธิ์ <b>Admin</b></li>
        <li>ไปที่เมนู <b>‘รายงาน’</b> > <b>‘การส่งออกข้อมูลรายงานอุบัติการณ์ขององค์กร (Excel File)’</b></li>
        <li>บันทึกเป็น <b>CSV UTF-8</b> แล้วอัปโหลดด้านล่าง</li>
        <li>✅ ถ้าไม่อัปโหลด สามารถกดปุ่ม <i>ใช้ไฟล์จาก GitHub</i> เพื่อดึง <code>Validate.csv</code> (RAW) มาใช้</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        uploaded_file = st.file_uploader("เลือกไฟล์ของคุณที่นี่ (.csv)", type=[".csv"])
    with c2:
        use_github = st.button("⬇️ ใช้ไฟล์จาก GitHub (Validate.csv)")
    with c3:
        reset_cache = st.button("🧹 ล้างข้อมูลที่บันทึกไว้")

    # ล้างพาร์เก็ตเก่า
    if reset_cache and PERSISTED_DATA_PATH.exists():
        PERSISTED_DATA_PATH.unlink(missing_ok=True)
        st.success("ล้างไฟล์ข้อมูลที่บันทึกไว้แล้ว (parquet)")

    # เส้นทางที่ 1: อัปโหลดไฟล์
    if uploaded_file:
        with st.spinner("กำลังอ่านไฟล์ที่อัปโหลด..."):
            try:
                uploaded_file.seek(0)
                df_raw = pd.read_csv(uploaded_file, keep_default_na=False, encoding='utf-8-sig', engine='python')
                st.success(f"อ่านไฟล์ที่อัปโหลดสำเร็จ — {len(df_raw):,} แถว")
            except Exception as e:
                st.error(f"อ่านไฟล์ที่อัปโหลดไม่สำเร็จ: {e}")
                return

        with st.spinner("กำลังประมวลผลข้อมูล..."):
            df = process_incident_dataframe(df_raw)
            if df.empty:
                st.error("ประมวลผลไม่สำเร็จ (ข้อมูลว่างหรือรูปแบบไม่ถูกต้อง)")
                return
            save_processed(df, note="จากไฟล์ที่อัปโหลด")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        return

    # เส้นทางที่ 2: ใช้ GitHub
    if use_github:
        with st.spinner("กำลังดึง CSV จาก GitHub (RAW) ..."):
            df_raw = load_csv_from_url_fallback(DEFAULT_CSV_URL)
            if df_raw.empty:
                st.error("โหลดไฟล์จาก GitHub ไม่สำเร็จ")
                return
            st.info(f"โหลดจาก GitHub ได้ {len(df_raw):,} แถว")

        with st.spinner("กำลังประมวลผลข้อมูล..."):
            df = process_incident_dataframe(df_raw)
            if df.empty:
                st.error("ประมวลผลไม่สำเร็จ (ข้อมูลว่างหรือรูปแบบไม่ถูกต้อง)")
                return
            save_processed(df, note="จาก GitHub RAW")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        return

    st.info("📎 อัปโหลดไฟล์ .csv หรือกด “ใช้ไฟล์จาก GitHub (Validate.csv)” ด้านบน")

# ==============================================================================
# DASHBOARD PAGES (Main Routing)
# ==============================================================================
def display_executive_dashboard():
    log_visit()

    # --- Sidebar title ---
    st.sidebar.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <img src="{LOGO_URL}" style="height: 32px; margin-right: 10px;">
            <h2 style="margin: 0; font-size: 1.7rem;">
                <span class="gradient-text">HOIA-RR Menu</span>
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # เมนู
    app_functions_list = ["คู่มือการใช้งาน", "RCA Helpdesk (AI Assistant)", "จัดการข้อมูล (Admin)"]
    dashboard_pages_list = [
        "แดชบอร์ดสรุปภาพรวม",
        "Incidents Analysis",
        "Risk Matrix (Interactive)",
        "Risk Register Assistant",
        "Heatmap รายเดือน",
        "Sentinel Events & Top 10",
        "Sankey: ภาพรวม",
        "Sankey: มาตรฐานสำคัญจำเป็นต่อความปลอดภัย 9 ข้อ",
        "สรุปอุบัติการณ์ตาม Safety Goals",
        "Persistence Risk Index",
        "Early Warning: อุบัติการณ์ที่มีแนวโน้มสูงขึ้น",
        "บทสรุปสำหรับผู้บริหาร",
    ]

    # default
    if 'selected_analysis' not in st.session_state:
        st.session_state.selected_analysis = "RCA Helpdesk (AI Assistant)"

    st.sidebar.markdown("---")
    for option in app_functions_list:
        button_clicked = st.sidebar.button(
            option,
            key=f"btn_{option}",
            type="primary" if st.session_state.selected_analysis == option else "secondary",
            use_container_width=True
        )
        if button_clicked:
            log_button_click(option)
            st.session_state.selected_analysis = option
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("เลือกส่วนที่ต้องการแสดงผล:")

    for option in dashboard_pages_list:
        button_clicked = st.sidebar.button(
            option,
            key=f"btn_{option}",
            type="primary" if st.session_state.selected_analysis == option else "secondary",
            use_container_width=True
        )
        if button_clicked:
            log_button_click(option)
            st.session_state.selected_analysis = option
            st.rerun()

    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
            **กิตติกรรมประกาศ:** 
            ขอขอบพระคุณ 
            - Prof. Shin Ushiro 
            - นพ.อนุวัฒน์ ศุภชุติกุล 
            - นพ.ก้องเกียรติ เกษเพ็ชร์ 
            - พญ.ปิยวรรณ ลิ้มปัญญาเลิศ 
            - ภก.ปรมินทร์ วีระอนันตวัฒน์    
            - ผศ.ดร.นิเวศน์ กุลวงค์ (อ.ที่ปรึกษา)

            เป็นอย่างสูง สำหรับการริเริ่ม เติมเต็ม สนับสนุน และสร้างแรงบันดาลใจ อันเป็นรากฐานสำคัญในการพัฒนาเครื่องมือนี้

            ขอบพระคุณผู้เชี่ยวชาญในการตรวจสอบเครื่องมือ
            - ศ.นพ.สงวนสิน รัตนเลิศ
            - ผศ.ดร.นพ. ปวีณ ตั้งจิตต์พิสุทธิ์
            - ผศ.ดร.นพ. อานนท์ จำลองกุล
            - พว.ศิริลักษณ์  โพธิกุล
            - พว.วราภรณ์ ภัทรมงคลเขตต์
            
            และขอบพระคุณโรงพยาบาลที่เข้าร่วมการวิจัยทุกแห่งเป็นอย่างสูง สำหรับความอนุเคราะห์และเอื้อเฟื้อข้อมูลอันเป็นประโยชน์ยิ่งต่องานวิจัยฉบับนี้ ได้แก่:
            - โรงพยาบาลบึงกาฬ จ.บึงกาฬ
            - โรงพยาบาลแม่สาย จ.เชียงราย 
            - โรงพยาบาลสวนผึ้ง จ.ราชบุรี
            - โรงพยาบาลเจ้าคุณไพบูลย์ พนมทวน จ.กาญจนบุรี
            - โรงพยาบาลชะอวด จ.นครศรีธรรมราช
            - โรงพยาบาลอุบลรักษ์ ธนบุรี จ.อุบลราชธานี
            - โรงพยาบาลเขาชะเมาเฉลิมพระเกียรติ ๘๐ พรรษา จ.ระยอง
            - โรงพยาบาลสมเด็จพระยุพราชเชียงของ จ. เชียงราย             
            - Kyushu University Hospital, Fukuoka, Japan (ศึกษาดูงาน)
            - โรงพยาบาลกรุงเทพ จันทบุรี, จ.จันทบุรี (ศึกษาดูงาน)
            - โรงพยาบาลศูนย์การแพทย์มหาวิทยาลัยแม่ฟ้าหลวง จ.เชียงราย (ต้นสังกัด)  
            """)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="font-size:12px; color:gray;">*เครื่องมือนี้เป็นส่วนหนึ่งของวิทยานิพนธ์ IMPLEMENTING THE  HOSPITAL OCCURRENCE/INCIDENT ANALYSIS & RISK REGISTER (HOIA-RR TOOL) IN THAI HOSPITALS: A STUDY ON EFFECTIVE ADOPTION โดย นางสาววิลาศินี  เขื่อนแก้ว นักศึกษาปริญญาโท สำนักวิชาวิทยาศาสตร์สุขภาพ มหาวิทยาลัยแม่ฟ้าหลวง</p>',
        unsafe_allow_html=True
    )

    selected_analysis = st.session_state.selected_analysis

    # ==============================================================================
    #  PART 1: Pages that do not require data
    # ==============================================================================
    if selected_analysis in app_functions_list:
        if selected_analysis == "คู่มือการใช้งาน":
            display_user_guide()

        elif selected_analysis == "RCA Helpdesk (AI Assistant)":
            st.markdown("<h4 style='color: #001f3f;'>AI Assistant: ที่ปรึกษาเคสอุบัติการณ์</h4>", unsafe_allow_html=True)

            AI_IS_CONFIGURED = False
            if genai:
                api_key = os.environ.get("GOOGLE_API_KEY")
                if api_key:
                    try:
                        genai.configure(api_key=api_key)
                        AI_IS_CONFIGURED = True
                    except Exception as e:
                        st.error(f"⚠️ เกิดข้อผิดพลาดในการตั้งค่า AI Assistant: {e}")
                else:
                    st.error("⚠️ ไม่สามารถตั้งค่า AI Assistant ได้: ไม่พบ 'GOOGLE_API_KEY' ใน Environment Variables")
            else:
                st.error("⚠️ ไม่สามารถตั้งค่า AI Assistant ได้: ไม่พบไลบรารี google.generativeai")

            if not AI_IS_CONFIGURED:
                st.stop()

            st.info("อธิบายรายละเอียดของอุบัติการณ์ที่เกิดขึ้น เพื่อให้ AI ช่วยให้คำปรึกษา")
            incident_description = st.text_area(
                "กรุณาอธิบายรายละเอียดอุบัติการณ์ที่นี่:",
                height=200,
                placeholder="เช่น ผู้ป่วยหญิงอายุ 65 ปี เป็นโรคเบาหวาน ได้รับยา losartan แต่เกิดผื่นขึ้นทั่วตัว...",
                key="rca_incident_input"
            )

            if st.button("ขอคำปรึกษาจาก AI", type="primary", use_container_width=True):
                log_button_click("ขอคำปรึกษาจาก AI")
                if incident_description.strip():
                    with st.spinner("AI กำลังวิเคราะห์และให้คำปรึกษา..."):
                        consultation = get_consultation_response(incident_description)
                        st.markdown("---")
                        st.markdown("### ผลการปรึกษาจาก AI:")
                        st.markdown(consultation)

        elif selected_analysis == "จัดการข้อมูล (Admin)":
            display_admin_page()

        return

    # ==============================================================================
    #  PART 2: Pages require data
    # ==============================================================================
    # Load parquet or fallback github
    try:
        df = pd.read_parquet(PERSISTED_DATA_PATH)
        df['Occurrence Date'] = pd.to_datetime(df['Occurrence Date'])
        st.caption(f"แหล่งข้อมูล: พาร์เก็ตที่บันทึกไว้ • {len(df):,} แถว")
    except FileNotFoundError:
        st.warning("ยังไม่มีข้อมูลที่บันทึกไว้ จะพยายามโหลดจาก GitHub (Validate.csv)")
        df_raw = load_csv_from_url_fallback(DEFAULT_CSV_URL)
        if df_raw.empty:
            st.error("โหลดจาก GitHub ไม่สำเร็จ กรุณาไปหน้า Admin เพื่ออัปโหลด/กดดึงจาก GitHub")
            return
        df = process_incident_dataframe(df_raw)
        if df.empty:
            st.error("ประมวลผลไฟล์จาก GitHub ไม่สำเร็จ")
            return
        save_processed(df, note="(auto-fallback)")
        st.caption(f"แหล่งข้อมูล: GitHub RAW (auto-fallback) • {len(df):,} แถว")

    # --- Filters ---
    st.sidebar.header("Filter by Date")
    min_date_in_data = df['Occurrence Date'].min().date()
    max_date_in_data = df['Occurrence Date'].max().date()
    today = datetime.now().date()

    filter_option = st.sidebar.selectbox(
        "เลือกช่วงเวลา:",
        ["ทั้งหมด", "ปีนี้", "ไตรมาสนี้", "เดือนนี้", "ปีที่แล้ว", "กำหนดเอง..."]
    )

    start_date, end_date = min_date_in_data, max_date_in_data
    if filter_option == "ปีนี้":
        start_date = today.replace(month=1, day=1)
        end_date = today
    elif filter_option == "ไตรมาสนี้":
        current_quarter = (today.month - 1) // 3 + 1
        start_date = datetime(today.year, 3 * current_quarter - 2, 1).date()
        end_date = today
    elif filter_option == "เดือนนี้":
        start_date = today.replace(day=1)
        end_date = today
    elif filter_option == "ปีที่แล้ว":
        last_year = today.year - 1
        start_date = datetime(last_year, 1, 1).date()
        end_date = datetime(last_year, 12, 31).date()
    elif filter_option == "กำหนดเอง...":
        start_date, end_date = st.sidebar.date_input(
            "เลือกระหว่างวันที่:",
            [min_date_in_data, max_date_in_data],
            min_value=min_date_in_data,
            max_value=max_date_in_data
        )

    df_filtered = df[
        (df['Occurrence Date'].dt.date >= start_date) &
        (df['Occurrence Date'].dt.date <= end_date)
    ].copy()

    df_filtered['Incident Type Name'] = df_filtered['Incident Type'].map(type_name).fillna(df_filtered['Incident Type'])

    if df_filtered.empty:
        st.sidebar.warning("ไม่พบข้อมูลในช่วงเวลาที่ท่านเลือก")
        st.warning("ไม่พบข้อมูลในช่วงเวลาที่ท่านเลือก กรุณาเลือกช่วงเวลาอื่น")
        return

    min_date_str = df_filtered['Occurrence Date'].min().strftime('%d/%m/%Y')
    max_date_str = df_filtered['Occurrence Date'].max().strftime('%d/%m/%Y')

    max_p = df_filtered['Occurrence Date'].max().to_period('M')
    min_p = df_filtered['Occurrence Date'].min().to_period('M')
    total_month = (max_p.year - min_p.year) * 12 + (max_p.month - min_p.month) + 1
    total_month = max(1, int(total_month))

    st.sidebar.markdown(f"**ช่วงข้อมูล:** {min_date_str} ถึง {max_date_str}")
    st.sidebar.markdown(f"**จำนวนเดือน:** {total_month} เดือน")
    st.sidebar.markdown(f"**จำนวนอุบัติการณ์:** {df_filtered.shape[0]:,} รายการ")

    # --- Metrics ---
    metrics_data = {}
    metrics_data['total_processed_incidents'] = df_filtered.shape[0]
    metrics_data['total_psg9_incidents_for_metric1'] = (
        df_filtered[df_filtered['รหัส'].isin(psg9_r_codes_for_counting)].shape[0]
        if psg9_r_codes_for_counting else 0
    )

    if sentinel_composite_keys:
        df_filtered['Sentinel code for check'] = df_filtered['รหัส'].astype(str).str.strip() + '-' + df_filtered['Impact'].astype(str).str.strip()
        metrics_data['total_sentinel_incidents_for_metric1'] = df_filtered[df_filtered['Sentinel code for check'].isin(sentinel_composite_keys)].shape[0]
    else:
        metrics_data['total_sentinel_incidents_for_metric1'] = 0

    severe_impact_levels_list = ['3', '4', '5']
    df_severe_incidents_calc = df_filtered[df_filtered['Impact Level'].isin(severe_impact_levels_list)].copy()
    metrics_data['total_severe_incidents'] = df_severe_incidents_calc.shape[0]

    if 'Resulting Actions' in df_filtered.columns:
        unresolved_conditions = df_severe_incidents_calc['Resulting Actions'].astype(str).isin(['None', '', 'nan'])
        df_severe_unresolved_calc = df_severe_incidents_calc[unresolved_conditions].copy()
        metrics_data['total_severe_unresolved_incidents_val'] = df_severe_unresolved_calc.shape[0]
        metrics_data['total_severe_unresolved_psg9_incidents_val'] = (
            df_severe_unresolved_calc[df_severe_unresolved_calc['รหัส'].isin(psg9_r_codes_for_counting)].shape[0]
            if psg9_r_codes_for_counting else 0
        )
    else:
        metrics_data['total_severe_unresolved_incidents_val'] = "N/A"
        metrics_data['total_severe_unresolved_psg9_incidents_val'] = "N/A"

    metrics_data['total_month'] = total_month

    df_freq = df_filtered['Incident'].value_counts().reset_index()
    df_freq.columns = ['Incident', 'count']

    # ==============================================================================
    # ROUTING: Render selected page
    # ==============================================================================
    if selected_analysis == "แดชบอร์ดสรุปภาพรวม":
        st.markdown("<h4 style='color: #001f3f;'>สรุปภาพรวมอุบัติการณ์:</h4>", unsafe_allow_html=True)

        with st.expander("แสดง/ซ่อน ตารางข้อมูลอุบัติการณ์ทั้งหมด (Full Data Table)"):
            safe_cols = ['Occurrence Date','Incident','Impact','รายละเอียดการเกิด_Anonymized','Resulting Actions']
            st.dataframe(
                df_filtered[safe_cols],
                hide_index=True,
                use_container_width=True,
                column_config={"Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด", format="DD/MM/YYYY")}
            )

        dashboard_expander_cols = ['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด_Anonymized', 'Resulting Actions']
        date_format_config = {"Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด", format="DD/MM/YYYY")}

        total_processed_incidents = metrics_data.get("total_processed_incidents", 0)
        total_psg9_incidents_for_metric1 = metrics_data.get("total_psg9_incidents_for_metric1", 0)
        total_sentinel_incidents_for_metric1 = metrics_data.get("total_sentinel_incidents_for_metric1", 0)
        total_severe_incidents = metrics_data.get("total_severe_incidents", 0)
        total_severe_unresolved_incidents_val = metrics_data.get("total_severe_unresolved_incidents_val", "N/A")
        total_severe_unresolved_psg9_incidents_val = metrics_data.get("total_severe_unresolved_psg9_incidents_val", "N/A")

        df_severe_incidents = df_filtered[df_filtered['Impact Level'].isin(['3', '4', '5'])].copy()
        total_severe_psg9_incidents = df_severe_incidents[df_severe_incidents['รหัส'].isin(psg9_r_codes_for_counting)].shape[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", f"{total_processed_incidents:,}")
        with col2:
            st.metric("มาตรฐานสำคัญจำเป็นฯ 9 ข้อ", f"{total_psg9_incidents_for_metric1:,}")
            with st.expander(f"ดูรายละเอียด ({total_psg9_incidents_for_metric1} รายการ)"):
                psg9_df = df_filtered[df_filtered['รหัส'].isin(psg9_r_codes_for_counting)]
                st.dataframe(psg9_df[dashboard_expander_cols], use_container_width=True, hide_index=True, column_config=date_format_config)
        with col3:
            st.metric("Sentinel", f"{total_sentinel_incidents_for_metric1:,}")
            with st.expander(f"ดูรายละเอียด ({total_sentinel_incidents_for_metric1} รายการ)"):
                if 'Sentinel code for check' in df_filtered.columns:
                    sentinel_df = df_filtered[df_filtered['Sentinel code for check'].isin(sentinel_composite_keys)]
                    st.dataframe(sentinel_df[dashboard_expander_cols], use_container_width=True, hide_index=True, column_config=date_format_config)

        col4, col5, col6, col7 = st.columns(4)
        with col4:
            st.metric("E-I & 3-5 [all]", f"{total_severe_incidents:,}")
            with st.expander(f"ดูรายละเอียด ({total_severe_incidents} รายการ)"):
                st.dataframe(df_severe_incidents[dashboard_expander_cols], use_container_width=True, hide_index=True, column_config=date_format_config)

        with col5:
            st.metric("E-I & 3-5 [มาตรฐานสำคัญฯ]", f"{total_severe_psg9_incidents:,}")
            with st.expander(f"ดูรายละเอียด ({total_severe_psg9_incidents} รายการ)"):
                severe_psg9_df = df_severe_incidents[df_severe_incidents['รหัส'].isin(psg9_r_codes_for_counting)]
                st.dataframe(severe_psg9_df[dashboard_expander_cols], use_container_width=True, hide_index=True, column_config=date_format_config)

        with col6:
            val_unresolved_all = f"{total_severe_unresolved_incidents_val:,}" if isinstance(total_severe_unresolved_incidents_val, int) else "N/A"
            st.metric("E-I & 3-5 [all] ที่ยังไม่ถูกแก้ไข", val_unresolved_all)

        with col7:
            val_unresolved_psg9 = f"{total_severe_unresolved_psg9_incidents_val:,}" if isinstance(total_severe_unresolved_psg9_incidents_val, int) else "N/A"
            st.metric("E-I & 3-5 [มาตรฐานสำคัญฯ] ที่ยังไม่ถูกแก้ไข", val_unresolved_psg9)

        st.markdown("---")

        monthly_counts = df_filtered.copy()
        monthly_counts['เดือน-ปี'] = monthly_counts['Occurrence Date'].dt.strftime('%Y-%m')
        incident_trend = monthly_counts.groupby('เดือน-ปี').size().reset_index(name='จำนวนอุบัติการณ์')
        incident_trend = incident_trend.sort_values(by='เดือน-ปี')

        total_incidents = metrics_data.get('total_processed_incidents', 0)
        resolved_incidents = df_filtered[~df_filtered['Resulting Actions'].astype(str).isin(['None', '', 'nan'])].shape[0]

        status_data = pd.DataFrame({
            'สถานะ': ['อุบัติการณ์ทั้งหมด', 'ที่แก้ไขแล้ว'],
            'จำนวน': [total_incidents, resolved_incidents]
        })

        fig_status = px.bar(
            status_data,
            x='จำนวน',
            y='สถานะ',
            orientation='h',
            title='ภาพรวมอุบัติการณ์ทั้งหมดเทียบกับที่แก้ไขแล้ว',
            text='จำนวน',
            color='สถานะ',
            labels={'สถานะ': '', 'จำนวน': 'จำนวนอุบัติการณ์'}
        )
        fig_status.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
        st.plotly_chart(fig_status, use_container_width=True)

        fig_trend = px.line(
            incident_trend,
            x='เดือน-ปี',
            y='จำนวนอุบัติการณ์',
            title='จำนวนอุบัติการณ์ทั้งหมดที่เกิดขึ้นในแต่ละเดือน',
            markers=True,
            labels={'เดือน-ปี': 'เดือน', 'จำนวนอุบัติการณ์': 'จำนวนครั้ง'},
            line_shape='spline'
        )
        fig_trend.update_traces(line=dict(width=3))
        st.plotly_chart(fig_trend, use_container_width=True)

    elif selected_analysis == "Incidents Analysis":
        st.markdown("<h4 style='color: #001f3f;'>Incidents Analysis</h4>", unsafe_allow_html=True)

        st.markdown("### สรุปตามรหัสอุบัติการณ์")
        code_summary = create_summary_table_by_code(df_filtered)
        if not code_summary.empty:
            st.dataframe(code_summary, use_container_width=True)
        else:
            st.info("ไม่พบข้อมูลสำหรับสร้างสรุปตามรหัส")

        st.markdown("---")
        st.markdown("### สรุปตามหมวด (AllCode)")
        if 'หมวด' in df_filtered.columns:
            cat_summary = create_summary_table_by_category(df_filtered, 'หมวด')
            if not cat_summary.empty:
                st.dataframe(cat_summary, use_container_width=True)
            else:
                st.info("ไม่พบข้อมูลหมวดสำหรับสรุป")
        else:
            st.warning("ไม่พบคอลัมน์ 'หมวด'")

        st.markdown("---")
        st.markdown("### สรุป PSG9")
        psg9_tbl = create_psg9_summary_table(df_filtered)
        if not psg9_tbl.empty:
            st.dataframe(psg9_tbl, use_container_width=True)
        else:
            st.info("ไม่พบข้อมูล PSG9 สำหรับสรุป")

    elif selected_analysis == "Heatmap รายเดือน":
        st.markdown("<h4 style='color: #001f3f;'>Heatmap: จำนวนอุบัติการณ์รายเดือน</h4>", unsafe_allow_html=True)
        st.info("Heatmap นี้แสดงความถี่ของการเกิดอุบัติการณ์แต่ละรหัสในแต่ละเดือน")

        heatmap_req_cols = ['รหัส', 'เดือน', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Month', 'หมวด']
        if not all(col in df_filtered.columns for col in heatmap_req_cols):
            st.warning(f"ไม่สามารถสร้าง Heatmap ได้ เนื่องจากขาดคอลัมน์: {', '.join(heatmap_req_cols)}")
        else:
            df_heat = df_filtered.copy()
            df_heat['incident_label'] = df_heat['รหัส'] + " | " + df_heat['ชื่ออุบัติการณ์ความเสี่ยง'].fillna('')

            total_counts = df_heat['incident_label'].value_counts().reset_index()
            total_counts.columns = ['incident_label', 'total_count']

            if len(total_counts) <= 1:
                top_n = 1
            else:
                top_n = st.slider(
                    "เลือกจำนวนอุบัติการณ์ (Top N) ที่ต้องการแสดง:",
                    min_value=1,
                    max_value=min(50, len(total_counts)),
                    value=min(20, len(total_counts)),
                    step=1
                )

            top_incident_labels = total_counts.nlargest(top_n, 'total_count')['incident_label']
            df_heat_view = df_heat[df_heat['incident_label'].isin(top_incident_labels)]

            heatmap_pivot = pd.pivot_table(
                df_heat_view,
                values='Incident',
                index='incident_label',
                columns='เดือน',
                aggfunc='count',
                fill_value=0
            )

            sorted_month_names = [v for k, v in sorted(month_label.items())]
            available_months = [m for m in sorted_month_names if m in heatmap_pivot.columns]
            if available_months:
                heatmap_pivot = heatmap_pivot[available_months]
                heatmap_pivot = heatmap_pivot.reindex(top_incident_labels).dropna()

            if not heatmap_pivot.empty:
                fig_heatmap = px.imshow(
                    heatmap_pivot,
                    labels=dict(x="เดือน", y="อุบัติการณ์", color="จำนวนครั้ง"),
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='Reds'
                )
                fig_heatmap.update_layout(
                    title_text=f"Heatmap ของอุบัติการณ์ Top {top_n}",
                    height=max(600, len(heatmap_pivot.index) * 25),
                    xaxis_title="เดือน",
                    yaxis_title="รหัส | ชื่ออุบัติการณ์"
                )
                fig_heatmap.update_xaxes(side="top")
                st.plotly_chart(fig_heatmap, use_container_width=True)

    elif selected_analysis == "Risk Matrix (Interactive)":
        st.subheader("Risk Matrix (Interactive)")

        matrix_data_counts = np.zeros((5, 5), dtype=int)
        impact_level_keys = ['5', '4', '3', '2', '1']
        freq_level_keys = ['1', '2', '3', '4', '5']

        matrix_df = df_filtered[
            df_filtered['Impact Level'].isin(impact_level_keys) &
            df_filtered['Frequency Level'].isin(freq_level_keys)
        ].copy()

        if not matrix_df.empty:
            risk_counts_df = matrix_df.groupby(['Impact Level', 'Frequency Level']).size().reset_index(name='counts')
            for _, row in risk_counts_df.iterrows():
                il_key, fl_key = str(row['Impact Level']), str(row['Frequency Level'])
                row_idx, col_idx = impact_level_keys.index(il_key), freq_level_keys.index(fl_key)
                matrix_data_counts[row_idx, col_idx] = int(row['counts'])

        impact_labels_display = {
            '5': "I / 5<br>Death",
            '4': "G-H / 4<br>Severe Harm",
            '3': "E-F / 3<br>Moderate Harm",
            '2': "C-D / 2<br>Low Harm",
            '1': "A-B / 1<br>No Harm"
        }
        freq_labels_display_short = {"1": "F1", "2": "F2", "3": "F3", "4": "F4", "5": "F5"}
        freq_labels_display_long = {
            "1": "Remote<br>(<2/mth)",
            "2": "Uncommon<br>(2-3/mth)",
            "3": "Occasional<br>(4-6/mth)",
            "4": "Probable<br>(7-29/mth)",
            "5": "Frequent<br>(>=30/mth)"
        }

        impact_to_color_row = {'5': 0, '4': 1, '3': 2, '2': 3, '1': 4}
        freq_to_color_col = {'1': 2, '2': 3, '3': 4, '4': 5, '5': 6}

        cols_header = st.columns([2.2, 1, 1, 1, 1, 1])
        with cols_header[0]:
            st.markdown(
                f"<div style='background-color:{colors2[6, 0]}; color:{get_text_color_for_bg(colors2[6, 0])}; padding:8px; text-align:center; font-weight:bold; border-radius:3px; margin:1px; height:60px; display:flex; align-items:center; justify-content:center;'>Impact / Frequency</div>",
                unsafe_allow_html=True
            )

        for i, fl_key in enumerate(freq_level_keys):
            with cols_header[i + 1]:
                header_freq_bg_color = colors2[5, freq_to_color_col.get(fl_key, 2) - 1]
                header_freq_text_color = get_text_color_for_bg(header_freq_bg_color)
                st.markdown(
                    f"<div style='background-color:{header_freq_bg_color}; color:{header_freq_text_color}; padding:8px; text-align:center; font-weight:bold; border-radius:3px; margin:1px; height:60px; display:flex; flex-direction: column; align-items:center; justify-content:center;'><div>{freq_labels_display_short.get(fl_key, '')}</div><div style='font-size:0.7em;'>{freq_labels_display_long.get(fl_key, '')}</div></div>",
                    unsafe_allow_html=True
                )

        for il_key in impact_level_keys:
            cols_data_row = st.columns([2.2, 1, 1, 1, 1, 1])
            row_idx_color = impact_to_color_row[il_key]

            with cols_data_row[0]:
                header_impact_bg_color = colors2[row_idx_color, 1]
                header_impact_text_color = get_text_color_for_bg(header_impact_bg_color)
                st.markdown(
                    f"<div style='background-color:{header_impact_bg_color}; color:{header_impact_text_color}; padding:8px; text-align:center; font-weight:bold; border-radius:3px; margin:1px; height:70px; display:flex; align-items:center; justify-content:center;'>{impact_labels_display[il_key]}</div>",
                    unsafe_allow_html=True
                )

            for i, fl_key in enumerate(freq_level_keys):
                with cols_data_row[i + 1]:
                    count = matrix_data_counts[impact_level_keys.index(il_key), freq_level_keys.index(fl_key)]
                    cell_bg_color = colors2[row_idx_color, freq_to_color_col[fl_key]]
                    text_color = get_text_color_for_bg(cell_bg_color)
                    st.markdown(
                        f"<div style='background-color:{cell_bg_color}; color:{text_color}; padding:5px; margin:1px; border-radius:3px; text-align:center; font-weight:bold; min-height:40px; display:flex; align-items:center; justify-content:center;'>{count}</div>",
                        unsafe_allow_html=True
                    )

                    if count > 0:
                        button_key = f"view_risk_{il_key}_{fl_key}"
                        if st.button("👁️", key=button_key, help=f"ดูรายการ - {count} รายการ", use_container_width=True):
                            st.session_state.clicked_risk_impact = il_key
                            st.session_state.clicked_risk_freq = fl_key
                            st.session_state.show_incident_table = True
                            st.rerun()
                    else:
                        st.markdown("<div style='height:38px; margin-top:5px;'></div>", unsafe_allow_html=True)

        if st.session_state.get('show_incident_table', False) and st.session_state.get('clicked_risk_impact') is not None:
            il_selected = st.session_state.clicked_risk_impact
            fl_selected = st.session_state.clicked_risk_freq

            df_incidents_in_cell = df_filtered[
                (df_filtered['Impact Level'].astype(str) == str(il_selected)) &
                (df_filtered['Frequency Level'].astype(str) == str(fl_selected))
            ].copy()

            expander_title = f"รายการอุบัติการณ์: Impact Level {il_selected}, Frequency Level {fl_selected} - พบ {len(df_incidents_in_cell)} รายการ"
            with st.expander(expander_title, expanded=True):
                st.dataframe(df_incidents_in_cell[display_cols_common], use_container_width=True, hide_index=True)
                if st.button("ปิดรายการ", key="clear_risk_selection_button"):
                    st.session_state.show_incident_table = False
                    st.session_state.clicked_risk_impact = None
                    st.session_state.clicked_risk_freq = None
                    st.rerun()

    elif selected_analysis == "Sentinel Events & Top 10":
        st.markdown("<h4 style='color: #001f3f;'>รายการ Sentinel Events ที่ตรวจพบ</h4>", unsafe_allow_html=True)

        if sentinel_composite_keys and 'Sentinel code for check' in df_filtered.columns:
            sentinel_events = df_filtered[df_filtered['Sentinel code for check'].isin(sentinel_composite_keys)].copy()

            if not sentinel_events.empty:
                if not Sentinel2024_df.empty and 'ชื่ออุบัติการณ์ความเสี่ยง' in Sentinel2024_df.columns:
                    sentinel_events = pd.merge(
                        sentinel_events,
                        Sentinel2024_df[['รหัส', 'Impact', 'ชื่ออุบัติการณ์ความเสี่ยง']].rename(
                            columns={'ชื่ออุบัติการณ์ความเสี่ยง': 'Sentinel Event Name'}
                        ),
                        on=['รหัส', 'Impact'],
                        how='left'
                    )

                display_sentinel_cols = [
                    'Occurrence Date', 'Incident', 'Impact',
                    'รายละเอียดการเกิด_Anonymized', 'Resulting Actions'
                ]
                if 'Sentinel Event Name' in sentinel_events.columns:
                    display_sentinel_cols.insert(2, 'Sentinel Event Name')

                st.dataframe(
                    sentinel_events[display_sentinel_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={"Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด", format="DD/MM/YYYY")}
                )
            else:
                st.info("ไม่พบ Sentinel Events ในช่วงเวลาที่เลือก")
        else:
            st.warning("ไม่สามารถตรวจสอบ Sentinel Events ได้ (ไฟล์ Sentinel2024.xlsx อาจไม่มีข้อมูล)")

        st.markdown("---")
        st.subheader("Top 10 อุบัติการณ์ (ตามความถี่)")

        if not df_freq.empty:
            df_freq_top10 = df_freq.nlargest(10, 'count')
            incident_names = df_filtered[['Incident', 'ชื่ออุบัติการณ์ความเสี่ยง']].drop_duplicates()
            df_freq_top10 = pd.merge(df_freq_top10, incident_names, on='Incident', how='left')

            st.dataframe(
                df_freq_top10[['Incident', 'ชื่ออุบัติการณ์ความเสี่ยง', 'count']],
                column_config={
                    "Incident": "รหัส",
                    "ชื่ออุบัติการณ์ความเสี่ยง": "ชื่ออุบัติการณ์",
                    "count": st.column_config.NumberColumn("จำนวนครั้ง", format="%d")
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("ไม่สามารถแสดง Top 10 อุบัติการณ์ได้")

    elif selected_analysis == "Sankey: ภาพรวม":
        st.markdown("<h4 style='color: #001f3f;'>Sankey Diagram: ภาพรวม</h4>", unsafe_allow_html=True)
        st.markdown("""
        <style>
            .plot-container .svg-container .sankey-node text {
                stroke-width: 0 !important; text-shadow: none !important; paint-order: stroke fill;
            }
        </style>
        """, unsafe_allow_html=True)

        req_cols = ['หมวด', 'Impact', 'Impact Level', 'Category Color']
        if not all(col in df_filtered.columns for col in req_cols):
            st.warning(f"ไม่พบคอลัมน์ที่จำเป็น ({', '.join(req_cols)}) สำหรับการสร้าง Sankey diagram")
        else:
            sankey_df = df_filtered.copy()
            placeholders = ['None', '', 'N/A', 'ไม่ระบุ',
                            'N/A (ข้อมูลจาก AllCode ไม่พร้อมใช้งาน)',
                            'N/A (ไม่พบรหัสใน AllCode หรือค่าว่างใน AllCode)']
            sankey_df = sankey_df[~sankey_df['หมวด'].astype(str).isin(placeholders)]
            if sankey_df.empty:
                st.warning("ไม่สามารถสร้าง Sankey Diagram ได้ เนื่องจากไม่มีข้อมูล 'หมวด' ที่ถูกต้องในช่วงเวลาที่เลือก")
            else:
                sankey_df['หมวด_Node'] = "หมวด: " + sankey_df['หมวด'].astype(str).str.strip()
                sankey_df['Impact_AI_Node'] = "Impact: " + sankey_df['Impact'].astype(str).str.strip()
                sankey_df.loc[sankey_df['Impact'].astype(str).isin(placeholders), 'Impact_AI_Node'] = "Impact: ไม่ระบุ A-I"

                impact_level_display_map = {
                    '1': "Level: 1 (A-B)", '2': "Level: 2 (C-D)",
                    '3': "Level: 3 (E-F)", '4': "Level: 4 (G-H)",
                    '5': "Level: 5 (I)", 'N/A': "Level: ไม่ระบุ"
                }
                sankey_df['Impact_Level_Node'] = sankey_df['Impact Level'].astype(str).str.strip().map(
                    impact_level_display_map
                ).fillna("Level: ไม่ระบุ")

                sankey_df['Risk_Category_Node'] = "Risk: " + sankey_df['Category Color'].astype(str).str.strip()

                node_cols = ['หมวด_Node', 'Impact_AI_Node', 'Impact_Level_Node', 'Risk_Category_Node']
                sankey_df.dropna(subset=node_cols, inplace=True)

                labels_muad = sorted(list(sankey_df['หมวด_Node'].unique()))
                impact_ai_order = [f"Impact: {i}" for i in list("ABCDEFGHI")] + ["Impact: ไม่ระบุ A-I"]
                labels_impact_ai = sorted(
                    list(sankey_df['Impact_AI_Node'].unique()),
                    key=lambda x: impact_ai_order.index(x) if x in impact_ai_order else 999
                )
                level_order_map = {
                    "Level: 1 (A-B)": 1, "Level: 2 (C-D)": 2, "Level: 3 (E-F)": 3,
                    "Level: 4 (G-H)": 4, "Level: 5 (I)": 5, "Level: ไม่ระบุ": 6
                }
                labels_impact_level = sorted(
                    list(sankey_df['Impact_Level_Node'].unique()),
                    key=lambda x: level_order_map.get(x, 999)
                )
                risk_order = ["Risk: Critical", "Risk: High", "Risk: Medium", "Risk: Low", "Risk: Undefined"]
                labels_risk_cat = sorted(
                    list(sankey_df['Risk_Category_Node'].unique()),
                    key=lambda x: risk_order.index(x) if x in risk_order else 999
                )

                all_labels_ordered = labels_muad + labels_impact_ai + labels_impact_level + labels_risk_cat
                all_labels = list(pd.Series(all_labels_ordered).unique())
                label_to_idx = {label: i for i, label in enumerate(all_labels)}

                source_indices, target_indices, values = [], [], []

                links1 = sankey_df.groupby(['หมวด_Node', 'Impact_AI_Node']).size().reset_index(name='value')
                for _, row in links1.iterrows():
                    source_indices.append(label_to_idx[row['หมวด_Node']])
                    target_indices.append(label_to_idx[row['Impact_AI_Node']])
                    values.append(row['value'])

                links2 = sankey_df.groupby(['Impact_AI_Node', 'Impact_Level_Node']).size().reset_index(name='value')
                for _, row in links2.iterrows():
                    source_indices.append(label_to_idx[row['Impact_AI_Node']])
                    target_indices.append(label_to_idx[row['Impact_Level_Node']])
                    values.append(row['value'])

                links3 = sankey_df.groupby(['Impact_Level_Node', 'Risk_Category_Node']).size().reset_index(name='value')
                for _, row in links3.iterrows():
                    source_indices.append(label_to_idx[row['Impact_Level_Node']])
                    target_indices.append(label_to_idx[row['Risk_Category_Node']])
                    values.append(row['value'])

                node_colors = []
                palette1 = px.colors.qualitative.Pastel1
                palette2 = px.colors.qualitative.Pastel2
                palette3 = px.colors.qualitative.Set3
                risk_color_map = {
                    "Risk: Critical": "red",
                    "Risk: High": "orange",
                    "Risk: Medium": "#F7DC6F",
                    "Risk: Low": "green",
                    "Risk: Undefined": "grey"
                }

                for label in all_labels:
                    if label.startswith("หมวด:"):
                        node_colors.append(palette1[labels_muad.index(label) % len(palette1)])
                    elif label.startswith("Impact:"):
                        node_colors.append(palette2[labels_impact_ai.index(label) % len(palette2)])
                    elif label.startswith("Level:"):
                        node_colors.append(palette3[labels_impact_level.index(label) % len(palette3)])
                    elif label.startswith("Risk:"):
                        node_colors.append(risk_color_map.get(label, 'grey'))
                    else:
                        node_colors.append('rgba(200,200,200,0.8)')

                link_colors_rgba = [
                    'rgba(200,200,200,0.3)' for _ in source_indices
                ]

                fig = go.Figure(data=[go.Sankey(
                    arrangement='snap',
                    node=dict(
                        pad=15,
                        thickness=18,
                        line=dict(color="rgba(0,0,0,0.6)", width=0.75),
                        label=all_labels,
                        color=node_colors,
                        hovertemplate='%{label} มีจำนวน: %{value}<extra></extra>'
                    ),
                    link=dict(
                        source=source_indices,
                        target=target_indices,
                        value=values,
                        color=link_colors_rgba,
                        hovertemplate='จาก %{source.label}<br />ไปยัง %{target.label}<br />จำนวน: %{value}<extra></extra>'
                    )
                )])

                fig.update_layout(
                    title_text="<b>แผนภาพ Sankey:</b> หมวด -> Impact (A-I) -> Impact Level (1-5) -> Risk Category",
                    font_size=12,
                    height=max(700, len(all_labels) * 18),
                    template='plotly_white',
                    margin=dict(t=60, l=10, r=10, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

    elif selected_analysis == "สรุปอุบัติการณ์ตาม Safety Goals":
        st.markdown("<h4 style='color: #001f3f;'>สรุปอุบัติการณ์ตามเป้าหมาย (Safety Goals)</h4>", unsafe_allow_html=True)

        goal_definitions = {
            "Patient Safety/ Common Clinical Risk": "P:Patient Safety Goals หรือ Common Clinical Risk Incident",
            "Specific Clinical Risk": "S:Specific Clinical Risk Incident",
            "Personnel Safety": "P:Personnel Safety Goals",
            "Organization Safety": "O:Organization Safety Goals"
        }

        for display_name, cat_name in goal_definitions.items():
            st.markdown(f"##### {display_name}")
            is_org_safety = (display_name == "Organization Safety")

            summary_table = create_goal_summary_table(
                df_filtered,
                cat_name,
                e_up_non_numeric_levels_param=[] if is_org_safety else ['A', 'B', 'C', 'D'],
                e_up_numeric_levels_param=['1', '2'] if is_org_safety else None,
                is_org_safety_table=is_org_safety
            )

            if summary_table is not None and not summary_table.empty:
                st.dataframe(summary_table, use_container_width=True)
            else:
                st.info(f"ไม่มีข้อมูลสำหรับ '{display_name}' ในช่วงเวลาที่เลือก")

    elif selected_analysis == "Persistence Risk Index":
        st.markdown("<h4 style='color: #001f3f;'>Persistence Risk Index</h4>", unsafe_allow_html=True)

        persistence_df = calculate_persistence_risk_score(df_filtered, total_month)
        if persistence_df.empty:
            st.info("ไม่มีข้อมูลเพียงพอสำหรับวิเคราะห์ความเสี่ยงเรื้อรัง")
        else:
            st.dataframe(persistence_df.head(50), use_container_width=True, hide_index=True)

    elif selected_analysis == "Early Warning: อุบัติการณ์ที่มีแนวโน้มสูงขึ้น":
        st.markdown("<h4 style='color: #001f3f;'>Early Warning</h4>", unsafe_allow_html=True)

        ew_df = prioritize_incidents_nb_logit_v2(df_filtered, horizon=3, min_months=4, min_total=5)
        if ew_df.empty:
            st.info("ไม่มีข้อมูลเพียงพอสำหรับวิเคราะห์ Early Warning")
        else:
            st.dataframe(ew_df.head(50), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("ดูแนวโน้มรายเดือน (Poisson Plot)")
            selected_code = st.selectbox("เลือกรหัสที่ต้องการดูกราฟ", ew_df['รหัส'].unique().tolist())
            fig_p = create_poisson_trend_plot(df_filtered, selected_code_for_plot=selected_code, show_ci=True)
            st.plotly_chart(fig_p, use_container_width=True)

    elif selected_analysis == "บทสรุปสำหรับผู้บริหาร":
        st.markdown("<h4 style='color: #001f3f;'>บทสรุปสำหรับผู้บริหาร</h4>", unsafe_allow_html=True)

        st.info("สามารถดาวน์โหลด PDF ได้ (ต้องมี weasyprint ติดตั้งใน environment)")

        if st.button("📄 สร้างรายงาน PDF", type="primary", use_container_width=True):
            pdf_bytes = generate_executive_summary_pdf(
                df_filtered=df_filtered,
                metrics_data=metrics_data,
                total_month=total_month,
                min_date_str=min_date_str,
                max_date_str=max_date_str,
                df_freq=df_freq
            )
            if pdf_bytes:
                st.download_button(
                    label="⬇️ ดาวน์โหลด PDF",
                    data=pdf_bytes,
                    file_name=f"HOIA-RR_ExecutiveSummary_{min_date_str.replace('/','-')}_to_{max_date_str.replace('/','-')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    else:
        st.info("หน้านี้อยู่ระหว่างการพัฒนา/หรือยังไม่ได้เปิดในโค้ดชุดนี้")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    display_executive_dashboard()


if __name__ == "__main__":
    main()

# analytics.py
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# --- การตั้งค่าการเชื่อมต่อ ---
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file"
]

# ใช้ st.secrets เพื่อความปลอดภัยเมื่อ deploy บน Render
CREDS_JSON = st.secrets["gcp_service_account"]
SHEET_NAME = "HOIA-RR-Analytics" # ชื่อ Google Sheet ที่คุณสร้าง

@st.cache_resource
def get_gspread_client():
    """สร้างและคืน client สำหรับเชื่อมต่อ Google Sheets"""
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_dict(CREDS_JSON, SCOPE)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"GSpread Connection Error: {e}")
        return None

def log_to_sheet(sheet_name, data_row: list):
    """ฟังก์ชันกลางสำหรับบันทึกข้อมูลลงชีต"""
    client = get_gspread_client()
    if client:
        try:
            sheet = client.open(SHEET_NAME).worksheet(sheet_name)
            sheet.append_row(data_row)
        except Exception as e:
            # ใน Production จริง อาจจะแค่ print log แทนการแสดง error
            print(f"Failed to log to sheet '{sheet_name}': {e}")

# --- ฟังก์ชันหลักสำหรับเรียกใช้ ---
def log_visit():
    """บันทึกการเข้าชมแอป (จะทำงานแค่ครั้งแรกของ Session)"""
    if 'visit_logged' not in st.session_state:
        now_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        log_to_sheet("Visits", [now_utc])
        st.session_state.visit_logged = True

def log_button_click(button_name: str):
    """บันทึกการคลิกปุ่ม"""
    now_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    log_to_sheet("Clicks", [now_utc, button_name])

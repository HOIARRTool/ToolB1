# analytics.py
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import os  
import json 

# --- การตั้งค่าการเชื่อมต่อ ---
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file"
]

# อ่านค่า JSON ทั้งหมดจาก Environment Variable ที่เราตั้งบน Render
GCP_CREDS_STR = os.environ.get("GCP_CREDS_JSON")
SHEET_NAME = "HRMS-analyzed" # ชื่อ Google Sheet ที่คุณสร้าง

@st.cache_resource
def get_gspread_client():
    """สร้างและคืน client สำหรับเชื่อมต่อ Google Sheets"""
    # ตรวจสอบว่ามีตัวแปรนี้อยู่จริงหรือไม่
    if not GCP_CREDS_STR:
        # ไม่แสดง error บนหน้าเว็บจริง แต่จะ print ไว้ใน log ของ Render
        print("CRITICAL ERROR: ไม่พบค่า GCP_CREDS_JSON ใน Environment Variables!")
        return None
    try:
        # แปลงข้อความ (string) JSON ที่ยาวๆ ให้กลายเป็น dictionary ที่ Python ใช้ได้
        creds_dict = json.loads(GCP_CREDS_STR)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        print(f"GSpread Connection Error: {e}")
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

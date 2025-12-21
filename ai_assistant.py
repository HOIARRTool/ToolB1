# File: ai_assistant.py

import os
import json
import re
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Set

import pandas as pd
import numpy as np
import statsmodels.api as sm

try:
    import google.generativeai as genai
except ImportError:
    genai = None



# --- Minimal RAG / top-k retriever (no extra dependencies) --------------------
# เก็บฐานความรู้ในไฟล์ JSONL (หนึ่งบรรทัด = 1 เอกสารย่อย) แล้วดึงเฉพาะส่วนที่เกี่ยวข้องมาใส่ prompt ต่อเคส
# ช่วยลด token/ค่าใช้จ่าย และเลี่ยงการยัด "ฐานความรู้ทั้งก้อน" เข้า prompt ทุกครั้ง
#
# โครงสร้างไฟล์: knowledge_base.jsonl (วางไว้โฟลเดอร์เดียวกับไฟล์นี้)
# แต่ละบรรทัดเป็น JSON เช่น:
# {"id":"code_def:...", "type":"code_def|kb3p", "codes":["CPE101"], "text":"..."}
#
# หมายเหตุ: ตัว retriever นี้ใช้การเทียบความคล้ายแบบ character n-gram (เหมาะกับภาษาไทย/อังกฤษแบบไม่ต้องตัดคำ)
_KB_CACHE: Optional[List[Dict]] = None
_KB_NGRAM_CACHE: Optional[List[Set[str]]] = None

def _kb_default_path() -> str:
    # ใช้ไฟล์ที่อยู่ข้าง ๆ สคริปต์นี้เป็นค่าเริ่มต้น (override ได้ด้วย env KB_PATH)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.getenv("KB_PATH", os.path.join(base_dir, "knowledge_base.jsonl"))

def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _char_ngrams(s: str, n: int = 2) -> Set[str]:
    s = _norm_text(s)
    s = s.replace(" ", "")
    if len(s) <= n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def _load_kb_once(kb_path: Optional[str] = None) -> None:
    global _KB_CACHE, _KB_NGRAM_CACHE
    if _KB_CACHE is not None and _KB_NGRAM_CACHE is not None:
        return

    path = kb_path or _kb_default_path()
    docs: List[Dict] = []
    ngrams: List[Set[str]] = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    txt = obj.get("text", "")
                    docs.append(obj)
                    ngrams.append(_char_ngrams(txt))
                except Exception:
                    # ข้ามบรรทัดที่ไม่ใช่ JSON
                    continue
    except FileNotFoundError:
        # ถ้าไม่มีไฟล์ KB จะยังทำงานได้ (retrieved_knowledge จะว่าง)
        docs, ngrams = [], []

    _KB_CACHE, _KB_NGRAM_CACHE = docs, ngrams

def retrieve_relevant_knowledge(query: str, top_k: int = 12, kb_path: Optional[str] = None, max_chars: int = 5000) -> str:
    """
    ดึงความรู้ที่เกี่ยวข้องกับ query แบบ top-k แล้วคืนเป็นข้อความ (พร้อมป้าย type/codes)
    - top_k: จำนวนเอกสารย่อยที่จะดึง
    - max_chars: ตัดความยาวรวมของ context เพื่อไม่ให้ prompt ใหญ่เกินจำเป็น
    """
    _load_kb_once(kb_path)
    if not _KB_CACHE or not _KB_NGRAM_CACHE:
        return "(ไม่พบไฟล์ฐานความรู้ knowledge_base.jsonl หรือฐานความรู้ยังว่าง)"

    qset = _char_ngrams(query)
    if not qset:
        return "(query ว่าง จึงไม่สามารถดึงฐานความรู้ได้)"

    scored: List[Tuple[float, int]] = []
    for i, dset in enumerate(_KB_NGRAM_CACHE):
        if not dset:
            continue
        inter = len(qset & dset)
        if inter == 0:
            continue
        # Jaccard similarity
        score = inter / (len(qset | dset) + 1e-9)
        scored.append((score, i))

    if not scored:
        return "(ไม่พบส่วนที่ match ในฐานความรู้ — จะวิเคราะห์จากข้อความเคสเป็นหลัก)"

    scored.sort(reverse=True, key=lambda x: x[0])
    picked = [i for _, i in scored[:max(1, top_k)]]

    chunks: List[str] = []
    total = 0
    for i in picked:
        doc = _KB_CACHE[i]
        dtype = doc.get("type", "kb")
        codes = doc.get("codes") or []
        codes_txt = f" | codes: {', '.join(codes)}" if codes else ""
        txt = (doc.get("text") or "").strip()

        # ตัดต่อชิ้นเพื่อกันยาวเกิน
        if len(txt) > 900:
            txt = txt[:900].rstrip() + "…"

        piece = f"- [{dtype}]{codes_txt}\n  {txt}"
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)

    return "\n".join(chunks) if chunks else "(ดึงฐานความรู้ได้ แต่ถูกตัดด้วย max_chars)"

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:html|markdown|md|json)?\s*|\s*```", re.IGNORECASE)

def _strip_code_fences(text: str) -> str:
    if not text:
        return ""
    return _FENCE_RE.sub("", text).strip()

def _get_api_key() -> Optional[str]:
    # ปรับชื่อตัวแปรตามที่คุณใช้จริงบน Render ได้เลย
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GENERATIVEAI_API_KEY")
    )

def _configure_genai_if_possible() -> None:
    """
    ปลอดภัยไว้ก่อน: ถ้ามี API key ใน env ก็ configure ให้
    (ถ้าคุณ configure ไว้ที่อื่นแล้ว ฟังก์ชันนี้จะไม่ทำให้พัง)
    """
    if not genai:
        return
    api_key = _get_api_key()
    if api_key:
        try:
            genai.configure(api_key=api_key)
        except Exception:
            # เงียบไว้เพื่อไม่ให้ทำให้ระบบเดิมล่ม
            pass


# ------------------------------------------------------------------------------
# NEW: Appendix to make Step-2 planning more reliable
# ------------------------------------------------------------------------------
PLANNING_INPUTS_APPENDIX = r"""
**ข้อกำหนดเพิ่มเติม (สำคัญมาก):**
หลังจากตอบครบทุกหัวข้อเดิมแล้ว ให้เพิ่มหัวข้อท้ายสุดชื่อ:

### 8. ประมวลผลเพื่อจัดทำแผนพัฒนา 
ให้สรุปเป็นรายการสั้นๆ (bullet) โดยต้องมีอย่างน้อย:
- Immediate containment (ทำทันทีภายใน 24-72 ชม.)
- Root drivers / system gaps (ตัวขับหลัก 3-5 ข้อ)
- “จุดเปลี่ยนสำคัญ” ที่ถ้าปรับจะลดความเสี่ยงได้มาก (Turning points 2-3 ข้อ)
- Data needed (ข้อมูลที่ต้องเก็บเพิ่มเพื่อยืนยันสมมติฐาน)
- Stakeholders/Owners (ใครควรเป็นเจ้าภาพหลัก)
"""


# ------------------------------------------------------------------------------
# NEW: Step-2 prompt (Executive Summary + Potential Change + KPI Plan)
# ------------------------------------------------------------------------------
EXECUTIVE_PLAN_PROMPT_TEMPLATE = r"""
คุณคือ "ที่ปรึกษาคุณภาพและความปลอดภัยของโรงพยาบาล" (บริบทประเทศไทย)

ผู้ใช้ให้รายละเอียดอุบัติการณ์ดังนี้:
{incident_description}

และมีผลวิเคราะห์เชิงเทคนิค (หัวข้อ 1-6) จาก AI แล้วด้านล่างนี้:
{consultation_text}

งานของคุณ: เขียน "ส่วนเพิ่มเติม" ต่อท้ายผลเดิม โดยต้องมีแค่ 2 หัวข้อดังนี้ และเขียนเป็น Markdown เท่านั้น (ห้ามเขียนโค้ดบล็อก)

### 9. แผนพัฒนา (Creative Solution & KPI)

ให้สรุป "Potential Change" จำนวน 3–5 ข้อ ที่ “ต่อยอด” จากข้อ 4–6 (ไม่ทวนข้อความเดิมซ้ำยาว ๆ)  
จากนั้นให้จัดทำ **ตารางแผนพัฒนา** ตามรูปแบบด้านล่างอย่างเคร่งครัด

**ข้อกำหนดสำคัญ**
- 1 Potential Change = 1 แถวในตาราง
- Key Actions ให้ใส่ 2–4 ข้อ (เขียนแบบสั้น กระชับ)
- SMART KPI ให้ใส่ 1–2 ตัว/แถว และต้องระบุ “นิยาม/สูตร” ให้ชัดเจน
- Timeline ให้ระบุช่วงเวลา เช่น “ภายใน 30 วัน”, “ไตรมาส 1/2569”, “ภายใน 6 เดือน” เป็นต้น
- Owner ระบุทีม/หน่วยงาน/บทบาทที่เหมาะสม (เช่น ER, Ward, QI, PTC, Nursing, RRT)

**รูปแบบการตอบ: ต้องเป็นตาราง Markdown เท่านั้น**

| Potential Change | Goal (เป้าหมาย) | Key Actions (2–4 ข้อ) | Owner | Timeline | KPI (นิยาม/สูตร) |
|---|---|---|---|---|---|
| ... | ... | 1) ...<br><ul><li>2) ...</li><li>3) ...<ul><li> | ... | ... | - KPI1: ... (นิยาม/สูตร: ...)<ul><li>- KPI2: ... (นิยาม/สูตร: ...)<ul><li> |

### 10. บทสรุปผู้บริหาร (Executive Summary)
- 5-7 บรรทัด อ่านเร็ว เข้าใจภาพรวม
- ครอบคลุม: เหตุการณ์/ความเสี่ยงหลัก, ความรุนแรงที่คาด, ประเด็นสาเหตุร่วมสำคัญ, มาตรการเร่งด่วน 1-2 ข้อ

ข้อกำหนดเพิ่มเติม:
- หลีกเลี่ยงความซ้ำซ้อนกับหัวข้อ 1-7 (อ้างอิงกลับได้ แต่ไม่คัดลอกซ้ำ)

"""


# ==============================================================================
# AI FUNCTION 2: CASE CONSULTATION (Upgraded)
# ==============================================================================
def get_consultation_response(incident_description: str) -> str:
    """
    อัปเกรดให้ทำงาน 2 ขั้น:
    Step 1: วิเคราะห์เชิงเทคนิค + ให้รหัส/ความรุนแรง/ปัจจัยร่วม (อิงฐานความรู้เดิม)
    Step 2: สร้างบทสรุปผู้บริหาร + potential change + KPI + PDSA + action plan
    """
    if not genai:
        return "ขออภัยครับ ไลบรารี google.generativeai ไม่ได้ถูกติดตั้ง"

    _configure_genai_if_possible()
 

    # --- RAG: ดึงฐานความรู้เฉพาะส่วนที่เกี่ยวข้อง (top-k) ---
    retrieved_knowledge = retrieve_relevant_knowledge(incident_description, top_k=12)
    master_prompt = f"""
    **บทบาท:**
    คุณคือ "ผู้ช่วย AI ด้านการจัดการความเสี่ยง" (AI Risk Management Assistant) มีหน้าที่ช่วยสรุปข้อมูลและให้ข้อเสนอแนะเบื้องต้น สำหรับการรายงานอุบัติการณ์ไปยังระบบ NRLS & HRMS
    
    **แนวทางการตอบ:**
    1.  **เสนอเป็นทางเลือก:** ให้คำตอบของคุณอยู่ในรูปแบบของ "ข้อเสนอแนะ", "แนวทางที่เป็นไปได้" หรือ "ข้อมูลเพื่อประกอบการพิจารณา" เสมอ ไม่ใช่คำสั่งหรือคำตอบที่สิ้นสุด
    2.  **ย้ำเตือนบทบาทของผู้ใช้:** ก่อนเริ่มคำแนะนำ ให้มีประโยคที่ส่งเสริมให้ผู้ใช้เป็นผู้ตัดสินใจเสมอ เช่น "โปรดใช้ข้อมูลนี้ร่วมกับวิจารณญาณและประสบการณ์ของผู้เชี่ยวชาญในการตัดสินใจขั้นสุดท้าย"
    3. **หากข้อมูลไม่พอ ให้ระบุ "ข้อมูลที่ควรถามเพิ่ม" แบบสั้น ๆ ด้านบนเลย (ทำตัวอักษรเอียง และไม่ต้องถามกลับผู้ใช้)
    
    **ข้อห้าม:**
    - ห้ามให้คำตอบที่เด็ดขาด ฟันธง หรือรับประกันความถูกต้อง 100%
    - ห้ามใช้ตำแหน่งที่สูงกว่าผู้ใช้งาน เช่น "ผู้จัดการ" หรือ "ผู้เชี่ยวชาญ"
    
    **ภารกิจ:**
    จาก "รายละเอียดอุบัติการณ์" ที่ผู้ใช้ป้อนเข้ามา จงวิเคราะห์และให้คำปรึกษาที่ครบถ้วนตามรูปแบบที่กำหนด โดยอ้างอิงจาก "ฐานข้อมูลความรู้" ที่ให้มาเท่านั้น
    คุณจะต้องให้คำปรึกษาใน 5 หัวข้อหลัก ได้แก่ 1. สรุปเหตุการณ์, 2. การให้รหัสอุบัติการณ์, 3. การประเมินระดับความรุนแรง, 4. การวิเคราะห์ปัจจัยร่วม, และ 5. ข้อเสนอแนะเบื้องต้น

    **ฐานข้อมูลความรู้:**
    ---
    [รหัส NRLS & HRMS]
    (ดึงเฉพาะส่วนที่เกี่ยวข้อง (top-k) ต่อเคสจากไฟล์ knowledge_base.jsonl)
    {retrieved_knowledge}

    [ระดับความรุนแรง] ในกลุ่มความเสี่ยงด้านคลินิก=รหัส NRLS & HRMS ที่ขึ้นต้นด้วย C และ GP จะเป็นระดับ A-I
    A (เกิดที่นี่): เกิดเหตุการณ์ขึ้นแล้วจากตัวเองและค้นพบได้ด้วยตัวเองสามารถปรับแก้ไขได้ไม่ส่งผลกระทบถึงผู้อื่นและผู้ป่วยหรือบุคลากร
    B (เกิดที่ไกล): เกิดเหตุการณ์/ ความผิดพลาดขึ้นแล้วโดยส่งต่อเหตุการณ์/ ความผิดพลาดนั้นไปที่ผู้อื่นแต่สามารถตรวจพบและแก้ไขได้ โดยยังไม่มีผลกระทบใดๆ ถึงผู้ป่วยหรือบุคลากร
    C (เกิดกับใคร): เกิดเหตุการณ์/ ความผิดพลาดขึ้นและมีผลกระทบถึงผู้ป่วยหรือบุคลากร แต่ไม่เกิดอันตรายหรือเสียหาย
    D (ให้ระวัง): เกิดความผิดพลาดขึ้น มีผลกระทบถึงผู้ป่วยหรือบุคลากร ต้องให้การดูแลเฝ้าระวังเป็นพิเศษว่าจะไม่เป็นอันตราย
    E (ต้องรักษา): เกิดความผิดพลาดขึ้น มีผลกระทบถึงผู้ป่วยหรือบุคลากร เกิดอันตรายชั่วคราวที่ต้องแก้ไข/ รักษาเพิ่มมากขึ้น
    F (เยียวยานาน): เกิดความผิดพลาดขึ้น มีผลกระทบที่ต้องใช้เวลาแก้ไขนานกว่าปกติหรือเกินกำหนด ผู้ป่วยหรือบุคลากร ต้องรักษา/ นอนโรงพยาบาลนานขึ้น
    G (ต้องพิการ): เกิดความผิดพลาดถึงผู้ป่วยหรือบุคลากร ทำให้เกิดความพิการถาวร หรือมีผลกระทบทำให้เสียชื่อเสียง/ ความเชื่อถือและ/ หรือมีการร้องเรียน
    H (ต้องการปั๊ม): เกิดความผิดพลาด ถึงผู้ป่วยหรือบุคลากร มีผลทำให้ต้องทำการช่วยชีวิต หรือกรณีทำให้เสียชื่อเสียงและ/ หรือมีการเรียกร้องค่าเสียหายจากโรงพยาบาล
    I (จำใจลา): เกิดความผิดพลาด ถึงผู้ป่วยหรือบุคลากร เป็นสาเหตุทำให้เสียชีวิต เสียชื่อเสียงโดยมีการฟ้องร้องทางศาล/ สื่อ

    และ ในกลุ่มความเสี่ยงทั่วไป=รหัส NRLS & HRMS ที่ขึ้นต้นด้วย O จะเป็นระดับ 1-5

    1   เกิดความผิดพลาดขึ้นแต่ไม่มีผลกระทบต่อผลสำเร็จหรือวัตถุประสงค์ของการดำเนินงาน (* เกิดผลกระทบที่มีมูลค่าความเสียหาย 0 - 10,000 บาท)
    2   เกิดความผิดพลาดขึ้นแล้ว โดยมีผลกระทบ (ที่ควบคุมได้) ต่อผลสำเร็จหรือวัตถุประสงค์ของการดำเนินงาน (* เกิดผลกระทบที่มีมูลค่าความเสียหาย 10,001 - 50,000 บาท)
    3   เกิดความผิดพลาดขึ้นแล้ว และมีผลกระทบ (ที่ต้องทำการแก้ไข) ต่อผลสำเร็จหรือวัตถุประสงค์ของการดำเนินงาน (* เกิดผลกระทบที่มีมูลค่าความเสียหาย 50,001 - 250,000 บาท)
    4   เกิดความผิดพลาดขึ้นแล้ว และทำให้การดำเนินงานไม่บรรลุผลสำเร็จตามเป้าหมาย (* เกิดผลกระทบที่มีมูลค่าความเสียหาย 250,001 – 10,000,000 บาท)
    5   เกิดความผิดพลาดขึ้นแล้ว และมีผลให้การดำเนินงานไม่บรรลุผลสำเร็จตามเป้าหมาย ทำให้ภารกิจขององค์กรเสียหายอย่างร้ายแรง  (* เกิดผลกระทบที่มีมูลค่าความเสียหายมากกว่า 10 ล้านบาท) 
   

    [Contributing factor]
    F0001 Staff Factors: Fatigue
    F0002 Staff Factors: Stress
    F0003 Staff Factors: Inattention
    F0004 Staff Factors: Knowledge, Skills
    F0005 Staff Factors: Cognitive bias
    F0006 Staff Factors: Competence, Experience
    F0007 Staff Factors: Health issues
    F0008 Staff Factors: Attitude, Cultural competency
    F0009 Staff Factors: Barrier to speaking up for safety.
    F0010 Patient Factors: Clinical condition
    F0011 Patient Factors: Medication
    F0012 Patient Factors: Language, sociocultural
    F0013 Patient Factors: Informed & literacy
    F0014 Nature of Work: Work process Complexity
    F0015 Nature of Work: Competing tasks
    F0016 Nature of Work: Interruptions
    F0017 Nature of Work: Physical/Cognitive requirements
    F0018 Team: Role clarity/Lack of responsibility
    F0019 Team: Briefing, Awareness
    F0020 Communication: Supervisor to staff
    F0021 Communication: Among staff or team members
    F0022 Communication: Staff to patient (or family)
    F0023 Supervision/support: Clinical supervision
    F0024 Supervision/support: Managerial supervision
    F0025 Policies & procedures/Clinical protocols: Presence of policies
    F0026 Policies & procedures/Clinical protocols: Clarity of policies
    F0027 Policies & procedures/Clinical protocols: Lack of compliance to policies, GL or SOP
    F0028 Data & Information: Availability
    F0029 Data & Information: Accuracy
    F0030 Data & Information: Legibility
    F0031 Equipment/device: Function
    F0032 Equipment/device: Design
    F0033 Equipment/device: Availability
    F0034 Equipment/device: Maintenance
    F0035 Environment: Culture of safety, Management, Poor engineer control
    F0036 Environment: Physical surroundings (e.g., lighting, noise)    

    NOWLEDGE BASE (ฐานข้อมูลความรู้ 3P Safety):
    (ดึงเฉพาะส่วนที่เกี่ยวข้อง (top-k) ต่อเคสจากไฟล์ knowledge_base.jsonl)
    {retrieved_knowledge}

    **รายละเอียดอุบัติการณ์จากผู้ใช้:**
    '''
    {incident_description}
    '''

    **คำสั่ง:**
    จงวิเคราะห์รายละเอียดอุบัติการณ์ แล้วให้คำปรึกษาตามหัวข้อและรูปแบบ Markdown ต่อไปนี้อย่างเคร่งครัด:
    ### 1. สรุปเหตุการณ์และประเด็นความเสี่ยงสำคัญ
    (จงสรุปเหตุการณ์โดย **ดึง Keyword ที่สอดคล้องกับรหัสความเสี่ยง** ออกมาเน้นย้ำ เขียนในลักษณะการจับประเด็นว่า "เกิดอะไรที่ไม่พึงประสงค์ขึ้นบ้าง" 
    เพื่อเป็นการปูทางไปสู่การให้รหัสในหัวข้อถัดไป เช่น แทนที่จะเล่าเรื่องเฉยๆ ให้ระบุว่า "ผู้ป่วยเกิดภาวะ [ชื่อภาวะ] ซึ่งนำไปสู่การ [ชื่อหัตถการฉุกเฉิน/ความผิดพลาด]" เป็นต้น)

    ### 2. รหัสอุบัติการณ์ (NRLS & HRMS) ที่แนะนำ
    (จากประเด็นความเสี่ยงที่สรุปในข้อ 1 จงเลือกรหัสที่ตรงที่สุดจากฐานข้อมูล 
     - ระบุรหัสและชื่อรหัส 
     - อธิบายเหตุผลโดยอ้างอิงกลับไปที่เนื้อหาในข้อ 1 ว่าเหตุการณ์ส่วนไหนที่ตรงกับรหัสนั้น)

    ### 3. ระดับความรุนแรงที่แนะนำ
    (เลือกระดับความรุนแรง A-I ที่เหมาะสมที่สุดจากฐานข้อมูล พร้อมอธิบายเหตุผลประกอบ)

    ### 4. ปัจจัยร่วมที่อาจเป็นสาเหตุ (Contributing Factors) พร้อมหลักฐาน
    (ให้เลือกปัจจัยจากฐานข้อมูลมา 1-3 ข้อ โดยต้อง **ระบุความเชื่อมโยง** ดังนี้:
     1. **ระบุปัจจัย:** (รหัสและชื่อปัจจัย)
     2. **หลักฐานจากเหตุการณ์:** (คัดลอกข้อความหรือระบุจุดเหตุการณ์เฉพาะเจาะจงที่ผู้ใช้เล่ามา ซึ่งสอดคล้องกับปัจจัยนี้)
     3. **คำอธิบาย:** (อธิบายสั้นๆ ว่าปัจจัยนี้ส่งผลให้เกิดอุบัติการณ์นี้ได้อย่างไร)
     *ตัวอย่างรูปแบบ:*
     * **F0010 Patient Factors: Clinical condition**
       * *หลักฐาน:* ผู้ป่วยมีโรคประจำตัวซับซ้อน (ESRD on HD, ประวัติ STEMI)
       * *คำอธิบาย:* สภาวะโรคเดิมของผู้ป่วยเป็นปัจจัยสำคัญที่ทำให้เกิดความเสี่ยงต่อภาวะ Hyperkalemia และหัวใจหยุดเต้นได้ง่ายกว่าปกติ)
       
    ### 5. Potential Change
    (ให้บอก “จุดเปลี่ยนสำคัญ” ที่ถ้าปรับจะลดความเสี่ยงได้มาก (Turning points 2-3 ข้อ))
    
    ### 6. ข้อเสนอแนะเบื้องต้น
    (เสนอแนวทางการแก้ไขเฉพาะหน้า และ/หรือ การป้องกันในระยะยาวที่เหมาะสมกับเหตุการณ์)

    ### 7. เรียนรู้จาก 3P Safety และมาตรฐาน HA ที่เกี่ยวข้อง
    จากรหัสอุบัติการณ์ที่กำหนดในข้อ 2 ให้คุณดำเนินการตามขั้นตอนต่อไปนี้:    
    ค้นหารหัสอุบัติการณ์ในฐานข้อมูลความรู้ และตรวจสอบข้อมูลในส่วน "เป้าหมายที่เกี่ยวข้อง"    
    จัดกลุ่มผลลัพธ์ ตาม "safety_GOAL" ทั้ง 3 ประเภท ได้แก่ "Patient Safety", "Personnel Safety", และ "People Safety"   
    สร้างรายงาน โดยแสดงผลลัพธ์แยกตามกลุ่มที่พบข้อมูล โดยใช้หัวข้อให้ชัดเจน    
    สำหรับแต่ละกลุ่มที่พบ ให้แสดงรายละเอียดของเป้าหมายที่เกี่ยวข้องทั้งหมด (ระบุเหตุผล, สรุปกระบวนการ, และมาตรฐาน HA)    
    หากอุบัติการณ์หนึ่งเกี่ยวข้องกับหลาย GOAL (เช่น กระทบทั้งบุคลากรและผู้ป่วย) ให้แสดงผลทั้งหมดโดยแยกเป็นหัวข้อของแต่ละ GOAL    
    หากค้นหาแล้ว ไม่พบข้อมูล ใน "เป้าหมายที่เกี่ยวข้อง" เลย (array ว่าง) ให้แสดงข้อความว่า: "สำหรับอุบัติการณ์นี้ ไม่พบเป้าหมายความปลอดภัยที่เกี่ยวข้องโดยตรงในฐานข้อมูล 3P Safety ควรพิจารณาตามบริบทขององค์กรและมาตรฐานวิชาชีพที่เกี่ยวข้อง"
    """


    # ✅ เพิ่ม appendix แบบไม่ต้องแก้ก้อนฐานความรู้
    master_prompt = master_prompt + "\n\n" + PLANNING_INPUTS_APPENDIX

    try:
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        model = genai.GenerativeModel(model_name)

        # -----------------------
        # Step 1: Technical consult
        # -----------------------
        consult_resp = model.generate_content(master_prompt)
        consultation_text = _strip_code_fences(getattr(consult_resp, "text", "") or "")

        # -----------------------
        # Step 2: Executive plan
        # -----------------------
        exec_prompt = EXECUTIVE_PLAN_PROMPT_TEMPLATE.format(
            incident_description=incident_description.strip(),
            consultation_text=consultation_text.strip(),
        )
        exec_resp = model.generate_content(exec_prompt)
        extra_markdown = (exec_resp.text or "").strip()
        combined_markdown = consultation_text.strip()
        if extra_markdown:
            combined_markdown = combined_markdown + "\n\n" + extra_markdown
    
        return combined_markdown

    except Exception as e:
        return f"ขออภัยครับ เกิดข้อผิดพลาดในการเชื่อมต่อกับ AI: {e}"

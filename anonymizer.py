# anonymizer.py
import re
from pathlib import Path

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from huggingface_hub import snapshot_download

# ====== ค่าตัวแทนเมื่อปกปิดข้อมูล ======
ENTITY_TO_ANONYMIZED_TOKEN_MAP = {
    "HN": "[HN_NUMBER]",
    "PERSON": "[PERSON]",
    "LOCATION": "[LOCATION]",
    "ORGANIZATION": "[ORGANIZATION]",
}

# ====== Privacy regex rules ======
# ห้ามใช้ \b หน้า HN เพราะภาษาไทยเป็น Unicode word character
# เช่น "ผู้ป่วยHN 123456" จะไม่ match ถ้าใช้ \b
#
# รองรับ:
# HN123456
# HN 123456
# HN.123456
# HN:123456
# HN-123456
# H.N.123456
# ผู้ป่วยHN 123456
# HN No. 123456
# HN NUMBER 123456
# รวมถึง zero-width spaces บางชนิด

_HN_SEP = r"[\s\u00A0\u200B\u200C\u200D\._:#/\\\-–—]*"

HN_PATTERN = re.compile(
    rf"(?:H{_HN_SEP}N|เอช{_HN_SEP}เอ็น)"
    rf"{_HN_SEP}(?:(?:เลข(?:ที่)?|NO\.?|NUMBER){_HN_SEP})?"
    rf"\d{{3,}}",
    re.IGNORECASE,
)

PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z_]+\]")


def redact_hn(text):
    """
    Redact HN in all supported formats.
    Safe to call repeatedly.
    """
    if not isinstance(text, str) or not text:
        return text

    return HN_PATTERN.sub(
        ENTITY_TO_ANONYMIZED_TOKEN_MAP["HN"],
        text
    )


def has_residual_hn(text) -> bool:
    """
    ตรวจว่าข้อความยังมี HN-like identifier เหลืออยู่หรือไม่
    """
    if not isinstance(text, str) or not text:
        return False

    return HN_PATTERN.search(text) is not None


def redact_hn_dataframe(df):
    """
    Defense-in-depth:
    ปิด HN ในทุกคอลัมน์ประเภทข้อความ
    ไม่ใช่เฉพาะ 'รายละเอียดการเกิด'
    """
    safe = df.copy()

    for col in safe.select_dtypes(include=["object", "string"]).columns:
        safe[col] = safe[col].map(
            lambda value: redact_hn(value)
            if isinstance(value, str)
            else value
        )

    return safe


def residual_hn_locations(df, max_items: int = 25):
    """
    Scan ทุก text column เพื่อหา HN ที่อาจยังหลงเหลือ

    คืนเฉพาะชื่อคอลัมน์และ row index
    โดยไม่คืนข้อความจริงที่อาจเป็น PHI
    """
    hits = []

    for col in df.select_dtypes(include=["object", "string"]).columns:
        mask = df[col].map(has_residual_hn)

        for idx in df.index[mask]:
            hits.append({
                "column": str(col),
                "row": str(idx)
            })

            if len(hits) >= max_items:
                return hits

    return hits


@st.cache_resource
def load_ner_model():
    """
    ดาวน์โหลด (ครั้งแรก) และโหลดโมเดล NER จาก Hugging Face
    แสดงสถานะด้วยกล่องเดียว (st.status)
    """
    with st.status(
        "🚀 กำลังโหลดโมเดล NER...",
        expanded=True
    ) as status:

        try:
            local_dir = Path("model")

            if (
                not local_dir.exists()
                or not any(local_dir.iterdir())
            ):
                st.write(
                    "🔽 กำลังดาวน์โหลดโมเดลจาก Hugging Face..."
                )

                snapshot_download(
                    repo_id="pythainlp/thainer-corpus-v2-base-model",
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                )

            st.write(
                "⚙️ กำลังโหลดโมเดลเข้าหน่วยความจำ..."
            )

            tokenizer = AutoTokenizer.from_pretrained(
                str(local_dir)
            )

            model = AutoModelForTokenClassification.from_pretrained(
                str(local_dir)
            )

            ner_pipeline = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                device=-1,
                aggregation_strategy="simple",
            )

            status.update(
                label="✅ โหลด NER pipeline เรียบร้อยแล้ว",
                state="complete"
            )

            return ner_pipeline

        except Exception as e:
            status.update(
                label=f"❌ โหลดโมเดลล้มเหลว: {e}",
                state="error"
            )

            return None


def anonymize_text(text: str, ner_model):
    """
    ปกปิดข้อมูลในหนึ่งข้อความแบบ defense-in-depth

    ขั้นตอน:
    1) deterministic HN redaction
    2) NER สำหรับ PERSON / LOCATION / ORGANIZATION
    3) deterministic HN redaction ซ้ำอีกครั้งก่อนคืนค่า

    ข้อสำคัญ:
    ต่อให้ NER fail ก็ยังต้องปิด HN ได้
    """

    if not isinstance(text, str) or not text.strip():
        return text

    # ---------- Pass 1: deterministic HN redaction ----------
    anonymized = redact_hn(text)

    # ต่อให้ NER โหลดไม่ได้ ก็ต้องคืนข้อความที่ปิด HN แล้ว
    if not ner_model:
        return redact_hn(anonymized)

    try:
        # ป้องกันไม่ให้ NER ไปแก้ placeholder ที่เราใส่ไว้
        protected_spans = [
            (m.start(), m.end())
            for m in PLACEHOLDER_PATTERN.finditer(anonymized)
        ]

        def overlaps(a, b):
            return not (
                a[1] <= b[0]
                or b[1] <= a[0]
            )

        ner_results = ner_model(anonymized)

        # ทำจากท้ายข้อความมาหน้า
        # เพื่อไม่ให้ตำแหน่ง start/end เปลี่ยนระหว่าง replace
        for ent in sorted(
            ner_results,
            key=lambda x: x["start"],
            reverse=True
        ):
            start = ent["start"]
            end = ent["end"]

            if any(
                overlaps((start, end), ps)
                for ps in protected_spans
            ):
                continue

            group = ent.get("entity_group")

            if group in ENTITY_TO_ANONYMIZED_TOKEN_MAP:
                token = ENTITY_TO_ANONYMIZED_TOKEN_MAP[group]

                anonymized = (
                    anonymized[:start]
                    + token
                    + anonymized[end:]
                )

                protected_spans.append(
                    (start, start + len(token))
                )

        # ---------- Pass 2: deterministic HN redaction ----------
        # เก็บตกอีกรอบหลัง NER
        return redact_hn(anonymized)

    except Exception:
        # ถ้า NER error ต้องไม่ทำให้ HN หลุด
        return redact_hn(anonymized)


def anonymize_column(
    df,
    text_col: str,
    ner_model,
    out_col: str = "รายละเอียดการเกิด_Anonymized"
):
    """
    ปกปิดทั้งคอลัมน์ พร้อม progress bar
    """

    if text_col not in df.columns:
        df[out_col] = df.get(text_col, "")
        return df

    with st.status(
        "🔒 กำลังปกปิดข้อมูลส่วนบุคคล…",
        expanded=True
    ) as status:

        n = len(df)
        pbar = st.progress(0)

        texts = df[text_col].astype(str).tolist()

        out = []

        for i, txt in enumerate(
            texts,
            start=1
        ):
            out.append(
                anonymize_text(
                    txt,
                    ner_model
                )
            )

            pbar.progress(
                int(
                    i * 100
                    / max(n, 1)
                )
            )

        df[out_col] = out

        # ---------- Privacy verification ----------
        residual_mask = df[out_col].map(
            has_residual_hn
        )

        residual_count = int(
            residual_mask.sum()
        )

        if residual_count > 0:
            status.update(
                label=(
                    f"❌ Privacy check failed: "
                    f"พบ HN คงเหลือ {residual_count} รายการ"
                ),
                state="error"
            )

            raise ValueError(
                "Residual HN detected after anonymization"
            )

        status.update(
            label="✅ ปกปิดข้อมูลส่วนบุคคลเรียบร้อย",
            state="complete"
        )

        return df

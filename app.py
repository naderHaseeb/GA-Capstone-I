from __future__ import annotations

import csv
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from flyer_pipeline import MODEL_ID, process_flyer


st.set_page_config(
    page_title="Flyer AI Reader",
    page_icon="🧾",
    layout="wide",
)

st.title("Flyer AI Reader")
st.caption(
    "Upload a supermarket flyer, review the AI extraction, "
    "correct mistakes, then approve or reject it."
)

DB_PATH = Path("flyer_data.db")
CORRECTIONS_PATH = Path("corrections.csv")


def get_api_key() -> str | None:
    """
    Do not hard-code API keys in GitHub.

    Locally you can use:
    export OPENROUTER_API_KEY="..."

    On Streamlit Cloud add OPENROUTER_API_KEY in App Settings -> Secrets.
    """
    try:
        key = st.secrets.get("OPENROUTER_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass

    return os.environ.get("OPENROUTER_API_KEY")


def init_database() -> None:
    """Create the small SQLite tables required by the capstone."""
    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS approved_products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id TEXT NOT NULL,
            reviewed_at TEXT NOT NULL,
            shop_name TEXT,
            campaign_name TEXT,
            region TEXT,
            branch TEXT,
            flyer_start_date TEXT,
            flyer_end_date TEXT,
            page INTEGER,
            product_name TEXT,
            quantity TEXT,
            price_before REAL,
            price_after REAL,
            currency TEXT,
            product_start_date TEXT,
            product_end_date TEXT,
            date_source TEXT,
            date_badge_text TEXT,
            bbox TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            review_id TEXT PRIMARY KEY,
            reviewed_at TEXT NOT NULL,
            status TEXT NOT NULL,
            shop_name TEXT,
            campaign_name TEXT,
            product_count INTEGER
        )
        """
    )

    conn.commit()
    conn.close()


def draw_bbox(image_path: str, bbox) -> Image.Image:
    """
    Draw the selected product's normalized 0-1000 box.

    This is the dashboard use of the bbox work from the model tests.
    """
    image = Image.open(image_path).convert("RGB")

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return image

    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return image

    width, height = image.size
    pixel_box = (
        x1 / 1000 * width,
        y1 / 1000 * height,
        x2 / 1000 * width,
        y2 / 1000 * height,
    )

    draw = ImageDraw.Draw(image)
    line_width = max(3, int(min(width, height) * 0.006))
    draw.rectangle(pixel_box, outline="red", width=line_width)
    return image


def editable_columns() -> list[str]:
    return [
        "product_name",
        "quantity",
        "price_before",
        "price_after",
        "currency",
        "product_start_date",
        "product_end_date",
        "date_source",
        "date_badge_text",
    ]


def log_corrections(
    original_df: pd.DataFrame,
    edited_df: pd.DataFrame,
    review_id: str,
) -> int:
    """Append every changed field to corrections.csv."""
    changes = []
    now = datetime.now(timezone.utc).isoformat()

    for idx in edited_df.index:
        if idx not in original_df.index:
            continue

        for field in editable_columns():
            old = original_df.at[idx, field] if field in original_df.columns else None
            new = edited_df.at[idx, field] if field in edited_df.columns else None

            old_cmp = "" if pd.isna(old) else str(old)
            new_cmp = "" if pd.isna(new) else str(new)

            if old_cmp != new_cmp:
                changes.append(
                    {
                        "review_id": review_id,
                        "timestamp": now,
                        "page": edited_df.at[idx, "page"],
                        "product_name": edited_df.at[idx, "product_name"],
                        "field": field,
                        "ai_value": old_cmp,
                        "corrected_value": new_cmp,
                    }
                )

    if changes:
        write_header = not CORRECTIONS_PATH.exists()

        with CORRECTIONS_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=changes[0].keys())
            if write_header:
                writer.writeheader()
            writer.writerows(changes)

    return len(changes)


def save_review(
    result: dict,
    edited_df: pd.DataFrame,
    status: str,
) -> tuple[str, int]:
    """
    Approve = save edited product rows to SQLite.
    Reject = save only the review status.

    Corrections are logged for either outcome so the audit trail remains visible.
    """
    review_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    reviewed_at = datetime.now(timezone.utc).isoformat()

    original_df = st.session_state["original_df"]
    correction_count = log_corrections(original_df, edited_df, review_id)

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        INSERT INTO reviews (
            review_id, reviewed_at, status,
            shop_name, campaign_name, product_count
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            review_id,
            reviewed_at,
            status,
            result.get("shop_name"),
            result.get("campaign_name"),
            len(edited_df),
        ),
    )

    if status == "approved":
        for _, row in edited_df.iterrows():
            bbox = row.get("bbox")
            if isinstance(bbox, (list, tuple)):
                bbox = json.dumps(list(bbox))
            elif pd.isna(bbox):
                bbox = None
            else:
                bbox = str(bbox)

            conn.execute(
                """
                INSERT INTO approved_products (
                    review_id, reviewed_at,
                    shop_name, campaign_name, region, branch,
                    flyer_start_date, flyer_end_date, page,
                    product_name, quantity, price_before, price_after,
                    currency, product_start_date, product_end_date,
                    date_source, date_badge_text, bbox
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    reviewed_at,
                    row.get("shop_name"),
                    row.get("campaign_name"),
                    row.get("region"),
                    row.get("branch"),
                    row.get("flyer_start_date"),
                    row.get("flyer_end_date"),
                    int(row.get("page")),
                    row.get("product_name"),
                    row.get("quantity"),
                    row.get("price_before"),
                    row.get("price_after"),
                    row.get("currency"),
                    row.get("product_start_date"),
                    row.get("product_end_date"),
                    row.get("date_source"),
                    row.get("date_badge_text"),
                    bbox,
                ),
            )

    conn.commit()
    conn.close()
    return review_id, correction_count


init_database()

api_key = get_api_key()
# TEMP TEST: verify that Streamlit can authenticate with OpenRouter
if api_key:
    test_response = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={
            "Authorization": f"Bearer {api_key}"
        },
        timeout=30
    )

    st.write("OpenRouter auth test:", test_response.status_code)

    if not test_response.ok:
        st.error(test_response.text)
if not api_key:
    st.warning(
        "OpenRouter API key is not configured yet. "
        "Add OPENROUTER_API_KEY to Streamlit secrets or your environment."
    )

uploaded_pdf = st.file_uploader(
    "Upload flyer PDF",
    type=["pdf"],
)

if uploaded_pdf is not None:
    st.write(f"**File:** {uploaded_pdf.name}")

    if st.button(
        "Process flyer",
        type="primary",
        disabled=not bool(api_key),
    ):
        # Store the temporary directory in session state so rendered page
        # images remain available while the reviewer uses the dashboard.
        session_dir = tempfile.mkdtemp(prefix="flyer_ai_")
        pdf_path = Path(session_dir) / uploaded_pdf.name
        pdf_path.write_bytes(uploaded_pdf.getvalue())

        progress = st.progress(0, text="Starting extraction...")

        try:
            progress.progress(10, text="Rendering PDF and reading Page 1...")
            with st.spinner(
                "Qwen3-VL-32B is reading the flyer. Multi-page PDFs can take a little while."
            ):
                result = process_flyer(
                    pdf_path=pdf_path,
                    work_dir=session_dir,
                    api_key=api_key,
                    model_id=MODEL_ID,
                )

            progress.progress(100, text="Extraction complete")

            st.session_state["flyer_result"] = result
            st.session_state["session_dir"] = session_dir

            df = pd.DataFrame(result["products"])
            st.session_state["original_df"] = df.copy(deep=True)
            st.session_state["edited_df"] = df.copy(deep=True)

        except Exception as exc:
            progress.empty()
            st.exception(exc)

if "flyer_result" in st.session_state:
    result = st.session_state["flyer_result"]
    original_df = st.session_state["original_df"]

    st.divider()

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Shop", result.get("shop_name") or "—")
    top2.metric("Products", len(original_df))
    top3.metric(
        "Flyer dates",
        f"{result.get('flyer_start_date') or '—'} → "
        f"{result.get('flyer_end_date') or '—'}",
    )

    total_cost = sum(float(x.get("cost", 0) or 0) for x in result.get("_usage", []))
    top4.metric("API cost", f"${total_cost:.4f}")

    pages = sorted(original_df["page"].dropna().astype(int).unique().tolist())
    if not pages:
        st.warning("No products were extracted.")
        st.stop()

    page = st.selectbox("Page to review", pages)

    page_original = original_df[original_df["page"].astype(int) == page].copy()

    left, right = st.columns([1.05, 1.25], gap="large")

    with right:
        st.subheader(f"Extracted products — Page {page}")
        st.caption("Edit any incorrect values directly in the table.")

        display_cols = [
            "product_name",
            "quantity",
            "price_before",
            "price_after",
            "currency",
            "product_start_date",
            "product_end_date",
            "date_source",
            "date_badge_text",
            "bbox",
        ]

        page_editor = st.data_editor(
            page_original[display_cols],
            use_container_width=True,
            hide_index=False,
            disabled=["bbox"],
            key=f"editor_page_{page}",
        )

        # Apply edits from this page back to the full edited dataframe.
        edited_full = st.session_state["edited_df"].copy()
        for idx in page_editor.index:
            for col in page_editor.columns:
                edited_full.at[idx, col] = page_editor.at[idx, col]
        st.session_state["edited_df"] = edited_full

        product_options = page_editor.index.tolist()
        selected_idx = st.selectbox(
            "Select a product to highlight",
            product_options,
            format_func=lambda idx: str(
                page_editor.at[idx, "product_name"]
                or f"Product {idx + 1}"
            ),
        )

    with left:
        st.subheader(f"Flyer — Page {page}")
        image_path = result["_page_images"][page - 1]

        selected_bbox = original_df.at[selected_idx, "bbox"]
        highlighted = draw_bbox(image_path, selected_bbox)
        st.image(highlighted, use_container_width=True)
        st.caption(
            "The red rectangle is the Qwen bounding box for the selected product."
        )

    st.divider()

    edited_df = st.session_state["edited_df"]

    approve_col, reject_col, spacer = st.columns([1, 1, 4])

    with approve_col:
        if st.button("Approve flyer", type="primary", use_container_width=True):
            review_id, corrections = save_review(
                result,
                edited_df,
                status="approved",
            )
            st.success(
                f"Approved and saved to SQLite. "
                f"{corrections} correction(s) logged. Review ID: {review_id}"
            )

    with reject_col:
        if st.button("Reject flyer", use_container_width=True):
            review_id, corrections = save_review(
                result,
                edited_df,
                status="rejected",
            )
            st.warning(
                f"Rejected. {corrections} correction(s) logged. "
                f"Review ID: {review_id}"
            )

    with st.expander("Run details"):
        st.write("Model:", result.get("_model"))
        st.dataframe(
            pd.DataFrame(result.get("_usage", [])),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download reviewed CSV",
            data=edited_df.to_csv(index=False).encode("utf-8"),
            file_name="reviewed_flyer_products.csv",
            mime="text/csv",
        )

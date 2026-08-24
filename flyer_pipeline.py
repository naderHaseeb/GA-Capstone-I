"""
Flyer extraction pipeline for the GA capstone.

Based on the final Qwen3-VL-32B notebook:
PDF -> page images -> Page 1 flyer context -> page extraction ->
date fallback / cleanup -> structured rows.

The model is called through OpenRouter; no ground truth is used here.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import fitz
import pandas as pd
import requests

MODEL_ID = "qwen/qwen3-vl-32b-instruct"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


FLYER_CONTEXT_PROMPT = """
You are reading PAGE 1 of a supermarket promotional flyer.

Extract ONLY flyer-wide campaign information.

Return VALID JSON ONLY.
No markdown, comments, code fences, or explanation.

Return exactly:
{
  "shop_name": null,
  "campaign_name": null,
  "flyer_start_date": null,
  "flyer_end_date": null,
  "region": null,
  "branch": null,
  "currency": null
}

RULES:
- Dates must use YYYY-MM-DD.
- Read the overall campaign validity dates, not product-specific badge dates.
- If the year is visible, use it.
- Use a three-letter currency code when identifiable.
- If a field cannot be determined confidently, return null.
- Do not invent information.
""".strip()


def image_to_data_url(path: str | Path) -> str:
    path = Path(path)
    ext = path.suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    with path.open("rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{data}"


def clean_json_text(text: str) -> str:
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start : end + 1]

    return text


def _post_openrouter(
    prompt: str,
    image_path: str | Path,
    api_key: str,
    model_id: str = MODEL_ID,
    timeout: int = 300,
) -> tuple[dict[str, Any], float]:
    if not api_key:
        raise ValueError(
            "OpenRouter API key is missing. Set OPENROUTER_API_KEY "
            "or add it to Streamlit secrets."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_id,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_path)},
                    },
                ],
            }
        ],
    }

    started = time.time()
    response = requests.post(
    OPENROUTER_URL,
    headers=headers,
    json=payload,
    timeout=timeout,
)

latency = time.time() - started

# NEW: show the actual OpenRouter error instead of only HTTPError
if not response.ok:
    raise RuntimeError(
        f"OpenRouter error {response.status_code}: {response.text}"
    )

return response.json(), latency

def render_pdf(
    pdf_path: str | Path,
    page_dir: str | Path,
    scale: float = 1.7,
) -> list[str]:
    """
    Same rendering approach as the notebook.

    scale=1.7 is intentionally kept from the final notebook so the app
    behaves like the version already tested.
    """
    page_dir = Path(page_dir)
    page_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    page_images: list[str] = []

    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                alpha=False,
            )
            path = page_dir / f"page-{i + 1}.jpg"
            pix.save(str(path))
            page_images.append(str(path))
    finally:
        doc.close()

    return page_images


def extract_flyer_context(
    image_path: str | Path,
    api_key: str,
    model_id: str = MODEL_ID,
) -> dict[str, Any]:
    response, _ = _post_openrouter(
        FLYER_CONTEXT_PROMPT,
        image_path,
        api_key,
        model_id=model_id,
    )
    raw = response["choices"][0]["message"]["content"]
    return json.loads(clean_json_text(raw))


def _flyer_year(start_date: Any, end_date: Any) -> int | None:
    for value in (start_date, end_date):
        if value:
            match = re.search(r"\b(20\d{2})\b", str(value))
            if match:
                return int(match.group(1))
    return None


def build_extraction_prompt(
    flyer_year: int | None,
    flyer_start_date: str | None,
    flyer_end_date: str | None,
    flyer_context: dict[str, Any],
) -> str:
    """
    Kept very close to the tested notebook prompt.

    Small additions:
    - passes Page 1 context to every page;
    - preserves raw date badge text;
    - explicitly rejects campaign mechanics as products.
    """
    return f"""
You are an expert supermarket-flyer information extraction model.

You are given ONE page from a multi-page supermarket flyer.

KNOWN FLYER CONTEXT FROM PAGE 1:
shop_name = {flyer_context.get("shop_name")}
campaign_name = {flyer_context.get("campaign_name")}
region = {flyer_context.get("region")}
branch = {flyer_context.get("branch")}
currency = {flyer_context.get("currency")}

The flyer year, when established from Page 1, is {flyer_year}.
Use this year whenever the flyer shows a month/day without explicitly printing the year.
If the year is unknown, do not invent a year.

Return VALID JSON ONLY.
No markdown.
No comments.
No explanation.
No code fences.

Extract these flyer-level fields when visible:
- shop_name
- campaign_name
- flyer_start_date
- flyer_end_date

Date format:
YYYY-MM-DD

If the flyer-wide validity dates are not visible on this page, return null for them.

Extract EVERY distinct visible product offer on the page.

For each product return:
- product_name
- quantity
- price_before
- price_after
- currency
- product_start_date
- product_end_date
- date_source
- date_badge_text
- bbox

PRODUCT RULES:
- One visible offer = one product object.
- Do not merge neighboring products.
- Do not create duplicates.
- Do not treat logos, QR codes, headers, footers, campaign graphics, store branding,
  department lists, fine print, voucher mechanics, or "spend X / get Y" campaign
  instructions as products.
- Read product names as accurately as possible.
- Keep brand + product description when visible.

QUANTITY RULES:
- Extract visible weight, volume, count, or pack size.
- Examples: "5kg", "4x1L", "500g", "30pcs".
- If no quantity is visible, return null.
- Do not guess.

PRICE RULES:
- price_after = current/promotional selling price.
- price_before = old/original price only if explicitly shown.
- Otherwise price_before = null.
- Return numeric values only.
- Do not confuse pack counts, percentages, badge numbers, voucher values, or dates with prices.

CURRENCY RULES:
- Return the three-letter currency code when identifiable.
- For Bahraini Dinar use "BHD".
- For Saudi Riyal use "SAR".
- Do not return currency symbols if the code is known.

DATE EXTRACTION RULES

This page belongs to a multi-page promotional flyer.

FLYER_START_DATE = {flyer_start_date}
FLYER_END_DATE = {flyer_end_date}

For EVERY product, determine its applicable start date, end date, and date source.

1. PRODUCT-SPECIFIC DATE OVERRIDES FLYER DATE
If product-specific date information exists, it overrides the flyer-wide validity period.
Set date_source = "product_badge".

2. RESOLVE RELATIVE DATE INFORMATION
Product dates may use exact dates, ranges, weekdays, abbreviated weekdays, day numbers,
"ONLY", "1 DAY OFFER", "2 DAY OFFER", "3 DAYS", or similar wording.
Interpret them using the known flyer validity period and visible flyer context.
Convert resolved dates to YYYY-MM-DD.
Do not invent dates that cannot be inferred.

3. INHERIT FLYER-WIDE DATES
If NO product-specific date information is associated with the product:
product_start_date = FLYER_START_DATE
product_end_date = FLYER_END_DATE
date_source = "flyer_default"

4. VISUAL ASSOCIATION
Only use a product-specific badge when the visual layout shows it belongs to that offer.
Do not assign a neighboring product's badge.

5. RAW BADGE TEXT
If a product has its own date wording, copy that wording into date_badge_text,
for example "SUN - MON - TUE" or "17, 18 & 19 AUG".
If no product-specific date wording exists, return null.

6. FINAL VALIDATION
- start date cannot be after end date
- do not invent a product-specific restriction
- if no product restriction exists, inherit the flyer-wide dates

BOUNDING BOX RULES:
bbox must cover the COMPLETE INDIVIDUAL PRODUCT OFFER, not only the product image.

Include all visible elements belonging to that product when present:
- full product image
- product name / description
- quantity / pack size
- old price
- promotional price
- product-specific date badge
- product-specific promotional badge

Do not include neighboring product content.

bbox format:
[x1, y1, x2, y2]

Coordinates are normalized from 0 to 1000.
Top-left = [0, 0]
Bottom-right = [1000, 1000]

Return exactly:
{{
  "shop_name": null,
  "campaign_name": null,
  "flyer_start_date": null,
  "flyer_end_date": null,
  "products": [
    {{
      "product_name": "",
      "quantity": null,
      "price_before": null,
      "price_after": null,
      "currency": null,
      "product_start_date": null,
      "product_end_date": null,
      "date_source": "flyer_default",
      "date_badge_text": null,
      "bbox": [0, 0, 0, 0]
    }}
  ]
}}
""".strip()


def call_qwen(
    image_path: str | Path,
    prompt: str,
    api_key: str,
    model_id: str = MODEL_ID,
) -> tuple[dict[str, Any], float]:
    return _post_openrouter(
        prompt,
        image_path,
        api_key,
        model_id=model_id,
    )


def parse_date(value: Any, fallback_year: int | None = None) -> str | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()
    dt = pd.to_datetime(text, errors="coerce")

    if pd.isna(dt):
        return None

    if fallback_year is not None:
        year_in_text = bool(re.search(r"\b20\d{2}\b", text))
        if not year_in_text:
            dt = dt.replace(year=fallback_year)

    return dt.strftime("%Y-%m-%d")


def valid_bbox(box: Any) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False

    try:
        x1, y1, x2, y2 = map(float, box)
    except Exception:
        return False

    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def normalize_currency(
    currency: Any,
    flyer_context: dict[str, Any],
) -> str | None:
    """
    Small deterministic cleanup added after the Ansar live-pipeline test.

    The model sometimes returned SAR/BD on a Bahrain flyer even though
    Page 1 context identified Bahrain. We only normalize when context
    makes the intended currency clear.
    """
    context_currency = str(flyer_context.get("currency") or "").upper().strip()
    region = str(flyer_context.get("region") or "").lower()

    if context_currency in {"BHD", "BD"}:
        return "BHD"
    if context_currency == "SAR":
        return "SAR"

    if "bahrain" in region:
        return "BHD"
    if "saudi" in region or "ksa" in region:
        return "SAR"

    if currency is None:
        return None

    code = str(currency).upper().strip()
    if code == "BD":
        return "BHD"
    return code or None


def process_flyer(
    pdf_path: str | Path,
    work_dir: str | Path,
    api_key: str | None = None,
    model_id: str = MODEL_ID,
    render_scale: float = 1.7,
) -> dict[str, Any]:
    """
    Run the complete extraction pipeline for one PDF.

    Returns the final structured result plus page image paths and usage
    information required by the review dashboard.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured.")

    work_dir = Path(work_dir)
    page_dir = work_dir / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)

    page_images = render_pdf(pdf_path, page_dir, scale=render_scale)
    if not page_images:
        raise ValueError("The uploaded PDF contains no pages.")

    flyer_context = extract_flyer_context(
        page_images[0],
        api_key=api_key,
        model_id=model_id,
    )

    flyer_start_date = flyer_context.get("flyer_start_date")
    flyer_end_date = flyer_context.get("flyer_end_date")
    flyer_year = _flyer_year(flyer_start_date, flyer_end_date)

    extraction_prompt = build_extraction_prompt(
        flyer_year,
        flyer_start_date,
        flyer_end_date,
        flyer_context,
    )

    page_predictions: list[dict[str, Any]] = []
    page_usage: list[dict[str, Any]] = []

    for page_num, image_path in enumerate(page_images, start=1):
        response, latency = call_qwen(
            image_path,
            extraction_prompt,
            api_key=api_key,
            model_id=model_id,
        )

        raw = response["choices"][0]["message"]["content"]
        prediction = json.loads(clean_json_text(raw))
        prediction["_page"] = page_num
        page_predictions.append(prediction)

        usage = response.get("usage", {}) or {}
        page_usage.append(
            {
                "page": page_num,
                "products": len(prediction.get("products", [])),
                "tokens": usage.get("total_tokens", 0) or 0,
                "cost": usage.get("cost", 0) or 0,
                "latency": latency,
            }
        )

    # Same fallback principle as the notebook, but without ground truth.
    global_start = parse_date(flyer_start_date, flyer_year)
    global_end = parse_date(flyer_end_date, flyer_year)

    for page in page_predictions:
        if global_start is None and page.get("flyer_start_date"):
            global_start = parse_date(page.get("flyer_start_date"), flyer_year)

        if global_end is None and page.get("flyer_end_date"):
            global_end = parse_date(page.get("flyer_end_date"), flyer_year)

    product_rows: list[dict[str, Any]] = []

    for page in page_predictions:
        page_num = int(page["_page"])

        for product in page.get("products", []):
            start = parse_date(product.get("product_start_date"), flyer_year)
            end = parse_date(product.get("product_end_date"), flyer_year)

            date_source = product.get("date_source")
            if date_source in {"flyer_default", "flyer_header", None, ""}:
                start = start or global_start
                end = end or global_end
                date_source = "flyer_default"

            bbox = product.get("bbox")
            if not valid_bbox(bbox):
                bbox = None

            row = {
                "shop_name": flyer_context.get("shop_name") or page.get("shop_name"),
                "campaign_name": flyer_context.get("campaign_name")
                or page.get("campaign_name"),
                "region": flyer_context.get("region"),
                "branch": flyer_context.get("branch"),
                "flyer_start_date": global_start,
                "flyer_end_date": global_end,
                "page": page_num,
                **product,
            }

            row["product_start_date"] = start
            row["product_end_date"] = end
            row["date_source"] = date_source
            row["currency"] = normalize_currency(
                row.get("currency"),
                flyer_context,
            )
            row["bbox"] = bbox

            product_rows.append(row)

    return {
        "shop_name": flyer_context.get("shop_name"),
        "campaign_name": flyer_context.get("campaign_name"),
        "flyer_start_date": global_start,
        "flyer_end_date": global_end,
        "region": flyer_context.get("region"),
        "branch": flyer_context.get("branch"),
        "currency": normalize_currency(
            flyer_context.get("currency"),
            flyer_context,
        ),
        "products": product_rows,
        "_page_images": page_images,
        "_usage": page_usage,
        "_model": model_id,
    }

"""
Flyer AI extraction pipeline.

This version deliberately mirrors the WORKING Colab/OpenRouter request style:
- requests.post(...)
- Authorization: Bearer <key>
- Content-Type: application/json
- qwen/qwen3-vl-32b-instruct
- temperature = 0
- image passed as a base64 data URL

No ground truth is used in the live pipeline.
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


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "qwen/qwen3-vl-32b-instruct"


def image_to_data_url(image_path: str | Path) -> str:
    image_path = str(image_path)
    extension = image_path.lower().split(".")[-1]

    if extension == "jpg":
        extension = "jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:image/{extension};base64,{encoded}"


def clean_json_text(text: str) -> str:
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    # Keep this small fallback because some models occasionally add
    # a sentence before/after the JSON despite the prompt.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]

    return text


def render_pdf(
    pdf_path: str | Path,
    page_dir: str | Path,
    scale: float = 1.3,
) -> list[str]:
    """
    Render PDF pages to JPG.

    scale=1.3 is used to keep requests comfortably below OpenRouter's
    request-size limit, matching the fix used during model testing.
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


def call_openrouter(
    image_path: str | Path,
    prompt: str,
    api_key: str,
    model_id: str = MODEL_ID,
    timeout: int = 240,
) -> tuple[dict[str, Any], float]:
    """
    IMPORTANT:
    This intentionally uses the same request structure as the working Colab notebook.
    """
    image_data_url = image_to_data_url(image_path)

    url = OPENROUTER_URL

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_id,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ]
    }

    start = time.time()

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout
    )

    latency = time.time() - start

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter HTTP {response.status_code}: {response.text}"
        )

    return response.json(), latency


FLYER_CONTEXT_PROMPT = """
You are reading PAGE 1 of a supermarket promotional flyer.

Extract flyer-wide campaign information only.

Return VALID JSON ONLY.
No markdown.
No comments.
No explanation.

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

Rules:
- Dates must use YYYY-MM-DD.
- Read the overall campaign validity dates, not a product-specific badge.
- If the year is visible, use it.
- For Bahraini Dinar return BHD.
- If a field cannot be determined, return null.
- Do not invent information.
""".strip()


def extract_flyer_context(
    image_path: str | Path,
    api_key: str,
    model_id: str = MODEL_ID,
) -> dict[str, Any]:
    response, _ = call_openrouter(
        image_path=image_path,
        prompt=FLYER_CONTEXT_PROMPT,
        api_key=api_key,
        model_id=model_id,
    )

    raw_answer = response["choices"][0]["message"]["content"]
    return json.loads(clean_json_text(raw_answer))


def find_year(start_date: Any, end_date: Any) -> int | None:
    for value in (start_date, end_date):
        if value:
            match = re.search(r"\b(20\d{2})\b", str(value))
            if match:
                return int(match.group(1))
    return None


def parse_date(value: Any, fallback_year: int | None = None) -> str | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()
    parsed = pd.to_datetime(text, errors="coerce")

    if pd.isna(parsed):
        return None

    if fallback_year is not None:
        if not re.search(r"\b20\d{2}\b", text):
            parsed = parsed.replace(year=fallback_year)

    return parsed.strftime("%Y-%m-%d")


def valid_bbox(box: Any) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False

    try:
        x1, y1, x2, y2 = [float(x) for x in box]
    except Exception:
        return False

    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def normalize_currency(currency: Any, context: dict[str, Any]) -> str | None:
    region = str(context.get("region") or "").lower()
    context_currency = str(context.get("currency") or "").upper().strip()

    if "bahrain" in region or context_currency in {"BHD", "BD"}:
        return "BHD"

    if "saudi" in region or context_currency == "SAR":
        return "SAR"

    if currency is None:
        return None

    code = str(currency).upper().strip()
    if code == "BD":
        return "BHD"

    return code or None


def build_product_prompt(
    flyer_context: dict[str, Any],
    flyer_year: int | None,
) -> str:
    return f"""
You are an expert supermarket-flyer information extraction model.

You are given ONE page from a multi-page supermarket flyer.

KNOWN FLYER CONTEXT FROM PAGE 1:
shop_name = {flyer_context.get("shop_name")}
campaign_name = {flyer_context.get("campaign_name")}
flyer_start_date = {flyer_context.get("flyer_start_date")}
flyer_end_date = {flyer_context.get("flyer_end_date")}
region = {flyer_context.get("region")}
branch = {flyer_context.get("branch")}
currency = {flyer_context.get("currency")}
flyer_year = {flyer_year}

Return VALID JSON ONLY.
No markdown.
No comments.
No explanation.
No code fences.

Extract EVERY distinct visible supermarket product offer on this page.

Do NOT treat these as products:
- store logos
- QR codes
- campaign titles
- department lists
- fine print
- voucher mechanics
- spend X / get Y campaign instructions
- general promotional banners

For each real product return:
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

PRICE RULES:
- price_after = promotional selling price
- price_before = old/original price only when explicitly shown
- return numeric values only

DATE RULES:
- A product-specific date badge ALWAYS overrides the flyer-wide dates.
- If product-specific date wording exists, set date_source = "product_badge".
- Resolve dates such as:
  - 15TH AUG
  - 15 & 16 AUG
  - 17, 18 & 19 AUG
  - SUN ONLY
  - SUN - MON - TUE
  - 1 DAY OFFER / 2 DAYS OFFER / 3 DAYS OFFER
  using the Page 1 flyer context whenever possible.
- Copy the visible date wording into date_badge_text.
- If there is NO product-specific date restriction:
  product_start_date = {flyer_context.get("flyer_start_date")}
  product_end_date = {flyer_context.get("flyer_end_date")}
  date_source = "flyer_default"
  date_badge_text = null
- Dates must use YYYY-MM-DD.
- Do not invent dates.

CURRENCY:
- Return BHD for Bahraini Dinar.
- Return SAR for Saudi Riyal.

BOUNDING BOX:
- bbox = [x1, y1, x2, y2]
- normalized 0 to 1000
- cover the COMPLETE individual offer: image, name, quantity, prices,
  and attached product-specific date badge
- exclude neighboring products

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


def process_flyer(
    pdf_path: str | Path,
    work_dir: str | Path,
    api_key: str,
    model_id: str = MODEL_ID,
) -> dict[str, Any]:
    """
    Full live pipeline:
    PDF -> Page 1 context -> every page -> product rows.
    """
    if not api_key:
        raise ValueError("OpenRouter API key is missing.")

    page_dir = Path(work_dir) / "pages"
    page_images = render_pdf(pdf_path, page_dir)

    if not page_images:
        raise ValueError("The PDF has no pages.")

    flyer_context = extract_flyer_context(
        page_images[0],
        api_key=api_key,
        model_id=model_id,
    )

    flyer_year = find_year(
        flyer_context.get("flyer_start_date"),
        flyer_context.get("flyer_end_date"),
    )

    product_prompt = build_product_prompt(
        flyer_context=flyer_context,
        flyer_year=flyer_year,
    )

    page_predictions = []
    usage_rows = []

    for page_num, image_path in enumerate(page_images, start=1):
        response, latency = call_openrouter(
            image_path=image_path,
            prompt=product_prompt,
            api_key=api_key,
            model_id=model_id,
        )

        raw_answer = response["choices"][0]["message"]["content"]
        prediction = json.loads(clean_json_text(raw_answer))
        prediction["_page"] = page_num
        page_predictions.append(prediction)

        usage = response.get("usage", {}) or {}
        usage_rows.append(
            {
                "page": page_num,
                "products": len(prediction.get("products", [])),
                "tokens": usage.get("total_tokens", 0) or 0,
                "cost": usage.get("cost", 0) or 0,
                "latency": latency,
            }
        )

    global_start = parse_date(
        flyer_context.get("flyer_start_date"),
        flyer_year,
    )
    global_end = parse_date(
        flyer_context.get("flyer_end_date"),
        flyer_year,
    )

    # Safety fallback only: if Page 1 missed a global date, use a later page.
    for page in page_predictions:
        if global_start is None:
            global_start = parse_date(
                page.get("flyer_start_date"),
                flyer_year,
            ) or global_start

        if global_end is None:
            global_end = parse_date(
                page.get("flyer_end_date"),
                flyer_year,
            ) or global_end

    rows = []

    for page in page_predictions:
        page_num = int(page["_page"])

        for product in page.get("products", []):
            start = parse_date(
                product.get("product_start_date"),
                flyer_year,
            )
            end = parse_date(
                product.get("product_end_date"),
                flyer_year,
            )

            date_source = product.get("date_source")

            if date_source in {None, "", "flyer_header", "flyer_default"}:
                start = start or global_start
                end = end or global_end
                date_source = "flyer_default"

            bbox = product.get("bbox")
            if not valid_bbox(bbox):
                bbox = None

            row = {
                "shop_name": flyer_context.get("shop_name")
                or page.get("shop_name"),
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

            rows.append(row)

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
        "products": rows,
        "_page_images": page_images,
        "_usage": usage_rows,
        "_model": model_id,
    }

# Flyer AI Reader

GA capstone project for extracting structured supermarket flyer data with a vision-language model and reviewing the result through a human-in-the-loop dashboard.

## What it does

1. Upload a multi-page flyer PDF.
2. Render the PDF into page images.
3. Read Page 1 to establish flyer-wide context such as shop and campaign dates.
4. Use **Qwen3-VL-32B-Instruct through OpenRouter** to extract products from each page.
5. Extract product name, quantity, old/new price, currency, product dates, date source, raw date-badge text, and normalized 0–1000 bounding boxes.
6. Apply the business rule: **product-specific dates override flyer-wide dates**.
7. Review the flyer and extracted rows side-by-side in Streamlit.
8. Select a product to highlight its bounding box.
9. Edit incorrect fields.
10. Approve or reject the extraction.
11. Save approved rows to SQLite and log corrections to `corrections.csv`.

## Project structure

```text
flyer-ai-capstone/
├── app.py
├── flyer_pipeline.py
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example
├── notebooks/
│   └── put your benchmark/final notebooks here
└── README.md
```

## Run locally

Create and activate a virtual environment if you want, then:

```bash
pip install -r requirements.txt
```

Set the OpenRouter API key:

```bash
export OPENROUTER_API_KEY="your-key-here"
```

On Windows PowerShell:

```powershell
$env:OPENROUTER_API_KEY="your-key-here"
```

Start Streamlit:

```bash
streamlit run app.py
```

## Streamlit Community Cloud

Push the repository to GitHub.

In the Streamlit app settings, add this secret:

```toml
OPENROUTER_API_KEY = "your-key-here"
```

Do **not** commit your real API key to GitHub.

## Files created by the app

- `flyer_data.db` — SQLite database containing approved product rows and review statuses.
- `corrections.csv` — audit trail of fields changed by the reviewer.

These files are intentionally ignored by Git in the starter `.gitignore`.

## Model

Default model:

```text
qwen/qwen3-vl-32b-instruct
```

The model ID can be changed in `flyer_pipeline.py` if needed.

## Notes

The bounding box format produced by the model is `[x1, y1, x2, y2]` normalized from 0 to 1000. The dashboard converts those values to display pixels when highlighting a selected product.

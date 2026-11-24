# Local Telugu OCR with Bodhan Indic OCR

This project runs Bodhan Indic OCR locally. No OCR API is used and input documents are not uploaded to an OCR service.

## Setup

First accept the model license on Hugging Face, then run:

```powershell
pip install -r requirements.txt
python src/ocr.py --image path\to\page.png
```

Save JSON output:

```powershell
python src/ocr.py --image path\to\page.png --json outputs\page.json
```

The first run downloads the model weights locally. Later runs use the local cache.

Line-level OCR for difficult handwriting:

```powershell
python src\line_ocr.py --image path\to\page.png --output outputs\line_ocr.json
```

## Streamlit app

Activate the main OCR environment:

```powershell
.venv\Scripts\activate
python -m streamlit run app.py
```

The app supports image preview, contrast preprocessing, line-by-line OCR,
editable Telugu text, TXT/JSON/CSV downloads, and local Telugu-to-English
translation. Translation runs in the separate `.venv_translate` environment so
its Transformers version does not conflict with Bodhan OCR.

For Hugging Face model access, create a local `.env` file from `.env.example`:

```env
HF_TOKEN=hf_your_read_token
```

Never commit or share `.env`.

## Train the lightweight Telugu handwriting model

This is a separate custom CNN+BiLSTM+CTC model. It is not the Bodhan checkpoint and can be fine-tuned on any Telugu image-label CSV manifest plus corrected handwriting labels.

Smoke test on 200 samples:

```powershell
python src\train_crnn.py --manifest path\to\labels.csv --limit 200 --epochs 2 --output models\telugu_crnn.pt
```

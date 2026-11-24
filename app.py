"""Streamlit interface for local Telugu handwritten OCR and translation."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import csv
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import streamlit as st
from huggingface_hub import snapshot_download  # type: ignore[reportUnknownVariableType]

# Load optional local environment settings before model initialization.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

SUPPORTED_TYPES = ["png", "jpg", "jpeg", "webp"]
TRANSLATOR_ENV = ".venv_translate"
TRANSLATOR_SCRIPT = Path(__file__).parent / "src" / "translate_local.py"


# -----------------------------------------------------------------------------
# Model loading and translation
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Bodhan Indic OCR model...")  # type: ignore[reportUntypedFunctionDecorator]
def load_ocr() -> Any:
    repo = str(snapshot_download(repo_id="bodhan-ai/indic-ocr"))  # type: ignore[reportUnknownVariableType]
    if repo not in sys.path:
        sys.path.insert(0, repo)
    module: Any = importlib.import_module("indic_ocr")
    return module.IndicOCR.from_pretrained(
        repo, device="cpu", table_format="markdown", max_new_tokens=512
    )


def translator_python() -> Path:
    """Return the isolated Python executable used by IndicTrans2."""
    executable = Path(__file__).parent / TRANSLATOR_ENV / "Scripts" / "python.exe"
    if not executable.exists():
        raise RuntimeError(
            "Translation environment missing. Expected .venv_translate."
        )
    return executable


def translate_telugu(text: str) -> str:
    """Translate Telugu text to English locally with IndicTrans2."""
    if not text.strip():
        return ""
    executable = translator_python()
    completed = subprocess.run(
        [str(executable), str(TRANSLATOR_SCRIPT)], input=text, text=True,
        capture_output=True, check=False, timeout=300,
        encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Local translation failed")
    return completed.stdout.strip()


def clean_result(result: dict[str, Any]) -> dict[str, Any]:
    marker = "\u2581"
    markdown: Any = result.get("markdown")
    result["markdown"] = str(markdown or "").replace(marker, " ").strip()
    blocks: list[Any] = list(result.get("blocks") or [])
    for block in blocks:
        block_data = as_result(block)
        text_value: Any = block_data.get("text")
        if isinstance(text_value, str):
            block_data["text"] = text_value.replace(marker, " ").strip()
    return result


def as_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    typed_value = cast(dict[Any, Any], value)
    for key, item in typed_value.items():
        result[str(key)] = item
    return result


def preprocess_image(source: Path, destination: Path) -> None:
    import cv2  # type: ignore[reportMissingImports]

    image = cv2.imread(str(source))
    if image is None:
        raise ValueError("Could not read the uploaded image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if not cv2.imwrite(str(destination), enhanced):
        raise ValueError("Could not write preprocessed image")


def process_file(
    ocr: Any, input_path: Path, work_dir: Path, use_preprocess: bool, index: int
) -> dict[str, Any]:
    ocr_path = input_path
    if use_preprocess:
        ocr_path = work_dir / f"preprocessed_{index}.png"
        preprocess_image(input_path, ocr_path)

    raw: Any = ocr.parse(str(ocr_path))
    return clean_result(as_result(raw))


def render_results(results: list[dict[str, Any]]) -> None:
    corrected: list[dict[str, str]] = []
    translated: list[dict[str, str]] = []
    for item in results:
        filename = str(item["file"])
        with st.expander(filename, expanded=True):
            if item.get("error"):
                st.error(str(item["error"]))
                continue
            value = str(item.get("markdown") or "")
            edited = st.text_area("Review and correct Telugu text", value, height=220, key=f"text_{filename}")
            corrected.append({"image": filename, "text": edited})
            if st.button("Translate to English", key=f"translate_{filename}"):
                try:
                    lines = [line.strip() for line in edited.splitlines() if line.strip()]
                    english_lines = translate_telugu("\n".join(lines)).splitlines()
                    aligned = [
                        {
                            "telugu": telugu_line,
                            "english": english_lines[index]
                            if index < len(english_lines)
                            else "",
                        }
                        for index, telugu_line in enumerate(lines)
                    ]
                    st.subheader("Sentence Alignment")
                    st.caption("Telugu OCR sentence → English translation")
                    for row in aligned:
                        left, right = st.columns(2)
                        with left:
                            st.markdown(row["telugu"])
                        with right:
                            st.markdown(row["english"])
                        st.divider()
                    english = "\n".join(row["english"] for row in aligned)
                    st.text_area(
                        "Complete English translation",
                        english,
                        height=180,
                        key=f"english_{filename}",
                    )
                    translated.append({"image": filename, "english": english})
                except Exception as exc:
                    st.error(f"Translation failed: {exc}")

    combined = "\n\n".join(
        f"### {item['file']}\n{item.get('markdown', '')}"
        for item in results if not item.get("error")
    )
    st.download_button("Download combined TXT", combined, "telugu_ocr_results.txt", "text/plain")
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    st.download_button("Download JSON", payload, "telugu_ocr_results.json", "application/json")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["image", "text"])
    writer.writeheader()
    writer.writerows(corrected)
    st.download_button("Download corrected labels CSV", buffer.getvalue(), "corrected_labels.csv", "text/csv")
    if translated:
        translation_buffer = io.StringIO()
        translation_writer = csv.DictWriter(translation_buffer, fieldnames=["image", "english"])
        translation_writer.writeheader()
        translation_writer.writerows(translated)
        st.download_button(
            "Download English translations CSV", translation_buffer.getvalue(),
            "english_translations.csv", "text/csv"
        )


# -----------------------------------------------------------------------------
# Application entry point
# -----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Telugu Handwritten OCR", page_icon="OCR", layout="wide")
    st.title("Telugu Handwritten OCR")
    uploads = st.file_uploader(
        "Upload one or more handwritten Telugu images",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
    )
    use_preprocess = st.checkbox("Apply contrast preprocessing", value=False)
    if not uploads:
        st.info("Upload an image to begin.")
        return

    st.write(f"Selected files: {len(uploads)}")
    st.subheader("Original uploaded images")
    columns = st.columns(min(3, len(uploads)))
    for index, upload in enumerate(uploads):
        with columns[index % len(columns)]:
            st.image(upload.getvalue(), caption=upload.name, width="stretch")

    if st.button("Extract Telugu Text", type="primary"):
        ocr: Any = load_ocr()
        results: list[dict[str, Any]] = []
        progress = st.progress(0)
        status = st.empty()
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            for index, upload in enumerate(uploads, 1):
                filename = Path(upload.name).name
                input_path = work_dir / filename
                input_path.write_bytes(upload.getvalue())
                status.write(f"Processing {index}/{len(uploads)}: {filename}")
                try:
                    result = process_file(ocr, input_path, work_dir, use_preprocess, index)
                    results.append({"file": filename, **result})
                except Exception as exc:
                    results.append({"file": filename, "error": str(exc), "markdown": "", "blocks": []})
                progress.progress(index / len(uploads))
        status.write("Completed")
        st.session_state["ocr_results"] = results

    stored_results = st.session_state.get("ocr_results")
    if isinstance(stored_results, list):
        render_results(stored_results)


if __name__ == "__main__":
    main()

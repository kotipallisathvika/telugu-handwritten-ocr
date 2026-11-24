"""Standalone IndicTrans2 Telugu-to-English translator."""
# Runtime dependencies are installed in the separate .venv_translate environment.
# Pylance may inspect this file with the main OCR environment instead.
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false
import sys
from typing import cast

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from IndicTransToolkit import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main() -> None:
    # Read raw UTF-8 bytes so Telugu text is preserved on Windows terminals.
    text = sys.stdin.buffer.read().decode("utf-8", errors="replace").strip()
    if not text:
        return
    name = "ai4bharat/indictrans2-indic-en-dist-200M"
    tokenizer = AutoTokenizer.from_pretrained(
        name, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        name, trust_remote_code=True, local_files_only=True
    )
    model.eval()
    processor = IndicProcessor(inference=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    prepared = processor.preprocess_batch(lines, src_lang="tel_Telu", tgt_lang="eng_Latn")
    inputs = tokenizer(prepared, padding="longest", truncation=True, max_length=256, return_tensors="pt")
    output = model.generate(**inputs, max_length=256, num_beams=5)
    decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
    translations = cast(list[str], processor.postprocess_batch(decoded, lang="eng_Latn"))
    print("\n".join(translations))


if __name__ == "__main__":
    main()

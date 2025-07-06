import sys
from pathlib import Path
from pypdf import PdfReader
import os

def is_signed(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        root = reader.trailer.get("/Root", {})
        acroform = root.get("/AcroForm", {})
        if "/SigFlags" in acroform:
            return True
        fields = acroform.get("/Fields", [])
        return any("/Sig" in str(field.get_object()) for field in fields)
    except Exception as e:
        print(f"Warning: could not read {pdf_path}: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_signed_pdfs.py /path/to/pdfdir")
        sys.exit(1)

    base_dir = Path(sys.argv[1]).resolve()
    signed_dir = base_dir / "SIGNED"
    other_dir  = base_dir / "OTHER"

    signed_dir.mkdir(exist_ok=True)
    other_dir.mkdir(exist_ok=True)

    for file in base_dir.iterdir():
        if file.is_file() and file.suffix.lower() == ".pdf":
            target_dir = signed_dir if is_signed(file) else other_dir
            link_path = target_dir / file.name
            try:
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()
                os.symlink(file, link_path)
                print(f"{'Signed' if target_dir==signed_dir else 'Other'}: {file.name}")
            except Exception as e:
                print(f"Could not create symlink for {file}: {e}")

if __name__ == "__main__":
    main()

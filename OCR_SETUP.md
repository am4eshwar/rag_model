# OCR Setup Guide for Windows

This guide explains how to set up OCR (Optical Character Recognition) support for processing scanned/photographed student assignments.

## Overview

The pipeline now supports two OCR engines:
1. **pytesseract** - Fast OCR for printed/typed text (scanned typed assignments)
2. **EasyOCR** - Deep learning OCR for handwritten text (photographed handwritten assignments)

## Installation Steps

### 1. Install Tesseract OCR Engine (Required for pytesseract)

#### Option A: Using Installer (Recommended)
1. Download Tesseract installer for Windows:
   - Visit: https://github.com/UB-Mannheim/tesseract/wiki
   - Download: `tesseract-ocr-w64-setup-v5.3.3.20231005.exe` (or latest version)

2. Run the installer:
   - Install to default location: `C:\Program Files\Tesseract-OCR`
   - ✅ Check "Add to PATH" during installation

3. Verify installation:
   ```cmd
   tesseract --version
   ```
   Should show: `tesseract 5.3.3` or similar

#### Option B: Using Chocolatey
```powershell
choco install tesseract
```

### 2. Install Python OCR Packages

Activate your virtual environment and install:

```cmd
cd "C:\program RAG model"
venv\Scripts\activate
pip install pytesseract pdf2image Pillow easyocr
```

### 3. Install Poppler (Required for pdf2image)

Poppler is needed to convert PDF pages to images for OCR.

#### Option A: Using Chocolatey (Easiest)
```powershell
choco install poppler
```

#### Option B: Manual Installation
1. Download poppler for Windows:
   - Visit: https://github.com/oschwartz10612/poppler-windows/releases
   - Download: `Release-XX.XX.X-X.zip` (latest)

2. Extract to: `C:\Program Files\poppler`

3. Add to PATH:
   - Open System Properties → Environment Variables
   - Edit "Path" variable
   - Add: `C:\Program Files\poppler\Library\bin`

4. Verify:
   ```cmd
   pdfinfo -v
   ```

### 4. Configure pytesseract (If Not in PATH)

If Tesseract is not in PATH, configure pytesseract to find it:

Create a file `tesseract_config.py` in your project root:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## Configuration

Edit `config.py` to customize OCR settings:

```python
# OCR Configuration
USE_OCR = True  # Enable/disable OCR
OCR_LANGUAGE = "eng"  # Tesseract language: eng, hin, spa, etc.
OCR_MIN_TEXT_THRESHOLD = 100  # Chars threshold to trigger OCR
USE_EASYOCR_FOR_HANDWRITING = True  # Use EasyOCR for handwriting
EASYOCR_LANGUAGES = ["en"]  # EasyOCR languages
OCR_DPI = 300  # Image quality (higher = better but slower)
```

### Language Support

**Tesseract Languages:**
- English: `eng` (default)
- Hindi: `hin`
- Spanish: `spa`
- Multiple: `eng+hin` (combines languages)

Download additional languages from:
https://github.com/tesseract-ocr/tessdata

Place `.traineddata` files in: `C:\Program Files\Tesseract-OCR\tessdata\`

**EasyOCR Languages:**
- English: `en`
- Hindi: `hi`
- Spanish: `es`
- See full list: https://www.jaided.ai/easyocr/

## Testing OCR

### Test with Sample Scanned PDF

1. Create a test scanned PDF:
   - Type or write text in Word/notepad
   - Save as PDF
   - Open in browser, print to PDF as image (scanned simulation)
   - Or photograph a handwritten page and convert to PDF

2. Place in `data/raw_submissions/`

3. Run ingestion:
   ```cmd
   python main.py --build
   ```

4. Check logs for OCR messages:
   ```
   INFO - Detected scanned PDF (45 chars < 100). Applying OCR...
   INFO - Converting PDF to images at 300 DPI...
   INFO - Processing page 1/1 with OCR...
   INFO - OCR extracted 523 chars (better than 45)
   ```

### Quick Test Script

Create `test_ocr.py`:
```python
from ingestion import DocumentIngester
from pathlib import Path

ingester = DocumentIngester()

# Test with your scanned PDF
pdf_path = Path("data/raw_submissions/scanned_assignment.pdf")
doc = ingester.ingest(pdf_path)

print(f"Extracted {len(doc.text)} characters")
print(f"Pages: {len(doc.pages)}")
print(f"First 200 chars:\n{doc.text[:200]}")
```

Run:
```cmd
python test_ocr.py
```

## Performance

### Speed Benchmarks (per page)
- **pytesseract**: ~1-2 seconds (printed text)
- **EasyOCR**: ~3-5 seconds (handwritten text, first run)
- **Normal PDF**: <0.1 seconds (no OCR needed)

### Recommendations
- **100 scanned assignments (5 pages each)**: ~15-20 minutes total
- Use `OCR_DPI=200` for faster processing (slightly lower quality)
- Use `OCR_DPI=400` for better quality (slower)

## Troubleshooting

### Error: "Tesseract not found"
```python
pytesseract.pytesseract.TesseractNotFoundError
```
**Solution:** Add Tesseract to PATH or set `tesseract_cmd` manually

### Error: "poppler not found"
```python
PDFInfoNotInstalledError
```
**Solution:** Install poppler and add to PATH

### Error: "EasyOCR CUDA not available"
```
EasyOCR will use CPU instead of GPU
```
**Solution:** This is normal on systems without NVIDIA GPU. CPU mode works fine.

### Poor OCR Quality
1. Increase DPI: `OCR_DPI = 400`
2. Ensure good image quality (not blurry, good lighting)
3. Try EasyOCR for handwriting: `USE_EASYOCR_FOR_HANDWRITING = True`

## How It Works

### Automatic Detection Flow

```
PDF Ingestion
    ↓
Try Normal Text Extraction (PyPDFLoader)
    ↓
Text < 100 chars? → YES → Scanned PDF detected
    ↓
Convert PDF pages to images (pdf2image)
    ↓
For each page:
    ↓
    Try pytesseract (fast, printed text)
        ↓
    Low quality (<50 chars)?
        ↓
    Try EasyOCR (handwriting)
    ↓
Return best result
```

### When OCR is Used
- ✅ Scanned typed assignments
- ✅ Photographed handwritten assignments
- ✅ PDF from phone camera
- ✅ Screenshots saved as PDF
- ❌ Normal digital PDFs (uses fast extraction)

## Next Steps

After OCR setup:
1. Test with a scanned document
2. Verify extraction quality
3. Adjust `OCR_DPI` if needed
4. Build index with mixed documents (scanned + digital)

The pipeline automatically detects and processes both types!

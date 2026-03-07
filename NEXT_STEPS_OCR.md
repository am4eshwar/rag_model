# OCR Integration - Next Steps

## ✅ What's Done

1. **Updated `requirements.txt`** with OCR dependencies:
   - `pytesseract` - Fast OCR for printed text
   - `pdf2image` - Convert PDF pages to images
   - `Pillow` - Image processing
   - `easyocr` - Deep learning OCR for handwriting

2. **Updated `config.py`** with OCR settings:
   ```python
   USE_OCR = True
   OCR_LANGUAGE = "eng"
   OCR_MIN_TEXT_THRESHOLD = 100
   USE_EASYOCR_FOR_HANDWRITING = True
   EASYOCR_LANGUAGES = ["en"]
   OCR_DPI = 300
   ```

3. **Updated `ingestion.py`** with OCR support:
   - Automatic detection of scanned PDFs
   - `_apply_ocr_to_pdf()` - OCR processing
   - `_apply_easyocr_to_image()` - Handwriting recognition
   - Smart fallback: normal extraction → pytesseract → EasyOCR

4. **Installed Python packages** ✅
   - All OCR dependencies are now in your virtual environment

## ⚠️ Required Manual Steps

### 1. Install Tesseract OCR Engine (Windows)

**Option A: Installer (Recommended)**
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Get: `tesseract-ocr-w64-setup-v5.3.3.20231005.exe`
3. Run installer, install to: `C:\Program Files\Tesseract-OCR`
4. ✅ Check "Add to PATH" during installation

**Option B: Chocolatey**
```powershell
choco install tesseract
```

**Verify installation:**
```cmd
tesseract --version
```

### 2. Install Poppler (for pdf2image)

**Option A: Chocolatey (Easiest)**
```powershell
choco install poppler
```

**Option B: Manual**
1. Download from: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract to: `C:\Program Files\poppler`
3. Add to PATH: `C:\Program Files\poppler\Library\bin`

**Verify:**
```cmd
pdfinfo -v
```

## 🧪 Testing OCR

### Test 1: Quick Verification
```cmd
cd "C:\program RAG model"
venv\Scripts\activate
python -c "import pytesseract; import pdf2image; import easyocr; print('✅ All OCR packages imported successfully')"
```

### Test 2: With Scanned Document

1. Create a test scanned PDF:
   - Type text in Notepad or Word
   - Print to PDF
   - Or photograph handwritten text and save as PDF

2. Place in `data/raw_submissions/scanned_test.pdf`

3. Run:
   ```cmd
   python main.py --build
   ```

4. Look for OCR messages in logs:
   ```
   INFO - Detected scanned PDF (45 chars < 100). Applying OCR...
   INFO - Converting PDF to images at 300 DPI...
   INFO - OCR extracted 523 chars
   ```

## 📊 How It Works

### Automatic Flow
```
PDF → Try normal extraction
        ↓
    Text < 100 chars?
        ↓ YES (scanned)
    Convert to images
        ↓
    pytesseract OCR (fast, printed)
        ↓
    Poor quality (<50 chars)?
        ↓ YES
    EasyOCR (handwriting)
        ↓
    Return best result
```

### When OCR Activates
- ✅ Scanned typed assignments
- ✅ Photographed handwritten assignments  
- ✅ PDF from phone camera
- ✅ Screenshots → PDF
- ❌ Normal digital PDFs (skips OCR, faster)

## 🎯 Current Configuration

Edit `config.py` to customize:

```python
USE_OCR = True                          # Enable/disable OCR
OCR_MIN_TEXT_THRESHOLD = 100            # Trigger OCR if < 100 chars
USE_EASYOCR_FOR_HANDWRITING = True      # Use EasyOCR fallback
OCR_DPI = 300                           # Image quality (200-400)
OCR_LANGUAGE = "eng"                    # Language code
```

### Performance Estimates
- **Normal PDF**: <0.1 seconds per page
- **Scanned (pytesseract)**: ~1-2 seconds per page
- **Handwritten (EasyOCR)**: ~3-5 seconds per page
- **100 assignments (5 pages, 50% scanned)**: ~15-20 minutes

## 🔧 Configuration Tips

### For Hindi + English Support
```python
OCR_LANGUAGE = "eng+hin"  # Tesseract
EASYOCR_LANGUAGES = ["en", "hi"]  # EasyOCR
```

### For Faster Processing (Lower Quality)
```python
OCR_DPI = 200  # Faster, slightly worse quality
```

### For Better Quality (Slower)
```python
OCR_DPI = 400  # Slower, better quality
```

### Disable EasyOCR (Faster, Printed-Only)
```python
USE_EASYOCR_FOR_HANDWRITING = False  # Skip handwriting detection
```

## 📖 Full Documentation

See `OCR_SETUP.md` for:
- Detailed installation instructions
- Troubleshooting guide
- Language setup
- Performance tuning
- Testing procedures

## 🚀 Ready to Use

Once Tesseract and Poppler are installed:

```cmd
# Build index with mixed documents (normal + scanned)
python main.py --build

# Query as usual - OCR happens transparently
python main.py --query "What is the main topic?"
```

The pipeline automatically:
- Detects scanned vs normal PDFs
- Applies appropriate OCR method
- Stores all text in the same index
- Works with 100s of mixed documents!

## Next Feature Implementation

After OCR setup, we can implement:
1. ✅ Document-specific queries (filter by student)
2. ✅ Batch processing for 100 submissions
3. ✅ Student-specific reports
4. ✅ Classroom analytics dashboard

OCR is ready - just install Tesseract + Poppler and test!

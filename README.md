# mokuro translator

Read Japanese manga with selectable text inside a browser or native desktop application.

This project is a personal branch from  [https://github.com/kha-white/mokuro](https://github.com/kha-white/mokuro "mokuro"). Mokuro has already given good function. However, for learning Japanese by using Manga, if a translator is combined in html, it will be much better. Then, I added a translator function to the Mokuro project.

## Mokuro GUI - GPU Accelerated Manga Reader

A native Python desktop application that provides GPU-accelerated translation for manga reading, replacing the browser-based HTML overlay system with a high-performance desktop interface.

### GUI Features

- **GPU-Accelerated Translation**: Uses PyTorch and Transformers with CUDA support for fast translation
- **Native Desktop Interface**: Built with PyQt5 for a responsive desktop experience
- **Multi-Language Support**: Supports Japanese ↔ Chinese, Japanese ↔ English, Chinese ↔ English
- **Interactive Manga Viewer**: Zoom, pan, and navigate through manga pages
- **Real-time Text Overlay**: Display original text, translations, or both simultaneously
- **OCR Integration**: Uses existing mokuro OCR pipeline with caching
- **Progress Tracking**: Visual progress bars for translation operations

### GUI Usage

```bash
python run_gui.py
```

### GUI Interface Overview

- **File Menu**: Open manga volumes (directories containing image files)
- **Toolbar**:
  - Volume indicator
  - Language selection (Source/Target)
  - Display mode (Original/Translation/Both)
  - Translate Page button
- **Main Viewer**: Interactive manga page display with zoom and pan
- **Side Panel**:
  - Page navigation controls
  - Zoom controls (Fit/Width/1:1)

### GUI Workflow

1. **Open Volume**: Use File → Open Volume to select a directory containing manga images
2. **Navigate Pages**: Use arrow buttons or keyboard shortcuts
3. **Run OCR**: OCR runs automatically on new pages (results are cached)
4. **Translate**: Click "Translate Page" to perform GPU-accelerated translation
5. **Interactive Reading**: Click on any text block to toggle between original and translated text
   - White background = Original text
   - Yellow background = Translated text
   - Hover over blocks to see red outline
   - Click to instantly switch languages

# Display
https://github.com/hkrds1996/mokuro_chinese/assets/16051414/9b618241-23ca-49ac-b812-196bea5b58b1


# Install
Requirements are the same as the original Mokuro project.

Instead of using pip3 install, we should do the following command
```commandline
python setup.py install
```

## GUI Requirements
For the desktop GUI application, additional dependencies are required:

```bash
pip install PyQt5 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Note:** GPU-enabled PyTorch is recommended for best translation performance. The GUI will automatically detect and use CUDA if available, falling back to CPU otherwise.

# Using
Using way is the same as the Mokuro project

## OCR Engine Options
Mokuro supports three OCR engines with different strengths:

- **`easyocr`** (recommended for Japanese manga): Supports 80+ languages including Japanese and English
- **`trocr`**: Transformer-based OCR with high accuracy for printed English text
- **`manga-ocr`** (default, but requires PyTorch >= 2.6): Specialized for Japanese manga text

**Note:** The default `manga-ocr` engine requires PyTorch >= 2.6 due to security fixes. For Japanese manga, use `easyocr` which provides excellent Japanese text recognition.

To use a different OCR engine:
```commandline
# Use EasyOCR for Japanese manga (recommended)
python -m mokuro /path/to/manga --ocr_engine easyocr

# Use TrOCR for printed English text
python -m mokuro /path/to/manga --ocr_engine trocr

# Use default manga-ocr (requires PyTorch >= 2.6)
python -m mokuro /path/to/manga
```

## Translation Language
If want to change translated language, go to the overlay_generator.py file and change dest's value in the translate_to_chinese function

```python
def translate_to_chinese(text):
    translator = Translator()
    translation = translator.translate(text, dest='zh-cn')
    return translation.text
```
# Contact
For any inquiries, please feel free to contact me at fasklas68@gmail.com

# BV
https://www.bilibili.com/video/BV1vm4y1a7o3

# Acknowledgments
- https://github.com/kha-white/mokuro
- https://github.com/dmMaze/comic-text-detector
- https://github.com/juvian/Manga-Text-Segmentation

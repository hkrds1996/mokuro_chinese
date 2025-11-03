# mokuro translator

Read Japanese manga with selectable text inside a browser or native desktop application.

This project is a personal branch from  [https://github.com/kha-white/mokuro](https://github.com/kha-white/mokuro "mokuro"). Mokuro has already given good function. However, for learning Japanese by using Manga, if a translator is combined in html, it will be much better. Then, I added a translator function to the Mokuro project.

## Mokuro GUI - GPU Accelerated Manga Reader

A native Python desktop application that provides GPU-accelerated translation for manga reading, replacing the browser-based HTML overlay system with a high-performance desktop interface.

### GUI Features

- **GPU-Accelerated Translation**: Uses Ollama with DeepSeek-R1 8B model for fast, high-quality translation
- **High-Quality Translation Model**: Leverages advanced LLM technology for superior translation quality
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

### Supported Image Formats

The GUI supports the following image formats:
- **JPG/JPEG**: Standard JPEG images
- **PNG**: Portable Network Graphics
- **WebP**: Modern web image format with better compression

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

## Ollama Setup for Translation

The GUI uses Ollama for GPU-accelerated translation. To use translation features, you need to:

### 1. Install Ollama

Download and install Ollama from [https://ollama.ai](https://ollama.ai)

### 2. Pull the Required Model

The GUI is configured to use the `deepseek-r1:8b` model. Pull it using:

```bash
ollama pull deepseek-r1:8b
```

**Note:** The model size is 8B parameters, which provides good balance between speed and quality. If you prefer a different model, you can modify the model name in `mokuro_gui.py`.

### 3. Start Ollama Server

Make sure Ollama is running before using translation features:

```bash
# On Linux/Mac
ollama serve

# On Windows, Ollama usually starts automatically
```

### 4. Verify Installation

Test that Ollama is working:

```bash
ollama list
```

You should see `deepseek-r1:8b` in the list of available models.

### Troubleshooting

- **Connection Issues**: Make sure Ollama is running on `http://localhost:11434`
- **Model Not Found**: Ensure you've pulled the correct model (`deepseek-r1:8b`)
- **Slow Performance**: The 8B model should be reasonably fast. If too slow, consider using a smaller model
- **Out of Memory**: If you get memory errors, try using a smaller model like `llama2:7b`

## Translation Quality

The GUI uses Ollama with DeepSeek-R1 8B model for high-quality translations. Key features include:

- **Advanced LLM Technology**: DeepSeek-R1 provides state-of-the-art translation capabilities
- **Full Text Processing**: Entire text blocks are processed as complete units for better context understanding
- **Line-by-line Formatting**: Translation results are properly formatted to match original manga text structure
- **Persistent Caching**: Translation results are cached as JSON files for instant loading
- **Context Preservation**: Maintains the visual layout and formatting of original manga text

The DeepSeek-R1 model offers excellent translation quality with good performance balance. For specialized manga translation needs, consider fine-tuning or using domain-specific models.

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

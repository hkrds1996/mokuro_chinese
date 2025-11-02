#!/usr/bin/env python3
"""
Mokuro GUI - Native Python desktop application with GPU-accelerated translation
"""

import sys
import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QComboBox, QProgressBar, QScrollArea,
                             QFrame, QSplitter, QMenuBar, QMenu, QAction, QFileDialog,
                             QStatusBar, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsTextItem, QGraphicsRectItem, QSizePolicy, QGraphicsItem)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QBrush, QColor, QFont, QTransform
from PyQt5.QtCore import Qt, QRectF, QPointF, QThread, pyqtSignal, QTimer

# Import mokuro components
from mokuro.manga_page_ocr import MangaPageOcr
from mokuro.utils import load_json

# Translation imports
from transformers import pipeline
import torch


class TranslationWorker(QThread):
    """Worker thread for GPU-accelerated translation"""
    progress = pyqtSignal(int, str)  # progress percentage, status message
    finished = pyqtSignal(dict)  # translated results
    error = pyqtSignal(str)

    def __init__(self, text_blocks: List[Dict], source_lang: str, target_lang: str):
        super().__init__()
        self.text_blocks = text_blocks
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.translator = None

    def run(self):
        try:
            # Initialize translator
            self.progress.emit(10, "Loading translation model...")

            # Language pair mapping using NLLB models (GPU-friendly and publicly available)
            # NLLB requires specific language codes
            nllb_lang_codes = {
                'ja': 'jpn_Jpan',  # Japanese
                'zh': 'zho_Hans',  # Simplified Chinese
                'en': 'eng_Latn',  # English
            }

            src_lang_code = nllb_lang_codes.get(self.source_lang)
            tgt_lang_code = nllb_lang_codes.get(self.target_lang)

            if not src_lang_code or not tgt_lang_code:
                raise ValueError(f"Unsupported language: {self.source_lang} or {self.target_lang}")

            device = 0 if torch.cuda.is_available() else -1
            self.translator = pipeline('translation', model='facebook/nllb-200-distilled-1.3B',
                                     src_lang=src_lang_code, tgt_lang=tgt_lang_code, device=device)

            self.progress.emit(30, "Translating text...")

            translated_blocks = []
            total_blocks = len(self.text_blocks)

            for i, block in enumerate(self.text_blocks):
                original_text = '\n'.join(block['lines'])
                if original_text.strip():
                    # Translate the text
                    result = self.translator(original_text, max_length=512)
                    translated_text = result[0]['translation_text']
                    translated_lines = translated_text.split('\n')
                else:
                    translated_lines = block['lines']

                # Create translated block
                translated_block = block.copy()
                translated_block['lines'] = translated_lines
                translated_blocks.append(translated_block)

                # Update progress
                progress = 30 + int(70 * (i + 1) / total_blocks)
                self.progress.emit(progress, f"Translating block {i + 1}/{total_blocks}...")

            self.progress.emit(100, "Translation complete")
            self.finished.emit({'blocks': translated_blocks})

        except Exception as e:
            self.error.emit(str(e))


class BatchTranslationWorker(QThread):
    """Worker thread for translating all pages in a volume"""
    page_progress = pyqtSignal(int, str, Path)  # page index, status message, image path
    overall_progress = pyqtSignal(int, str)  # overall percentage, status message
    page_finished = pyqtSignal(dict, Path)  # translated results, image path
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, image_paths: List[Path], ocr_results: Dict, source_lang: str, target_lang: str, volume_path: Path):
        super().__init__()
        self.image_paths = image_paths
        self.ocr_results = ocr_results
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.volume_path = volume_path
        self.translator = None
        self.is_cancelled = False

    def cancel(self):
        """Cancel the batch translation"""
        self.is_cancelled = True

    def run(self):
        try:
            # Initialize translator
            self.overall_progress.emit(5, "Loading translation model...")

            # Language pair mapping using NLLB models (GPU-friendly and publicly available)
            # NLLB requires specific language codes
            nllb_lang_codes = {
                'ja': 'jpn_Jpan',  # Japanese
                'zh': 'zho_Hans',  # Simplified Chinese
                'en': 'eng_Latn',  # English
            }

            src_lang_code = nllb_lang_codes.get(self.source_lang)
            tgt_lang_code = nllb_lang_codes.get(self.target_lang)

            if not src_lang_code or not tgt_lang_code:
                raise ValueError(f"Unsupported language: {self.source_lang} or {self.target_lang}")

            device = 0 if torch.cuda.is_available() else -1
            self.translator = pipeline('translation', model='facebook/nllb-200-distilled-1.3B',
                                     src_lang=src_lang_code, tgt_lang=tgt_lang_code, device=device)

            self.overall_progress.emit(10, "Starting batch translation...")

            total_pages = len(self.image_paths)
            completed_pages = 0

            for page_idx, image_path in enumerate(self.image_paths):
                if self.is_cancelled:
                    break

                # Check if translation already exists
                results_dir = self.volume_path.parent / '_ocr' / self.volume_path.name
                translation_json_path = results_dir / f"{image_path.relative_to(self.volume_path).with_suffix('')}_{self.source_lang}_{self.target_lang}.json"

                if translation_json_path.exists():
                    # Skip already translated pages
                    self.page_progress.emit(page_idx, f"Skipping page {page_idx + 1}/{total_pages} (already translated)...", image_path)
                    completed_pages += 1
                    overall_progress = 10 + int(90 * completed_pages / total_pages)
                    self.overall_progress.emit(overall_progress, f"Completed {completed_pages}/{total_pages} pages")
                    continue

                self.page_progress.emit(page_idx, f"Processing page {page_idx + 1}/{total_pages}...", image_path)

                # Load OCR data if not already loaded
                if image_path not in self.ocr_results:
                    # Load cached OCR results
                    results_dir = self.volume_path.parent / '_ocr' / self.volume_path.name
                    ocr_json_path = results_dir / image_path.relative_to(self.volume_path).with_suffix('.json')

                    if ocr_json_path.exists():
                        try:
                            ocr_data = load_json(ocr_json_path)
                            self.ocr_results[image_path] = ocr_data
                        except Exception as e:
                            print(f"Failed to load OCR data for {image_path}: {e}")
                            completed_pages += 1
                            continue
                    else:
                        # Skip pages without OCR data (they need to be processed first)
                        completed_pages += 1
                        continue

                ocr_data = self.ocr_results[image_path]

                # Translate all blocks on this page
                translated_blocks = []
                total_blocks = len(ocr_data['blocks'])

                for i, block in enumerate(ocr_data['blocks']):
                    if self.is_cancelled:
                        break

                    original_text = '\n'.join(block['lines'])
                    if original_text.strip():
                        # Translate the text
                        result = self.translator(original_text, max_length=512)
                        translated_text = result[0]['translation_text']
                        translated_lines = translated_text.split('\n')
                    else:
                        translated_lines = block['lines']

                    # Create translated block
                    translated_block = block.copy()
                    translated_block['lines'] = translated_lines
                    translated_blocks.append(translated_block)

                if not self.is_cancelled:
                    # Emit page completion
                    result = {'blocks': translated_blocks}
                    self.page_finished.emit(result, image_path)

                completed_pages += 1
                overall_progress = 10 + int(90 * completed_pages / total_pages)
                self.overall_progress.emit(overall_progress, f"Completed {completed_pages}/{total_pages} pages")

            if not self.is_cancelled:
                self.overall_progress.emit(100, "Batch translation complete")
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))


class OCRWorker(QThread):
    """Worker thread for OCR processing"""
    progress = pyqtSignal(int, str)  # progress percentage, status message
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, volume_path: Path, ocr_engine: str):
        super().__init__()
        self.volume_path = volume_path
        self.ocr_engine = ocr_engine
        self.process = None

    def run(self):
        try:
            import subprocess
            import sys
            import select
            import os

            # Get the Python executable path
            python_exe = sys.executable

            # Build the mokuro command
            cmd = [
                python_exe, "-m", "mokuro",
                str(self.volume_path),
                "--ocr_engine", self.ocr_engine,
                "--disable_confirmation"
            ]

            print(f"Running OCR command: {' '.join(cmd)}")
            print(f"Working directory: {self.volume_path.parent}")

            # Start the subprocess
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stdout and stderr
                text=True,
                cwd=str(self.volume_path.parent),
                bufsize=1,  # Line buffered
                universal_newlines=True
            )

            # Read output line by line
            while True:
                if self.process.poll() is not None:
                    break

                # Try to read a line
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()
                    print(f"OCR: {line}")

                    # Parse progress from output
                    if "Processing pages" in line and "|" in line:
                        # Extract progress percentage from tqdm output like:
                        # "Processing pages...:  10%|...| 23/230"
                        try:
                            # Find percentage
                            if "%" in line:
                                percent_str = line.split("%")[0].split()[-1]
                                if percent_str.isdigit():
                                    progress = int(percent_str)
                                    self.progress.emit(progress, f"OCR Progress: {progress}%")
                        except:
                            pass
                    elif "Processing" in line and ".jpg" in line:
                        # Show which file is being processed
                        self.progress.emit(-1, f"Processing: {line.split('Processing')[-1].strip()}")
                    elif "Skipping" in line and ".jpg" in line:
                        # Show skipped files
                        self.progress.emit(-1, f"Skipping: {line.split('Skipping')[-1].strip()}")

            # Get final return code
            return_code = self.process.wait()

            if return_code == 0:
                self.progress.emit(100, "OCR processing complete")
                self.finished.emit()
            else:
                # Read any remaining output
                remaining = self.process.stdout.read()
                if remaining:
                    print(f"OCR remaining output: {remaining}")
                self.error.emit(f"OCR failed with return code {return_code}")

        except Exception as e:
            print(f"OCR worker error: {e}")
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def cancel(self):
        """Cancel the OCR processing"""
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                # Wait a bit then kill if still running
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            except:
                pass


class MangaGraphicsView(QGraphicsView):
    """Custom graphics view for manga page display with zoom and pan"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)

        # Zoom factors
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        zoom_in = event.angleDelta().y() > 0
        factor = 1.2 if zoom_in else 1/1.2

        new_zoom = self.zoom_factor * factor
        if self.min_zoom <= new_zoom <= self.max_zoom:
            self.scale(factor, factor)
            self.zoom_factor = new_zoom

    def fit_to_view(self):
        """Fit the manga page to the view"""
        if self.scene():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
            self.zoom_factor = self.transform().m11()  # Get current scale

    def zoom_to_width(self):
        """Zoom to fit width"""
        if self.scene():
            rect = self.scene().sceneRect()
            view_width = self.viewport().width()
            scale = view_width / rect.width()
            self.resetTransform()
            self.scale(scale, scale)
            self.zoom_factor = scale

    def reset_zoom(self):
        """Reset to original size"""
        self.resetTransform()
        self.zoom_factor = 1.0


class TextBlockItem(QGraphicsRectItem):
    """Clickable text block item that can toggle between original and translated text"""

    def __init__(self, block_data, translated_data, img_width, img_height, parent=None):
        # Use the exact OCR bounding box coordinates
        xmin, ymin, xmax, ymax = block_data['box']
        super().__init__(xmin, ymin, xmax-xmin, ymax-ymin)

        self.block_data = block_data
        self.translated_data = translated_data
        self.img_width = img_width
        self.img_height = img_height

        # Set up appearance - fully opaque
        self.setBrush(QBrush(QColor(255, 255, 255, 255)))
        self.setPen(QPen(QColor(0, 0, 0, 100)))
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

        # Create text item
        self.text_item = QGraphicsTextItem(self)
        self.update_text_display(show_original=True)

        # Set initial position
        self.update_position()

        # Store current state
        self.showing_original = True

    def _arrange_text_to_fit_block(self, text_lines, block_width, block_height, is_vertical=False):
        """Arrange text to fit within block dimensions with proper line breaks and columns"""
        if not text_lines:
            return ""
            
        full_text = ' '.join(text_lines) if not is_vertical else ''.join(text_lines)
        
        if is_vertical:
            return self._arrange_vertical_text(full_text, block_width, block_height)
        else:
            return self._arrange_horizontal_text(full_text, block_width, block_height)

    def _arrange_horizontal_text(self, text, block_width, block_height):
        """Arrange horizontal text with line breaks based on block width"""
        if not text:
            return ""
            
        # Estimate character width (rough approximation)
        avg_char_width = 10  # conservative estimate
        max_chars_per_line = max(1, int(block_width / avg_char_width))
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length <= max_chars_per_line:
                current_line.append(word)
                current_length += word_length
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
                
        if current_line:
            lines.append(' '.join(current_line))
            
        # If text is still too tall, reduce number of lines by increasing chars per line
        estimated_height = len(lines) * 20  # approx line height
        if estimated_height > block_height and max_chars_per_line < len(text):
            # Try with fewer lines by allowing more characters per line
            return self._arrange_horizontal_text(text, block_width, block_height * 1.5)
            
        return '\n'.join(lines)

    def _arrange_vertical_text(self, text, block_width, block_height):
        """Arrange vertical text by keeping meaningful character groups together"""
        if not text:
            return ""

        # For vertical text, keep the text as horizontal lines but stack them vertically
        # This preserves reading context while maintaining vertical layout
        lines = []
        current_line = ""

        for char in text:
            # Start new line for major punctuation or when line gets too long
            if char in '。、！？' or len(current_line) >= 8:  # Max 8 chars per line
                if current_line:
                    lines.append(current_line)
                current_line = char
            else:
                current_line += char

        if current_line:
            lines.append(current_line)

        # Join lines with newlines for vertical stacking
        return '\n'.join(lines)

    def _calculate_optimal_font_size(self, text, block_width, block_height, is_vertical=False):
        """Calculate optimal font size to fit text within block by testing actual rendering"""
        # Start with detected font size or default (ensure integer)
        detected_size = self.block_data.get('font_size', 16)
        font_size = max(8, min(72, int(detected_size)))  # Reasonable bounds, ensure int

        # Test if current font size fits, if not, reduce it
        test_font = QFont()
        test_font.setPixelSize(font_size)

        # Create a temporary text item to measure
        temp_text_item = QGraphicsTextItem()
        temp_text_item.setFont(test_font)
        temp_text_item.setPlainText(text)
        temp_text_item.setTextWidth(block_width if not is_vertical else block_width)

        # Check if it fits
        text_rect = temp_text_item.boundingRect()

        # If text is too wide or too tall, reduce font size
        while (text_rect.width() > block_width * 1.1 or text_rect.height() > block_height * 1.1) and font_size > 8:
            font_size -= 1  # Keep as int
            test_font.setPixelSize(font_size)
            temp_text_item.setFont(test_font)
            text_rect = temp_text_item.boundingRect()

        return max(8, font_size)  # Ensure minimum readable size, return int

    def update_text_display(self, show_original=True):
        """Update the text display and resize block accordingly"""
        # Get OCR block dimensions
        xmin, ymin, xmax, ymax = self.block_data['box']
        ocr_width = xmax - xmin
        ocr_height = ymax - ymin

        # Get the text content
        if show_original:
            text_lines = self.block_data['lines']
            bg_color = QColor(255, 255, 255, 255)  # White background for original
        else:
            if self.translated_data:
                text_lines = self.translated_data['lines']
            else:
                text_lines = self.block_data['lines']
            bg_color = QColor(255, 255, 0, 255)  # Yellow background for translation

        is_vertical = self.block_data.get('vertical', False)
        
        # Arrange text to fit block
        arranged_text = self._arrange_text_to_fit_block(
            text_lines, ocr_width, ocr_height, is_vertical
        )
        
        # Calculate optimal font size
        font_size = self._calculate_optimal_font_size(
            arranged_text, ocr_width, ocr_height, is_vertical
        )

        # Set text and font
        self.text_item.setPlainText(arranged_text)
        font = QFont()
        font.setPixelSize(font_size)
        self.text_item.setFont(font)

        # Set text properties based on orientation
        if is_vertical:
            # Vertical text - constrain width to block width for proper horizontal line layout
            self.text_item.setRotation(0)
            self.text_item.setTransformOriginPoint(QPointF(0, 0))
            self.text_item.setTextWidth(ocr_width)  # Constrain to block width
            self.text_item.setPos(0, 0)
        else:
            # Horizontal text - enable word wrap within block width
            self.text_item.setRotation(0)
            self.text_item.setTransformOriginPoint(QPointF(0, 0))
            self.text_item.setTextWidth(ocr_width)
            self.text_item.setPos(0, 0)

        # Update background color
        self.setBrush(QBrush(bg_color))

        # Auto-resize and position text within block
        self.auto_resize_block()

    def auto_resize_block(self):
        """Ensure text fits properly within the OCR bounding box"""
        xmin, ymin, xmax, ymax = self.block_data['box']
        ocr_width = xmax - xmin
        ocr_height = ymax - ymin

        # Set block to exact OCR dimensions
        self.setRect(0, 0, ocr_width, ocr_height)

        # Center text within the OCR bounding box
        text_rect = self.text_item.boundingRect()
        text_x = (ocr_width - text_rect.width()) / 2
        text_y = (ocr_height - text_rect.height()) / 2
        
        # Ensure text doesn't go outside block
        text_x = max(0, min(text_x, ocr_width - text_rect.width()))
        text_y = max(0, min(text_y, ocr_height - text_rect.height()))
        
        self.text_item.setPos(text_x, text_y)

    def update_position(self):
        """Update the position of the text item - use exact OCR coordinates"""
        # Position is already set in constructor using OCR box coordinates
        pass

    def mousePressEvent(self, event):
        """Handle mouse click to toggle text"""
        if event.button() == Qt.LeftButton:
            self.showing_original = not self.showing_original
            self.update_text_display(self.showing_original)
            # Add visual feedback
            self.setPen(QPen(QColor(255, 0, 0, 150), 2))
            QTimer.singleShot(200, lambda: self.setPen(QPen(QColor(0, 0, 0, 100))))
        super().mousePressEvent(event)

    def hoverEnterEvent(self, event):
        """Handle mouse hover enter"""
        self.setPen(QPen(QColor(255, 0, 0, 120), 2))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Handle mouse hover leave"""
        self.setPen(QPen(QColor(0, 0, 0, 100)))
        super().hoverLeaveEvent(event)

class MokuroGUI(QMainWindow):
    """Main GUI application for mokuro"""

    def __init__(self):
        super().__init__()
        self.current_volume_path = None
        self.current_page_idx = 0
        self.image_paths = []
        self.ocr_results = {}  # Cache for OCR results
        self.translated_results = {}  # Cache for translated results
        self.mpocr = None
        self.text_blocks = []  # Store text block items for current page
        self.current_display_mode = 'Original'  # Track current display mode

        # Load persistent translation settings
        self.config_file = Path.home() / '.mokuro_gui_config.json'
        self.load_translation_config()

        self.translation_worker = None

        self.init_ui()
        self.init_menus()
        # Initialize OCR lazily when needed

    def load_translation_config(self):
        """Load translation settings from config file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.source_lang = config.get('source_lang', 'ja')
                    self.target_lang = config.get('target_lang', 'zh')
                    self.ocr_engine = config.get('ocr_engine', 'trocr')
            else:
                # Default settings
                self.source_lang = 'ja'
                self.target_lang = 'zh'
                self.ocr_engine = 'trocr'
        except Exception as e:
            print(f"Failed to load config: {e}")
            self.source_lang = 'ja'
            self.target_lang = 'zh'
            self.ocr_engine = 'trocr'

    def save_translation_config(self):
        """Save translation settings to config file"""
        try:
            config = {
                'source_lang': self.source_lang,
                'target_lang': self.target_lang,
                'ocr_engine': self.ocr_engine
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Mokuro GUI - GPU Accelerated Manga Reader")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Top toolbar
        toolbar_layout = QHBoxLayout()

        # Volume selection
        self.volume_label = QLabel("No volume loaded")
        toolbar_layout.addWidget(self.volume_label)

        toolbar_layout.addStretch()

        # Language selection
        toolbar_layout.addWidget(QLabel("Source:"))
        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItems(['ja', 'zh', 'en'])
        self.source_lang_combo.setCurrentText(self.source_lang)
        self.source_lang_combo.currentTextChanged.connect(self.on_source_lang_changed)
        toolbar_layout.addWidget(self.source_lang_combo)

        toolbar_layout.addWidget(QLabel("Target:"))
        self.target_lang_combo = QComboBox()
        self.target_lang_combo.addItems(['zh', 'ja', 'en'])
        self.target_lang_combo.setCurrentText(self.target_lang)
        self.target_lang_combo.currentTextChanged.connect(self.on_target_lang_changed)
        toolbar_layout.addWidget(self.target_lang_combo)

        # OCR Engine selection
        toolbar_layout.addWidget(QLabel("OCR:"))
        self.ocr_engine_combo = QComboBox()
        self.ocr_engine_combo.addItems(['manga-ocr', 'trocr'])
        self.ocr_engine_combo.setCurrentText('trocr')  # Default to trocr as user prefers
        self.ocr_engine_combo.currentTextChanged.connect(self.on_ocr_engine_changed)
        toolbar_layout.addWidget(self.ocr_engine_combo)

        # Display options
        toolbar_layout.addWidget(QLabel("Display:"))
        self.display_combo = QComboBox()
        self.display_combo.addItems(['Original', 'Translation', 'Both', 'Hidden'])
        self.display_combo.setCurrentText('Original')
        self.display_combo.currentTextChanged.connect(self.on_display_mode_changed)
        toolbar_layout.addWidget(self.display_combo)

        # Translate buttons
        self.translate_btn = QPushButton("Translate Page")
        self.translate_btn.clicked.connect(self.translate_current_page)
        self.translate_btn.setEnabled(False)
        toolbar_layout.addWidget(self.translate_btn)

        self.translate_all_btn = QPushButton("Translate All")
        self.translate_all_btn.clicked.connect(self.translate_all_pages)
        self.translate_all_btn.setEnabled(False)
        toolbar_layout.addWidget(self.translate_all_btn)

        main_layout.addLayout(toolbar_layout)

        # Progress bar for translation
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Main content area
        content_splitter = QSplitter(Qt.Horizontal)

        # Manga viewer
        self.graphics_view = MangaGraphicsView()
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        content_splitter.addWidget(self.graphics_view)

        # Side panel for page navigation
        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)

        # Page navigation
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀")
        self.prev_btn.clicked.connect(self.prev_page)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)

        self.page_label = QLabel("Page: 0/0")
        nav_layout.addWidget(self.page_label)

        self.next_btn = QPushButton("▶")
        self.next_btn.clicked.connect(self.next_page)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)

        side_layout.addLayout(nav_layout)

        # Zoom controls
        zoom_layout = QHBoxLayout()
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.clicked.connect(self.graphics_view.fit_to_view)
        zoom_layout.addWidget(self.fit_btn)

        self.width_btn = QPushButton("Width")
        self.width_btn.clicked.connect(self.graphics_view.zoom_to_width)
        zoom_layout.addWidget(self.width_btn)

        self.original_btn = QPushButton("1:1")
        self.original_btn.clicked.connect(self.graphics_view.reset_zoom)
        zoom_layout.addWidget(self.original_btn)

        side_layout.addLayout(zoom_layout)

        side_layout.addStretch()

        content_splitter.addWidget(side_panel)
        content_splitter.setSizes([800, 200])

        main_layout.addWidget(content_splitter)

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

    def init_menus(self):
        """Initialize menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        open_action = QAction('Open Volume', self)
        open_action.triggered.connect(self.open_volume)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        ocr_action = QAction('Process Volume with OCR', self)
        ocr_action.triggered.connect(self.process_volume_with_ocr)
        ocr_action.setEnabled(False)
        self.ocr_action = ocr_action
        file_menu.addAction(ocr_action)

        file_menu.addSeparator()

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu('View')

        fit_action = QAction('Fit to Screen', self)
        fit_action.triggered.connect(self.graphics_view.fit_to_view)
        view_menu.addAction(fit_action)

        width_action = QAction('Fit to Width', self)
        width_action.triggered.connect(self.graphics_view.zoom_to_width)
        view_menu.addAction(width_action)

        original_action = QAction('Original Size', self)
        original_action.triggered.connect(self.graphics_view.reset_zoom)
        view_menu.addAction(original_action)

    def init_ocr(self):
        """Initialize OCR engine"""
        try:
            self.mpocr = MangaPageOcr(force_cpu=False)  # Use GPU if available
            self.status_bar.showMessage("OCR engine initialized")
        except Exception as e:
            self.status_bar.showMessage(f"OCR initialization failed: {e}")

    def open_volume(self):
        """Open a volume directory"""
        volume_path = QFileDialog.getExistingDirectory(self, "Select Volume Directory")
        if volume_path:
            self.load_volume(volume_path)

    def load_volume(self, volume_path: str):
        """Load a manga volume"""
        self.current_volume_path = Path(volume_path)

        # Find image files
        self.image_paths = []
        for ext in ['.jpg', '.jpeg', '.png']:
            self.image_paths.extend(self.current_volume_path.glob(f'**/*{ext}'))

        # Remove duplicates (in case of case-insensitive file systems)
        self.image_paths = list(dict.fromkeys(self.image_paths))

        # Natural sort the image paths (1.jpg, 2.jpg, 10.jpg, not 1.jpg, 10.jpg, 2.jpg)
        def natural_sort_key(path):
            # Extract filename and split into parts
            filename = path.name
            # Split on numbers and non-numbers
            parts = re.split(r'(\d+)', filename)
            # Convert numeric parts to integers for proper sorting
            return [int(part) if part.isdigit() else part.lower() for part in parts]

        self.image_paths = sorted(self.image_paths, key=natural_sort_key)

        if not self.image_paths:
            self.status_bar.showMessage("No image files found in the selected directory")
            return

        self.current_page_idx = 0
        self.volume_label.setText(f"Volume: {self.current_volume_path.name}")
        self.update_navigation_buttons()
        self.translate_all_btn.setEnabled(True)
        self.ocr_action.setEnabled(True)
        self.load_page(0)

    def load_page(self, page_idx: int):
        """Load and display a page"""
        if not self.image_paths or page_idx >= len(self.image_paths):
            return

        image_path = self.image_paths[page_idx]

        # Load image
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.status_bar.showMessage(f"Failed to load image: {image_path}")
            return

        # Clear scene and add image
        self.scene.clear()
        pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(pixmap_item)

        # Load OCR results
        self.load_ocr_results(image_path)

        # Update UI
        self.page_label.setText(f"Page: {page_idx + 1}/{len(self.image_paths)}")
        self.graphics_view.fit_to_view()
        self.translate_btn.setEnabled(True)

    def load_ocr_results(self, image_path: Path):
        """Load OCR results and translations for the current page"""
        results_dir = self.current_volume_path.parent / '_ocr' / self.current_volume_path.name
        ocr_json_path = results_dir / image_path.relative_to(self.current_volume_path).with_suffix('.json')
        translation_json_path = results_dir / f"{image_path.relative_to(self.current_volume_path).with_suffix('')}_{self.source_lang}_{self.target_lang}.json"

        # Load cached OCR results
        ocr_data = None
        if ocr_json_path.exists():
            try:
                ocr_data = load_json(ocr_json_path)
                self.ocr_results[image_path] = ocr_data
            except Exception as e:
                print(f"Failed to load OCR data from {ocr_json_path}: {e}")
                self.status_bar.showMessage(f"Warning: Corrupted OCR data for {image_path.name}")
                # Remove the corrupted file so it can be re-processed
                try:
                    ocr_json_path.unlink()
                except:
                    pass
                # Fall through to run OCR below

        if ocr_data is None:
            # No OCR data available - create empty structure for display
            # User must manually run OCR via menu option
            ocr_data = {'blocks': [], 'img_width': 100, 'img_height': 100}

        # Load cached translation results if they exist
        if translation_json_path.exists():
            try:
                translated_data = load_json(translation_json_path)
                self.translated_results[image_path] = translated_data
                self.status_bar.showMessage("Loaded cached translation")
            except Exception as e:
                print(f"Failed to load cached translation from {translation_json_path}: {e}")
                self.status_bar.showMessage(f"Warning: Corrupted translation data for {image_path.name}")
                # Remove the corrupted file so it can be re-processed
                try:
                    translation_json_path.unlink()
                except:
                    pass

        # Display text blocks (will use cached translation if available)
        self.display_text_blocks(ocr_data)

    def display_text_blocks(self, ocr_data: Dict, display_mode: str = None):
        """Display text blocks on the scene"""
        if 'blocks' not in ocr_data:
            return

        if display_mode is None:
            display_mode = self.current_display_mode

        W, H = ocr_data['img_width'], ocr_data['img_height']

        # Clear existing text blocks safely
        for block in self.text_blocks[:]:  # copy the list
            try:
                if block.scene() is not None:
                    self.scene.removeItem(block)
            except RuntimeError:
                # already deleted, ignore
                pass
        self.text_blocks = []
        # Determine which text to show
        image_path = self.image_paths[self.current_page_idx]
        translated_data = self.translated_results.get(image_path)

        for i, block in enumerate(ocr_data['blocks']):
            translated_block = None
            if translated_data and i < len(translated_data['blocks']):
                translated_block = translated_data['blocks'][i]

            # Create clickable text block item
            text_block_item = TextBlockItem(block, translated_block, W, H)

            # Set initial display state based on current display mode
            if display_mode == 'Hidden':
                text_block_item.setVisible(False)
            else:
                text_block_item.setVisible(True)
                if display_mode == 'Original':
                    text_block_item.showing_original = True
                    text_block_item.update_text_display(show_original=True)
                elif display_mode == 'Translation':
                    text_block_item.showing_original = False
                    text_block_item.update_text_display(show_original=False)
                elif display_mode == 'Both':
                    # Both mode - blocks are clickable to toggle
                    text_block_item.showing_original = True  # Default to original
                    text_block_item.update_text_display(show_original=True)

            text_block_item.setPos(block['box'][0], block['box'][1])  # x, y of OCR box
            self.scene.addItem(text_block_item)
            self.text_blocks.append(text_block_item)



    def translate_current_page(self):
        """Translate the current page"""
        if not self.image_paths or self.current_page_idx >= len(self.image_paths):
            return

        image_path = self.image_paths[self.current_page_idx]
        if image_path not in self.ocr_results:
            self.status_bar.showMessage("No OCR results available for translation")
            return

        ocr_data = self.ocr_results[image_path]

        # Start translation in background thread
        self.progress_bar.setVisible(True)
        self.translate_btn.setEnabled(False)

        self.translation_worker = TranslationWorker(
            ocr_data['blocks'], self.source_lang, self.target_lang
        )
        self.translation_worker.progress.connect(self.on_translation_progress)
        self.translation_worker.finished.connect(self.on_translation_finished)
        self.translation_worker.error.connect(self.on_translation_error)
        self.translation_worker.start()

    def on_translation_progress(self, percentage: int, message: str):
        """Handle translation progress updates"""
        self.progress_bar.setValue(percentage)
        self.status_bar.showMessage(message)

    def on_translation_finished(self, result: Dict):
        """Handle translation completion"""
        self.progress_bar.setVisible(False)
        self.translate_btn.setEnabled(True)

        image_path = self.image_paths[self.current_page_idx]
        self.translated_results[image_path] = result

        # Save translation results to JSON file
        results_dir = self.current_volume_path.parent / '_ocr' / self.current_volume_path.name
        translation_json_path = results_dir / f"{image_path.relative_to(self.current_volume_path).with_suffix('')}_{self.source_lang}_{self.target_lang}.json"

        try:
            results_dir.mkdir(parents=True, exist_ok=True)
            with open(translation_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save translation results: {e}")

        # Update existing text blocks with translated data
        for i, text_block in enumerate(self.text_blocks):
            if i < len(result['blocks']):
                text_block.translated_data = result['blocks'][i]
                # If currently showing translation, update display
                if not text_block.showing_original:
                    text_block.update_text_display(show_original=False)

        self.status_bar.showMessage("Translation complete")

    def on_translation_error(self, error_msg: str):
        """Handle translation errors"""
        self.progress_bar.setVisible(False)
        self.translate_btn.setEnabled(True)
        self.translate_all_btn.setEnabled(True)
        self.status_bar.showMessage(f"Translation failed: {error_msg}")

    def translate_all_pages(self):
        """Translate all pages in the current volume"""
        if not self.image_paths:
            return

        # Check if batch translation is already running
        if hasattr(self, 'batch_translation_worker') and self.batch_translation_worker and self.batch_translation_worker.isRunning():
            self.status_bar.showMessage("Batch translation already running")
            return

        # Start batch translation
        self.progress_bar.setVisible(True)
        self.translate_btn.setEnabled(False)
        self.translate_all_btn.setEnabled(False)
        self.translate_all_btn.setText("Translating...")

        self.batch_translation_worker = BatchTranslationWorker(
            self.image_paths, self.ocr_results, self.source_lang, self.target_lang, self.current_volume_path
        )
        self.batch_translation_worker.overall_progress.connect(self.on_batch_overall_progress)
        self.batch_translation_worker.page_progress.connect(self.on_batch_page_progress)
        self.batch_translation_worker.page_finished.connect(self.on_batch_page_finished)
        self.batch_translation_worker.finished.connect(self.on_batch_finished)
        self.batch_translation_worker.error.connect(self.on_batch_error)
        self.batch_translation_worker.start()

    def on_batch_overall_progress(self, percentage: int, message: str):
        """Handle overall batch translation progress"""
        self.progress_bar.setValue(percentage)
        self.status_bar.showMessage(message)

    def on_batch_page_progress(self, page_idx: int, message: str, image_path: Path):
        """Handle individual page progress during batch translation"""
        # Update current page display if we're viewing the page being translated
        if page_idx == self.current_page_idx:
            self.status_bar.showMessage(f"Translating current page: {message}")

    def on_batch_page_finished(self, result: Dict, image_path: Path):
        """Handle completion of individual page translation"""
        self.translated_results[image_path] = result

        # Save translation results to JSON file
        results_dir = self.current_volume_path.parent / '_ocr' / self.current_volume_path.name
        translation_json_path = results_dir / f"{image_path.relative_to(self.current_volume_path).with_suffix('')}_{self.source_lang}_{self.target_lang}.json"

        try:
            results_dir.mkdir(parents=True, exist_ok=True)
            with open(translation_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save translation results: {e}")

        # Update display if this is the current page
        page_idx = self.image_paths.index(image_path)
        if page_idx == self.current_page_idx:
            # Refresh the current page display with new translations
            self.display_text_blocks(self.ocr_results[image_path])

    def on_batch_finished(self):
        """Handle completion of batch translation"""
        self.progress_bar.setVisible(False)
        self.translate_btn.setEnabled(True)
        self.translate_all_btn.setEnabled(True)
        self.translate_all_btn.setText("Translate All")
        self.status_bar.showMessage("Batch translation complete")

    def on_batch_error(self, error_msg: str):
        """Handle batch translation errors"""
        self.progress_bar.setVisible(False)
        self.translate_btn.setEnabled(True)
        self.translate_all_btn.setEnabled(True)
        self.translate_all_btn.setText("Translate All")
        self.status_bar.showMessage(f"Batch translation failed: {error_msg}")

    def on_source_lang_changed(self, lang: str):
        """Handle source language change"""
        self.source_lang = lang
        self.save_translation_config()

    def on_target_lang_changed(self, lang: str):
        """Handle target language change"""
        self.target_lang = lang
        self.save_translation_config()

    def on_ocr_engine_changed(self, engine: str):
        """Handle OCR engine change"""
        self.ocr_engine = engine
        self.save_translation_config()

    def process_volume_with_ocr(self):
        """Process the current volume with OCR using mokuro CLI"""
        if not self.current_volume_path:
            self.status_bar.showMessage("No volume loaded")
            return

        # Check if OCR is already running
        if hasattr(self, 'ocr_worker') and self.ocr_worker and self.ocr_worker.isRunning():
            self.status_bar.showMessage("OCR processing already running")
            return

        # Disable the OCR action during processing
        self.ocr_action.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("Starting OCR processing...")

        # Start OCR processing in background thread
        self.ocr_worker = OCRWorker(self.current_volume_path, self.ocr_engine)
        self.ocr_worker.progress.connect(self.on_ocr_progress)
        self.ocr_worker.finished.connect(self.on_ocr_finished)
        self.ocr_worker.error.connect(self.on_ocr_error)
        self.ocr_worker.start()

    def on_ocr_progress(self, percentage: int, message: str):
        """Handle OCR progress updates"""
        if percentage >= 0:
            self.progress_bar.setValue(percentage)
        self.status_bar.showMessage(message)

    def on_ocr_finished(self):
        """Handle OCR completion"""
        self.progress_bar.setVisible(False)
        self.ocr_action.setEnabled(True)
        self.status_bar.showMessage("OCR processing complete")
        # Refresh the current page to load new OCR data
        self.load_page(self.current_page_idx)

    def on_ocr_error(self, error_msg: str):
        """Handle OCR errors"""
        self.progress_bar.setVisible(False)
        self.ocr_action.setEnabled(True)
        self.status_bar.showMessage(f"OCR processing failed: {error_msg}")

    def on_display_mode_changed(self, mode: str):
        """Handle display mode change"""
        self.current_display_mode = mode  # Store the current display mode

        # Update visibility of all text blocks based on display mode
        for text_block in self.text_blocks:
            if mode == 'Hidden':
                text_block.setVisible(False)
            else:
                text_block.setVisible(True)
                if mode == 'Original':
                    text_block.showing_original = True
                    text_block.update_text_display(show_original=True)
                elif mode == 'Translation':
                    text_block.showing_original = False
                    text_block.update_text_display(show_original=False)
                elif mode == 'Both':
                    # Both mode - blocks are clickable to toggle
                    text_block.showing_original = True  # Default to original
                    text_block.update_text_display(show_original=True)

    def prev_page(self):
        """Go to previous page"""
        if self.current_page_idx > 0:
            self.current_page_idx -= 1
            self.load_page(self.current_page_idx)
            self.update_navigation_buttons()

    def next_page(self):
        """Go to next page"""
        if self.current_page_idx < len(self.image_paths) - 1:
            self.current_page_idx += 1
            self.load_page(self.current_page_idx)
            self.update_navigation_buttons()

    def update_navigation_buttons(self):
        """Update navigation button states"""
        self.prev_btn.setEnabled(self.current_page_idx > 0)
        self.next_btn.setEnabled(self.current_page_idx < len(self.image_paths) - 1)


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Mokuro GUI")
    app.setApplicationVersion("1.0.0")

    # Set application style
    app.setStyle('Fusion')

    window = MokuroGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

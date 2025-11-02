import cv2
import numpy as np
from PIL import Image
from loguru import logger
from scipy.signal.windows import gaussian

from comic_text_detector.inference import TextDetector
from mokuro import __version__
from mokuro.cache import cache
from mokuro.utils import imread

# Conditional imports
try:
    from manga_ocr import MangaOcr
    MANGA_OCR_AVAILABLE = True
except ImportError:
    MANGA_OCR_AVAILABLE = False
    logger.warning("manga_ocr not available - import failed")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("easyocr not available - import failed")

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    logger.warning("trocr not available - import failed")


class MangaPageOcr:
    def __init__(self,
                 pretrained_model_name_or_path='kha-white/manga-ocr-base',
                 force_cpu=False,
                 detector_input_size=1024,
                 text_height=64,
                 max_ratio_vert=16,
                 max_ratio_hor=8,
                 anchor_window=2,
                 ocr_engine='manga-ocr',
                 ):

        self.text_height = text_height
        self.max_ratio_vert = max_ratio_vert
        self.max_ratio_hor = max_ratio_hor
        self.anchor_window = anchor_window
        self.ocr_engine = ocr_engine

        logger.info('Initializing text detector')
        self.text_detector = TextDetector(model_path=cache.comic_text_detector, input_size=detector_input_size,
                                          device='cuda' if not force_cpu else 'cpu',
                                          act='leaky')

        # Automatic fallback chain: manga-ocr -> easyocr -> trocr
        if ocr_engine == 'manga-ocr':
            if not MANGA_OCR_AVAILABLE:
                logger.warning("MangaOCR not available, trying EasyOCR...")
                ocr_engine = 'easyocr'
                self.ocr_engine = ocr_engine
            else:
                logger.info('Initializing MangaOCR')
                try:
                    self.ocr = MangaOcr(pretrained_model_name_or_path, force_cpu)
                except Exception as e:
                    logger.warning(f"MangaOCR failed: {e}")
                    logger.warning("Trying EasyOCR...")
                    ocr_engine = 'easyocr'
                    self.ocr_engine = ocr_engine

        if ocr_engine == 'easyocr':
            if not EASYOCR_AVAILABLE:
                logger.warning("EasyOCR not available, trying TrOCR...")
                ocr_engine = 'trocr'
                self.ocr_engine = ocr_engine
            else:
                logger.info('Initializing EasyOCR')
                try:
                    self.ocr = easyocr.Reader(['ja', 'en'], gpu=not force_cpu)
                except Exception as e:
                    logger.warning(f"EasyOCR failed: {e}")
                    logger.warning("Trying TrOCR...")
                    ocr_engine = 'trocr'
                    self.ocr_engine = ocr_engine

        if ocr_engine == 'trocr':
            if not TROCR_AVAILABLE:
                raise RuntimeError("No OCR engines available! Please install at least one of: manga-ocr, easyocr, or transformers")
            logger.info('Initializing TrOCR (Transformer-based OCR)')
            try:
                device = 'cpu' if force_cpu else 'cuda'
                self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
                self.ocr = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
                self.ocr.to(device)
            except Exception as e:
                raise RuntimeError(f"TrOCR failed to initialize: {e}. No working OCR engines available.")

        logger.info(f"Using OCR engine: {self.ocr_engine}")

    def __call__(self, img_path):
        img = imread(img_path)
        H, W, *_ = img.shape
        mask, mask_refined, blk_list = self.text_detector(img, refine_mode=1, keep_undetected_mask=True)

        result = {'version': __version__, 'img_width': W, 'img_height': H, 'blocks': []}

        for blk_idx, blk in enumerate(blk_list):

            result_blk = {
                'box': list(blk.xyxy),
                'vertical': blk.vertical,
                'font_size': blk.font_size,
                'lines_coords': [],
                'lines': []
            }

            for line_idx, line in enumerate(blk.lines_array()):
                if blk.vertical:
                    max_ratio = self.max_ratio_vert
                else:
                    max_ratio = self.max_ratio_hor

                line_crops, cut_points = self.split_into_chunks(
                    img, mask_refined, blk, line_idx,
                    textheight=self.text_height, max_ratio=max_ratio, anchor_window=self.anchor_window)

                line_text = ''
                for line_crop in line_crops:
                    if blk.vertical:
                        line_crop = cv2.rotate(line_crop, cv2.ROTATE_90_CLOCKWISE)

                    if self.ocr_engine == 'manga-ocr':
                        line_text += self.ocr(Image.fromarray(line_crop))
                    elif self.ocr_engine == 'easyocr':
                        # EasyOCR expects RGB, convert from BGR if needed
                        if line_crop.shape[2] == 3:  # BGR to RGB
                            line_crop_rgb = cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB)
                        else:
                            line_crop_rgb = line_crop
                        results = self.ocr.readtext(line_crop_rgb, detail=0)
                        line_text += ' '.join(results) if results else ''
                    elif self.ocr_engine == 'trocr':
                        # TrOCR expects RGB PIL Image
                        if line_crop.shape[2] == 3:  # BGR to RGB
                            line_crop_rgb = cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB)
                        else:
                            line_crop_rgb = line_crop
                        pil_image = Image.fromarray(line_crop_rgb)

                        # Process image and generate text
                        pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
                        device = next(self.ocr.parameters()).device
                        pixel_values = pixel_values.to(device)

                        generated_ids = self.ocr.generate(pixel_values, max_length=50, num_beams=4)
                        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        line_text += generated_text
                
                result_blk['lines_coords'].append(line.tolist())
                result_blk['lines'].append(line_text)

            result['blocks'].append(result_blk)

        return result

    @staticmethod
    def split_into_chunks(img, mask_refined, blk, line_idx, textheight, max_ratio=16, anchor_window=2):
        line_crop = blk.get_transformed_region(img, line_idx, textheight)

        h, w, *_ = line_crop.shape
        ratio = w / h

        if ratio <= max_ratio:
            return [line_crop], []

        else:
            k = gaussian(textheight * 2, textheight / 8)

            line_mask = blk.get_transformed_region(mask_refined, line_idx, textheight)
            num_chunks = int(np.ceil(ratio / max_ratio))

            anchors = np.linspace(0, w, num_chunks + 1)[1:-1]

            line_density = line_mask.sum(axis=0)
            line_density = np.convolve(line_density, k, 'same')
            line_density /= line_density.max()

            anchor_window *= textheight

            cut_points = []
            for anchor in anchors:
                anchor = int(anchor)

                n0 = np.clip(anchor - anchor_window // 2, 0, w)
                n1 = np.clip(anchor + anchor_window // 2, 0, w)

                p = line_density[n0:n1].argmin()
                p += n0

                cut_points.append(p)

            return np.split(line_crop, cut_points, axis=1), cut_points

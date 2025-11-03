import cv2
import numpy as np
from PIL import Image
from loguru import logger
from scipy.signal.windows import gaussian
import torch
import os
from transformers import AutoModel, AutoTokenizer

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

try:
    from transformers import AutoModel
    from transformers import AutoProcessor, AutoModelForVision2Seq
    DEEPSEEK_OCR_AVAILABLE = True
except ImportError:
    DEEPSEEK_OCR_AVAILABLE = False
    logger.warning("deepseek-ocr not available - import failed")


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

        # Automatic fallback chain: manga-ocr -> easyocr -> trocr -> deepseek-ocr
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
                logger.warning("TrOCR not available, trying DeepSeek-OCR...")
                ocr_engine = 'deepseek-ocr'
                self.ocr_engine = ocr_engine
            else:
                logger.info('Initializing TrOCR (Transformer-based OCR)')
                try:
                    device = 'cpu' if force_cpu else 'cuda'
                    self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
                    self.ocr = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
                    self.ocr.to(device)
                except Exception as e:
                    logger.warning(f"TrOCR failed: {e}")
                    logger.warning("Trying DeepSeek-OCR...")
                    ocr_engine = 'deepseek-ocr'
                    self.ocr_engine = ocr_engine

        if ocr_engine == 'deepseek-ocr':
            if not DEEPSEEK_OCR_AVAILABLE:
                raise RuntimeError("No OCR engines available! Please install at least one of: manga-ocr, easyocr, transformers, or deepseek-ocr support")
            logger.info('Initializing DeepSeek-OCR')
            try:
                device = 'cpu' if force_cpu else 'cuda'
                os.environ["CUDA_VISIBLE_DEVICES"] = '0' if device == 'cuda' else ''
                self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-ocr", trust_remote_code=True)
                self.ocr = AutoModel.from_pretrained("deepseek-ai/deepseek-ocr", trust_remote_code=True, use_safetensors=True)
                self.ocr = self.ocr.eval().to(device).to(torch.bfloat16 if device == 'cuda' else torch.float32)
            except Exception as e:
                raise RuntimeError(f"DeepSeek-OCR failed to initialize: {e}. No working OCR engines available.")

        logger.info(f"Using OCR engine: {self.ocr_engine}")

    def __call__(self, img_path):
        img = imread(img_path)
        H, W, *_ = img.shape
        mask, mask_refined, blk_list = self.text_detector(img, refine_mode=1, keep_undetected_mask=True)

        result = {'version': __version__, 'img_width': W, 'img_height': H, 'blocks': []}

        for blk_idx, blk in enumerate(blk_list):

            result_blk = {
                'box': [int(x) for x in blk.xyxy],  # Convert numpy int32 to Python int
                'vertical': bool(blk.vertical),     # Convert numpy bool to Python bool
                'font_size': int(blk.font_size),    # Convert numpy int to Python int
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

                    logger.debug(f"Processing line crop with shape: {line_crop.shape}, engine: {self.ocr_engine}")
                    logger.debug(f"Line text before OCR: '{line_text}' (length: {len(line_text)})")
                    try:
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
                            logger.debug(f"TrOCR extracted text: '{generated_text}'")
                        elif self.ocr_engine == 'deepseek-ocr':
                            # DeepSeek-OCR uses custom infer method
                            if line_crop.shape[2] == 3:  # BGR to RGB
                                line_crop_rgb = cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB)
                            else:
                                line_crop_rgb = line_crop
                            pil_image = Image.fromarray(line_crop_rgb)

                            # Save temp image for DeepSeek-OCR
                            import tempfile
                            import os
                            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                                pil_image.save(tmp_file.name, 'JPEG')
                                temp_image_path = tmp_file.name

                            try:
                                # Use DeepSeek-OCR infer method
                                prompt = "<image>\n<|grounding|>Convert the document to markdown."
                                with tempfile.TemporaryDirectory() as tmp_dir:
                                    ocr_result = self.ocr.infer(
                                        self.tokenizer,
                                        prompt=prompt,
                                        image_file=temp_image_path,
                                        output_path=tmp_dir,
                                        base_size=1024,
                                        image_size=640,
                                        crop_mode=True,
                                        save_results=False,
                                        test_compress=False
                                    )
                                    # Debug: Print the DeepSeek result structure
                                    logger.info(f"DeepSeek OCR result type: {type(ocr_result)}")
                                    logger.info(f"DeepSeek OCR result keys: {list(ocr_result.keys()) if isinstance(ocr_result, dict) else 'Not a dict'}")
                                    if isinstance(ocr_result, dict):
                                        for key, value in ocr_result.items():
                                            logger.info(f"DeepSeek result[{key}]: type={type(value)}, value={repr(value)[:200]}...")

                                    # Extract text from result - ensure we only get string data
                                    if ocr_result:
                                        if 'markdown' in ocr_result:
                                            text_content = ocr_result['markdown']
                                            if isinstance(text_content, str):
                                                line_text += text_content.strip()
                                            else:
                                                # Convert numpy arrays or other types to string
                                                line_text += str(text_content).strip()
                                        elif 'text' in ocr_result:
                                            text_content = ocr_result['text']
                                            if isinstance(text_content, str):
                                                line_text += text_content.strip()
                                            else:
                                                # Convert numpy arrays or other types to string
                                                line_text += str(text_content).strip()
                            finally:
                                # Clean up temp file
                                try:
                                    os.unlink(temp_image_path)
                                except:
                                    pass
                    except Exception as e:
                        logger.error(f"OCR failed for line crop (engine: {self.ocr_engine}): {e}")
                        logger.error(f"Line crop shape: {line_crop.shape if 'line_crop' in locals() else 'unknown'}")
                        # Continue with empty text for this chunk
                        continue
                
                result_blk['lines_coords'].append([[int(coord) for coord in point] for point in line])
                result_blk['lines'].append(line_text)
                logger.debug(f"Final line text for line {line_idx}: '{line_text}'")

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

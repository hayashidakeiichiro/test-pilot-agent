from browser_use import Controller, ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.dom.views import DOMBaseNode, DOMElementNode, DOMTextNode
from browser_use.dom.service import DomService
from Levenshtein import distance
from typing import List, Tuple, Optional
import asyncio
import datetime
import json
import os
import base64
import re
import base64
import io
from dotenv import load_dotenv
from PIL import Image
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from browser_use import ActionResult, Agent, Browser, BrowserConfig, Controller
from langchain_core.language_models.chat_models import BaseChatModel
import re
import functools
from langchain_core.prompts import PromptTemplate
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage, BaseMessageChunk
from itertools import islice

from PIL import Image, ImageDraw, ImageFont
from math import ceil


from PIL import Image, ImageDraw, ImageFont
from typing import Dict
import io
import base64
from math import ceil
logger = logging.getLogger(__name__)
load_dotenv()
# スクショ保存ディレクトリ
LOG_DIR = "logs"
IMAGE_DIR = os.path.join(LOG_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# LLMログファイル
LLM_LOG_FILE = os.path.join(LOG_DIR, "llm_log.txt")


def save_image_from_base64(data_uri):
    """Base64エンコードされた画像を保存"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    try:
        base64_data = data_uri.split(",")[1] if "," in data_uri else data_uri
        image_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_data))
        image_filename = os.path.join(IMAGE_DIR, f"image_{timestamp}.png")
        image.save(image_filename)
        print(f"Image saved: {image_filename}")
        return image_filename
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def draw_index_label(draw, x, y, index, font, bg_color="black", text_color="white", padding=4):
    text = f"{index}"

    # textsize の代わりに textbbox を使用
    bbox = draw.textbbox((x, y), text, font=font)  # (left, top, right, bottom)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    box_width = text_width + padding * 2
    box_height = text_height + padding * 2

    # 背景の四角形を描画
    draw.rectangle(
        [x, y, x + box_width, y + box_height],
        fill=bg_color
    )

    # テキストの描画
    draw.text(
        (x + padding, y + padding),
        text,
        fill=text_color,
        font=font
    )


def create_canvas_from_crops(crops, items_per_row=5, thumb_size=(100, 100)):
    if not crops:
        return Image.new("RGB", thumb_size, "white")

    try:
        rows = ceil(len(crops) / items_per_row)
        canvas_width = thumb_size[0] * items_per_row
        canvas_height = thumb_size[1] * rows
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("arial.ttf", 148)
        except IOError:
            font = ImageFont.load_default()

        for i, (index, crop_img) in enumerate(sorted(crops.items())):
            try:
                x = (i % items_per_row) * thumb_size[0]
                y = (i // items_per_row) * thumb_size[1]
                resized_crop = crop_img.resize(thumb_size)
                canvas.paste(resized_crop, (x, y))

                # 改善されたインデックス描画
                draw_index_label(draw, x, y, index, font)
            except Exception as e:
                print(f"Warning: Skipping index {index} due to error: {e}")
                continue

        return canvas

    except Exception as e:
        print(f"Failed to create canvas: {e}")
        return Image.new("RGB", thumb_size, "white")
    
# --- スクリーンショット＋DOMノードからサムネイル画像生成 ---
async def generate_selector_thumbnail_grid_with_scaling(
    screenshot_bytes: bytes,
    selector_map: Dict[int, DOMElementNode],
    thumb_size=(100, 100),
    items_per_row=5,
    browser: BrowserContext = None,
) -> Image.Image:
    screenshot_image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
    crops = {}
    img_width, img_height = screenshot_image.size
    scale_x = 1
    scale_y = 1
    page = await browser.get_agent_current_page()

    for idx, node in selector_map.items():
        try:
            xpath = node.xpath
            if not xpath:
                continue

            result = await page.evaluate('''
                (xpath) => {
                const result = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                const element = result.singleNodeValue;
                if (!element) return null;
                const rect = element.getBoundingClientRect();
                return {
                    x: rect.x,
                    y: rect.y,
                    width: rect.width,
                    height: rect.height
                };
                }
                ''', xpath)

            # 取得に失敗したか、数値が0または負で無効な場合をスキップ
            if not result or result["width"] <= 0 or result["height"] <= 0:
                logger.info(f"Skipping idx {idx}: invalid rect: {result}")
                continue

            # JavaScriptで得た絶対座標を画像スケーリングに合わせて変換
            left = int(max(0, min(result["x"] * scale_x, img_width - 1)))
            top = int(max(0, min(result["y"] * scale_y, img_height - 1)))
            right = int(max(left + 1, min((result["x"] + result["width"]) * scale_x, img_width)))
            bottom = int(max(top + 1, min((result["y"] + result["height"]) * scale_y, img_height)))

            if right <= left or bottom <= top:
                logger.info(f"Skipping idx {idx}: zero-sized crop area.")
                continue

            cropped = screenshot_image.crop((left, top, right, bottom))
            crops[idx] = cropped
        except Exception as e:
            logger.error(f"Error cropping idx {idx}: {e}")
            continue
    return create_canvas_from_crops(
        crops=crops,
        items_per_row=items_per_row,
        thumb_size=thumb_size
    )

def extract_none_text_selector_map(selector_map: Dict[int, DOMElementNode]) -> Dict[int, DOMElementNode]:
    none_text_selector_map = {}
    for idx, node in selector_map.items():
        is_image_buton = False
        for child in node.children:
            if isinstance(child, DOMElementNode) and (child.tag_name == 'img' or child.tag_name == 'svg'):
                is_image_buton = True
                break
        if node.get_all_text_till_next_clickable_element(include_attr_for_img=False) and not is_image_buton:
            continue
        none_text_selector_map[idx] = node
    return none_text_selector_map

from io import BytesIO
from PIL import ImageFile, Image
MAX_PIXELS = 178_956_970
# Image.MAX_IMAGE_PIXELS = None

def compress_if_too_large(image_bytes: bytes, max_pixels=178_956_970) -> bytes:
    parser = ImageFile.Parser()
    parser.feed(image_bytes)
    img = parser.close()

    width, height = img.size
    pixel_count = width * height

    if pixel_count <= max_pixels:
        return image_bytes

    print(f"[⚠️] Image is too large: {pixel_count} pixels. Resizing...")

    scale = (max_pixels / pixel_count) ** 0.5
    new_size = (int(width * scale), int(height * scale))

    img = img.resize(new_size, Image.LANCZOS)
    output = BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()
# --- Base64エンコードしてData URL形式で返す ---
async def generate_selector_thumbnail_grid_base64(
    screenshot: str,
    selector_map: Dict[int, DOMElementNode],
    thumb_size=(100, 100),
    items_per_row=5,
    browser: BrowserContext = None,
) -> str:
    print("Generating thumbnail grid...")
    # base64文字列からヘッダー(data:image/png;base64,...)を取り除く
    save_image_from_base64(screenshot)
    base64_data = re.sub("^data:image/[^;]+;base64,", "", screenshot)
    screenshot_bytes = base64.b64decode(base64_data)
    screenshot_bytes = compress_if_too_large(screenshot_bytes)
    canvas = await generate_selector_thumbnail_grid_with_scaling(
        screenshot_bytes,
        selector_map,
        thumb_size,
        items_per_row,
        browser
    )
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_url = f"data:image/png;base64,{encoded_image}"
    save_image_from_base64(data_url)
    return data_url


def attach_generate_icon_list_image(controller: Controller):
    @controller.action("Find target element by context_block and target")
    async def generate_icon_list_image(
        browser: BrowserContext,
    ):
        page = await browser.get_agent_current_page()
        dom_service = DomService(page)
        # スクリーンショット取得（bytes）
        screenshot = await browser.take_screenshot(full_page=True)


        # clickable elements の取得
        content = await dom_service.get_clickable_elements(
            focus_element=-1,
            viewport_expansion=-1,
            highlight_elements=True,
        )
        selector_map = extract_none_text_selector_map(content.selector_map)
    
        # Base64形式のサムネイルグリッド画像を生成
        data_url = await generate_selector_thumbnail_grid_base64(
            screenshot=screenshot,
            selector_map=selector_map,
            thumb_size=(100, 100),
            items_per_row=5,
            browser=browser
        )
        return ActionResult(extracted_content="ok")

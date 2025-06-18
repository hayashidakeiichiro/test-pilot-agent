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
from PIL import Image
from examples.custom_functions.generate_icon_list_image import generate_selector_thumbnail_grid_base64, extract_none_text_selector_map

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

# Content object representing a segment candidate (paper's Co)
@dataclass
class ContentObject:
    element: DOMElementNode
    geometry: Tuple[float, float, float, float]
    category: str
    children: List['ContentObject'] = field(default_factory=list)

# Logic object representing a final block (paper's Lo)
class LogicObject:
    include_attributes: list[str] = [
		'title',
		'type',
		'name',
		'role',
		'tabindex',
		'aria-label',
		'placeholder',
		'value',
		'alt',
		'aria-expanded',
		# test-pilot-img-attribute
		# 'src',
		# 'class',
		# test-pilot
	]
    def element(self) -> Optional[DOMElementNode]:
        """Representative DOM element of this logic block (first content object)."""
        return self.content_objects[0].element if self.content_objects else None

    def __init__(self,
                 content_objects: Optional[List[ContentObject]] = None,
                 geometry: Optional[Tuple[float, float, float, float]] = None,
                 label: Optional[str] = None,
                 children: Optional[List['LogicObject']] = None,
                 follows: Optional['LogicObject'] = None):
        self.content_objects = content_objects or []
        self.geometry = geometry or (0.0, 0.0, 0.0, 0.0)
        self.label = label
        self.children = children or []
        self.follows = follows
    def get_clickable_elements_text(self) -> str:
        """Get text of all clickable elements in this logic object."""
        texts = []
        for co in self.content_objects:
            if isinstance(co.element, DOMElementNode):
                texts.append(co.element.clickable_elements_to_string(include_attributes=self.include_attributes) or "")
        return " ".join(texts).strip()


    def summarize_for_ai(self, text_snippet_length: int = 100) -> str:
        co_texts = []
        for co in self.content_objects:
            if isinstance(co.element, DOMElementNode):
                text = co.element.clickable_elements_to_string(include_attributes=getattr(self, 'include_attributes', None)) or ""
                co_texts.append(text.strip())

        max_co_count = 5  # AIに渡す最大の co 数

        if len(co_texts) > max_co_count:
            head_count = 1
            middle_count = 3
            tail_count = 1

            # 先頭
            head = co_texts[:head_count]

            # 中央
            mid_start = max(0, len(co_texts) // 2 - middle_count // 2)
            mid_end = mid_start + middle_count
            middle = co_texts[mid_start:mid_end]

            # 末尾
            tail = co_texts[-tail_count:]

            selected = []
            for text in head + middle + tail:
                if len(text) > text_snippet_length:
                    snippet = text[:text_snippet_length//2] + "..." + text[-text_snippet_length//2:]
                else:
                    snippet = text
                selected.append(snippet)
        else:
            # 少ない場合は全部使うが長さ制限は緩める（例：最大 text_snippet_length * 5）
            max_length = text_snippet_length * max_co_count
            snippet_part = max_length // 3 

            selected = []
            for text in co_texts:
                if len(text) > max_length:
                    # 中央部分の範囲計算
                    mid_start = max(0, (len(text) // 2) - (snippet_part // 2))
                    mid_end = mid_start + snippet_part

                    snippet = (
                        text[:snippet_part] + "..."
                        + text[mid_start:mid_end] + "..."
                        + text[-snippet_part:]
                    )
                else:
                    snippet = text

                selected.append(snippet)

        balanced_summary = " | ".join(selected)

        # Geometry info
        x, y, w, h = self.geometry

        return (
            f"Position: x: {x:.2f}, y: {y:.2f}, width: {w:.2f}, height: {h:.2f}\n"
            f"Sampled Texts: \"{balanced_summary}\""
        )

    # Diagonal of this Lo
    def diagonal(self) -> float:
        _, _, w, h = self.geometry
        return (w**2 + h**2)**0.5
    # rDiagonal relative to parent Lo
    def rDiagonal(self, parent: 'LogicObject') -> float:
        return self.diagonal() / max(1.0, parent.diagonal())
    # Distance to other Lo
    def distance(self, other: 'LogicObject') -> float:
        x1, y1, w1, h1 = self.geometry
        x2, y2, w2, h2 = other.geometry
        cx1, cy1 = x1 + w1/2, y1 + h1/2
        cx2, cy2 = x2 + w2/2, y2 + h2/2
        return ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5
    # Alignment check (horizontal or vertical)
    def aligned(self, other: 'LogicObject') -> bool:
        x1, y1, w1, h1 = self.geometry
        x2, y2, w2, h2 = other.geometry
        return abs(y1 - y2) < 10 or abs(x1 - x2) < 10
    # Visual cues present: placeholder using CSS class
    def visualCuesPresent(self) -> bool:
        # for co in self.content_objects:
        #     if not isinstance(co.element, DOMElementNode):
        #         print('Warning: co.element is not a DOMElementNode:', type(co.element))
        #         continue
        #     if co.element.attributes.get('class'):
        #         return True
        return False
    # Accept as leaf
    def accept(self):
        self.children = []
    # Merge with other Lo
    def mergeWith(self, other: 'LogicObject'):
        x1, y1, w1, h1 = self.geometry
        x2, y2, w2, h2 = other.geometry
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1+w1, x2+w2)
        y_max = max(y1+h1, y2+h2)
        self.geometry = (x_min, y_min, x_max-x_min, y_max-y_min)
        self.content_objects.extend(other.content_objects)
        # adopt other's children? simplistic: clear others
        self.children = []

# Utility functions

async def get_geometry(node: DOMElementNode, page) -> Tuple[float, float, float, float]:
    # Use viewport or page coordinates
    xpath = node.xpath
    if xpath:
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
        if result is None:
            # 要素が見つからない場合は (0, 0, 0, 0) など適当なデフォルトにする
            return (0, 0, 0, 0)
        return (result.get("x", 0), result.get("y", 0), result.get("width", 0), result.get("height", 0))
    return (0, 0, 0, 0)

CATEGORY_MAP = {
    # Simplified mapping
    'header': 'sections', 'nav': 'grouping', 'section': 'sections',
    'article': 'sections', 'div': 'grouping', 'p': 'text-level', 'a': 'links',
    'img': 'embedded', 'footer': 'sections'
}

def get_category(node: DOMElementNode) -> str:
    return CATEGORY_MAP.get(node.tag_name.lower(), 'sections')

# Phase 1: Page Analysis -> build Content Structure (d2c)

async def d2c(node: DOMElementNode, page) -> Optional[ContentObject]:
    # If only one child, descend
    if node.tag_name.lower() == "iframe":
        return None
    if len(node.children) == 1 and isinstance(node.children[0], DOMElementNode):
        return  await d2c(node.children[0], page)
    # Check visibility / validity
    geom =  await get_geometry(node, page)
    # A simple valid check: positive area and in viewport
    if node.xpath != "/body" and (geom[2] <= 0 or geom[3] <= 0):
        return None
    co = ContentObject(
        element=node,
        geometry=geom,
        category=get_category(node)
    )
    for child in node.children:
        if isinstance(child, DOMElementNode):
            child_co = await d2c(child, page)
            if child_co:
                co.children.append(child_co)
    return co

# Phase 2: Page Understanding -> build Logic Structure (c2l)

def diagonal(geom: Tuple[float, float, float, float]) -> float:
    # Calculate diagonal length of bbox
    _, _, w, h = geom
    return (w**2 + h**2) ** 0.5

def relative_diagonal(lo: LogicObject, parent: LogicObject) -> float:
    return diagonal(lo.geometry) / max(1.0, diagonal(parent.geometry))
def getLabelFromContent(lo: LogicObject) -> str:
    if lo.content_objects and hasattr(lo.content_objects[0].element, 'tag_name'):
        tag = lo.content_objects[0].element.tag_name.lower()
        if tag in ('header','nav','section','article','footer'):
            return tag
    return 'none'
def c2l(co: ContentObject, parent: Optional[LogicObject], pG: float, epsilon: float = 50) -> LogicObject:
    lo = LogicObject(content_objects=[co], geometry=co.geometry)
    for child_co in co.children:
        child_lo = c2l(child_co, lo, pG)
        lo.children.append(child_lo)
    if parent is None or lo.rDiagonal(parent) >= pG:
        lo.label = getLabelFromContent(lo)
    else:
        if lo.visualCuesPresent():
            lo.accept()
            lo.label = getLabelFromContent(lo)
        else:
            i = 0
            while i < len(lo.children):
                j = i + 1
                while j < len(lo.children):
                    c1, c2 = lo.children[i], lo.children[j]
                    if c1.distance(c2) < epsilon and c1.aligned(c2):
                        c1.mergeWith(c2)
                        lo.children.pop(j)
                        continue
                    j += 1
                i += 1
            for child in lo.children:
                c2l(child, lo, pG)
            if all(child.rDiagonal(lo) < pG for child in lo.children):
                lo.accept()
                lo.label = getLabelFromContent(lo)
    return lo
def extract_large_sections_relative_auto(
    lo: LogicObject,
    viewport_width: float,
    viewport_height: float,
    min_area_ratio_start: float = 0.5,
    min_area_ratio_step: float = 0.1,
    target_min: int = 5,
    target_max: int = 15,
    max_iterations: int = 5
) -> List[LogicObject]:
    current_area_ratio = min_area_ratio_start
    iterations = 0

    while iterations < max_iterations:
        candidates = _extract_large_sections_relative(
            lo,
            viewport_width,
            viewport_height,
            current_area_ratio
        )

        if target_min <= len(candidates) <= target_max:
            return candidates

        if len(candidates) > target_max:
            current_area_ratio = min(0.9, current_area_ratio + min_area_ratio_step)
        else:
            current_area_ratio = max(0.1, current_area_ratio - min_area_ratio_step)

        iterations += 1

    return candidates


def _extract_large_sections_relative(
    lo: LogicObject,
    viewport_width: float,
    viewport_height: float,
    min_area_ratio: float,
) -> List[LogicObject]:
    candidates = []
    w, h = lo.geometry[2], lo.geometry[3]
    area_ratio = (w * h) / (viewport_width * viewport_height)

    if not lo.children or (area_ratio <= min_area_ratio and w * h > 0):
        candidates.append(lo)
    else:
        for child in lo.children:
            candidates.extend(_extract_large_sections_relative(
                child,
                viewport_width,
                viewport_height,
                min_area_ratio,
            ))
    return candidates


def extract_sections(lo: LogicObject) -> List[LogicObject]:
    # これがセグメントの最終単位（意味ブロック）
    if not lo.children:
        return [lo]
    result = []
    for child in lo.children:
        result.extend(extract_sections(child))
    return result

def extract_sections_at_depth(lo: LogicObject, target_depths: list[int], current_depth: int = 0) -> list[LogicObject]:
    results = []
    
    # 階層が一致する場合はこのノードを回収
    if current_depth in target_depths:
        results.append(lo)
    
    # 子ノードを再帰的に探索
    for child in lo.children:
        results.extend(
            extract_sections_at_depth(child, target_depths, current_depth + 1)
        )
    
    return results

# Main segmentation API
async def segment_page(browser: BrowserContext, dom_root: DOMElementNode, pG: float=0.1, epsilon: float=50) -> List[LogicObject]:
    """
    Given a DOMElementNode tree, return list of LogicObject sections.
    """
    page = await browser.get_agent_current_page()

    content_root = await d2c(dom_root, page)
    if not content_root:
        return []
    logic_root = c2l(content_root, None, pG)
    viewport = await page.evaluate('''
        () => {
            return {
                scrollWidth: document.documentElement.scrollWidth,
                scrollHeight: document.documentElement.scrollHeight,
                viewportWidth: window.innerWidth,
                viewportHeight: window.innerHeight
            };
        }
    ''')

    return extract_large_sections_relative_auto(logic_root, viewport_width=viewport.get("scrollWidth", 0), viewport_height=viewport.get("scrollHeight", 0))  # 1階層目のセクションを抽出
def build_summary_text(indexed_summary_json: str) -> str:
    prompt = '''
    You are provided with a structured JSON summary of a website's main sections.
    Each entry has:
    - An index (string)
    - The section's summarized description (raw summary text)

    ---

    ## Summary of Page Sections:
    {summary_text}

    ---

    ## Instructions:

    1. Analyze the provided JSON summary.
    2. For each section (by index), generate a short, clear human-friendly explanation describing what this section seems to represent or contain, based on the provided summary.

    ---

    Respond in this exact JSON format:

    {{
        "<index>": "<AI-generated summary for this section>",
        "<index>": "<AI-generated summary for this section>",
        ...
    }}
    ＊Respond with a **pure JSON object only**, without any markdown formatting such as ```json or ``` blocks. Do not add any explanations, comments, or additional text—only return the JSON itself.
    '''

    return PromptTemplate(
        input_variables=['summary_text'],
        template=prompt
    ).format(
        summary_text=indexed_summary_json or ""
    )

async def get_summary_text(
    llm: BaseChatModel,
    segments: List[LogicObject],
    browser: BrowserContext
) -> dict:
    summary_dict = {}

    # Step 1: 各セグメントの簡易サマリーを集める
    for idx, lo in enumerate(segments):
        summary = lo.summarize_for_ai()
        summary_dict[str(idx)] = summary

    # Step 2: AI用のプロンプトを作る
    json_summary = json.dumps(summary_dict, indent=2, ensure_ascii=False)
    prompt = build_summary_text(json_summary)
    # print("Prompt for LLM:", prompt)

    # Step 3: スクリーンショット取得
    state = await browser.get_state(cache_clickable_elements_hashes=True)

    # Step 4: AIに投げる
    message = HumanMessage(
        content=[
            {'type': 'text', 'text': prompt},
            {
                'type': 'image_url',
                'image_url': {'url': f'data:image/png;base64,{state.screenshot}'},
            }
        ]
    )
    output = await llm.ainvoke([message])

    # Step 5: AIの応答をパース
    try:
        ai_summary_dict = json.loads(output.content)
        print("AI Summary Dict:", ai_summary_dict)
    except Exception as e:
        print("Failed to parse AI output:", e)
        return {}

    # Step 6: 最終オブジェクトを構築
    result = {}
    for idx, lo in enumerate(segments):
        idx_str = str(idx)
        result[idx_str] = {
            "position": {
                "x": lo.geometry[0],
                "y": lo.geometry[1],
                "width": lo.geometry[2],
                "height": lo.geometry[3]
            },
            "summary": ai_summary_dict.get(idx_str, "")        
        }

    return result

async def get_summary_text_to_ai(browser: BrowserContext, llm: BaseChatModel) -> Tuple[str]:
    page = await browser.get_agent_current_page()
    dom_service = DomService(page)

    # ページからクリック可能要素を取得
    content = await dom_service.get_clickable_elements(
        focus_element=-1,
        viewport_expansion=-1,
        highlight_elements=True,
    )
    html_tree = content.element_tree
    segments = await segment_page(browser=browser, dom_root=html_tree)
    summary_text = await get_summary_text(
        llm=llm,
        segments=segments,
        browser=browser)
    return summary_text
def attach_generate_site_summary_v3(
        controller: Controller, 
    ):
    @controller.action("Find target element by context_block and target")
    async def generate_site_summary_v3(
        browser: BrowserContext,
        page_extraction_llm: BaseChatModel,
    ):
        page = await browser.get_agent_current_page()
        dom_service = DomService(page)

        # ページからクリック可能要素を取得
        content = await dom_service.get_clickable_elements(
            focus_element=-1,
            viewport_expansion=-1,
            highlight_elements=True,
        )
        html_tree = content.element_tree
        segments = await segment_page(browser=browser, dom_root=html_tree)
        summary_text = await get_summary_text(
            llm=page_extraction_llm,
            segments=segments,
            browser=browser)
        print("Summary Text:", summary_text)
        return ActionResult(
            extracted_content= json.dumps(
                summary_text, indent=2, ensure_ascii=False
            ),
            include_in_memory=True
        )

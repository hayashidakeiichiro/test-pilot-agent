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

def xpath_sort_key(xpath: str):
    parts = xpath.strip("/").split("/")
    key = []
    for part in parts:
        m = re.match(r"([a-zA-Z0-9_\-]+)(?:\[(\d+)\])?", part)
        if m:
            tag = m.group(1)
            index = int(m.group(2) or 1)
            key.append((tag, index))
    return key

def hierarchical_compare(a, b):
    for (tag1, idx1), (tag2, idx2) in zip(a, b):
        if tag1 != tag2:
            return 0  # タグ構造が違う → 比較対象ではない
        if idx1 != idx2:
            return idx1 - idx2  # indexで比較
    return len(a) - len(b)  # 残り階層が浅い方が先



def find_all_heading_nodes(node: DOMElementNode) -> list[DOMElementNode]:
    headings = []
    if isinstance(node, DOMElementNode) and is_title_node(node) and node.is_visible:
        headings.append(node)
    # if isinstance(node, DOMElementNode) and node.is_visible:
    for child in getattr(node, "children", []):
        if isinstance(child, DOMElementNode):
            headings.extend(find_all_heading_nodes(child))
    return headings
TITLE_TAGS_ALWAYS = {"h1", "h2", "h3", "h4", "h5", "h6", "header"}
TITLE_KEYWORDS = [
    "title", "heading", "head", "header",
    "section-title", "block-title", "card-title",
    "product-title", "article-title", "entry-title", "page-title",
    "ranking-title", "category-title",
    # 略語
    "ttl", "hd", "hdr", "cap", "caption",
    "headline", "lbl", "label", "txt-title", "mod-ttl", "unit-ttl"
]
def looks_like_title(node: DOMElementNode) -> bool:
    cls = node.attributes.get("class", "").lower()
    role = node.attributes.get("role", "").lower()
    return (
        any(keyword in cls for keyword in TITLE_KEYWORDS) or
        role == "heading"
    )
def is_title_node(node: DOMElementNode) -> bool:
    # tag = node.tag_name.lower()
    return True


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
		'src',
		'class',
		# test-pilot
	]

from langchain.prompts import PromptTemplate
import json

def build_element_prompt(target: str, context_block: str, hint: str, candidates: str) -> str:
    # prompt = '画像のサイトにはクリックできる要素に枠線と番号のラベルが追加されています。確認できるラベルの番号とその要素に書いてある文字を列挙して'
    prompt = '''
        You are given a list of clickable candidate elements from a web page. Each element has the following metadata:
        - `text`: The visible label text (may include images or be empty)
        - `tag`: The HTML tag (e.g., a, button, li)
        - `xpath`: The XPath indicating the element’s position in the DOM

        Your task is to select the **single most relevant element to click** based on the user's intent.

        ---
        ## Important Notes:

        - Your goal is to understand and interpret the user's intent deeply, not mechanically apply selection rules.
        - Evaluate elements based on what the user most likely wants to click, considering semantic context, visual positioning, and content type.
        - Do not prioritize elements solely because they contain text matching context blocks; image-only elements or other visually significant items may be more relevant.
        - When positional hints are given, interpret them according to typical webpage layouts and user expectations, not just DOM order.


        ## User Instruction:
        - Target label: The exact visible text label that the user wants to click. If empty, the user has not specified a direct label.
        - Context block: The semantic section or category on the page where the target element is located. Use this to narrow down the search area when the target label is empty or ambiguous.
        - Hint: Additional clues provided by the user to identify the target element within the context block. This can include positional information (e.g., 'topmost'), purpose (e.g., 'click the featured item'), or other guidance to help select the correct element.


        ## User Instruction Input:
        - Target label: "{target}"
        - Context block: "{context_block}"
        - Hint: "{hint}"

        ---

        ## Candidate Elements:
        {candidates}

        ---

        ## Selection Guidelines:

        1. If `target` is provided and exactly matches the `text` of an element, select that element.
        2. If `target` is empty or ambiguous:
            - Use `context_block` and `hint` to identify the relevant **section or category** on the page.
            - From that section, choose the most relevant element. If the hint suggests “click the first item” or “click the featured content,” prefer actual content (e.g., product links or main items).
        4. Choose the earliest matching element based on visual or DOM order if multiple are valid.
        5. If no element clearly matches, return `null`.

        ---

        ## Output Format:

        Return a JSON object using this exact structure:

        {{
            "index": <index of the selected element> | null,
            "reason": "<brief explanation of why this element was selected>"
        }}

        ＊Respond with a **pure JSON object only**, without any markdown formatting such as ```json or ``` blocks. Do not add any explanations, comments, or additional text—only return the JSON itself.
    '''


    return PromptTemplate(
        input_variables=["target", "context_block", "hint", "candidates"],
        template=prompt
    ).format(
        target=target,
        context_block=context_block,
        hint=hint or "",
        candidates=candidates
    )
def build_select_candidates_prompt(target: str, context_block: str, hint: str, candidates: str) -> str:
    # prompt = '画像のサイトにはクリックできる要素に枠線と番号のラベルが追加されています。確認できるラベルの番号とその要素に書いてある文字を列挙して'
    prompt = '''
        You are given a list of clickable candidate elements from a web page. Each element has the following metadata:
        - `text`: visible label text
        - `tag`: HTML tag (e.g., a, button)
        - `xpath`: XPath indicating its position in the DOM

        Your task is to identify the **most relevant interactive element** based on the user's instruction.

        ---

        ## User Instruction:

        - Target label (exact match): "{target}"
        - Context block: "{context_block}"
        - Header hint: "{hint}"

        ---

        ## Candidate Elements:
        {candidates}

        ---

        ## Instructions:

        1. If `target` is provided and matches exactly with the `text` of an element, return **that element’s XPath** and explain why.
        2. If `target` is empty or ambiguous:
            - Use `context_block` and `hint` to locate the relevant **section** (e.g., a heading or container representing a subsection of the page).
            - Return the XPath of the **section title element** (typically `header` or `h-` like `h1`,`h2` tag) that best matches the context.
        3. Do not make guesses. If no suitable element or section can be confidently identified, return `null` with a clear reason.

        ---

        Respond in this exact JSON format:

        {{
            "selected_xpath": "<XPath of the selected element>" | null,
            "reason": "<short explanation of why this element was selected, or why nothing could be selected>"
            "match_type": "target" | "section" | "none"
        }}
        ＊Respond with a **pure JSON object only**, without any markdown formatting such as ```json or ``` blocks. Do not add any explanations, comments, or additional text—only return the JSON itself.
        ---
        '''
    return PromptTemplate(
        input_variables=['target', 'context_block', 'hint', 'candidates'],
        template=prompt
    ).format(
        target=target,
        context_block=context_block,
        hint=hint or "",
        candidates="\n\n".join([
                f"\n  text: {node.get_all_text_till_next_clickable_element()}"
                f"\n  tag: {node.tag_name}"
                f"\n  score: {score:.3f}"
                f"\n  xpath: {node.xpath}"
                for i, (score, node) in enumerate(candidates)
            ])
        ) 
def format_selector_map(selector_map: dict[int, DOMElementNode]) -> str:
    lines = []
    for idx, node in sorted(selector_map.items()):
        tag = node.tag_name or "unknown"
        text = node.get_all_text_till_next_clickable_element()
        lines.append(f"[{idx}] <{tag}> {text}")
    return "\n".join(lines)
def build_select_candidates_by_hint(hint: str, candiddates_str: str) -> str:
    prompt = '''
        You are given a list of clickable candidate elements extracted from a web page. Each element includes the following metadata:
        - `index`: A numeric identifier used to reference the element
        - `tag`: The HTML tag (e.g., a, button, div)
        - `text`: The visible label text (if any)

        Your goal is to identify the **most relevant element to click** based on the user’s natural-language hint. The hint may describe visual position (e.g., "top-right arrow"), purpose (e.g., "next button"), or structure (e.g., "first item in slider").
        
        You are also provided with a composite image containing visual thumbnails of interactive elements extracted from the web page.

        Each visual element in the image:
        - Is a cropped screenshot of a clickable element
        - Is displayed in a grid layout
        - Has its numeric index clearly shown in the **bottom-right corner**
        - Matches the same index used in the metadata below

        Use this image to visually interpret the appearance of elements (e.g., arrows, icons, buttons) alongside the metadata.

        ---

        ## User Hint:
        "{hint}"

        ---

        ## Candidate Elements:
        {candidates}

        ---

        ## Instructions:

        1. Carefully read the hint and interpret the user's intention as precisely as possible.
        2. From the candidates, select the **single element** that most clearly and confidently matches the hint.
        3. ⚠️ If **no element clearly matches the user's intent**, return `null`. Do **not** try to force a match or guess.
        4. Do **not** select an element just because it is the "closest" if it's still ambiguous or weakly related.
        5. Do not assume anything beyond what is visible in the candidates list.

        ---

        Respond with a **pure JSON object** only, using the following format:

        [{{
            "selected_index": <index of the selected element> | null,
            "reason": "<brief explanation of why this element was selected, or why nothing could be confidently selected>"
        }}]

        ＊Respond with a **pure JSON object only**, without any markdown formatting such as ```json or ``` blocks. Do not add any explanations, comments, or additional text—only return the JSON itself.
            ---
        '''
    return PromptTemplate(
        input_variables=['hint', 'candidates'],
        template=prompt
    ).format(
        hint=hint or "",
        candidates=candiddates_str
    )

def normalize_text(text: str) -> str:
    return (text or "").strip().replace("\n", "").replace("\r", "").replace(" ", "").lower()

def similarity_levenshtein(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return 1 - distance(a, b) / max(len(a), len(b))


async def find_elements_by_target(
    selector_map: dict[int, DOMElementNode],
    target: str,
) -> List[Tuple[float, DOMElementNode]]:
    """
    target（ラベル）に近いテキストを持つクリック可能要素を検索し候補リストを返す。
    visible_selector_map を使い表示範囲の判定も行える（必要に応じて追加処理可能）。
    """
    normalized_target = normalize_text(target)
    candidates = []

    for index, node in selector_map.items():
        text = node.get_all_text_till_next_clickable_element()
        normalized_text = (text or "").lower().strip().replace(" ", "")
        if not normalized_text:
            continue
        score = similarity_levenshtein(normalized_target, normalized_text)
        if score > 0.2:
            candidates.append((score, node))

    return candidates

async def find_elements_by_context_block(
    html_tree: DOMElementNode,
    context_block: str,
) -> List[Tuple[float, DOMElementNode]]:
    """
    ページ全体のDOMツリーから見出しノードを抽出し、
    context_blockとのテキスト類似度で候補を返す。
    """
    normalized_context = normalize_text(context_block)
    candidates = []

    headings = find_all_heading_nodes(html_tree)
    for node in headings:
        text = node.get_all_text_till_next_clickable_element()
        normalized_text = normalize_text(text)
        if not normalized_text:
            continue
        score = similarity_levenshtein(normalized_context, normalized_text)
        if score > 0.2:
            candidates.append((score, node))
    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    return candidates

async def find_elements_by_hint(
    llm: BaseChatModel,
    hint: str,
    browser: BrowserContext
) -> dict[str, str] | None:
    """
    ページ全体のDOMツリーからヒントノードを抽出し、
    context_blockとのテキスト類似度で候補を返す。
    """
    page = await browser.get_agent_current_page()
    dom_service = DomService(page)
    candidates = []
    page = await browser.get_agent_current_page()
    await page.evaluate("""
        () => {
        // container全削除（ID重複対応）
        document.querySelectorAll('#playwright-highlight-container').forEach(el => el.remove());

        // ラベルも削除
        document.querySelectorAll('.playwright-highlight-label').forEach(el => el.remove());
        }
        """)

    screenshot = await browser.take_screenshot(full_page=True)
    


    # clickable elements の取得
    content = await dom_service.get_clickable_elements(
        focus_element=-1,
        viewport_expansion=-1,
        highlight_elements=True,
    )
    none_text_selector_map = extract_none_text_selector_map(content.selector_map)

    # Base64形式のサムネイルグリッド画像を生成
    data_url = await generate_selector_thumbnail_grid_base64(
        screenshot=screenshot,
        selector_map=none_text_selector_map,
        thumb_size=(100, 100),
        items_per_row=5,
        browser=browser
    )
    prompt = build_select_candidates_by_hint(hint=hint, selector_map=none_text_selector_map)
    message = HumanMessage(
        content=[
            {'type': 'text', 'text': prompt},
            {
                'type': 'image_url',
                'image_url': {'url': data_url},
            },
        ]
    )
    output = await llm.ainvoke([message])
    # print(prompt, "output", output)
    try:
        result = json.loads(output.content)
        for r in result:
            candidates.append({"index": r.get("selected_index", None), "reason": r.get("reason", "")})
    except:
        pass

    return candidates


def attach_generate_site_summary(controller: Controller):
    @controller.action("Find target element by context_block and target")
    async def generate_site_summary(
        browser: BrowserContext,
        page_extraction_llm: BaseChatModel,
        target: str,
        context_block: str,
        hint: Optional[str] = None
    ):
        page = await browser.get_agent_current_page()
        dom_service = DomService(page)

        # ページからクリック可能要素を取得
        content = await dom_service.get_clickable_elements(
            focus_element=-1,
            viewport_expansion=-1,
            highlight_elements=True,
        )


        await page.evaluate("""
            () => {
            // container全削除（ID重複対応）
            document.querySelectorAll('#playwright-highlight-container').forEach(el => el.remove());

            // ラベルも削除
            document.querySelectorAll('.playwright-highlight-label').forEach(el => el.remove());
            }
            """)
        # screenshot = await browser.take_screenshot(full_page=True)
    
        # clickable elements の取得
        content = await dom_service.get_clickable_elements(
            focus_element=-1,
            viewport_expansion=-1,
            highlight_elements=True,
        )
        none_text_selector_map = extract_none_text_selector_map(content.selector_map)
        html_tree = content.element_tree

        # Base64形式のサムネイルグリッド画像を生成
        # data_url = await generate_selector_thumbnail_grid_base64(
        #     screenshot=screenshot,
        #     selector_map=none_text_selector_map,
        #     thumb_size=(100, 100),
        #     items_per_row=5,
        #     browser=browser
        # )
        prompt = build_select_candidates_by_hint(hint=hint, candiddates_str=html_tree.clickable_elements_to_string_compact(include_attributes=include_attributes, max_chars=4000))
        # message = HumanMessage(
        #     content=[
        #         {'type': 'text', 'text': prompt},
        #         # {
        #         #     'type': 'image_url',
        #         #     'image_url': {'url': data_url},
        #         # },
        #     ]
        # )
        # output = await page_extraction_llm.ainvoke([message])
        print(prompt)

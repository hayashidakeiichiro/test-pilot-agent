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
from examples.custom_functions.generate_icon_list_image import generate_selector_thumbnail_grid_base64
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
    if isinstance(node, DOMElementNode) and node.is_visible:
        headings.append(node)
    # if isinstance(node, DOMElementNode) and node.is_visible:
    for child in getattr(node, "children", []):
        if isinstance(child, DOMElementNode):
            headings.extend(find_all_heading_nodes(child))
    return headings

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
def build_select_candidates_by_hint(hint: str, candidates_str: str) -> str:
    # prompt = '''
    #     You are given a list of clickable candidate elements extracted from a web page. Each element includes the following metadata:
    #     - `index`: A numeric identifier used to reference the element
    #     - `text`: The visible label text (if any)
    #     - `tag`: The HTML tag (e.g., a, button, div)
    #     - `attributes`: Selected HTML attributes (e.g., class, role, aria-label)

    #     Your goal is to identify the **most relevant element to click** based on the user’s natural-language hint. The hint may describe visual position (e.g., "top-right arrow"), purpose (e.g., "next button"), or structure (e.g., "first item in slider").

    #     ---

    #     ## User Hint:
    #     "{hint}"

    #     ---

    #     ## Candidate Elements:
    #     {candidates}

    #     ---

    #     ## Instructions:

    #     1. Read the hint carefully and interpret the user's intention as precisely as possible.
    #     2. Select **exactly one** element that matches the hint clearly and unambiguously.
    #     3. ⚠️ If **no element clearly and confidently matches the user's hint**, return `null`. 
    #     - This includes cases where the correct element may be **off-screen or not yet loaded**.
    #     - Do **not** try to guess or loosely match just because something is visible.
    #     4. It is completely acceptable—and often correct—to return `null` when:
    #     - All visible elements are ambiguous or weakly related.
    #     - The user's hint refers to something that might be off-screen.
    #     5. Your job is not to force a match but to be precise and cautious. If you make an incorrect selection, the user will be misled.

    #     ---

    #     Respond in this exact JSON format:

    #     {{
    #     "selected_index": <index of the selected element> | null,
    #     "reason": "<brief explanation of why this element was selected, or why nothing could be confidently selected>"
    #     }}

    #     ＊Respond with a **pure JSON object only**, without any markdown formatting such as ```json or ``` blocks. Do not add any explanations, comments, or additional text—only return the JSON itself.
    #     '''
    prompt = '''
    You are shown a screenshot of a web page and a user's instruction.

    Your task is to determine whether the element the user is looking for is **visually present in the screenshot**.

    ---

    ## User Hint:
    "{hint}"

    ## Screenshot:
    (See image)

    ---

    ### Instructions:

    1. Carefully analyze the screenshot.
    2. Try to find any visual element or section that appears to match the user's hint.
    3. ⚠️ If you are **not confident** that the element is present in the image, respond "not_found".
    4. If you see a clearly matching element, respond "found".

    ---

    Respond in this exact JSON format:

    {{
        "find": true | false,
        "reason": "<short explanation of why this element was selected, or why nothing could be selected>"
    }}
    ＊Respond with a **pure JSON object only**, without any markdown formatting such as ```json or ``` blocks. Do not add any explanations, comments, or additional text—only return the JSON itself.
    ---
    '''

    return PromptTemplate(
        input_variables=['hint', 'candidates'],
        template=prompt
    ).format(
        hint=hint or "",
        candidates=candidates_str
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

# async def find_elements_by_hint(
#     llm: BaseChatModel,
#     hint: str,
#     browser: BrowserContext
# ) -> dict[str, str] | None:
#     """
#     ページ全体のDOMツリーからヒントノードを抽出し、
#     context_blockとのテキスト類似度で候補を返す。
#     """
#     page = await browser.get_agent_current_page()
#     dom_service = DomService(page)
#     candidate = None
#     for i in range(10):     
#         visible_content = await dom_service.get_clickable_elements(
#             focus_element=-1,
#             viewport_expansion=0,
#             highlight_elements=True,
#         )
#         visible_element_tree = visible_content.element_tree
#         visible_element_tree_str = visible_element_tree.clickable_elements_to_string(include_attributes=include_attributes)
            
#         prompt = build_select_candidates_by_hint(hint=hint, candidates_str=visible_element_tree_str)
#         state = await browser.get_state(cache_clickable_elements_hashes=True)
#         message = HumanMessage(
#             content=[
#                 {'type': 'text', 'text': prompt},
#                 {
#                     'type': 'image_url',
#                     'image_url': {'url': f'data:image/png;base64,{state.screenshot}'},
#                 },
#             ]
#         )
#         output = await llm.ainvoke([message])
#         print("hint step", i, output.content)
#         try:
#             result = json.loads(output.content)
#             if(result.get("selected_index", "")):
#                 candidate = {"index": result.get("selected_index", ""), "reason": result.get("reason", "")}
#         except:
#             pass
#         if candidate is not None:
#             break
#         else:
#             at_bottom = await page.evaluate("""
#                 () => {
#                     return (window.scrollY + window.innerHeight) >= document.body.scrollHeight;
#                 }
#             """)
#             if at_bottom:
#                 break

#             # スクロール実行
#             scroll_offset = await page.evaluate("() => window.innerHeight")
#             await page.evaluate(f"""
#                 () => {{
#                     window.scrollBy({{ top: {scroll_offset}, behavior: 'auto' }});
#                 }}
#             """)
#             await asyncio.sleep(0.5)

#     return candidate
async def find_elements_by_hint(
    llm: BaseChatModel,
    hint: str,
    browser: BrowserContext
) -> bool:
    """
    ページ全体のDOMツリーからヒントノードを抽出し、
    context_blockとのテキスト類似度で候補を返す。
    """
    page = await browser.get_agent_current_page()
    dom_service = DomService(page)
    find_target = False
    for i in range(4):     
        visible_content = await dom_service.get_clickable_elements(
            focus_element=-1,
            viewport_expansion=0,
            highlight_elements=True,
        )
        visible_element_tree = visible_content.element_tree
        visible_element_tree_str = visible_element_tree.clickable_elements_to_string(include_attributes=include_attributes)
            
        prompt = build_select_candidates_by_hint(hint=hint, candidates_str=visible_element_tree_str)
        state = await browser.get_state(cache_clickable_elements_hashes=True)
        message = HumanMessage(
            content=[
                {'type': 'text', 'text': prompt},
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/png;base64,{state.screenshot}'},
                },
            ]
        )
        output = await llm.ainvoke([message])
        try:
            result = json.loads(output.content)
            print("reason", result, i)
            if(result.get("find", False)):
                find_target = True
        except:
            pass
        if find_target:
            break
        else:
            at_bottom = await page.evaluate("""
                () => {
                    return (window.scrollY + window.innerHeight) >= document.body.scrollHeight;
                }
            """)
            if at_bottom:
                break

            # スクロール実行
            scroll_offset = await page.evaluate("() => window.innerHeight")
            await page.evaluate(f"""
                () => {{
                    window.scrollBy({{ top: {scroll_offset}, behavior: 'auto' }});
                }}
            """)
            await asyncio.sleep(0.5)
    if not find_target:
        msg = f'not_found'
        logger.info(msg)
        return ActionResult(extracted_content=msg)
    visible_content = await dom_service.get_clickable_elements(
        focus_element=-1,
        viewport_expansion=0,
        highlight_elements=True,
    )
    visible_element_tree = visible_content.element_tree
    visible_element_tree_str = visible_element_tree.clickable_elements_to_string(include_attributes=include_attributes)
    prompt = build_element_prompt("", "", hint or "", visible_element_tree_str)
    state = await browser.get_state(cache_clickable_elements_hashes=True)
    message = HumanMessage(
        content=[
            {'type': 'text', 'text': prompt},
            {
                'type': 'image_url',
                'image_url': {'url': f'data:image/png;base64,{state.screenshot}'},
            },
        ]
    )
    output = await llm.ainvoke([message])
    result = json.loads(output.content)
    extracted_content = f"""
        selected_index: {result["index"]},
        reason: {result["reason"]},
        ✅ Element has been identified successfully. You may now proceed to the next action.
    """

    return ActionResult(
        extracted_content=extracted_content,
        include_in_memory=True
    )


def attach_find_target_v2(controller: Controller):
    @controller.action("Find target element by context_block and target")
    async def find_target_v2(
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
        html_tree = content.element_tree

        selector_map = content.selector_map

        visible_content = await dom_service.get_clickable_elements(
            focus_element=-1,
            viewport_expansion=0,
            highlight_elements=True,
        )

        target_candidates = []
        context_candidates = []
        if target:
            target_candidates = await find_elements_by_target(selector_map = selector_map, target=target)
        if context_block:
            context_candidates = await find_elements_by_context_block(html_tree=html_tree, context_block=context_block)
        if not target and not context_block and hint:
            result = await find_elements_by_hint(browser = browser, llm = page_extraction_llm, hint = hint)
            # if result is not None or not result.get("index"):
            #     extracted_content = f"""
            #         selected_index: {result["index"]},
            #         reason: {result["reason"]},
            #     ✅ Element has been identified successfully. You may now proceed to the next action.
            #     """

            #     return ActionResult(
            #         extracted_content=extracted_content,
            #         include_in_memory=True
            #     )
            return result

        candidates = target_candidates + context_candidates
        if not candidates:
            msg = f'❌ No candidates found for target {target}, context_block: {context_block}, hint: {hint}'
            logger.info(msg)
            return ActionResult(extracted_content=msg)
        # XPath階層でソート
        sorted_candidates = sorted(
            candidates,
            key=functools.cmp_to_key(
                lambda a, b: hierarchical_compare(
                    xpath_sort_key(a[1].xpath),
                    xpath_sort_key(b[1].xpath),
                )
            ),
        )
        
        formatted_prompt = build_select_candidates_prompt(
            target=target,
            context_block=context_block,
            hint=hint or "",
            candidates=sorted_candidates
        )
        print(formatted_prompt)
        message = HumanMessage(
            content=[
                {'type': 'text', 'text': formatted_prompt},
            ]
        )
        try:
            output = await page_extraction_llm.ainvoke([message])
            data = json.loads(output.content)
            xpath = data.get("selected_xpath", None)
            msg = xpath
            if xpath is not None:
                locator = page.locator(f'xpath={xpath}')
                if await locator.count() > 0:
                    first = locator.first
                    if await first.is_visible():
                        await first.evaluate("""
                            el => {
                                const rect = el.getBoundingClientRect();
                                const offset = 80;  // 上から80pxの余白を空けたい
                                window.scrollBy({
                                top: rect.top - offset,
                                behavior: 'auto'
                                });
                            }
                        """)
                        await asyncio.sleep(0.5)  # Wait for scroll to complete
            visible_content = await dom_service.get_clickable_elements(
                focus_element=-1,
                viewport_expansion=0,
                highlight_elements=True,
            )
            visible_element_tree = visible_content.element_tree
            visible_element_tree_str = visible_element_tree.clickable_elements_to_string(include_attributes=include_attributes)
            prompt = build_element_prompt(target, context_block, hint or "", visible_element_tree_str)
            state = await browser.get_state(cache_clickable_elements_hashes=True)
            message = HumanMessage(
                content=[
                    {'type': 'text', 'text': prompt},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/png;base64,{state.screenshot}'},
                    },
                ]
            )
            output = await page_extraction_llm.ainvoke([message])
            result = json.loads(output.content)
            extracted_content = f"""
                selected_index: {result["index"]},
                reason: {result["reason"]},
               ✅ Element has been identified successfully. You may now proceed to the next action.
            """

            return ActionResult(
                extracted_content=extracted_content,
                include_in_memory=True
            )
        except Exception as e:
            logger.debug(f'Error extracting content: {e}')
            msg = f'📄 Extraction fallback content:\n{e}'
            logger.info(msg)
            return ActionResult(extracted_content=msg)

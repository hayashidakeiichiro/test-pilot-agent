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
from examples.custom_functions.generate_site_summary_v3 import get_summary_text_to_ai


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
		# 'src',
		'class',
		# test-pilot
	]

from langchain.prompts import PromptTemplate
import json
def build_scan_and_direction_prompt(target: str, context_block: str, hint: str, candidates_str: str) -> str:
    """
    Prompt to scan the viewport for a hinted element and decide next scroll direction.
    If found, return its index. If not, return a scroll direction.
    """
    prompt = '''
You are shown two related images of a web page and given a user's instruction (hint). Your task is to:

1. Determine whether the element described in the hint is visually present.
2. If it is found, return its index.
3. If it is not found, decide whether to scroll "up" or "down" next to continue the search.

---

## Image Descriptions:
- **Image 1**: Screenshot of the current visible portion.
- Index labels correspond between both images.

---
## User Instruction:
- Target label: The exact visible text label that the user wants to click. If empty, the user has not specified a direct label.
- Context block: The semantic section or category on the page where the target element is located. Use this to narrow down the search area when the target label is empty or ambiguous.
- Hint: Additional clues provided by the user to identify the target element within the context block. This can include positional information (e.g., 'topmost'), purpose (e.g., 'click the featured item'), or other guidance to help select the correct element.

## User Instruction:
- Target label (exact match): "{target}"
- Context block: "{context_block}"
- Header hint: "{hint}"

ユーザーが指定したターゲットのdom情報（dom構造やテキストが変更されている可能性があるのであくまで参考）
currentエレメントがターゲットです
--- Parent Element ---
XPath: /html/body/div[1]/div[15]/div[1]/div[1]/section[6]/div[1]/div[1]/div[4]/div[2]/div[4]/div[2]/div[1]
Tag: div
Attributes: id='rnkDailyPrevButton_rnkDailyGenreLink_0' class='rnkApiBtLeft rnkBanner-prev' style='padding:29px 7px 0 0; top: 88px; margin-top: -24px;'
Rect: x: 448, y: 493.9921875, width: 22, height: 29

--- Current Element ---
XPath: /html/body/div[1]/div[15]/div[1]/div[1]/section[6]/div[1]/div[1]/div[4]/div[2]/div[4]/div[2]/div[1]/a[1]
Tag: a
Attributes: class='prev-button' href='#'
Rect: x: 448, y: 493.9921875, width: 40, height: 40

--- Child Element ---
XPath: /html/body/div[1]/div[15]/div[1]/div[1]/section[6]/div[1]/div[1]/div[4]/div[2]/div[4]/div[2]/div[1]/a[1]/button[1]
Tag: button
Attributes: class='ai-copy-button' style='position: absolute; top: 0px; right: 0px; z-index: 9999; font-size: 10px; padding: 2px; background: blue; color: white; border: none; cursor: pointer;'
Rect: x: 473, y: 493.9921875, width: 15, height: 15

---

## Interactive Candidates (text list from Image 2):
{candidates}

---

## Instructions:
1. Examine both images carefully.
2. If you see an element that clearly matches the hint:
   - Respond `"find": true`
   - Include its `"index": <number>`
3. If you do not see it or are uncertain:
   - Respond `"find": false`
   - Choose a scroll `"direction": "up"` or `"down"`
4. In all cases, include a brief `"reason"` for your decision.

---

Respond in this exact JSON format (no markdown or extra text):

{{
  "find": true | false,
  "index": <number>,        // only if find is true
  "direction": "up" | "down", // only if find is false
  "reason": "<short explanation>"
}}
＊Respond with a **pure JSON object only**, without any markdown formatting such as ```json or ``` blocks. Do not add any explanations, comments, or additional text—only return the JSON itself.
'''
    return PromptTemplate(
        input_variables=["target", "context_block", "hint", "candidates"],
        template=prompt
    ).format(
        target=target or "",
        context_block=context_block or "",
        hint=hint or "",
        candidates=candidates_str
    )

def build_xpath_candidates_prompt(target: str, context_block: str, hint: str, candidates_str: str) -> str:
    if not target and not context_block and not hint:
        raise ValueError("At least one of target, context_block, or hint must be provided.")
    if target:
        prompt = '''
        You are given a user's instruction (hint) and a JSON list of webpage section summaries.
        Your task is to identify the most promising XPath corresponding to the element or content described 
        by the target label and hint.

        ---
        ## User Instruction:
        - Target label: The exact visible text label that the user wants to click. If empty, the user has not specified a direct label.
        - Context block: The semantic section or category on the page where the target element is located. Use this to narrow down the search area when the target label is empty or ambiguous.
        - Hint: Additional clues provided by the user to identify the target element within the context block. This can include positional information (e.g., 'topmost'), purpose (e.g., 'click the featured item'), or other guidance to help select the correct element.


        - Target label (exact match): "{target}"
        - Context block: "{context_block}"
        - Header hint: "{hint}"


        ### Section Summaries (JSON format):
        {candidates}

        Each section summary includes:
        - "position": {{ "x": <number>, "y": <number>, "width": <number>, "height": <number> }}
        - "summary": "<natural language description of the section>"
        - "targets_and_contexts": [ {{ "xpath": <string>, "y": <number>, "text": <string>, "tag": <string>, "type": <'target'|'context'> }} ]

        ---

        ## Instructions:

        1. Analyze the provided target label and hint.
        2. Search through the section summaries and their associated elements.
        3. Identify the **single most likely XPath** that matches the target label, considering the hint and context.
        4. Return only that XPath.

        ---

        Respond in this exact JSON format:

        {{
            "candidates": [
                {{
                "xpath": "<element_xpath_or_null>",
                "section_index": "<index_or_null>",
                "position_in_section": "top" | "middle" | "bottom" | null,
                "reason": "<short explanation>"
                }},
                ...
            ]
        }}

        ＊Respond with a **pure JSON object only**, without any markdown formatting such as ```json or ``` blocks. 
        Do not add any explanations, comments, or additional text—only return the JSON itself.
        ---
        '''
    elif context_block:
        prompt = '''
        You are given a user's instruction (hint), a context block label, and a JSON list of webpage section summaries.
        Your task is to identify the most promising XPath corresponding to the element or content described 
        by the context block and hint (there is no exact target label).

        ---

        ## User Instruction:
        - Context block: The semantic section or category on the page where the target element is located. Use this to narrow down the search area when the target label is empty or ambiguous.
        - Hint: Additional clues provided by the user to identify the target element within the context block. This can include positional information (e.g., 'topmost'), purpose (e.g., 'click the featured item'), or other guidance to help select the correct element.

        - Context block: "{context_block}"
        - Header hint: "{hint}"

        ### Section Summaries (JSON format):
        {candidates}

        Each section summary includes:
        - "position": {{ "x": <number>, "y": <number>, "width": <number>, "height": <number> }}
        - "summary": "<natural language description of the section>"
        - "targets_and_contexts": [ {{ "xpath": <string>, "y": <number>, "text": <string>, "tag": <string>, "type": <'target'|'context'> }} ]

        ---

        ## Instructions:

        1. Analyze the provided context block and hint.
        2. Search through the section summaries and their associated elements.
        3. Identify the **single most likely XPath** that matches the intent described by the context block and hint.
        4. Return only that XPath.

        ---

        Respond in this exact JSON format:

        {{
            "candidates": [
                {{
                "xpath": "<element_xpath_or_null>",
                "section_index": "<index_or_null>",
                "position_in_section": "top" | "middle" | "bottom" | null,
                "reason": "<short explanation>"
                }},
                ...
            ]
        }}

        ＊Respond with a **pure JSON object only**, without any markdown formatting such as ```json or ``` blocks. 
        Do not add any explanations, comments, or additional text—only return the JSON itself.
        ---
        '''
    else:
        prompt = '''
        You are given a user's instruction (hint) and a JSON list of webpage section summaries.
        Your task is to identify:
        1. Which section(s) on the page are the most promising to contain the element or content described in the hint.
        2. For each selected section, where within the section (top, middle, bottom) is the best position to scroll to.

        ---

        ## User Instruction:

        - Hint: Additional clues provided by the user to identify the target element within the context block. This can include positional information (e.g., 'topmost'), purpose (e.g., 'click the featured item'), or other guidance to help select the correct element.
        - Header hint: "{hint}"

        ### Section Summaries (JSON format):
        {candidates}

        Each section summary includes:
        - "position": {{ "x": <number>, "y": <number>, "width": <number>, "height": <number> }}
        - "summary": "<natural language description of the section>"

        ---

        ## Instructions:

        1. Analyze the provided hint carefully.
        2. Compare the hint against the section summaries.
        3. Select one or more promising sections.
        4. For each section, determine where (top, middle, or bottom) is the most likely scroll target.

        ---

        Respond in this exact JSON format:

        {{
            "candidates": [
                {{
                "xpath": "<element_xpath_or_null>",
                "section_index": "<index_or_null>",
                "position_in_section": "top" | "middle" | "bottom" | null,
                "reason": "<short explanation>"
                }},
                ...
            ]
        }}

        ＊Respond with a **pure JSON object only**, without any markdown formatting such as ```json or ``` blocks. 
        Do not add any explanations, comments, or additional text—only return the JSON itself.
        ---
        '''



    return PromptTemplate(
        input_variables=['hint', 'candidates', 'target', 'context_block'],
        template=prompt
    ).format(
        hint=hint or "",
        candidates=candidates_str,
        target=target or "",
        context_block=context_block or ""
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
) -> List[dict[float, DOMElementNode]]:
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
            candidates.append({"score": score, "node": node})
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates

async def find_elements_by_context_block(
    html_tree: DOMElementNode,
    context_block: str,
) -> List[dict[float, DOMElementNode]]:
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
            candidates.append({"score": score, "node": node})
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates

async def find_elements_from_candidates(
    llm: BaseChatModel,
    target: str,
    context_block: str,
    hint: str,
    browser: BrowserContext,
    candidates_data: dict,
    summary_text_dict: dict
) -> ActionResult:
    page = await browser.get_agent_current_page()
    dom_service = DomService(page)
    candidates = candidates_data.get("candidates", [])
    mode = candidates_data.get("mode", "hint")

    find_target = False
    selected_index = None

    for candidate in candidates:
        xpath = candidate.get("xpath")
        section_index = candidate.get("section_index")
        position_in_section = candidate.get("position_in_section")
        reason = candidate.get("reason", "")

        if xpath:
            try:
                # 要素の位置を取得してスクロールのみ
                box = await page.evaluate(f'''
                    () => {{
                        const el = document.evaluate("{xpath}", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                        if (!el) return null;
                        const rect = el.getBoundingClientRect();
                        return {{
                            x: rect.x, y: rect.y, width: rect.width, height: rect.height
                        }};
                    }}
                ''')
                if box:
                    scroll_y = max(0, box['y'] - page.viewport_size["height"] / 3)
                    logger.info(f"Scrolling to xpath position y={scroll_y}")
                    await page.evaluate(f'window.scrollTo(0, {scroll_y})')
                    await asyncio.sleep(0.8)
                else:
                    logger.warning(f"Xpath {xpath} not found on page, skipping.")
                    continue
            except Exception as e:
                logger.error(f"Error handling xpath {xpath}: {e}")
                continue
        else:
            # fallback: scroll based on y and section
            section_info = summary_text_dict.get(section_index, {}).get("position", {})
            y_base = section_info.get("y", 0)
            section_height = section_info.get("height", 0)
            if position_in_section == "middle":
                y_base += section_height * 0.5
            elif position_in_section == "bottom":
                y_base += section_height * 0.95

            scroll_y = max(0, y_base - page.viewport_size["height"] / 3)
            logger.info(f"Scrolling to section y={scroll_y} (mode: {mode}, reason: {reason})")
            await page.evaluate(f'window.scrollTo(0, {scroll_y})')
            await asyncio.sleep(0.8)


        # Try up to 4 scroll-adjust attempts
        for attempt in range(4):
            state = await browser.get_state(cache_clickable_elements_hashes=True)
            visible_content = await dom_service.get_clickable_elements(
                focus_element=-1,
                viewport_expansion=0,
                highlight_elements=True,
            )
            visible_element_tree = visible_content.element_tree
            visible_element_tree_str = visible_element_tree.clickable_elements_to_string_with_tag(include_attributes=include_attributes)

            # Ask AI to scan current view
            prompt = build_scan_and_direction_prompt(target= target, context_block=context_block, hint=hint, candidates_str=visible_element_tree_str)
            print(f"AI Prompt: {prompt}")
            message = HumanMessage(
                content=[
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{state.screenshot}'}}
                ]
            )
            output = await llm.ainvoke([message])

            try:
                result = json.loads(output.content)
                if result.get("find", False):
                    find_target = True
                    selected_index = result.get("index")
                    break
                direction = result.get("direction", "down")
                offset = page.viewport_size["height"] / 2
                if direction == "down":
                    await page.evaluate(f'window.scrollBy(0, {offset})')
                else:
                    await page.evaluate(f'window.scrollBy(0, {-offset})')
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"Failed to parse AI response: {e}")
                continue
        if find_target:
            break

    if not find_target:
        msg = f'not_found_after_all_scrolls'
        logger.info(msg)
        return ActionResult(extracted_content=msg)

    extracted_content = f"""
        selected_index: {selected_index},
        ✅ Element has been identified successfully after guided scrolling.
    """
    print(f"extracted_content: {extracted_content}")

    return ActionResult(
        extracted_content=extracted_content,
        include_in_memory=True
    )

async def assign_targets_and_contexts(summary_text_dict, targets: list[dict[float, DOMElementNode]], browser: BrowserContext):
    # Initialize combined list in each section
    for idx_str in summary_text_dict.keys():
        summary_text_dict[idx_str]["targets_and_contexts"] = []
    resolved_targets = []
    page = await browser.get_agent_current_page()
    for target in targets:
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
        ''', target["node"].xpath)

        if result:
            resolved_targets.append({
                "xpath": target["node"].xpath,
                "x": result.get("x", 0),
                "y": result.get("y", 0),
                "width": result.get("width", 0),
                "height": result.get("height", 0),
                "text": target["node"].get_all_text_till_next_clickable_element(),
                "tag": target["node"].tag_name,
                "type": target["type"]
            })

    # Process targets
    for item in resolved_targets:
        assigned = False
        for idx_str, section in summary_text_dict.items():
            sec_x = section["position"]["x"]
            sec_y = section["position"]["y"]
            sec_w = section["position"]["width"]
            sec_h = section["position"]["height"]

            if (sec_x <= item["x"] <= sec_x + sec_w) and (sec_y <= item["y"] <= sec_y + sec_h):
                summary_text_dict[idx_str]["targets_and_contexts"].append({
                    "xpath": item["xpath"],
                    "y": item["y"],
                    "text": item["text"],
                    "tag":  item["tag"],
                    "type": item["type"]
                })
                assigned = True
                break
        if not assigned:
            summary_text_dict.setdefault("_unassigned", {}).setdefault("targets_and_contexts", []).append({
                "xpath": item["xpath"],
                "y": item["y"],
                "text": item["text"],
                "tag":  item["tag"],
                "type": item["type"]
            })

    # Sort each section's combined list by y coordinate (top-to-bottom)
    for section in summary_text_dict.values():
        if "targets_and_contexts" in section:
            section["targets_and_contexts"].sort(key=lambda e: e["y"])

    return summary_text_dict

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

        target_candidates = []
        context_candidates = []
        if target:
            target_candidates = await find_elements_by_target(selector_map = selector_map, target=target)
        if context_block:
            context_candidates = await find_elements_by_context_block(html_tree=html_tree, context_block=context_block)

        candidates = [
            {**item, "type": "target"} for item in target_candidates
        ] + [
            {**item, "type": "context"} for item in context_candidates
        ]
        # XPath階層でソート
        sorted_candidates = sorted(
            candidates,
            key=functools.cmp_to_key(
                lambda a, b: hierarchical_compare(
                    xpath_sort_key(a["node"].xpath),
                    xpath_sort_key(b["node"].xpath),
                )
            ),
        )
        summary_text_dict = await get_summary_text_to_ai(browser=browser, llm=page_extraction_llm)
        summary_text_dict = await assign_targets_and_contexts(browser=browser, summary_text_dict=summary_text_dict, targets=sorted_candidates)
        # print(f"summary_text_dict: {json.dumps(summary_text_dict, indent=2, ensure_ascii=False)}")

        
        formatted_prompt = build_xpath_candidates_prompt(
            target=target,
            context_block=context_block,
            hint=hint or "",
            candidates_str=json.dumps(summary_text_dict, indent=2, ensure_ascii=False)
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
            print(f"Output from LLM: {data}")
            final_result = await find_elements_from_candidates(
                llm=page_extraction_llm,
                target=target,
                context_block=context_block,
                hint=hint,
                browser=browser,
                candidates_data=data,
                summary_text_dict=summary_text_dict
            )
            return final_result
        except Exception as e:
            logger.debug(f'Error extracting content: {e}')
            msg = f'📄 Extraction fallback content:\n{e}'
            logger.info(msg)
            return ActionResult(extracted_content=msg)

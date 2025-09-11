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

async def get_scroll_remaining(page):
    """
    ページのスクロール状況を取得して残り量を返す。
    返り値:
      {
        "y": 現在のscrollTop,
        "ih": ビューポート高さ,
        "sh": ドキュメント全体の高さ,
        "remainDown": 下端までの残りピクセル,
        "remainUp": 上端までの距離,
        "bottomY": 最下部へ移動する際の目標scrollTop
      }
    """
    return await page.evaluate("""() => {
      const el = document.scrollingElement || document.documentElement || document.body;
      const y  = el.scrollTop | 0;
      const ih = window.innerHeight | 0;
      const sh = el.scrollHeight | 0;
      const remainDown = Math.max(0, sh - (y + ih));
      const remainUp   = y;
      const bottomY    = Math.max(0, sh - ih);
      return { y, ih, sh, remainDown, remainUp, bottomY };
    }""")

from langchain.prompts import PromptTemplate

def build_select_candidates_by_hint(
    hint: str,
    candidates_str: str,
    actions_str: str,
    page_scroll_ctx_str: str,  # ← {"y","ih","sh","remainDown","remainUp"} を文字列で渡す
) -> str:
    prompt = """
    You are shown a screenshot of a web page and a user's instruction.

    Your task:
    1) Decide if the target element seems **visually present** now.
    2) If not confident, **recommend a scroll direction** using the page scroll context.

    ---

    ## User Hint
    "{hint}"

    ## Page Scroll Context (JSON)
    {page_scroll_ctx}

    # Keys:
    # y: current scrollTop (px from top)
    # ih: viewport height
    # sh: document scrollHeight
    # remainDown = max(0, sh - (y + ih))
    # remainUp   = y

    ## Screenshot
    (See image)

    ---

    ### Instructions

    1. Analyze the screenshot carefully.
    2. If a clearly matching element is visible, set "find": true and choose the best **action** (e.g., click_element_by_index). Set "scroll": null.
    3. If you are **not confident** it is present:
    - Set "find": false.
    - Choose a **scroll direction** using the Page Scroll Context:
        - If remainDown > ih*0.2 → "down"
        - If 0 < remainDown ≤ ih*0.2 → "bottom"
        - If remainDown ≤ 0 and remainUp > 0 → "up"
        - If remainUp ≤ 0 → "down" (at top)
    - Also pick a concrete scroll action:
        - "down"/"up": use "scroll_down"/"scroll_up" with amount ≈ round(min(0.9*ih, remainDown or remainUp))
        - "bottom": use "scroll_down" with amount = remainDown
        - "top": (rare) use "scroll_up" with amount = remainUp
    4. Return JSON ONLY.

    ---

    ### Available Actions
    {actions_str}

    ---

    ### Response format (JSON ONLY)

    {{
    "find": true | false,
    "scroll": "down" | "up" | "top" | "bottom" | null,
    "action": {{
        go_to_url: {{url: {{type: string}}}} |
        go_back: {{}} |
        wait: {{seconds: {{default: 3, type: integer}}}} |
        click_element_by_index: {{index: {{type: integer}}, xpath: {{anyOf: [{{type: string}}, {{type: null}}], default: null}}}} |
        input_text: {{index: {{type: integer}}, text: {{type: string}}, xpath: {{anyOf: [{{type: string}}, {{type: null}}], default: null}}}} |
        switch_tab: {{page_id: {{type: integer}}}} |
        open_tab: {{url: {{type: string}}}} |
        close_tab: {{page_id: {{type: integer}}}} |
        extract_content: {{goal: {{type: string}}, should_strip_link_urls: {{type: boolean}}}} |
        scroll_down: {{amount: {{anyOf: [{{type: integer}}, {{type: null}}], default: null}}}} |
        scroll_up: {{amount: {{anyOf: [{{type: integer}}, {{type: null}}], default: null}}}} |
        send_keys: {{keys: {{type: string}}}} |
        scroll_to_text: {{text: {{type: string}}}} |
        get_dropdown_options: {{index: {{type: integer}}}} |
        select_dropdown_option: {{index: {{type: integer}}, text: {{type: string}}}} |
        drag_drop: {{
            element_source: {{anyOf: [{{type: string}}, {{type: null}}], default: null}},
            element_target: {{anyOf: [{{type: string}}, {{type: null}}], default: null}},
            element_source_offset: {{anyOf: [{{type: object}}, {{type: null}}], default: null}},
            element_target_offset: {{anyOf: [{{type: object}}, {{type: null}}], default: null}},
            coord_source_x: {{anyOf: [{{type: integer}}, {{type: null}}], default: null}},
            coord_source_y: {{anyOf: [{{type: integer}}, {{type: null}}], default: null}},
            coord_target_x: {{anyOf: [{{type: integer}}, {{type: null}}], default: null}},
            coord_target_y: {{anyOf: [{{type: integer}}, {{type: null}}], default: null}},
            steps: {{anyOf: [{{type: integer}}, {{type: null}}], default: 10}},
            delay_ms: {{anyOf: [{{type: integer}}, {{type: null}}], default: 5}}
        }}
    }}
    }}
    ＊Respond with a **pure JSON object only**, without any markdown formatting such as ```json or ``` blocks. Do not add any explanations, comments, or additional text—only return the JSON itself.
    """
    return PromptTemplate(
        input_variables=['hint', 'candidates', 'actions_str', 'page_scroll_ctx'],
        template=prompt
    ).format(
        hint=hint or "",
        candidates=candidates_str,
        actions_str=actions_str,
        page_scroll_ctx=page_scroll_ctx_str or '{"y":0,"ih":0,"sh":0,"remainDown":0,"remainUp":0}',
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
    browser: BrowserContext,
    actions_str: str,
) -> bool:
    """
    ページ全体のDOMツリーからヒントノードを抽出し、
    context_blockとのテキスト類似度で候補を返す。
    """
    page = await browser.get_agent_current_page()
    dom_service = DomService(page)
    find_target = False
    selected_action = None
    max_steps = 4

    for _ in range(max_steps):
        # 1) 画面内の候補を取得
        visible_content = await dom_service.get_clickable_elements(
            focus_element=-1,
            viewport_expansion=0,
            highlight_elements=True,
        )
        visible_element_tree = visible_content.element_tree
        visible_element_tree_str = visible_element_tree.get_all_text_till_next_clickable_element()
        print(visible_element_tree.get_all_text_till_next_clickable_element())

        # 2) ページスクロール状況をJSON文字列で用意
        page_ctx = await get_scroll_remaining(page)
        page_scroll_ctx_str = json.dumps(
            {
                "y": page_ctx["y"],
                "ih": page_ctx["ih"],
                "sh": page_ctx["sh"],
                "remainDown": page_ctx["remainDown"],
                "remainUp": page_ctx["remainUp"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

        # 3) プロンプト（スクロール判断付き）
        prompt = build_select_candidates_by_hint(
            hint=hint,
            candidates_str=visible_element_tree_str,
            actions_str=actions_str,
            page_scroll_ctx_str=page_scroll_ctx_str,  # ← 追加
        )

        # 4) 画像＋テキストで問い合わせ
        state = await browser.get_state(cache_clickable_elements_hashes=True)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{state.screenshot}"},
                },
            ]
        )
        output = await llm.ainvoke([message])

        # 5) 応答の解釈
        result = None
        try:
            result = json.loads(output.content)
        except Exception as e:
            logger.warning(f"LLM parse error: {e}")
        print(result, output)

        if isinstance(result, dict):
            if result.get("find", False):
                # 要素が見つかった
                find_target = True
                selected_action = result.get("action", None)
                break

            # 見つからない → スクロール指示を実行
            scroll_dir = (result.get("scroll") or "").lower()  # "down"|"up"|"top"|"bottom"|null
            action_obj = result.get("action") or {}

            # 量はLLMの action が持っていれば採用、無ければこちらで決定
            amount = None
            if scroll_dir in ("down", "bottom"):
                if "scroll_down" in action_obj and isinstance(action_obj["scroll_down"], dict):
                    amount = action_obj["scroll_down"].get("amount")
                if amount is None:
                    amount = page_ctx["remainDown"] if scroll_dir == "bottom" else min(
                        int(0.9 * page_ctx["ih"]), page_ctx["remainDown"]
                    )
            elif scroll_dir in ("up", "top"):
                if "scroll_up" in action_obj and isinstance(action_obj["scroll_up"], dict):
                    amount = action_obj["scroll_up"].get("amount")
                if amount is None:
                    amount = page_ctx["remainUp"] if scroll_dir == "top" else min(
                        int(0.9 * page_ctx["ih"]), page_ctx["remainUp"]
                    )

            amount = int(amount or 0)

            # 終了条件チェック
            at_bottom = page_ctx["remainDown"] <= 0
            at_top = page_ctx["remainUp"] <= 0

            if scroll_dir in ("bottom", "down"):
                if at_bottom or amount <= 0:
                    # もう下に行けない
                    break
                # 実行
                if scroll_dir == "bottom":
                    await page.evaluate(
                        """(y)=>{ const el=document.scrollingElement||document.documentElement||document.body;
                                el.scrollTo({top:y,behavior:'instant'}); }""",
                        page_ctx["bottomY"],
                    )
                else:
                    await page.evaluate(
                        """(dy)=>{ const el=document.scrollingElement||document.documentElement||document.body;
                                el.scrollBy({top:dy,behavior:'instant'}); }""",
                        amount,
                    )

            elif scroll_dir in ("top", "up"):
                if at_top or amount <= 0:
                    # もう上に行けない
                    break
                if scroll_dir == "top":
                    await page.evaluate(
                        """()=>{ const el=document.scrollingElement||document.documentElement||document.body;
                                el.scrollTo({top:0,behavior:'instant'}); }"""
                    )
                else:
                    await page.evaluate(
                        """(dy)=>{ const el=document.scrollingElement||document.documentElement||document.body;
                                el.scrollBy({top:-Math.abs(dy),behavior:'instant'}); }""",
                        amount,
                    )
            else:
                # スクロール指示がない/不正 → 打ち切り
                break

            await asyncio.sleep(0.4)
            # 次ループへ（再度DOMを見て判断）

        else:
            # 応答が不正なら終了
            break

    # 6) 結果
    print(find_target, selected_action)
    if not find_target:
        return {}
    return selected_action


def attach_run_instruction_based_test(controller: Controller):
    @controller.action("Find target element by context_block and target")
    async def run_instruction_based_test(
        browser: BrowserContext,
        page_extraction_llm: BaseChatModel,
        action: str
    ):
        actions_str = controller.registry.get_prompt_description()
        selected_action = await find_elements_by_hint(browser = browser, llm = page_extraction_llm, hint = action, actions_str = actions_str)
        print(f'selected_action: {selected_action}')
        if selected_action is None:
            extracted_content = {"xpath": None, "log": f""}
            return ActionResult(extracted_content=json.dumps(extracted_content))
        xpath = None
        for key, value in selected_action.items():
            if 'index' in value and value['index'] is not None:
                page = await browser.get_agent_current_page()
                dom_service = DomService(page)

                # ページからクリック可能要素を取得
                content = await dom_service.get_clickable_elements()
                selector_map = content.selector_map
                index = value['index']
                node = selector_map[index]
                xpath = node.xpath
        node.get_all_text_till_next_clickable_element()
        
        extracted_content = {"next_action": selected_action, "xpath": xpath, "log": f""}
        return ActionResult(
            extracted_content=json.dumps(extracted_content),
        )

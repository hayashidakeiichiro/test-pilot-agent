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
def build_assertion_prompt(assertion: str, candidates_str: str, url: str) -> str:
    prompt = '''
    You are shown a screenshot of a web page and an expected test result indicator.

    Your task is to determine whether the test has succeeded, failed, or is still undetermined based on visible content.

    ---

    ## Expected Indicator:
    "{assertion}"

    ---

    ## URL:
    {url}

    ---

    ## Contents:
    {candidates}

    ---

    ### Instructions:

    1. Carefully analyze the screenshot.
    2. If you see an indicator that the test was successful (e.g., "Test passed", ✅, 完了), return: `"status": "success"`.
    3. If you see an indicator that the test failed (e.g., "Test failed", ❌, エラー), return: `"status": "failure"`.
    4. If you cannot tell yet, return: `"status": "uncertain"` and suggest one of the following scroll directions:
        - "down", "up", "top", "bottom"

    Respond in this exact JSON format:

    {{
        "status": "success" | "failure" | "uncertain",
        "reason": "explanation for the decision",
        "scroll": "down" | "up" | "top" | "bottom" | null
    }}

    ＊Respond with a **pure JSON object only**, without any markdown formatting such as ```json or ``` blocks. Do not add any explanations, comments, or additional text—only return the JSON itself.
    '''
    return PromptTemplate(
        input_variables=['assertion', 'candidates'],
        template=prompt
    ).format(
        assertion=assertion or "",
        candidates=candidates_str,
        url=url
    )


async def assert_result_on_screen(
    llm: BaseChatModel,
    assertion: str,
    browser: BrowserContext,
    max_attempts: int = 4,
) -> ActionResult:
    page = await browser.get_agent_current_page()
    dom_service = DomService(page)

    for attempt in range(max_attempts):     
        visible_content = await dom_service.get_clickable_elements(
            focus_element=-1,
            viewport_expansion=0,
            highlight_elements=True,
        )
        tree_str = visible_content.element_tree.clickable_elements_to_string(include_attributes=["class", "aria-label"])

        prompt = build_assertion_prompt(assertion=assertion, candidates_str=tree_str, url=page.url)
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

        try:
            output = await llm.ainvoke([message])
            result = json.loads(output.content)
            print(f"LLM response: {result}")
            status = result.get("status", None)
            scroll = result.get("scroll", None)
            reason = result.get("reason", "")

            if status == "success":
                return ActionResult(extracted_content=f"✅ Test success indicator found: {reason}", success=True, is_done=True)
            elif status == "failure":
                return ActionResult(extracted_content=f"❌ Test failure indicator found: {reason}", success=False, is_done=True)

            # status == "uncertain" → scroll if LLM suggests
            if scroll == "bottom":
                await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            elif scroll == "top":
                await page.evaluate("() => window.scrollTo(0, 0)")
            elif scroll == "down":
                await page.evaluate("() => window.scrollBy({ top: window.innerHeight, behavior: 'auto' })")
            elif scroll == "up":
                await page.evaluate("() => window.scrollBy({ top: -window.innerHeight, behavior: 'auto' })")
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.warning(f"LLM response error or parsing failure: {e}")

    # max_attemptsまで試しても success にならなかった
    return ActionResult(extracted_content=f"❌ Test result not confirmed after scrolling: {reason}", success=False, is_done=True)


def attach_assert_test_success(controller: Controller):
    @controller.action("assert test success")
    async def assert_test_success(
        browser: BrowserContext,
        page_extraction_llm: BaseChatModel,
        assertion: str
    ):
        result = await assert_result_on_screen(
            browser=browser,
            llm=page_extraction_llm,
            assertion=assertion
        )
        return result



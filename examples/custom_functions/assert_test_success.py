from browser_use import Controller, ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.dom.views import DOMBaseNode, DOMElementNode, DOMTextNode
from browser_use.dom.service import DomService
import asyncio
import json
from browser_use import ActionResult, Controller
from langchain_core.language_models.chat_models import BaseChatModel
import re
from langchain_core.prompts import PromptTemplate
import logging
from langchain_core.messages import HumanMessage

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
def build_assertion_prompt(
    assertion: str,
    candidates_str: str,
    url: str,
    context: str | dict | None = None,
) -> str:
    ctx_json = context if isinstance(context, str) else json.dumps(context or {}, ensure_ascii=False)

    prompt = '''
        You are a tester responsible for determining whether a web application's test has passed or failed based on the visible content and, when applicable, the current URL. Your task is to determine whether the test has succeeded, failed, or is still undetermined using the expected test result indicator as an assertion.

    ---

    ## Expected Indicator (Assertion):
    "{assertion}"

    ---

    ## Current URL:
    {url}

    ---

    ## Context (JSON):
    {ctx_json}

    ---

    ## Contents:
    {candidates}

    ---

    ### Instructions:

    1. **Analyze the Content**:
    - Carefully analyze the screenshot or content provided.
    - Check if the expected indicator (assertion) is visible. This could be specific text, a button, image, or other elements relevant to the test case.
    - If the expected indicator (assertion) is found in the content, mark the test as "success."

    2. **When URL is Relevant**:
    - If the test case involves verifying that a specific page has loaded, check the URL to ensure it matches the expected one (e.g., an article page, a login page).
    - If the URL is incorrect or the page is not as expected, return `"status": "failure"`.

    3. **Use Context**:
    - Use the provided Context (JSON) when relevant (e.g., last action kind/args, target_url hints).

    4. **Determine Success or Failure**:
    - If the content matches the expected indicator (assertion) or the URL matches and the content is as expected, return: `"status": "success"`.
    - If the expected indicator (assertion) is missing or the URL is incorrect, return: `"status": "failure"`.

    5. **Uncertain Status**:
    - If the content is not clear or incomplete, and the result is not obvious, return: `"status": "uncertain"`. Suggest one of the following scroll directions:
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
        input_variables=['assertion', 'candidates', 'url', 'ctx_json'],
        template=prompt
    ).format(
        assertion=assertion or "",
        candidates=candidates_str or "",
        url=url or "",
        ctx_json=ctx_json or "{}",
    )


async def assert_result_on_screen(
    llm: BaseChatModel,
    assertion: str,
    context: str,
    browser: BrowserContext,
    max_attempts: int = 4,
) -> ActionResult:
    page = await browser.get_agent_current_page()
    dom_service = DomService(page)
    reason = ""

    for attempt in range(max_attempts):     
        visible_content = await dom_service.get_clickable_elements(
            focus_element=-1,
            viewport_expansion=0,
            highlight_elements=True,
        )
        tree_str = visible_content.element_tree.get_all_text_till_next_clickable_element()

        prompt = build_assertion_prompt(assertion=assertion, candidates_str=tree_str, url=page.url, context=context)
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
                output_dict = {"success": True, "log": reason}
                return ActionResult(extracted_content=json.dumps(output_dict), success=True, is_done=True)
            elif status == "failure":
                output_dict = {"success": False, "log": reason}
                return ActionResult(extracted_content=json.dumps(output_dict), success=False, is_done=True)

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
    output_dict = {"success": False, "log": reason}
    return ActionResult(extracted_content=json.dumps(output_dict), success=False, is_done=True)

def attach_assert_test_success(controller: Controller):
    @controller.action("assert test success")
    async def assert_test_success(
        browser: BrowserContext,
        page_extraction_llm: BaseChatModel,
        assertion: str,
        action_context: str
    ):
        result = await assert_result_on_screen(
            browser=browser,
            llm=page_extraction_llm,
            assertion=assertion,
            context=action_context
        )
        return result



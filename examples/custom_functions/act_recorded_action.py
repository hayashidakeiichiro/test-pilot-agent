import math
from rapidfuzz import fuzz
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, ValidationError
from browser_use.dom.service import DomService
from browser_use import Controller
from browser_use.browser.context import BrowserContext
from browser_use.dom.views import DOMElementNode, DOMTextNode
import re
from playwright.async_api import Page
from patchright.async_api import ElementHandle, Page
import json
from browser_use import Controller, ActionResult
import pickle
import base64

class Position(BaseModel):
    x: float
    y: float
    width: float
    height: float

class TestStep(BaseModel):
    type: str
    selector: str
    xpath: Optional[str] = None
    value: Optional[str] = None
    innerText: Optional[str] = None
    parentSummary: Optional[str] = None
    childrenSummary: Optional[List[str]] = None
    attributes: Optional[Dict[str, Optional[str]]] = None
    timestamp: int
    href: str
    position: Optional['Position'] = None
    isButton: Optional[bool] = None
    siblingTexts: Optional[List[str]] = None
    tabId: Optional[int] = None
    url: Optional[str] = None

    def set_children_summary_from_dom(self, node: 'DOMElementNode', max_depth: int = 3):
        def traverse(n: 'DOMElementNode', prefix: str, level: int):
            tag = n.tag_name.lower()
            path = f"{prefix} > {tag}" if prefix else tag
            children = [c for c in n.children if isinstance(c, DOMElementNode)]

            if not children or level >= max_depth:
                return path

            return traverse(children[0], path, level + 1)

        # node の直下の子要素たちそれぞれに対して再帰構造パスを得る
        children_paths = []
        for child in node.children:
            if isinstance(child, DOMElementNode):
                path = traverse(child, "", 1)
                if path:
                    children_paths.append(path)

        self.childrenSummary = children_paths


async def click_element(element_node: DOMElementNode, browser: BrowserContext):
    session = await browser.get_session()
    initial_pages = len(session.context.pages)
    page = await browser.get_agent_current_page()

    # if element has file uploader then dont click
    if await browser.is_file_uploader(element_node):
        return False

    try:
        await browser._click_element_node(element_node)
        if len(session.context.pages) > initial_pages:
            await browser.switch_to_tab(-1)
        return True
    except Exception as e:
        return False
async def input_text(element_node: DOMElementNode, text: str, browser: BrowserContext):
    await browser._input_text_element_node(element_node, text)
    return True
async def select_dropdown_option(
    dom_element: DOMElementNode,
    text: str,
    browser: BrowserContext,
) -> bool:
    """Select dropdown option by the text of the option you want to select"""
    page = await browser.get_current_page()

    # Validate that we're working with a select element
    if dom_element.tag_name != 'select':
        return False

    try:
        frame_index = 0
        for frame in page.frames:
            try:

                # First verify we can find the dropdown in this frame
                find_dropdown_js = """
                    (xpath) => {
                        try {
                            const select = document.evaluate(xpath, document, null,
                                XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                            if (!select) return null;
                            if (select.tagName.toLowerCase() !== 'select') {
                                return {
                                    error: `Found element but it's a ${select.tagName}, not a SELECT`,
                                    found: false
                                };
                            }
                            return {
                                id: select.id,
                                name: select.name,
                                found: true,
                                tagName: select.tagName,
                                optionCount: select.options.length,
                                currentValue: select.value,
                                availableOptions: Array.from(select.options).map(o => o.text.trim())
                            };
                        } catch (e) {
                            return {error: e.toString(), found: false};
                        }
                    }
                """

                dropdown_info = await frame.evaluate(find_dropdown_js, dom_element.xpath)

                if dropdown_info:
                    if not dropdown_info.get('found'):
                        continue
                    await frame.locator('//' + dom_element.xpath).nth(0).select_option(label=text, timeout=1000)
                    
                    return True

            except Exception as frame_e:
                return False

            frame_index += 1

        return False

    except Exception as e:
        return False

    
def attach_act_recorded_action(controller):
    @controller.action("Find target element by context_block and target")
    async def act_recorded_action(browser: BrowserContext, action_json: Dict):
        action_type = action_json["action_type"]
        node: DOMElementNode | None = None
        value = None
        try:
            node = pickle.loads(base64.b64decode(action_json.get("node", None)))
            value = action_json.get("value", "")
        except:
            pass
        if not node:
            raise ValueError("No node provided")

        if action_type == 'click':
            await click_element(node, browser)
        elif action_type == 'input':
            await input_text(node, value, browser)
        elif action_type == 'select':
            await select_dropdown_option(node, value, browser)
        label = node.get_all_text_till_next_clickable_element(include_attr_for_img=False) if node.get_all_text_till_next_clickable_element(include_attr_for_img=False) else f"<{node.tag_name}>"
        extracted_content={"xpath": node.xpath, "log": f"Executed action: {action_type} on {label}"}

        return ActionResult(
            extracted_content=json.dumps(extracted_content),
        )

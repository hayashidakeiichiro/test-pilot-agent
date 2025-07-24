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


def clean_text(text: str | None) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()

def normalize_href(href: str | None) -> str:
    return re.sub(r'^https?://[^/]+', '', href or '')

def normalized_xpath_levenshtein(s1, s2):
    def simplify(xpath: str) -> str:
        return re.sub(r'\[1\]', '', re.sub(r'\[(\d+)\]', r'[\1]', xpath or ''))
    return fuzz.token_set_ratio(simplify(s1), simplify(s2)) / 100

def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def area(width, height):
    return width * height

def shape_ratio(width, height):
    return width / height if height != 0 else 0

def similarity_score(target: TestStep, candidate: TestStep) -> float:
    score = 0.0
    weight_total = 0.0

    def add_score(sim, weight):
        nonlocal score, weight_total
        score += sim * weight
        weight_total += weight

    high_weight = 1.5
    low_weight = 0.5

    add_score(1.0 if target.selector == candidate.selector else 0.0, high_weight)

    # IDとnameは論文的には補助扱いなので low_weight に変更
    if target.attributes.get('id') and candidate.attributes.get('id'):
        add_score(1.0 if target.attributes['id'] == candidate.attributes['id'] else 0.0, low_weight)

    if target.attributes.get('name') and candidate.attributes.get('name'):
        add_score(1.0 if target.attributes['name'] == candidate.attributes['name'] else 0.0, low_weight)
      
    if target.attributes.get('class') and candidate.attributes.get('class'):
        add_score(fuzz.token_set_ratio(clean_text(target.attributes.get('class')), clean_text(candidate.attributes.get('class'))) / 100, low_weight)

    if target.attributes.get('href') and candidate.attributes.get('href'):
        add_score(fuzz.token_set_ratio(normalize_href(target.attributes.get('href')), normalize_href(candidate.attributes.get('href'))) / 100, low_weight)

    if target.attributes.get('alt') and candidate.attributes.get('alt'):
        add_score(fuzz.token_set_ratio(target.attributes.get('alt'), candidate.attributes.get('alt')) / 100, low_weight)

    if target.xpath and candidate.xpath:
        add_score(normalized_xpath_levenshtein(target.xpath, candidate.xpath), low_weight)

    if target.innerText and candidate.innerText:
        add_score(fuzz.token_set_ratio(clean_text(target.innerText), clean_text(candidate.innerText)) / 100, high_weight)

    if target.siblingTexts and candidate.siblingTexts:
        target_neighbors = clean_text(' '.join(target.siblingTexts))
        candidate_neighbors = clean_text(' '.join(candidate.siblingTexts))
        add_score(fuzz.token_set_ratio(target_neighbors, candidate_neighbors) / 100, high_weight)
    if target.childrenSummary and candidate.childrenSummary:
        def simplify(summary: list[str]) -> list[str]:
            return [s.split('.')[0] for s in summary]

        simplified_target = clean_text(' '.join(simplify(target.childrenSummary)))
        simplified_candidate = clean_text(' '.join(simplify(candidate.childrenSummary)))

        add_score(
            fuzz.token_set_ratio(simplified_target, simplified_candidate) / 100,
            low_weight
        )
    if target.position and candidate.position:
        dist = euclidean_distance(target.position, candidate.position)
        pos_sim = max(0, 1 - dist / 100)
        add_score(pos_sim, low_weight)

        target_area = area(target.position.width, target.position.height)
        candidate_area = area(candidate.position.width, candidate.position.height)
        area_sim = max(0, 1 - abs(target_area - candidate_area) / max(target_area, candidate_area, 1))
        add_score(area_sim, low_weight)

        target_shape = shape_ratio(target.position.width, target.position.height)
        candidate_shape = shape_ratio(candidate.position.width, candidate.position.height)
        shape_sim = max(0, 1 - abs(target_shape - candidate_shape) / max(target_shape, candidate_shape, 1))
        add_score(shape_sim, low_weight)

    return score / weight_total if weight_total > 0 else 0

async def click_element(element_node: DOMElementNode, browser: BrowserContext):
    session = await browser.get_session()
    initial_pages = len(session.context.pages)
    page = await browser.get_agent_current_page()

    # if element has file uploader then dont click
    if await browser.is_file_uploader(element_node):
        return False

    try:
        await scroll_to_element_by_xpath(page, element_node.xpath)
        await browser._click_element_node(element_node)
        if len(session.context.pages) > initial_pages:
            await browser.switch_to_tab(-1)
        return True
    except Exception as e:
        return False
async def input_text(element_node: DOMElementNode, text: str, browser: BrowserContext):
    page = await browser.get_agent_current_page()
    await scroll_to_element_by_xpath(page, element_node.xpath)
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
        await scroll_to_element_by_xpath(page, dom_element.xpath)
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

async def scroll_to_element_by_xpath(page: Page, xpath: str) -> None:
    found = await page.evaluate(
        """(xpath) => {
            const result = document.evaluate(
                xpath,
                document,
                null,
                XPathResult.FIRST_ORDERED_NODE_TYPE,
                null
            );
            const el = result.singleNodeValue;
            if (!el) return false;

            const rect = el.getBoundingClientRect();
            const inViewport = (
                rect.top >= 0 &&
                rect.left >= 0 &&
                rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
                rect.right <= (window.innerWidth || document.documentElement.clientWidth)
            );

            if (!inViewport) {
                el.scrollIntoView({ behavior: 'auto', block: 'center', inline: 'center' });
            }

            return true;
        }""",
        xpath
    )

    if not found:
        raise Exception(f"Element not found for XPath: {xpath}")

    
def attach_find_target_v3(controller):
    @controller.action("Find target element by context_block and target")
    async def find_target(browser: BrowserContext, test_steps_json: Dict):
        if not test_steps_json:
            raise ValueError("No test steps provided")

        target_item = test_steps_json["action"]
        if "position" in target_item and target_item["position"] is not None:
            target_item["position"] = Position(**target_item["position"])
        target_step = TestStep(**target_item)

        page = await browser.get_agent_current_page()
        dom_service = DomService(page)
        content = await dom_service.get_clickable_elements(
            focus_element=-1,
            viewport_expansion=0,
            highlight_elements=True,
        )
        html_tree = content.element_tree

        candidates = []

        async def get_element_position(xpath):
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
                }''', xpath)
            return Position(**result) if result else None

        async def traverse(node):
            if isinstance(node, DOMElementNode):
                if node.is_visible:
                    position = await get_element_position(node.xpath)
                    candidate_data = TestStep(
                        type=target_item.get('type', 'none'),
                        selector=node.tag_name,
                        childrenSummary=[],
                        xpath=node.xpath,
                        value=target_item.get('value', ''),
                        innerText=' '.join(
                            c.text for c in node.children if isinstance(c, DOMTextNode)
                        ),
                        attributes=node.attributes,
                        timestamp=0,
                        href='',
                        position=position,
                        isButton=node.is_interactive,
                        siblingTexts=[
                            c.text for c in node.parent.children if isinstance(c, DOMTextNode) and c != node
                        ] if node.parent else []
                    )
                    candidate_data.set_children_summary_from_dom(node)
                    sim_score = similarity_score(target_step, candidate_data)
                    candidates.append({
                        'score': sim_score,
                        'node': node,
                        'test_step': candidate_data
                    })
                for child in node.children:
                    await traverse(child)

        await traverse(html_tree)

        # スコア順にソート
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # 最良の候補を取得
        best_entry = candidates[0] if candidates else None
            
        if best_entry["test_step"].type == 'click':
            await click_element(best_entry['node'], browser)
        elif best_entry["test_step"].type == 'input':
            if not best_entry["test_step"].value:
                raise ValueError("No value provided for input action")
            await input_text(best_entry['node'], best_entry["test_step"].value, browser)
        elif best_entry["test_step"].type == 'select':
            await select_dropdown_option(best_entry['node'], best_entry["test_step"].value, browser)
        label = best_entry['node'].get_all_text_till_next_clickable_element(include_attr_for_img=False) if best_entry['node'].get_all_text_till_next_clickable_element(include_attr_for_img=False) else f"<{best_entry['node'].tag_name}>"
        extracted_content={"xpath": best_entry['node'].xpath, "log": f"Executed action: {best_entry['test_step'].type} on {label}"}

        return ActionResult(
            extracted_content=json.dumps(extracted_content),
        )

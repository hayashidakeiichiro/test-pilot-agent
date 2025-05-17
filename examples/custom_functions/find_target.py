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
    tag = node.tag_name.lower()
    return tag in TITLE_TAGS_ALWAYS or looks_like_title(node)



def attach_find_target(controller: Controller):
    @controller.action("Find target element by context_block and target")
    async def find_target(
        browser: BrowserContext,
        target: str,
        context_block: str,
        header_hint: Optional[str] = None
    ):
        page = await browser.get_agent_current_page()
        dom_service = DomService(page)
        content = await dom_service.get_clickable_elements(
            focus_element=-1,
            viewport_expansion=-1,
            highlight_elements=True,
        )
        html_tree = content.element_tree

        def normalize(text: str) -> str:
            return (text or "").strip().replace("\n", "").replace("\r", "").replace(" ", "").lower()

        def similarity_levenshtein(a: str, b: str) -> float:
            return 1 - distance(a, b) / max(len(a), len(b))

        normalized_target = normalize(context_block)
        normalized_alt = normalize(header_hint or "")

        headings = find_all_heading_nodes(html_tree)

        candidates: list[tuple[float, DOMElementNode]] = []

        for node in headings:
            text = node.get_all_text_till_next_clickable_element()
            normalized_text = normalize(text)
            if not normalized_text:
                continue

            score_main = similarity_levenshtein(normalized_target, normalized_text)
            score_alt = similarity_levenshtein(normalized_alt, normalized_text) if header_hint else 0.0
            score = max(score_main, score_alt)

            if score > 0.2:
                candidates.append((score, node))

        dom_service = DomService(page)
        content = await dom_service.get_clickable_elements(
            focus_element=-1,
            viewport_expansion=-1,
            highlight_elements=True,
        )
        selector_map = content.selector_map
        visible_content = await dom_service.get_clickable_elements(
            focus_element=-1,
            viewport_expansion=0,
            highlight_elements=True,
        )
        visible_selector_map = visible_content.selector_map

        # 正規化済みターゲット（大文字小文字と空白無視）
        def normalize(text):
            return (text or "").replace("\n", "").replace("\r", "").strip().lower()
        def similarity_levenshtein(a, b):
            return 1 - distance(a, b) / max(len(a), len(b))
        def in_viewport(index: int) -> bool:   
            return  visible_selector_map.get(index) is not None

        normalized_target = normalize(target)

        for index, node in selector_map.items():
            text = node.get_all_text_till_next_clickable_element()
            normalized_text = text.lower().strip().replace(" ", "")
            if not normalized_text:
                continue

            score = similarity_levenshtein(normalized_target, normalized_text)
            if score > 0.2:
                candidates.append((score, node))
        print("result",[
                    f"[{i}]"
                    f"\n  text: {node.get_all_text_till_next_clickable_element()}"
                    f"\n  tag: {node.tag_name}"
                    f"\n  score: {score:.3f}"
                    f"\n  xpath: {node.xpath}"
                    for i, (score, node) in enumerate(candidates)
                ])

        return ActionResult(
            extracted_content=(
                f"🧭 Found {len(candidates)} heading candidates for context block '{context_block}'"
                f"{f' (or header hint: {header_hint})' if header_hint else ''}.\n\n"
                "📋 Candidate headings:\n" +
                "\n\n".join([
                    f"[{i}]"
                    f"\n  text: {node.get_all_text_till_next_clickable_element()}"
                    f"\n  tag: {node.tag_name}"
                    f"\n  score: {score:.3f}"
                    f"\n  xpath: {node.xpath}"
                    for i, (score, node) in enumerate(candidates)
                ]) +
                "\n\n"
                "🧠 Once you select the best matching heading:\n"
                "- You MUST scroll to it before performing any further action.\n"
                "- To scroll the selected heading into view, call the custom action:\n"
                "  `scroll_to_xpath` with the `xpath` of the selected heading.\n"
                "- Only proceed to search for elements within that section AFTER it has been scrolled into view.\n"
                "- Do NOT interact with headings or sections that are outside the viewport.\n"
            )
        )



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
def clear_logs():
    """logsフォルダを空にする（フォルダ自体は残す）"""
    if os.path.exists(LOG_DIR):
        for file in os.listdir(LOG_DIR):
            file_path = os.path.join(LOG_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                for sub_file in os.listdir(file_path):
                    os.remove(os.path.join(file_path, sub_file))
                # サブフォルダも消したい場合は↓を有効化
                # os.rmdir(file_path)
    else:
        os.makedirs(LOG_DIR)


def extract_base64_image(text):
    """Base64画像データを抽出"""
    split_text = text.split("data:image/png;base64,")
    if len(split_text) > 1:
        return split_text[1].split("'")[0]
    return None


class LoggingCallbackHandler(BaseCallbackHandler):
    """LLMリクエストとレスポンスを記録"""

    def on_llm_start(self, serialized, prompts, **kwargs):
        log_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "args": prompts,
        }
        with open(LLM_LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write("=== LLM Request ===\n")
            log_file.write(json.dumps(log_data, indent=4, ensure_ascii=False) + "\n")

        # スクショが含まれていたら保存
        for prompt in prompts:
            img_data = extract_base64_image(prompt)
            if img_data:
                save_image_from_base64(f"data:image/png;base64,{img_data}")

    def on_llm_end(self, response, **kwargs):
        generations_serializable = []
        for generation_list in response.generations:
            for generation in generation_list:
                generations_serializable.append({
                    "text": generation.text,
                    "generation_info": generation.generation_info,
                    "message": {
                        "content": generation.message.content,
                        "additional_kwargs": generation.message.additional_kwargs,
                        "response_metadata": generation.message.response_metadata
                    }
                })

                # スクショが含まれていたら保存
                img_data = extract_base64_image(generation.message.content)
                if img_data:
                    save_image_from_base64(f"data:image/png;base64,{img_data}")

        with open(LLM_LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write("=== LLM Response ===\n")
            log_file.write(json.dumps(generations_serializable, indent=4, ensure_ascii=False) + "\n\n")


from browser_use.browser.context import BrowserContext, BrowserContextConfig
async def main_task():

    url = "https://www.rakuten.co.jp/"
    # url = "https://zenn.dev"
    task_prompt = f"""
    context_block:人気商品ランキング
    の一位の要素をクリックして
    最初にfind_section_by_context_blockを使えばいい"""
    clear_logs()
    controller = Controller()
    attach_find_section_by_context_block(controller)
    browser = Browser(config=BrowserConfig(headless=True))
    browser_context = BrowserContext(config=BrowserContextConfig(user_agent='foobarfoo'), browser=browser)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, callbacks=[LoggingCallbackHandler()])

    agent = Agent(
        task=task_prompt,
        llm=llm,
        controller=controller,
        browser_context=browser_context,
        initial_actions=[{'open_tab': {'url': url}}]
    )

    result = await agent.run(max_steps=5)

if __name__ == "__main__":
    asyncio.run(main_task())

    

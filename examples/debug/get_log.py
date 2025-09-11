import asyncio
import datetime
import json, re
import os
import base64
import io
from dotenv import load_dotenv
from PIL import Image
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from browser_use import ActionResult, Agent, Browser, BrowserConfig, Controller
from examples.custom_functions.find_target_v3 import attach_find_target_v3
from examples.custom_functions.run_instruction_based_test import attach_run_instruction_based_test
from examples.custom_functions.run_recorded_test import attach_run_recorded_test
from examples.custom_functions.assert_test_success import attach_assert_test_success
from examples.custom_functions.act_recorded_action import attach_act_recorded_action
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs, unquote, urljoin


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

def _safe_attrs(action: Dict[str, Any]) -> Dict[str, Any]:
    allow = {"id","class","name","type","role","href","alt","aria-label","placeholder","value"}
    attrs = (action.get("attributes") or {})
    return {k: v for k, v in attrs.items() if k in allow and v not in (None, "", [])}

def _as_actions_list(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """rec が actions配列 or 単一 recorded アクションのどちらでも扱えるように整形"""
    if isinstance(rec.get("actions"), list):
        return rec["actions"]
    if rec.get("type") == "recorded" and isinstance(rec.get("action"), dict):
        return [rec]
    return []

def build_assertion_context_str_from_recording(
    rec: Dict[str, Any],
) -> str:
    actions: List[Dict[str, Any]] = _as_actions_list(rec)

    last = actions[-1] if actions else {}
    last_type = last.get("type", "")
    last_act: Dict[str, Any] = last.get("action") or {}
    if last_type == "manual":
        ctx: Dict[str, Any] = {
            "last_action": last_act or None,
        }
        return json.dumps(ctx, ensure_ascii=False, separators=(",", ":"))

    last_url: Optional[str] = last.get("url")


    # --- last_action 要約（click 以外にも汎用で持たせる）---
    la_attrs = _safe_attrs(last_act)
    last_action = {
        "type": (last_act.get("type") or "").lower() or None,
        "selector": last_act.get("selector") or None,
        "xpath": last_act.get("xpath") or None,
        "innerText": (last_act.get("innerText") or None),
        "attributes": la_attrs or None,
        "isButton": last_act.get("isButton") if "isButton" in last_act else None,
        "position": last_act.get("position") or None,
        "timestamp": last_act.get("timestamp") or None,
        "tabId": last.get("tabId") or None,
    }
    last_action = {k: v for k, v in last_action.items() if v is not None}

    ctx: Dict[str, Any] = {
        "prev_url": last_url,
        "last_action": last_action or None,
    }
    ctx = {k: v for k, v in ctx.items() if v is not None}
    return json.dumps(ctx, ensure_ascii=False, separators=(",", ":"))


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
            # if img_data:
                # save_image_from_base64(f"data:image/png;base64,{img_data}")

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
                # if img_data:
                #     save_image_from_base64(f"data:image/png;base64,{img_data}")

        with open(LLM_LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write("=== LLM Response ===\n")
            log_file.write(json.dumps(generations_serializable, indent=4, ensure_ascii=False) + "\n\n")


from browser_use.browser.context import BrowserContext, BrowserContextConfig
async def main_task():

    # url = "https://www.w3schools.com/html/html5_draganddrop.asp"
    # url = "https://www.rakuten.co.jp/"
    url = "https://qiita.com"
    task_prompt = f"""

    以下のjsonをもとに、find_target_v2で要素を特定し、アクション実行してください
    [{{
        "action": "click_element",
        "target": "",
        "context_block": "人気商品ランキング",
        "hint": "人気商品ランキングのスライダーを戻す左側のナビゲーション矢印をクリックしてください"
    }}]

    ❌ Do NOT substitute `hint` or `context_block` as `target` if `target` is empty.
    ❌ Do NOT invent or infer a new `target`.
    """
    # task_prompt = f"""
    # generate_site_summary_v3というアクションをステップ1で必ず実行して
    # """
    clear_logs()
    controller = Controller()
    attach_run_recorded_test(controller)
    attach_run_instruction_based_test(controller)
    attach_assert_test_success(controller)
    attach_act_recorded_action(controller)
    # attach_generate_icon_list_image(controller)
    # attach_generate_site_summary_v3(controller)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, callbacks=[LoggingCallbackHandler()])
    page_extraction_llm = ChatOpenAI(model="gpt-4o", temperature=0.1, callbacks=[LoggingCallbackHandler()])

    async def callback(state, model_output, steps: int, page):
        extracted_content=model_output.action_result[0].extracted_content
        try:
            xpath=json.loads(extracted_content).get("xpath","")
        except:
            xpath = None
        # await page.evaluate('''
        #     // 既存のオーバーレイ削除
        #     const overlays = document.querySelectorAll("#playwright-highlight-container");
        #     overlays.forEach(el => el.remove());
        # ''')
        # 目立つオーバーレイを追加
        if(xpath):
            await page.evaluate('''
                (xpath) => {
                    // xpathから要素取得
                    const result = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                    const element = result.singleNodeValue;
                    if (!element) return;

                    const rect = element.getBoundingClientRect();

                    // オーバーレイdiv作成（赤い破線枠）
                    const overlay = document.createElement("div");
                    overlay.id = "playwright-highlight-container";
                    overlay.style.position = "absolute";
                    overlay.style.top = (rect.top + window.scrollY) + "px";
                    overlay.style.left = (rect.left + window.scrollX) + "px";
                    overlay.style.width = rect.width + "px";
                    overlay.style.height = rect.height + "px";
                    overlay.style.border = "4px dashed red";
                    overlay.style.backgroundColor = "transparent";  // ←背景は透明
                    overlay.style.zIndex = "9999";
                    overlay.style.pointerEvents = "none";
                    overlay.style.boxSizing = "border-box";

                    document.body.appendChild(overlay);
                }
            ''', xpath)


        screenshot = await page.screenshot(
            full_page=False,
            animations='disabled',
        )

        screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
        save_image_from_base64(screenshot_b64)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "steps.json")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        test_steps_json_list = json_data.get("actions", [])
        meta_data = json_data.get("metaData", {})
        viewport = meta_data.get("viewport", {"width": 1200, "height": 800})
        
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file: {e}")

    browser = Browser(config=BrowserConfig(headless=True))
    browser_context = BrowserContext(config=BrowserContextConfig(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
        window_width=viewport["width"], 
        window_height=viewport["height"],
     ), browser=browser)
    agent = Agent(
        task=task_prompt,
        llm=llm,
        page_extraction_llm = page_extraction_llm,
        controller=controller,
        browser_context=browser_context,
        initial_actions=[{'open_tab': {'url': url}}],
        register_end_step_callback=callback
    )
    
    actions = []
    i = 0

    while i < len(test_steps_json_list):
        step = test_steps_json_list[i]
        if step["type"] == "recorded":
            actions.append({
                "run_recorded_test": {
                    "test_steps_json": step
                }
            })
        else:
            actions.append({
                "run_instruction_based_test": {
                    "action": step.get("action", "")
                }
            })
        i += 1
    ctx = build_assertion_context_str_from_recording(json_data)
    print(f"actions:", ctx)
    actions.append({
        "assert_test_success": {
            "assertion": "メアドを入力しろ的な文言が出る",
            "action_context": ctx   
        }
    })

    result = await agent.run_actions(actions=actions)




if __name__ == "__main__":
    asyncio.run(main_task())

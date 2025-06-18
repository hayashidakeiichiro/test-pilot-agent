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
from examples.custom_functions.find_section_by_context import attach_find_section_by_context_block
from examples.custom_functions.find_target_v3 import attach_find_target_v3, attach_drag_and_drop
from examples.custom_functions.generate_icon_list_image import attach_generate_icon_list_image
from examples.custom_functions.generate_site_summary_v3 import attach_generate_site_summary_v3


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

    url = "https://zenn.dev"
    # url = "https://zenn.dev"
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
    attach_find_target_v3(controller)
    attach_drag_and_drop(controller)
    # attach_generate_icon_list_image(controller)
    # attach_generate_site_summary_v3(controller)
    browser = Browser(config=BrowserConfig(headless=True))
    browser_context = BrowserContext(config=BrowserContextConfig(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
     ), browser=browser)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, callbacks=[LoggingCallbackHandler()])
    page_extraction_llm = ChatOpenAI(model="gpt-4o", temperature=0.1, callbacks=[LoggingCallbackHandler()])

    async def callback(state, model_output, steps: int):
        screenshot_base64=state.screenshot
        save_image_from_base64(screenshot_base64)
    agent = Agent(
        task=task_prompt,
        llm=llm,
        page_extraction_llm = page_extraction_llm,
        controller=controller,
        browser_context=browser_context,
        initial_actions=[{'open_tab': {'url': url}}],
        register_new_step_callback=callback,
        register_done_callback=callback,
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "steps.json")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        test_steps_json_list = json_data
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file: {e}")
    
    actions = []
    i = 0

    while i < len(test_steps_json_list):
        step = test_steps_json_list[i]
        if step["type"] == "recorded":
            actions.append({
                "find_target": {
                    "test_steps_json": step
                }
            })
        else:
            actions.append({
                "find_target": {
                    "test_steps_json": step
                }
            })
        i += 1

    print(f"actions: {actions}")

    result = await agent.run_actions(actions=actions)




if __name__ == "__main__":
    asyncio.run(main_task())

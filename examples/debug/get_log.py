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

from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig

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


async def main_task():

    url = "https://zenn.dev/"
    # url = "https://zenn.dev"
    task_prompt = f"""
    techセクションの最初の記事のユーザーアイコンをクリックして
    """
    clear_logs()
    browser = Browser(config=BrowserConfig(headless=True))
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8, callbacks=[LoggingCallbackHandler()])

    agent = Agent(
        task=task_prompt,
        llm=llm,
        browser=browser,
        initial_actions=[{'open_tab': {'url': url}}]
    )

    result = await agent.run(max_steps=2)


if __name__ == "__main__":
    asyncio.run(main_task())

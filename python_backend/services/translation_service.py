# translation_service.py

import asyncio
import json
import logging
from typing import Optional

import aioredis
from transformers import AutoTokenizer
from langchain.llms.base import LLM
from pydantic import BaseModel, Extra
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

LLAMA_MODEL_NAME = "ensemble"
TRITON_SERVER_URL = "localhost:8000"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
TRANSLATION_CHANNEL = "translations"
TRANSCRIPTION_CHANNEL = "transcriptions"

redis = aioredis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}/0")

class TritonLLM(LLM):
    llm_url: str = f"http://{TRITON_SERVER_URL}/v2/models/{LLAMA_MODEL_NAME}/generate"

    class Config:
        extra = 'forbid'  

    @property
    def _llm_type(self) -> str:
        return "Triton LLM"

    def _call(
        self,
        prompt: str,
        temperature: float,
        stop: Optional[list] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        payload = {
            "text_input": prompt,
            "parameters": {
                "max_tokens": 200,
                "temperature": temperature,
                "top_k": 50,
            }
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.llm_url, json=payload, headers=headers)
            response.raise_for_status()
            translation = response.json().get('text_output', '')
            if not translation:
                raise ValueError("No 'text_output' field in the response.")
            return translation
        except requests.exceptions.RequestException as e:
            logging.error(f"LLM request failed: {e}")
            return ""
        except ValueError as ve:
            logging.error(f"LLM response error: {ve}")
            return ""

    @property
    def _identifying_params(self) -> dict:
        return {"llmUrl": self.llm_url}

llm = TritonLLM()

try:
    tokenizer = AutoTokenizer.from_pretrained("/home/chris/engines/Meta-Llama-3.1-8B-Instruct")
    logging.info("Tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load tokenizer: {e}")
    exit(1)

def translate_text(transcription: str) -> Optional[str]:
    """
    Translates the transcription from German to English using the LLM.
    """
    messages = [
        {"role": "system", "content": "You are an AI translator working for the German company P&I which builds the LOGA platform, and your task is to translate given sentences written in German to English. Just answer the English translation without any other explanation or comments."},
        {"role": "user", "content": transcription},
    ]

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    translation = llm(prompt=prompt, temperature=0.0)
    return translation.strip() if translation else None

async def publish_message(message: dict, channel: str):
    """
    Publishes a JSON message to the specified Redis channel.
    """
    try:
        await redis.publish(channel, json.dumps(message))
        logging.info(f"Published message to {channel}: {message}")
    except Exception as e:
        logging.error(f"Redis Publish Error: {e}")

async def translate_and_publish(message: dict):
    """
    Translates the transcription and publishes the translation.
    """
    transcription = message.get('transcription')
    if not transcription:
        logging.warning("Received message without 'transcription' field.")
        return

    translation = translate_text(transcription)
    if translation:
        logging.info(f"Translation (English): {translation}")

        translated_message = {
            'translation': translation
        }
        await publish_message(translated_message, TRANSLATION_CHANNEL)
    else:
        logging.warning("Translation failed.")

async def listen_to_redis():
    """
    Listens to the transcriptions Redis channel and processes translations.
    """
    try:
        pubsub = redis.pubsub()
        await pubsub.subscribe(TRANSCRIPTION_CHANNEL)
        logging.info(f"Subscribed to Redis channel: {TRANSCRIPTION_CHANNEL}")

        async for message in pubsub.listen():
            if message['type'] == 'message':
                data = message['data']
                try:
                    msg = json.loads(data)
                    logging.info(f"Received transcription: {msg}")
                    await translate_and_publish(msg)
                except json.JSONDecodeError:
                    logging.error("Failed to decode JSON message.")
    except asyncio.CancelledError:
        logging.info("Redis listener task cancelled.")
    except Exception as e:
        logging.error(f"Error in Redis listener: {e}")

async def main():
    await listen_to_redis()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Translation service stopped by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

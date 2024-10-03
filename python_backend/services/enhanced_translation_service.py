import asyncio
import json
import logging
import time
from typing import Optional, List

import aioredis
from transformers import AutoTokenizer
from langchain.llms.base import LLM
from pydantic import BaseModel, Extra
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
from difflib import SequenceMatcher  

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

LLAMA_MODEL_NAME = "ensemble"
TRITON_SERVER_URL = "localhost:8000"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
TRANSLATION_CHANNEL = "better_translations"
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
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        payload = {
            "text_input": prompt,
            "parameters": {
                "max_tokens": 500,
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

transcription_buffer = []
buffer_lock = asyncio.Lock()  
translation_history = []  
MAX_BUFFER_SIZE = 5  
MIN_TRANSLATION_INTERVAL = 1.0  
last_translation_time = time.time()

def translate_text(transcription: str) -> Optional[str]:
    """
    Translates the transcription from German to English using the LLM.
    """
    messages = [
        {"role": "system", "content": "You are an AI translator working for the German company P&I, which builds the LOGA platform. Your task is to translate the following text from German to English. Just provide the English translation without any additional explanation or comments."},
        {"role": "user", "content": transcription.strip()},
    ]

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    translation = llm(prompt=prompt, temperature=0.0)
    return translation.strip() if translation else None

def sentences_similarity(s1: str, s2: str) -> float:
    """
    Calculates the similarity between two sentences.
    """
    return SequenceMatcher(None, s1, s2).ratio()

def is_semantically_complete(text: str) -> bool:
    """
    Uses the LLM to determine if the text is semantically complete.
    """
    messages = [
        {"role": "system", "content": "Determine whether the following German text is a complete sentence or expresses a complete thought. Answer 'Yes' or 'No' without any additional text."},
        {"role": "user", "content": text.strip()},
    ]

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    response = llm(prompt=prompt, temperature=0.0).strip().lower()

    logging.info(f"Semantic completeness check response: {response}")

    return response.startswith('yes')

async def publish_message(message: dict, channel: str):
    """
    Publishes a JSON message to the specified Redis channel.
    """
    try:
        await redis.publish(channel, json.dumps(message))
        logging.info(f"Published message to {channel}: {message}")
    except Exception as e:
        logging.error(f"Redis Publish Error: {e}")

async def process_buffer():
    """
    Processes the transcription buffer to translate when appropriate.
    """
    global transcription_buffer, translation_history, last_translation_time

    async with buffer_lock:
        if not transcription_buffer:
            return

        current_text = ' '.join(transcription_buffer).strip()

        buffer_is_full = len(transcription_buffer) >= MAX_BUFFER_SIZE
        semantically_complete = is_semantically_complete(current_text)

        time_since_last_translation = time.time() - last_translation_time

        if semantically_complete or buffer_is_full or time_since_last_translation >= MIN_TRANSLATION_INTERVAL:
            translation = translate_text(current_text)
            if translation:
                logging.info(f"Translation (English): {translation}")

                if not translation_history or sentences_similarity(translation, translation_history[-1]['translation']) < 0.9:
                    should_finalize = semantically_complete or buffer_is_full

                    translated_message = {
                        'translation': translation,
                        'finalized': should_finalize  
                    }
                    await publish_message(translated_message, TRANSLATION_CHANNEL)
                    translation_history.append({'translation': translation, 'finalized': should_finalize})
                else:
                    logging.info("Translation is similar to the previous one. Skipping publication.")

                last_translation_time = time.time()

                if semantically_complete or buffer_is_full:
                    transcription_buffer.clear()
            else:
                logging.warning("Translation failed.")

async def translate_and_publish(message: dict):
    """
    Adds transcription to the buffer and processes it.
    """
    transcription = message.get('transcription')
    if not transcription:
        logging.warning("Received message without 'transcription' field.")
        return

    async with buffer_lock:
        transcription_buffer.append(transcription.strip())

    await process_buffer()

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

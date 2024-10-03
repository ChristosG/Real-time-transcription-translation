# transcription_service.py

import asyncio
import json
import logging
import time
import numpy as np
import sounddevice as sd
import webrtcvad
import collections
import aioredis
import threading

from typing import Optional
from transformers import AutoTokenizer
from langchain.llms.base import LLM
from pydantic import BaseModel
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

WHISPER_PROMPT = "<|startoftranscript|><|de|><|transcribe|><|notimestamps|>"
LANGUAGE_CODE = "de"
WHISPER_MODEL_NAME = "whisper"
TRITON_SERVER_URL = "localhost:8001"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
TRANSCRIPTION_CHANNEL = "transcriptions"
SAMPLE_RATE = 16000  # Hz
FRAME_DURATION = 30  # ms
OVERLAP_DURATION = 0.3  # sec
COOLDOWN_PERIOD = 0.2  # sec
MAX_AUDIO_DURATION = 3.8  # secs
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000) * 2  # 16-bit audio

# REdis init
redis = aioredis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}/0")

class TritonLLM(LLM):
    llm_url: str = f"http://{TRITON_SERVER_URL}/v2/models/llama3.1/generate"

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
                "max_tokens": 100,
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

def transcribe_audio(
    audio_data: np.ndarray,
    whisper_prompt: str,
    language: str,
    model_name: str = "whisper-large-v3",
    server_url: str = "localhost:8001"
) -> Optional[str]:
    """
    Sends audio data to the Triton server via gRPC for transcription.
    """
    try:
        triton_client = grpcclient.InferenceServerClient(url=server_url, verbose=False)

        if not triton_client.is_server_live():
            logging.error("Triton server is not live.")
            return None
        if not triton_client.is_model_ready(model_name):
            logging.error(f"Model {model_name} is not ready on Triton server.")
            return None

        samples = audio_data.astype(np.float32)
        samples = np.expand_dims(samples, axis=0) 

        inputs = []

        input_wav = grpcclient.InferInput("WAV", samples.shape, np_to_triton_dtype(samples.dtype))
        input_wav.set_data_from_numpy(samples)
        inputs.append(input_wav)

        input_text = grpcclient.InferInput("TEXT_PREFIX", [1, 1], "BYTES")
        input_text.set_data_from_numpy(np.array([[whisper_prompt.encode()]], dtype=object))
        inputs.append(input_text)

        outputs = [grpcclient.InferRequestedOutput("TRANSCRIPTS")]

        response = triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
        )

        transcription = response.as_numpy("TRANSCRIPTS")[0]
        if isinstance(transcription, np.ndarray):
            transcription = b" ".join(transcription).decode("utf-8")
        else:
            transcription = transcription.decode("utf-8")

        logging.debug(f"Raw Transcription: {transcription}")
        return transcription

    except Exception as e:
        logging.error(f"Transcription Error: {e}")
        return None

def normalize_audio(audio_int16: np.ndarray) -> Optional[np.ndarray]:
    """
    Normalizes audio data to [-1.0, 1.0] and clamps values.
    """
    try:
        audio_float32 = audio_int16.astype(np.float32) / np.iinfo(np.int16).max  

        max_val = np.max(np.abs(audio_float32))
        if max_val > 0:
            audio_normalized = audio_float32 / max_val
            audio_normalized = np.clip(audio_normalized, -1.0, 1.0)
        else:
            audio_normalized = audio_float32

        if not np.all(np.abs(audio_normalized) <= 1.0):
            logging.error("Audio normalization failed to clamp values within [-1.0, 1.0].")
            return None

        return audio_normalized
    except Exception as e:
        logging.error(f"Audio normalization error: {e}")
        return None

async def publish_message(message: dict, channel: str):
    """
    Publishes a JSON message to the specified Redis channel.
    """
    try:
        await redis.publish(channel, json.dumps(message))
        logging.info(f"Published message to {channel}: {message}")
    except Exception as e:
        logging.error(f"Redis Publish Error: {e}")

def record_audio_and_publish(
    whisper_prompt: str,
    language: str,
    whisper_model_name: str,
    transcription_channel: str,
    sample_rate: int,
    frame_duration_ms: int,
    overlap_duration_s: float,
    max_audio_duration_s: float,
    cooldown_period_s: float,
    loop: asyncio.AbstractEventLoop
):
    """
    Records audio from the microphone, applies VAD to detect speech segments,
    transcribes detected speech using Triton, and publishes
    transcriptions to Redis.
    """
    vad = webrtcvad.Vad(1) 
    channels = 1
    dtype = 'int16'
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    ring_buffer = collections.deque(maxlen=int(overlap_duration_s / (frame_duration_ms / 1000)))
    triggered = False
    buffer_bytes = b''
    buffer_duration = 0.0
    last_transcription_time = 0
    total_frames = 0

    transcribe_lock = threading.Lock()

    async def process_transcription(buffer_bytes: bytes):
        nonlocal last_transcription_time
        with transcribe_lock:
            current_time = time.time()
            if current_time - last_transcription_time < cooldown_period_s:
                logging.debug("Cooldown period active. Skipping transcription.")
                return

            audio_int16 = np.frombuffer(buffer_bytes, dtype=np.int16)
            audio_normalized = normalize_audio(audio_int16)
            if audio_normalized is None:
                return

            transcription = transcribe_audio(
                audio_data=audio_normalized,
                whisper_prompt=whisper_prompt,
                language=language,
                model_name=whisper_model_name,
                server_url=TRITON_SERVER_URL
            )

            if transcription and 'Vielen Dank.' not in transcription and 'Tschüss' not in transcription and len(transcription.strip()) > 2:
                logging.info("=== New Transcription ===")
                logging.info(f"Transcription (German): {transcription}")

                message = {'transcription': transcription}
                asyncio.run_coroutine_threadsafe(
                    publish_message(message, transcription_channel),
                    loop
                )

                last_transcription_time = current_time
            else:
                logging.warning(f"Transcription failed or contains 'Vielen Dank.': {transcription}")

    def audio_callback(indata, frames, time_info, status):
        nonlocal triggered, buffer_bytes, buffer_duration, total_frames

        if status:
            logging.warning(f"Recording Status: {status}")
        frame = indata.flatten().tobytes()
        is_speech = vad.is_speech(frame, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.7 * ring_buffer.maxlen:
                triggered = True
                buffer_bytes = b''.join([f for f, s in ring_buffer])
                buffer_duration = len(buffer_bytes) / (sample_rate * 2)
                ring_buffer.clear()
                logging.info("Speech detected. Triggering transcription.")
        else:
            buffer_bytes += frame
            buffer_duration += frame_duration_ms / 1000.0
            total_frames += 1

            if buffer_duration >= max_audio_duration_s:
                logging.info("Maximum buffer duration exceeded. Initiating transcription.")
                asyncio.run_coroutine_threadsafe(
                    process_transcription(buffer_bytes),
                    loop
                )
                buffer_bytes = b''
                buffer_duration = 0.0
                triggered = False
                ring_buffer.clear()
            else:
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.7 * ring_buffer.maxlen:
                    logging.info("End of speech detected. Initiating transcription.")
                    asyncio.run_coroutine_threadsafe(
                        process_transcription(buffer_bytes),
                        loop
                    )
                    buffer_bytes = b''
                    buffer_duration = 0.0
                    triggered = False
                    ring_buffer.clear()

    async def record():
        try:
            with sd.InputStream(
                callback=audio_callback,
                channels=channels,
                samplerate=sample_rate,
                dtype=dtype,
                blocksize=frame_size
            ):
                logging.info("Recording with VAD... Press Ctrl+C to stop.")
                print("Recording with VAD... Press Ctrl+C to stop.")
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"Error initializing audio stream: {e}")
            print(f"Error initializing audio stream: {e}")

    asyncio.run_coroutine_threadsafe(record(), loop)

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        recording_thread = threading.Thread(
            target=record_audio_and_publish,
            args=(
                WHISPER_PROMPT,
                LANGUAGE_CODE,
                WHISPER_MODEL_NAME,
                TRANSCRIPTION_CHANNEL,
                SAMPLE_RATE,
                FRAME_DURATION,
                OVERLAP_DURATION,
                MAX_AUDIO_DURATION,
                COOLDOWN_PERIOD,
                loop
            )
        )
        recording_thread.start()

        loop.run_forever()
    except KeyboardInterrupt:
        logging.info("Recording stopped by user.")
        print("\nStopping recording...")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")
    finally:
        loop.stop()
        recording_thread.join()


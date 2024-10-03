# app.py

import asyncio
import json
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import aioredis
import logging

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://zelime.duckdns.org"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_HOST = "localhost"
REDIS_PORT = 6379
TRANSLATION_CHANNEL = "translations"
TRANSCRIPTION_CHANNEL = "transcriptions"
BETTER_TRANSLATION_CHANNEL = "better_translations"  

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.active_connections.append(websocket)
        logger.info(f"Client connected: {websocket.client}")

    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"Client disconnected: {websocket.client}")

    async def broadcast(self, message: str):
        async with self.lock:
            for connection in self.active_connections.copy():
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending message to {connection.client}: {e}")
                    await self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            #handle data from client? maybe TODO their mic ? or possibly their v_sink?
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket)

@app.on_event("startup")
async def startup_event():
    try:
        app.state.redis = aioredis.from_url(
            f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
            decode_responses=True,
            max_connections=10  
        )
        app.state.pubsub = app.state.redis.pubsub()
        await app.state.pubsub.subscribe(TRANSLATION_CHANNEL, TRANSCRIPTION_CHANNEL, BETTER_TRANSLATION_CHANNEL)
        logger.info(f"Subscribed to Redis channels: {TRANSLATION_CHANNEL}, {TRANSCRIPTION_CHANNEL}, {BETTER_TRANSLATION_CHANNEL}")
        
        app.state.listener_task = asyncio.create_task(listen_to_redis(app.state.pubsub))
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    await manager.broadcast("Server is shutting down.")
    
    app.state.listener_task.cancel()
    try:
        await app.state.listener_task
    except asyncio.CancelledError:
        logger.info("Redis listener task cancelled.")
    
    await app.state.pubsub.unsubscribe(TRANSLATION_CHANNEL, TRANSCRIPTION_CHANNEL, BETTER_TRANSLATION_CHANNEL)
    await app.state.redis.close()
    logger.info("Redis connection closed.")

async def listen_to_redis(pubsub):
    logger.info("Listening to Redis channels for messages...")
    try:
        async for message in pubsub.listen():
            if message['type'] == 'message':
                msg = message['data']
                channel = message['channel']
                logger.info(f"Received message from Redis channel '{channel}': {msg}")
                message_to_send = json.dumps({
                    'channel': channel,
                    'data': msg
                })
                await manager.broadcast(message_to_send)
    except asyncio.CancelledError:
        logger.info("Redis listener task cancelled.")
    except Exception as e:
        logger.error(f"Error in Redis listener: {e}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7000, reload=False, workers=1)

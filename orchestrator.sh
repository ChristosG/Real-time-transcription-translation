#!/bin/bash

pactl load-module module-null-sink sink_name=Virtual_Sink sink_properties=device.description=Virtual_Sink
pacmd set-default-source Virtual_Sink.monitor

start_triton(){
	echo "Starting Triton..."
	gnome-terminal -- bash -c "cd ./python_backend  || exit; ./start_triton.sh; exec bash"
}


start_yarn() {
    echo "Starting yarn..."
    gnome-terminal -- bash -c "yarn start; exec bash"
}

start_transcription_service(){
	echo "Starting Transcription Service..."
        gnome-terminal -- bash -c "cd ./python_backend/services || exit; python transcription_service.py; exec bash"
}

start_translation_service(){
	echo "Starting Translation Service..."
        gnome-terminal -- bash -c "cd ./python_backend/services || exit; python translation_service.py; exec bash"
}

start_better_translation_service(){
	echo "Starting Translation Service..."
        gnome-terminal -- bash -c "cd ./python_backend/services || exit; python enhanced_translation_service.py; exec bash"
}


start_user_proxying(){
	echo "Starting Socket Connector..."
        gnome-terminal -- bash -c "cd ./python_backend || exit; gunicorn new_fast_api:app -k uvicorn.workers.UvicornWorker --workers 4 --bind 0.0.0.0:7000; exec bash"
}

start_triton
start_transcription_service
start_translation_service
start_better_translation_service
start_user_proxying
start_yarn

echo "All processes have been started in separate terminals."

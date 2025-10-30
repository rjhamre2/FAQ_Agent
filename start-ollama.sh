#!/bin/sh

# Start Ollama server in background
ollama serve &

# Wait for server to be ready
sleep 10

# Pull the required model
ollama pull nomic-embed-text

# Keep the script running
wait

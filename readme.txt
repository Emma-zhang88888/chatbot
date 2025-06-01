1. Install Ollama
https://github.com/ollama/ollama/blob/main/README.md#quickstart
pull the following models from your terminal:
ollama pull llama3:8b
ollama pull mxbai-embed-large
ollama pull nomic-embed-text
Please make sure Ollama is running when you are testing the code.

Install necessary python modules.

2. Use the main.py under the code folder to run chatbots
There are 3 chatbots. Type 1 for llama3 model, type 2 for mxbai-embed-large multilingual embedding model, type anything else for nomic-embed-text embedding mode(best accuracy).
Type 'exit' if you want to stop.
After responding to each question, a survey question will be asked about the answer's correctness. Before exiting, you can provide feedback.

3. Testing questions and logging files are saved under the doc folder

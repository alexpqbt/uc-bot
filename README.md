# UC Atlas
This is made to learn building simple RAG chatbot using data from the [university](https://www.universityofcebu.net/)

# How to use
1. Clone the repo
2. Create an `.env` file and add whatever is needed provided from the `.env.example`
3. [`uv`](https://docs.astral.sh/uv/getting-started/installation/) is needed for this project to run `uv sync`
4. then run `uv run agent.py` to run the chatbot
5. access [`http://localhost:7860`](http://localhost:7860) to open the chat interface

# Status
Satisfied but the code is messy

# Things that I used
- LangChain for everything dealing with document conversion, embedding, making agents, etc.
- Gemini and Llama3.1 for the LLM
- Gradio for UI
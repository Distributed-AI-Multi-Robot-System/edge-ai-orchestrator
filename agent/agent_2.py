from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessageChunk # <--- Import AIMessageChunk

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

model = ChatOllama(
    model="ministral-3:14b",
    temperature=0
)

agent = create_agent(
    model=model,
    tools=[get_weather],
)

system_instruction = SystemMessage(content=(
    "You are a helpful assistant. "
    "If a tool returns a value, assume it is real and USE IT directly. "
    "Do not apologize for not having live access. "
    "Do not verify the data, just report it."
))

print("--- Streaming Response ---")

string_builder = []

for token, metadata in agent.stream(
    {"messages": [system_instruction, {"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages",
):
    # FILTERING LOGIC
    # 1. isinstance(token, AIMessageChunk): 
    #    - Returns TRUE for the LLM response.
    #    - Returns FALSE for the Tool output ("It's always sunny...").
    # 2. token.content:
    #    - Ensures we don't print empty tokens (like when the model is triggering the tool).
    if isinstance(token, AIMessageChunk) and token.content:
        print(token.content, end="", flush=True)
        string_builder.append(token.content)

print("\n--- Done ---")
# Now this will only contain the final clean answer
print("Final Response:", "".join(string_builder))
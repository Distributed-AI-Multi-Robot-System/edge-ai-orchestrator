from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessageChunk # <--- Import AIMessageChunk

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def get_hotel_price(days: int, city: str) -> str:
    """Get hotel price for a given city for a number of days. The supported cities are San Francisco, New York, and Los Angeles."""
    cities = {"San Francisco": 200, "New York": 250, "Los Angeles": 180}
    if city in cities:
        price = cities[city] * days
        return f"The hotel price in {city} for {days} days is ${price}."
    return f"Sorry, we don't have data for {city}."

model = ChatOllama(
    model="ministral-3:14b",
    temperature=0
)

agent = create_agent(
    model=model,
    tools=[get_weather, get_hotel_price],
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
    {"messages": [system_instruction, {"role": "user", "content": "What is the weather in SF? Also tell me the hotel price for 4 days in San Fransico. I want to visit the Golden Gate Bridge."}]},
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
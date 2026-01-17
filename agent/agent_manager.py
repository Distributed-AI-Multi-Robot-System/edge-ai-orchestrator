import re
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessageChunk # <--- Import AIMessageChunk
import emoji

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

class AgentManager:
    def __init__(self):
        model = ChatOllama(
            model="ministral-3:14b",
            temperature=0
        )

        # Regex to remove standard Markdown symbols
        # Matches: **bold**, __italic__, `code`, # Headers, [Links](...)
        self.md_pattern = re.compile(r'\*+|_+|`+|#+|\[([^\]]+)\]\([^\)]+\)')
        self.system_instruction = SystemMessage(content=(
        "You are a helpful assistant. "
        "If a tool returns a value, assume it is real and USE IT directly. "
        "Do not apologize for not having live access. "
        "Do not verify the data, just report it."
        ))
        self.agent = create_agent(
            model=model,
            tools=[get_weather, get_hotel_price],
        )


    async def stream_agent_response(self, user_message: str):
        for token, _ in self.agent.stream(
            {"messages": [self.system_instruction, {"role": "user", "content": user_message}]},
            stream_mode="messages",
        ):
            if isinstance(token, AIMessageChunk) and token.content:
                # remove hashtags and emojis, leave german and french and italian characters
                sanitized_from_emojies = emoji.replace_emoji(token.content, replace="")
                sanitized_token_content = self.clean_llm_text_for_tts(sanitized_from_emojies)
                if not sanitized_token_content:
                    continue
                
                yield sanitized_token_content

    def clean_llm_text_for_tts(self, text: str) -> str:
        # 1. Handle Links: Convert [Text](URL) -> Text
        # We do this first so the brackets don't get messed up by later steps.
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # 2. Handle Symbols: Remove *, _, #, `, ~
        # We replace these with an empty string.
        return re.sub(r'[\*\_#`~]', '', text)
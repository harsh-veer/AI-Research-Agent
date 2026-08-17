import os
import certifi
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent


# =====================================================
# CONFIGURATION
# =====================================================

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# =====================================================
# STREAMLIT PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🤖",
    layout="centered"
)


# =====================================================
# CHECK API KEYS
# =====================================================

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is missing from .env")

if not TAVILY_API_KEY:
    st.error("TAVILY_API_KEY is missing from .env")

if not WEATHERSTACK_API_KEY:
    st.warning("WEATHERSTACK_API_KEY is missing from .env")


# =====================================================
# HEADER
# =====================================================

st.title("🤖 AI Research Agent")

st.markdown(
    """
    Ask me questions and I can use:

    - 🧠 **Gemini** for reasoning
    - 🔎 **Tavily** for web search
    - 🌤️ **WeatherStack** for current weather
    """
)


# =====================================================
# TAVILY SEARCH TOOL
# =====================================================

search_tool = TavilySearch(
    max_results=4
)


# =====================================================
# WEATHER TOOL
# =====================================================

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    if not WEATHERSTACK_API_KEY:
        return "WeatherStack API key is not configured."

    url = "https://api.weatherstack.com/current"

    params = {
        "access_key": WEATHERSTACK_API_KEY,
        "query": city
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        if "current" not in data:
            return f"Could not fetch weather data for {city}."

        weather_description = (
            data["current"]["weather_descriptions"][0]
            if data["current"]["weather_descriptions"]
            else "Unknown"
        )

        return (
            f"City: {city}\n"
            f"Temperature: {data['current']['temperature']}°C\n"
            f"Weather: {weather_description}\n"
            f"Humidity: {data['current']['humidity']}%\n"
            f"Wind Speed: {data['current']['wind_speed']} km/h"
        )

    except Exception as e:
        return f"Could not fetch weather data for {city}: {str(e)}"


# =====================================================
# GEMINI MODEL
# =====================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
    api_key=GEMINI_API_KEY
)


# =====================================================
# AGENT TOOLS
# =====================================================

tools = [
    search_tool,
    get_weather_data
]


# =====================================================
# CREATE AGENT
# =====================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
    You are a helpful AI research assistant.

    You have access to:
    1. Tavily web search for current information.
    2. WeatherStack for current weather.

    Use Tavily when the user asks for current, recent,
    factual, or web-based information.

    Use the weather tool when the user asks for current
    weather information.

    When a user asks multiple things, complete all tasks
    and provide a clear final answer.
    """
)


# =====================================================
# CHAT HISTORY
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================================
# DISPLAY PREVIOUS MESSAGES
# =====================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =====================================================
# USER INPUT
# =====================================================

user_input = st.chat_input(
    "Ask something... e.g. Find the capital of India and its weather"
)


# =====================================================
# PROCESS USER QUESTION
# =====================================================

if user_input:

    # Display user message
    st.chat_message("user").markdown(user_input)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = agent.invoke(
                    {
                        "messages": st.session_state.messages
                    }
                )

                content = response["messages"][-1].content
                if isinstance(content, list):
                    assistant_response = "\n".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                else:
                    assistant_response = str(content)
                st.markdown(assistant_response)
                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_response
                    }
                )

            except Exception as e:

                st.error(
                    f"An error occurred: {str(e)}"
                )
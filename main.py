
import os
import certifi
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
#from langchain import hub
from langchain.tools import tool
import requests

#%pip install -U tavily-python langchain-community
#%pip install -U langchain langchain-google-genai langchain-community tavily-python
import langchain
from langchain.agents import create_agent
#=====================
# Load environment variables
#=====================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("Tavily_API_KEY")
WEATHERSTACK_API_KEY=os.getenv("WEATHERSTACK_API_KEY")

search_tool=TavilySearchResults(max_results=4)

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )
result=search_tool.invoke("Find the capital of India"
        "and then find its current weather.")
result
#================
#LLM
#================
llm=ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
    api_key=GEMINI_API_KEY
)
import os
from dotenv import load_dotenv

load_dotenv()

print("API key loaded:", bool(os.getenv("GEMINI_API_KEY")))
response = llm.invoke("WHO is current CM Of UP?")
print(response.content)
#=======
#prompt
#=======

#prompt=hub.pull("hwchase17/react")
#prompt

#=======
#tools
#=======

tools=[search_tool,get_weather_data]
#=========
#create agent
#=========

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant. Use the search tool when you need current information."
)
##############
# run
response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Find the capital of India and then find its current weather ."
        }
    ]
})

print(response["messages"][-1].content)
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
# Tool calling agent with custom tools 
from tools import search_tool, wiki_tool, save_tool

# from tools import DuckDuckGoSearchRun, WikipediaQueryRun, WikipediaAPIWrapper


load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# fields for LLM csll
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    
# set up an LLM, but give it to an agent
# llm = ChatOpenAI(model="gpt-4o")
llm = ChatGoogleGenerativeAI(model="gemini-pro")
# llm = ChatAnthropic(model="claude-3-7-sonnet-20250219")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)


# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper about problems in Agriculture.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# response = llm.invoke("What are some critical problems in Agriculture Sector?")
# print(response)

# Tools list
tools=[search_tool, wiki_tool, save_tool]
# Agent
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = input("What can I help you research about Agriculture?")

# Use agent executer to generate some kind of responds
raw_response = agent_executor.invoke({"query": query})
# print(raw_response)
# print(raw_response["output"])

# using the parser to parse the output in a structured format
try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print(f"Error parsing response:", e, "Raw response - ", raw_response)

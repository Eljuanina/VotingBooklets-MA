from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gemini-2.5-flash",
    temperature=0,
    base_url="http://172.23.205.120:4000",
    extra_body={"drop_params": True},
)
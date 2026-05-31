import glob
from dotenv import load_dotenv
from langchain.agents import create_agent
from llm_config import llm
from agent_tools import TOOLS

load_dotenv()

agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=(
        "You are an expert corpus linguist building multilingual parallel corpora "
        "from Swiss federal voting booklets.\n\n"

        "Your goal is to produce the cleanest paragraph-level parallel corpus you can "
        "from the PDFs you are given. Use your tools and judgment freely — you decide "
        "what to do and in what order based on what you observe.\n\n"

        "HARD LIMITS\n"
        "- refine_alignment: at most 2 calls per year. Stop immediately if it returns REFINE_SKIPPED.\n"
        "- ocr_pdf_gemini costs ~20–50x more per page than ocr_pdf_tesseract. "
        "Only escalate if check_ocr_quality gives you a clear reason to — never as a default.\n"
        "- Do not retry a failed tool more than once. Move on.\n\n"

        "WHAT TO ACCEPT\n"
        "- Gaps in source documents mean some rows will have empty translations. This is normal.\n"
        "- check_alignment_quality returns REFINE_SKIPPED when the corpus is already good enough. Trust it.\n\n"

        "OUTPUT\n"
        "A JSONL file with one aligned paragraph group per line, "
        "language codes (de/fr/it/rm) as keys, plus a 'date' field."
    ),
)


def run_agent(message: str):
    for step in agent.stream(
        {"messages": [{"role": "user", "content": message}]},
        stream_mode="updates"
    ):
        for node_name, node_output in step.items():
            print(f"\n{'='*40}")
            print(f"STEP: {node_name}")
            print(f"{'='*40}")
            for msg in node_output.get("messages", []):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print("🤔 THINKING → calling tool(s):")
                    for tc in msg.tool_calls:
                        print(f"   🔧 Tool: {tc['name']}")
                        print(f"   📥 Args: {tc['args']}")
                elif hasattr(msg, "name") and msg.name:
                    print(f"✅ TOOL RESULT [{msg.name}]:")
                    print(f"   {str(msg.content)[:300]}")
                elif hasattr(msg, "content") and msg.content:
                    print(f"💬 RESPONSE: {msg.content}")


if __name__ == "__main__":
    for year in ["1977", "1985", "2007"]:
        pdfs = sorted(set(glob.glob(f"data/booklets/**/*{year}*.pdf", recursive=True)))
    
        if not pdfs:
            print(f"❌ No PDFs found for year {year} in data/booklets/")
        else:
            print(f"\n📚 Found {len(pdfs)} PDF(s) for {year}:\n")
            for p in pdfs:
                print(f"   {p}")

            pdf_list = "\n".join(f"- {p}" for p in pdfs)
            message = (
                f"Build a multilingual parallel corpus for the Swiss federal vote of {year}.\n\n"
                f"PDFs:\n{pdf_list}"
            )

            print("\n--- AGENT RUNNING ---\n")
            run_agent(message)

    print("\n--- DONE ---")



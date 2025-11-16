import os
from dotenv import load_dotenv
import sys
import kagglesdk
import kaggle
kaggle.api.authenticate()
import json
load_dotenv()
from holistic_ai_bedrock import get_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Callable
from huggingface_hub import HfApi
from datetime import datetime
import tiktoken
import codecarbon
import time
import langsmith 
import uuid
from langsmith import traceable
from langchain_valyu import ValyuSearchTool
from valyu import Valyu

def get_agent():
    return create_react_agent(
        llm,
        tools=[searchKaggle, search_huggingface_hub, searchValyu],
        prompt=system_prompt,
        response_format=userInputCheck
    )

process_logs = []

def log_process(message: str, log_callback: Callable[[str], None] = None):
    """Log a process step and optionally call a callback."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    process_logs.append(log_entry)
    print(log_entry)
    if log_callback:
        log_callback(log_entry)


encoding = tiktoken.encoding_for_model('gpt-5-mini')

# Helper function to count tokens
def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(encoding.encode(text))




@tool
def searchValyu(n: str) -> str:
    """
    Valyu DeepSearch surfaces real-time web content, research, financial data, and proprietary datasets; use it for specific, up-to-date search across domains.
    """
    
    valyu = Valyu()

    # Use the input argument 'n'
    response = valyu.search(
        query=n,
        max_num_results=1,
        search_type="all",
        response_length=500,
    )

    results_list = response.results

    if results_list:
     
        return results_list[0].content
    else:
        return "No results found."



@tool
def searchKaggle(n: str) -> List[Dict[str, str]]:
    """Search and return datasets in Kaggle website. A keyword is used to search through the database of kaggle

    Args: 
        n: Keyword to initiate search

    Return:
        A list of dictionaries with dataset information from kaggle search.
    """

    log_process("Search Kaggle Called")

    try:
        results = kaggle.api.dataset_list(search=n)
        
        processed_results = []
        for dataset in results[:4]:
            try:
                title = str(dataset.ref)
                
                description = getattr(dataset, 'description', None)
                if description and len(description) > 150:
                    description = description[:150] + "..."
                elif not description:
                    description = "No description available"
                
                author = getattr(dataset, 'creator_name', None) or 'Unknown'
                
                last_updated = getattr(dataset, 'last_updated', None)
                if last_updated:
                    last_updated_str = str(last_updated)
                else:
                    last_updated_str = 'Unknown'
                
                processed_results.append({
                    "title": title,
                    "author": author,
                    "description": description,
                    "last_updated_at": last_updated_str,
                    "link": f"https://www.kaggle.com/datasets/{title}"
                })
                
            except Exception as e:
                print(f"[Kaggle Warning] Skipped dataset: {e}")
                continue
        
        log_process(f"Kaggle Found {len(processed_results)} datasets")

        print(f"[Kaggle] Found {len(processed_results)} datasets")


        return processed_results
        
    except Exception as e:
        print(f"[Kaggle ERROR] {e}")
        return []



@tool
def search_huggingface_hub(query: str) -> List[Dict[str, str]]:
    """Search and return datasets on the Hugging Face Hub. A keyword is used 
    to search the Hub for matching datasets.

    Args: 
        query: Keyword to initiate search.

    Return:
        A list of dictionaries with dataset information from hugging face search.
    """

    log_process("Search Hugging Face Called")

    
    try:
        api = HfApi()
        results_generator = api.list_datasets(search=query, limit=4)
        top_results = list(results_generator)
        
        processed_results = []
        for dataset in top_results:
            try:
                # Safely get all attributes
                author = getattr(dataset, 'author', 'Unknown')
                
                # Get and truncate description
                description = getattr(dataset, 'description', 'No description available')
                if description and len(description) > 150:
                    description = description[:150] + "..."
                elif not description:
                    description = "No description available"
                
                # Handle lastModified safely
                last_modified = getattr(dataset, 'lastModified', None)
                if last_modified and isinstance(last_modified, datetime):
                    last_updated_str = last_modified.isoformat()
                elif last_modified:
                    last_updated_str = str(last_modified)
                else:
                    last_updated_str = "Unknown"
                
                processed_results.append({
                    "title": dataset.id,
                    "author": author,
                    "description": description,
                    "last_updated_at": last_updated_str,
                    "link": f"https://huggingface.co/datasets/{dataset.id}"
                })
                
            except Exception as e:
                print(f"[HF Warning] Skipped dataset: {e}")
                continue
        
        print(f"\n[HF] Found {len(processed_results)} datasets\n")

        log_process(f"Hugging Face Found {len(processed_results)} datasets")

        
        return processed_results
        
    except Exception as e:
        print(f"[HF ERROR] {e}")
        return []





ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
SYSTEM_PROMPT_PATH = os.path.join(ROOT_DIR, "system_prompt.txt")

if not os.path.exists(SYSTEM_PROMPT_PATH):
    print(f"Error: system_prompt.txt not found at {SYSTEM_PROMPT_PATH}")
    sys.exit(1)

with open(SYSTEM_PROMPT_PATH, "r") as f:
    system_prompt = f.read()


llm = get_chat_model(
    "claude-3-5-sonnet",
    max_tokens=4096  
)

log_process("Chat model selected")

langsmith_project = os.getenv('LANGSMITH_PROJECT', 'default')
run_id = str(uuid.uuid4())



class userInputCheck(BaseModel):
    continueProcess: bool = Field(description="return false if input is invalid")
    searchQueries: List[str] = Field(description="Potential search queries to search on net")
    reasoning: str = Field(description="High-level explanation of why the system chose specific tools or actions")
    valyuResults: str = Field(description="Results of Valyu search tool for our search query")
    kaggleResults: List[Dict[str, str]] = Field(description="Results of search kaggle tool for our search queries")
    huggingFaceResults: List[Dict[str, str]] = Field(description="Results of search hugging face hub tool for our search queries")


agent = create_react_agent(
    llm,
    tools=[searchKaggle, search_huggingface_hub, searchValyu],
    prompt=system_prompt,
    response_format=userInputCheck
) 

log_process("ReAct Agent Created")

@traceable
def track_agent_with_tokens(agent, question: str) -> dict:
    """Run agent and track comprehensive metrics."""
    
    # Count input tokens
    input_tokens = count_tokens(question)

    log_process("Query Recieved")
    
    # Run agent and measure time
    start_time = time.time()
    result = agent.invoke({"messages": [HumanMessage(content=question)]}, {"run_id": run_id, "tags": ["tutorial", "observability"]})
    usage = {"prompt_tokens": 120, "completion_tokens": 58, "total": 178}

    log_process("Agent Invoked")

    elapsed = time.time() - start_time

    log_process("Agent execution completed")
    
    # Get response and count output tokens
    response = result['messages'][-1].content
    output_tokens = count_tokens(str(response))
    total_tokens = input_tokens + output_tokens
    
    # Calculate cost (Claude (via Bedrock) Nano pricing)
    # Input: $0.15 per 1M tokens, Output: $0.60 per 1M tokens
    input_cost = (input_tokens / 1_000_000) * 0.15
    output_cost = (output_tokens / 1_000_000) * 0.60
    total_cost = input_cost + output_cost

    log_process("Process Completed")
    
    return {
        'time': elapsed,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
        'cost': total_cost,
        'tokens_per_second': total_tokens / elapsed if elapsed > 0 else 0,
        'answer': response,
        "metadata": {"token_usage": usage}
        
    }

print("Monitoring function ready!")
print("  Tracks: latency, tokens (accurate), cost, throughput")



if __name__ == "__main__":
    print("\n" + "="*70)
    print("RUNNING TEST EXECUTION")
    print("="*70 + "\n")
    
    message = "I want to find datasets to make a ml algorithm for house prices"
    fresh_agent = get_agent()
    metrics = track_agent_with_tokens(fresh_agent, message)

    result = metrics["answer"]

    

    # Display results
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    print(f"Latency:          {metrics['time']:.3f}s")
    print(f"Input Tokens:     {metrics['input_tokens']}")
    print(f"Output Tokens:    {metrics['output_tokens']}")
    print(f"Total Tokens:     {metrics['total_tokens']}")
    print(f"Tokens/Second:    {metrics['tokens_per_second']:.2f}")
    print(f"Estimated Cost:   ${metrics['cost']:.6f}")
    print()
    print(f"Response: {metrics['answer'][:150]}...")
    print("="*70)

    

    json_string_content = result

    parsed_output = userInputCheck.model_validate_json(json_string_content)

    reasoning_text = parsed_output.reasoning


    continue_value = parsed_output.continueProcess

    if not continue_value:
        print("\n Invalid input detected. Please provide a dataset-related query.")
        sys.exit(1)

    query_values = parsed_output.searchQueries
    kaggle_return_values = parsed_output.kaggleResults
    hugging_face_return_values = parsed_output.huggingFaceResults

    print(f"\nKAGGLE RESULTS ({len(kaggle_return_values)} datasets)")
    print("="*70)
    for i, item in enumerate(kaggle_return_values, 1):
        print(f"\n{i}. {item['title']}")
        print(f"   Author: {item['author']}")
        print(f"   Description: {item['description']}")
        print(f"   Last Updated: {item['last_updated_at']}")
        print(f"   Link: {item['link']}")

    print("\n" + "="*70)
    print(f"HUGGING FACE RESULTS ({len(hugging_face_return_values)} datasets)")
    print("="*70)
    for i, item in enumerate(hugging_face_return_values, 1):
        print(f"\n{i}. {item['title']}")
        print(f"   Author: {item['author']}")
        print(f"   Description: {item['description']}")
        print(f"   Last Updated: {item['last_updated_at']}")
        print(f"   Link: {item['link']}")

    if os.getenv('LANGSMITH_API_KEY'):
        try:
            from langsmith import Client
            ls_client = Client()
            run_url = ls_client.read_run(run_id).url
            print(f"\n  {run_url}")
        except Exception:
            print(f"\n  https://smith.langchain.com")
            print(f"  Project: {os.getenv('LANGSMITH_PROJECT')}")
            print(f"  Search for run ID: {run_id}")
    else:
        print("\n  (LangSmith API key not set - tracing disabled)")
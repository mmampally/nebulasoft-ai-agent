import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from tools import get_all_tools

# Load environment variables
load_dotenv()

# Sentiment detection keywords
ANGRY_KEYWORDS = ["angry", "frustrated", "terrible", "awful", "hate", "worst", "useless", "horrible"]
HAPPY_KEYWORDS = ["great", "awesome", "excellent", "love", "amazing", "fantastic", "wonderful", "perfect"]


def detect_sentiment(user_input: str) -> str:
    """
    Detects user sentiment from their message.
    Returns: 'angry', 'happy', or 'neutral'
    """
    user_input_lower = user_input.lower()
    
    if any(keyword in user_input_lower for keyword in ANGRY_KEYWORDS):
        return "angry"
    elif any(keyword in user_input_lower for keyword in HAPPY_KEYWORDS):
        return "happy"
    else:
        return "neutral"


def create_system_prompt(sentiment: str) -> str:
    """
    Creates the system prompt based on detected sentiment.
    """
    base_persona = """You are a Tier-1 Technical Support Representative for NebulaSoft, a fictional software company.

YOUR IDENTITY:
- Your name is Mynko, and you've been with NebulaSoft support for 3 years
- You are helpful, professional, and knowledgeable
- You represent NebulaSoft and must maintain company standards

STRICT RULES YOU MUST FOLLOW:
1. **Never use general knowledge** to answer technical questions about NebulaSoft
2. **Always use the search_documentation tool first** for any technical question
3. **Always cite the source document** when providing information from documentation (e.g., "According to nebula_manual.txt...")
4. **Only escalate tickets** if the documentation search cannot answer the question
5. **Use the calculate_pricing tool** for any pricing or cost questions
6. You cannot make up features, error codes, or solutions - only use what's in the documentation

WORKFLOW:
1. For technical questions → Use search_documentation tool
2. For pricing questions → Use calculate_pricing tool  
3. If documentation doesn't help → Use escalate_ticket tool

RESPONSE STYLE:
"""
    
    # Adjust tone based on sentiment
    if sentiment == "angry":
        tone = """- Be APOLOGETIC and empathetic
- Acknowledge their frustration immediately
- Use phrases like "I sincerely apologize", "I understand how frustrating this must be"
- Prioritize resolving their issue quickly
- Offer to escalate if needed"""
    
    elif sentiment == "happy":
        tone = """- Be ENTHUSIASTIC and friendly
- Match their positive energy
- Use phrases like "That's great to hear!", "I'm so glad!", "Wonderful!"
- Be warm and encouraging"""
    
    else:
        tone = """- Be professional and helpful
- Maintain a friendly but focused tone
- Be clear and concise"""
    
    return base_persona + tone


def run_agent_with_tools(llm, tools, system_prompt: str, user_input: str):
    """
    Manually run the agent with tools.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    
    max_iterations = 5
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Get LLM response with tool binding
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(messages)
        
        # Add assistant's response to messages
        messages.append(response)
        
        # Check if LLM wants to use tools
        if not response.tool_calls:
            # No tool calls, return final answer
            return response.content
        
        # Execute tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if iteration == 1:
                print(f"\n🔧 Using tool: {tool_name}")
                print(f"   Args: {tool_args}")
            
            # Find and execute the tool
            tool_result = None
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        tool_result = tool._run(**tool_args)
                    except Exception as e:
                        tool_result = f"Error executing tool: {str(e)}"
                    break
            
            if tool_result is None:
                tool_result = f"Tool {tool_name} not found"
            
            # Add tool result to messages
            from langchain_core.messages import ToolMessage
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )
            )
    
    return "I apologize, but I've reached the maximum number of iterations. Let me escalate this to Tier-2 support."


def main():
    """
    Main interaction loop.
    """
    print("~" * 110)
    print("NebulaSoft AI Support Agent")
    print("~" * 110)
    print("Hi! I'm Mynko from NebulaSoft support. How can I help you today?")
    print("(Type 'quit' or 'exit' to end the conversation)")
    print("~" * 110)
    print()
    
    # Initialize LLM
    llm = ChatOpenAI(
    model="openai/gpt-4o-mini",  # Supports tools, very cheap (~$0.075 per 1M tokens)
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
    )
    
    # Get tools
    tools = get_all_tools()
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\nMynko: Thank you for contacting NebulaSoft support. See you next time! ")
            break
        
        # Detect sentiment
        sentiment = detect_sentiment(user_input)
        
        # Create system prompt based on sentiment
        system_prompt = create_system_prompt(sentiment)
        
        # Run agent
        try:
            print("\n Thinking...")
            response = run_agent_with_tools(llm, tools, system_prompt, user_input)
            print(f"\nAlex: {response}\n")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
            import traceback
            traceback.print_exc()
            print("Mynko: I apologize, but I encountered an error. Let me escalate this to Tier-2 support.\n")


if __name__ == "__main__":
    main()
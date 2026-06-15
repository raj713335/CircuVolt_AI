import operator
from typing import Annotated, Sequence, TypedDict, Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, create_model

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class DynamicAgentBuilder:
    def __init__(self):
        # We use a standard chat model
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    def compile_custom_tools(self, tool_configs: List[Dict[str, str]]) -> List[StructuredTool]:
        """
        Dynamically compiles Python code into LangChain StructuredTools.
        tool_configs is a list of dicts:
        {
            "name": "func_name",
            "description": "...",
            "code": "def func_name(arg1: str) -> str:\n    return 'done'"
        }
        """
        tools = []
        for config in tool_configs:
            name = config.get("name")
            description = config.get("description")
            code_str = config.get("code")

            if not all([name, description, code_str]):
                continue

            # A safe execution environment for the dynamic code
            local_scope = {}
            import requests as _requests
            import json as _json
            global_scope = {
                "__builtins__": __builtins__,
                "Dict": Dict,
                "List": List,
                "Any": Any,
                "print": print,
                "Exception": Exception,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "requests": _requests,
                "json": _json
            }

            try:
                # Compile and execute the user's tool code
                exec(code_str, global_scope, local_scope)
                
                # Retrieve the defined function by name
                if name in local_scope and callable(local_scope[name]):
                    func = local_scope[name]
                    # Wrap the function in a LangChain StructuredTool
                    tool = StructuredTool.from_function(
                        func=func,
                        name=name,
                        description=description
                    )
                    tools.append(tool)
            except Exception as e:
                print(f"Failed to compile tool {name}: {e}")
                
        return tools

    def build_agent(self, system_prompt: str, tool_configs: List[Dict[str, str]], mcp_urls: List[str] = None, a2a_urls: List[str] = None):
        tools = self.compile_custom_tools(tool_configs)
        
        mcp_urls = mcp_urls or []
        a2a_urls = a2a_urls or []

        # Add dynamic MCP proxy tools
        for i, url in enumerate(mcp_urls):
            if not url.strip(): continue
            def make_mcp_proxy(target_url):
                def mcp_proxy(action: str, payload: str) -> str:
                    """Send a request to the configured MCP Server to leverage its tools.
                    Provide the 'action' (tool name) and 'payload' (JSON arguments)."""
                    import requests
                    try:
                        res = requests.post(f"{target_url}/call", json={"action": action, "payload": payload}, timeout=10)
                        if res.status_code == 200:
                            return str(res.json())
                        return f"MCP Error: {res.status_code} - {res.text}"
                    except Exception as e:
                        return f"MCP Connection Failed: {str(e)}"
                return mcp_proxy
            
            mcp_tool = StructuredTool.from_function(
                func=make_mcp_proxy(url),
                name=f"call_mcp_server_{i}",
                description=f"Connects to the external MCP server at {url} to access additional tools."
            )
            tools.append(mcp_tool)
            system_prompt += f"\n\nYou also have access to an external MCP Server at {url}. Use 'call_mcp_server_{i}' to interact with it."

        # Add dynamic A2A proxy tools
        for i, url in enumerate(a2a_urls):
            if not url.strip(): continue
            def make_a2a_proxy(target_url):
                def a2a_proxy(task_message: str) -> str:
                    """Send a natural language task to another AI agent via the A2A protocol.
                    Provide a clear 'task_message' describing what you want the agent to do."""
                    import requests
                    import uuid
                    try:
                        # Convert agent discovery URL to task send URL if needed
                        send_url = target_url
                        if send_url.endswith("/.well-known/agent.json"):
                            send_url = send_url.replace("/.well-known/agent.json", "/a2a/tasks/send")
                        elif not send_url.endswith("/a2a/tasks/send"):
                            send_url = f"{send_url.rstrip('/')}/a2a/tasks/send"

                        payload = {
                            "jsonrpc": "2.0",
                            "method": "a2a.tasks.send",
                            "id": str(uuid.uuid4()),
                            "params": {
                                "message": {"parts": [{"type": "text", "text": task_message}]},
                                "sessionId": "a2a-delegation"
                            }
                        }
                        res = requests.post(send_url, json=payload, timeout=30)
                        if res.status_code == 200:
                            data = res.json()
                            if "result" in data and "response" in data["result"]:
                                return data["result"]["response"]
                            return str(data)
                        return f"A2A Error: {res.status_code} - {res.text}"
                    except Exception as e:
                        return f"A2A Delegation Failed: {str(e)}"
                return a2a_proxy
            
            a2a_tool = StructuredTool.from_function(
                func=make_a2a_proxy(url),
                name=f"delegate_to_agent_{i}",
                description=f"Delegates a natural language task to the external A2A Agent at {url}."
            )
            tools.append(a2a_tool)
            system_prompt += f"\n\nYou can also delegate tasks to an external A2A Agent at {url}. Use 'delegate_to_agent_{i}' with a natural language task_message."
        
        # Bind tools to the model if any exist
        llm_with_tools = self.llm.bind_tools(tools) if tools else self.llm

        def call_model(state: AgentState):
            messages = state["messages"]
            # Prepend system prompt if it's not already there
            if system_prompt and not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=system_prompt)] + list(messages)
            
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)

        if tools:
            tool_node = ToolNode(tools)
            workflow.add_node("action", tool_node)
            workflow.add_conditional_edges("agent", tools_condition, {"tools": "action", "__end__": END})
            workflow.add_edge("action", "agent")
        
        workflow.set_entry_point("agent")
        return workflow.compile()

import logging
from pydantic import BaseModel, create_model

from typing import Literal, Type, Dict, List, Any

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from .text_generation import TextAgent
from .scheduler_agent import SchedulerAgent
from .price_agent import PriceAgent
from .email_agent import EmailAgent
from .image_agent import ImageAgent
from .transactions_agent import TransactionsAgent
from .silo_agent import SiloAgent
import datetime
import json

logger = logging.getLogger("agent/shiami")

class State(MessagesState):
    next: str
    context: Dict[str, Any] = {}  # Added to store the context for each agent

class TaskManager:
    def __init__(self, llm, agents: list[str], prompts: dict[str, str]):
        self._llm = llm
        self._agents = agents
        self._prompts = prompts
        self._system_prompt = (
            "You are a task manager responsible for distributing work among specialized agents. "
            f"You have the following agents available: {agents}. "
            "For each agent, generate appropriate context and specific tasks based on the user request. "
            "Be specific about what each agent needs to do and provide relevant information."
        )
    
    def generate_context(self, user_input: str) -> Dict[str, str]:
        """Generate specific context and tasks for each agent based on the user input"""
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"Generate appropriate context and tasks for each agent based on this request: {user_input}"}
        ]
        
        # Create a structured output for the task distribution
        AgentTasks = create_model(
            "AgentTasks",
            **{agent: (str, ...) for agent in self._agents}
        )
        
        response = self._llm.with_structured_output(AgentTasks).invoke(messages)
        
        # Convert the response to a dictionary
        return {agent: getattr(response, agent) for agent in self._agents}

class Shiami:
    def __init__(self, agent, agents: list[str], llm, prompt: str, prompts: dict[str, str], data: dict[str, str], agent_id: str = None):
        if (len(agents) > 5): raise ValueError("Shiami can only manage up to 5 agents")
        self._agents = agents
        self._system_prompt = (
            "You are a Shiami tasked with managing a conversation between the"
            f" following workers: {agents}. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When error occurs,"
            " respond with FINISH."
            " if some agent doesn't do it's task well, ask to redo. and if some agent replies with anything that is the job of other agent then retry by refining the prompt specific to that agent."
            f" Here's some info for your current task: {prompt}"
            f" Prompts for workers: {prompts}"
        )
        self._data = data
        self._llm = llm
        self._prompts = prompts
        self._agent = agent
        self._agent_id = agent_id
        self._task_manager = TaskManager(llm, agents, prompts)
        self.graph = self.create_graph()

    def create_graph(self):
        builder = StateGraph(State)
        builder.add_edge(START, "shiami")
        builder.add_node("shiami", self.node)
        
        for agent in self._agents:
            agent_type = self._data[agent]["name"]
            if agent_type == "scheduler":
                builder.add_node(agent, SchedulerAgent(
                    llm=self._llm, 
                    name=agent, 
                    prompt=self._prompts[agent], 
                    next=self._data[agent]["next"], 
                    run_manager=self.execute_task,
                    agent_id=self._agent_id).node)
                continue
            if agent_type == "image":
                builder.add_node(agent, ImageAgent(
                    llm=self._llm,
                    name=agent,
                    prompt=self._prompts[agent],
                    next=self._data[agent]["next"],
                    agent=self._agent
                ).node)
                continue
            if agent_type == "transactions":
                builder.add_node(agent, TransactionsAgent(
                    llm=self._llm,
                    name=agent,
                    prompt=self._prompts[agent],
                    next=self._data[agent]["next"],
                    agent=self._agent
                ).node)
                continue
            if agent_type == "silo":
                builder.add_node(agent, SiloAgent(
                    llm=self._llm,
                    name=agent,
                    prompt=self._prompts[agent],
                    next=self._data[agent]["next"],
                    agent=self._agent
                ).node)
                continue
            agent_class = self._name_to_class(agent_type)
            logger.info(self._data[agent]["next"])
            logger.info(agent)
            builder.add_node(
                agent, 
                agent_class(
                    llm=self._llm, 
                    name=agent, 
                    prompt=self._prompts[agent],
                    next=self._data[agent]["next"]
                ).node
            )

        return builder.compile()

    def node(self, state: State):
        messages = state["messages"]
        
        # Generate context if this is the first call or if context needs updating
        if not state.get("context") and len(messages) > 0:
            # Properly extract the user input from first message
            user_input = ""
            for msg in messages:
                # Handle different message formats
                if hasattr(msg, 'type') and msg.type == 'human':
                    user_input = msg.content
                    break
                elif isinstance(msg, dict) and msg.get("role") == "user":
                    user_input = msg.get("content", "")
                    break
                elif isinstance(msg, tuple) and len(msg) > 1 and msg[0] == "user":
                    user_input = msg[1]
                    break
            
            if user_input:
                context = self._task_manager.generate_context(user_input)
                state["context"] = context
        
        system_prompt = self._system_prompt
        
        # If there's context available, enhance the system prompt with agent-specific context
        if state.get("context"):
            context_info = "\nAgent-specific context:\n"
            for agent, task in state["context"].items():
                context_info += f"- {agent}: {task}\n"
            system_prompt += context_info
        
        # Convert messages to dict format for LLM input
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'type'):
                # Handle LangChain message objects
                if msg.type == 'human':
                    formatted_messages.append({"role": "user", "content": msg.content})
                elif msg.type == 'ai':
                    formatted_messages.append({"role": "assistant", "content": msg.content})
                # Skip system messages for Gemini
            elif isinstance(msg, dict):
                if msg.get("role") != "system":  # Skip system messages for Gemini
                    formatted_messages.append(msg)
            elif isinstance(msg, tuple) and len(msg) > 1:
                if msg[0] != "system":  # Skip system messages for Gemini
                    formatted_messages.append({"role": msg[0], "content": msg[1]})
        
        messages_with_system = [
            {"role": "user", "content": f"System: {system_prompt}"},
        ] + formatted_messages
        
        Router: Type[BaseModel] = create_model(
            "Router",
            next=(Literal[*self._agents, "FINISH"], ...)
        )
        
        response = self._llm.with_structured_output(
            Router).invoke(messages_with_system)
        goto = response.next
        
        if goto == "FINISH":
            goto = END
        elif goto in self._agents and state.get("context"):
            agent_context = state["context"].get(goto, "")
            if agent_context:
                from langchain_core.messages import HumanMessage
                state["messages"].append(HumanMessage(content=f"[Context for {goto}]: {agent_context}"))
        
        return Command(goto=goto, update={"next": goto})

    async def execute_task(self, message: str):
        try:
            result = {}
            logs = []
            from langchain_core.messages import HumanMessage, AIMessage

            async for s in self.graph.astream(
                {
                    "messages": [HumanMessage(content=message)],
                    "context": {}
                },
                subgraphs=True,
            ):
                logger.info(s)
                logger.info("----")
                # Handle regular chatbot messages
                chatbot_data = s[1].get("chatbot", {})
                ai_messages = chatbot_data.get("messages", [])

                for msg in ai_messages:
                    if isinstance(msg, AIMessage) and msg.content.strip():
                        logs.append({
                            'content': msg.content,
                            'timestamp': str(datetime.datetime.now())
                        })

                result["result"] = s
                result["logs"] = logs
                logger.info(logs)

        except Exception as e:
            logger.error(f"Error executing task: {e}", exc_info=True)
            return {"error": str(e)}

        logger.info(result)
        return result


    @staticmethod
    def _name_to_class(class_name: str):
        if class_name == "text":
            return TextAgent
        elif class_name == "scheduler":
            return SchedulerAgent
        elif class_name == "email":
            return EmailAgent
        elif class_name == "price_predictor":
            return PriceAgent
        elif class_name == "image":
            return ImageAgent
        elif class_name == "transactions":
            return TransactionsAgent
        elif class_name == "silo":
            return SiloAgent
        return None

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from sqlalchemy.orm import Session
from src.models.image import GeneratedImage
from src.database import get_db

from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Type
import logging
import asyncio
import signal
import threading
from pathlib import Path
from src.cli import ZerePyCLI
from fastapi.responses import StreamingResponse

from src.agents.shiami import Shiami
from src.connections.llm_base_connection import LLMBaseConnection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server/app")


class ActionRequest(BaseModel):
    """Request model for agent actions"""
    connection: str
    action: str
    params: Optional[List[str]] = []


class ServerState:
    """Simple state management for the server"""

    def __init__(self):
        self.cli = ZerePyCLI()
        self.agent_running = False
        self.agent_task = None
        self._stop_event = threading.Event()

    def _run_agent_loop(self):
        """Run agent loop in a separate thread"""
        try:
            log_once = False
            while not self._stop_event.is_set():
                if self.cli.agent:
                    try:
                        if not log_once:
                            logger.info("Loop logic not implemented")
                            log_once = True

                    except Exception as e:
                        logger.error(f"Error in agent action: {e}")
                        if self._stop_event.wait(timeout=30):
                            break
        except Exception as e:
            logger.error(f"Error in agent loop thread: {e}")
        finally:
            self.agent_running = False
            logger.info("Agent loop stopped")

    async def start_agent_loop(self):
        """Start the agent loop in background thread"""
        if not self.cli.agent:
            raise ValueError("No agent loaded")

        if self.agent_running:
            raise ValueError("Agent already running")

        self.agent_running = True
        self._stop_event.clear()
        self.agent_task = threading.Thread(target=self._run_agent_loop)
        self.agent_task.start()

    async def stop_agent_loop(self):
        """Stop the agent loop"""
        if self.agent_running:
            self._stop_event.set()
            if self.agent_task:
                self.agent_task.join(timeout=5)
            self.agent_running = False


class ZerePyServer:
    def __init__(self):
        self.app = FastAPI(title="ZerePy Server")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.state = ServerState()
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/")
        async def root():
            """Server status endpoint"""
            return {
                "status": "running",
                "agent": self.state.cli.agent.name if self.state.cli.agent else None,
                "agent_running": self.state.agent_running
            }

        @self.app.get("/agents")
        async def list_agents():
            """List available agents"""
            try:
                agents = []
                agents_dir = Path("agents")
                if agents_dir.exists():
                    for agent_file in agents_dir.glob("*.json"):
                        if agent_file.stem != "general":
                            agents.append(agent_file.stem)
                return {"agents": agents}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/connections")
        async def list_connections():
            """List all available connections"""
            if not self.state.cli.agent:
                raise HTTPException(status_code=400, detail="No agent loaded")

            try:
                connections = {}
                for name, conn in self.state.cli.agent.connection_manager.connections.items():
                    connections[name] = {
                        "configured": conn.is_configured(),
                        "is_llm_provider": conn.is_llm_provider
                    }
                return {"connections": connections}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/agent/create")
        async def create_agent():
            model_provider = self.state.cli.agent.model_provider
            llm_class: Type[LLMBaseConnection] = self.state.cli.agent.connection_manager._class_name_to_type(
                model_provider)

            config = next(
                (config for config in self.state.cli.agent.connection_manager.config if config["name"] == model_provider), None)

            llm = llm_class(config, self.state.cli.agent, False)._get_client()
            shiami = Shiami(
                agents=["scheduler", "price", "text", "email"],
                llm=llm,
                prompt="""You are an AI assistant managing email scheduling and cryptocurrency price reports for users. Your role is to:
1. Understand user requests related to scheduling price reports or getting price predictions.
2. Use the price agent to generate cryptocurrency price predictions when needed.
3. Use the text agent to format the price information into a well-structured email.
4. Use the email agent to send the formatted information.
5. Use the scheduler agent only when explicitly asked to schedule recurring tasks.
6. Always confirm actions with the user before executing them.
7. If a request is unclear or outside your capabilities, ask for clarification.
""",
                prompts={
                    "scheduler": """You are a scheduling assistant. Your role is to:
1. Only schedule tasks when explicitly requested by the user.
2. Use cron expressions for scheduling (e.g., "0 9 * * *" for daily at 9 AM).
3. Confirm the schedule details with the user before setting it up.
4. Provide clear feedback on the scheduled task, including the next run time.
5. If a scheduling request is unclear, ask for more details.
Example: "Schedule a daily Bitcoin price report at 9 AM."
""",
                    "price": """You are a cryptocurrency price prediction agent. Your role is to:
1. Provide price predictions only for the requested cryptocurrency.
2. Use historical data and your tools to make accurate predictions.
3. Always include confidence levels with your predictions.
4. Explain any significant factors influencing your prediction.
5. If asked about an unsupported cryptocurrency, inform the user and suggest alternatives.
Example: "Predict the price of Ethereum."
""",
                    "text": """You are an email content formatting agent. Your role is to:
1. Take the provided information (e.g., price predictions) and format it into a clear, professional email.
2. Structure the email with a proper greeting, body, and closing.
3. Use bullet points or tables for clarity when presenting data.
4. Ensure the content is concise and easy to read.
5. Do not add any information that wasn't provided to you.
Output format: email(recipient, subject, body)
""",
                    "email": """You are an email sending agent. Your role is to:
1. Send emails using the provided email(recipient, subject, body) format.
2. Confirm that the email was sent successfully.
3. If there are any issues with sending the email, provide a clear error message.
4. Do not modify the email content provided to you.
5. Respect privacy and do not disclose email addresses or content unnecessarily.
"""
                }
            )
            
            shiami.execute_task("Send a price report of S to parth.eng1210@gmail.com")

        @self.app.post("/agent/action")
        async def agent_action(action_request: ActionRequest):
            """Execute a single agent action"""
            if not self.state.cli.agent:
                raise HTTPException(status_code=400, detail="No agent loaded")

            try:
                action = self.state.cli.agent.connection_manager.connections[action_request.connection].actions[action_request.action]
                
                kwargs = {}
                param_index = 0

                # Add provided parameters up to the number provided
                for i, param in enumerate(action.parameters):
                    if param_index < len(action_request.params):
                        kwargs[param.name] = action_request.params[param_index]
                        param_index += 1

                # Validate all required parameters are present
                missing_required = [
                    param.name
                    for param in action.parameters
                    if param.required and param.name not in kwargs
                ]

                if missing_required:
                    logging.error(
                        f"\nError: Missing required parameters: {', '.join(missing_required)}"
                    )
                    return {"status": "error", "message": f"Missing required parameters: {', '.join(missing_required)}"}
                action_method = self.state.cli.agent.connection_manager.connections[action_request.connection].perform_action
                if asyncio.iscoroutinefunction(action_method):
                    result = await action_method(
                        action_name=action_request.action,
                        kwargs=kwargs
                    )
                else:
                    result = await asyncio.to_thread(
                        action_method,
                        action_name=action_request.action,
                        kwargs=kwargs
                    )
                return {"status": "success", "result": result}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/agent/chat")
        async def agent_chat(action_request: ActionRequest):
            """Chat with the agent"""
            if not self.state.cli.agent:
                raise HTTPException(status_code=400, detail="No agent loaded")

            try:
                result = await self.state.cli.agent.perform_action(
                    connection=action_request.connection,
                    action="generate-text",
                    params=action_request.params
                )

                if hasattr(result, "__aiter__"):
                    return StreamingResponse(result, media_type="text/plain")

                return {"status": "success", "result": result}

            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        @self.app.post("/agent/start")
        async def start_agent():
            """Start the agent loop"""
            if not self.state.cli.agent:
                raise HTTPException(status_code=400, detail="No agent loaded")

            try:
                await self.state.start_agent_loop()
                return {"status": "success", "message": "Agent loop started"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/agent/stop")
        async def stop_agent():
            """Stop the agent loop"""
            try:
                await self.state.stop_agent_loop()
                return {"status": "success", "message": "Agent loop stopped"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/image/{filename}")
        async def get_image(filename: str, db: Session = Depends(get_db)):
            # Get image record from database
            image = db.query(GeneratedImage).filter(
                GeneratedImage.filename == filename).first()

            if not image:
                raise HTTPException(status_code=404, detail="Image not found")

            # Return the image file
            return FileResponse(image.file_path)


def create_app():
    server = ZerePyServer()
    return server.app

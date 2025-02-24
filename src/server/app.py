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
        async def create_agent(background_tasks: BackgroundTasks):
            model_provider = self.state.cli.agent.model_provider
            llm_class: Type[LLMBaseConnection] = self.state.cli.agent.connection_manager._class_name_to_type(
                model_provider)

            config = next(
                (config for config in self.state.cli.agent.connection_manager.config if config["name"] == model_provider), None)

            llm = llm_class(config, self.state.cli.agent, False)._get_client()
            shiami = Shiami(
                agents=["scheduler", "price",
                        "email_formatter", "report", "email", "image"],
                llm=llm,
                prompt="""You are an AI assistant manageing conversations between the agents. If you get ipfs hash = ba343ei.... then URL = https://ipfs.io/ipfs/ba343ei....""",
                prompts={
                    "scheduler": """You are a scheduling assistant.""",
                    "price": """You are a cryptocurrency price prediction agent.""",
                    "report": """You are a price report report generating agent based on the given data.""",
                    "email_formatter": """You are an email content formatting agent.""",
                    "email": """You are an email sending agent. Send email to parth.eng1210@gmail.com""",
                    "image": """You are an image generation agent. Focus on the image prompt and ignore anything else. Output the image URL - https://ipfs.io/ipfs/[ipfs_hash]
                    Example - generate an image of roses and send its ipfs URL to....
                    Here, just focus on "generate an image of roses" and ignore the rest.

                    """
                },
                data={
                    "scheduler": {"name": "scheduler", "next": "price"},
                    "price": {"name": "price", "next": "report"},
                    "report": {"name": "text", "next": "email_formatter"},
                    "email_formatter": {"name": "text", "next": "email"},
                    "email": {"name": "email", "next": "FINISH"},
                    "image": {"name": "image", "next": "email"}
                },
                agent=self.state.cli.agent
            )

            background_tasks.add_task(
                shiami.execute_task,
                "generate an image of roses and send its ipfs URL to parth.eng1210@gmail.com. URL = https://ipfs.io/ipfs/[ipfs_hash]"
            )
        @self.app.post("/agent/action")
        async def agent_action(action_request: ActionRequest):
            """Execute a single agent action"""
            if not self.state.cli.agent:
                raise HTTPException(status_code=400, detail="No agent loaded")

            try:
                action = self.state.cli.agent.connection_manager.connections[
                    action_request.connection].actions[action_request.action]

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
                action_method = self.state.cli.agent.connection_manager.connections[
                    action_request.connection].perform_action
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

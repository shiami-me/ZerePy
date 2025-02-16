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
            llm_class: Type[LLMBaseConnection] = self.state.cli.agent.connection_manager._class_name_to_type(model_provider)
            
            config = next((config for config in self.state.cli.agent.connection_manager.config if config["name"] == model_provider), None)
            
            llm = llm_class(config, self.state.cli.agent, False)._get_client()
            shiami = Shiami(
                agents=["python_repl", "text", "rag_agent", "price_prediction"],
                llm=llm,
                prompts={
                    "python_repl": "You are a math agent",
                    "text": "You are a researcher. DO NOT do any math.",
                    "rag_agent": "You are a RAG agent that utilizes knowledge graph for retrieving result from the uplaoded document and answer user queries",
                    "price_prediction": "you are a Price prediction agent that predicts price for given cryptocurrency or token."
                }
            )
            shiami.execute_task("Write a research about addition and do 2 + 2")
        @self.app.post("/agent/action")
        async def agent_action(action_request: ActionRequest):
            """Execute a single agent action"""
            if not self.state.cli.agent:
                raise HTTPException(status_code=400, detail="No agent loaded")
            
            try:
                result = await asyncio.to_thread(
                    self.state.cli.agent.perform_action,
                    connection=action_request.connection,
                    action=action_request.action,
                    params=action_request.params
                )
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
            image = db.query(GeneratedImage).filter(GeneratedImage.filename == filename).first()
            
            if not image:
                raise HTTPException(status_code=404, detail="Image not found")
            
            # Return the image file
            return FileResponse(image.file_path)

def create_app():
    server = ZerePyServer()
    return server.app
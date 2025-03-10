from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from web3 import Web3
from eth_account.messages import encode_defunct
from pydantic import BaseModel, Field

from sqlalchemy.orm import Session
from src.models.image import GeneratedImage
from src.database import get_db, engine, Base
from src.models.agent import Agent

from typing import Optional, List, Dict, Any, Type
import logging
import asyncio
import signal
import threading
import datetime
from pathlib import Path
from src.cli import ZerePyCLI

from src.agents.shiami import Shiami
from src.connections.llm_base_connection import LLMBaseConnection
from src.tools.scheduler_tool import SchedulerTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server/app")


class ActionRequest(BaseModel):
    """Request model for agent actions"""
    connection: str
    action: str
    params: Optional[List[str]] = []


class CreateAgentRequest(BaseModel):
    """Request model for creating an agent"""
    agents: List[str] = []
    prompts: Dict[str, str] = {}
    data: Dict[str, Dict[str, Any]] = {}
    task: str = ""
    user_address: str
    name: Optional[str] = None
    is_one_time: bool = True
    signature: str = Field(..., description="Signature of 'I authorize creating an agent with my address: {user_address}'")


class DeleteAgentRequest(BaseModel):
    """Request model for deleting an agent"""
    signature: str = Field(..., description="Signature of 'I authorize deleting agent with ID: {agent_id}'")
    user_address: str


class InteractAgentRequest(BaseModel):
    """Request model for interacting with an agent"""
    task: str
    user_address: str


class ServerState:
    """Simple state management for the server"""

    def __init__(self):
        Base.metadata.create_all(engine)
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
        self.active_connections = set()
        
    def verify_signature(self, message: str, signature: str, address: str) -> bool:
        """
        Verify that a message was signed by the given address
        
        Args:
            message: The message that was signed
            signature: The signature to verify
            address: The address that supposedly signed the message
            
        Returns:
            bool: True if the signature is valid and from the address
        """
        try:
            # Convert to checksum address
            address = Web3.to_checksum_address(address)
            
            # Encode the message
            w3 = Web3()
            message_hash = encode_defunct(text=message)
            
            # Recover the address
            recovered_address = w3.eth.account.recover_message(message_hash, signature=signature)
            
            # Compare (case-insensitive)
            return recovered_address.lower() == address.lower()
        except Exception as e:
            logger.error(f"Error verifying signature: {str(e)}")
            return False

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
        async def list_agents(user_address: str = None, db: Session = Depends(get_db)):
            """List agents for a specific user address"""
            try:
                # If no user address provided, list available agent templates
                if not user_address:
                    agents = []
                    agents_dir = Path("agents")
                    if agents_dir.exists():
                        for agent_file in agents_dir.glob("*.json"):
                            if agent_file.stem != "general":
                                agents.append(agent_file.stem)
                    return {"agents": agents}
                
                # If user address provided, list their saved agents from the database
                query = db.query(Agent).filter(Agent.user_address == user_address)
                agents = query.all()
                
                return {
                    "agents": [
                        {
                            "id": agent.id,
                            "name": agent.name,
                            "created_at": agent.created_at,
                            "updated_at": agent.updated_at,
                            "agents_list": agent.agents_list,
                            "prompts": agent.prompts,
                            "data": agent.data
                        }
                        for agent in agents
                    ]
                }
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
        async def create_agent(request: CreateAgentRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
            # Verify signature to ensure the user address is authentic
            expected_message = f"I authorize creating an agent with my address: {request.user_address}"
            if not self.verify_signature(expected_message, request.signature, request.user_address):
                raise HTTPException(status_code=401, detail="Invalid signature. Authorization failed.")
                
            model_provider = self.state.cli.agent.model_provider
            llm_class: Type[LLMBaseConnection] = self.state.cli.agent.connection_manager._class_name_to_type(
                model_provider)

            config = next(
                (config for config in self.state.cli.agent.connection_manager.config if config["name"] == model_provider), None)

            llm = llm_class(config, self.state.cli.agent, False)._get_client()
            
            # For one-time agents, agent_id is None
            agent_id = None
            
            if not request.is_one_time:
                # For persistent agents, create DB record first to get ID
                if not request.user_address or not request.name:
                    raise HTTPException(status_code=400, detail="User address and name are required for persistent agents")
                
                # Create a new agent record
                db_agent = Agent(
                    user_address=request.user_address,
                    name=request.name,
                    is_one_time=False,
                    agents_list=request.agents,
                    prompts=request.prompts,
                    data=request.data
                )
                
                # Save to database
                db.add(db_agent)
                db.commit()
                db.refresh(db_agent)
                agent_id = str(db_agent.id)
            
            shiami = Shiami(
                agents=request.agents,
                llm=llm,
                prompt="""You are an AI assistant managing conversations between the agents. If you get ipfs hash = ba343ei.... then URL = https://ipfs.io/ipfs/ba343ei....""",  # Keep this prompt as is
                prompts=request.prompts,
                data=request.data,
                agent=self.state.cli.agent,
                agent_id=agent_id
            )
            
            # For one-time agents, just execute the task and don't store it
            if request.is_one_time:
                if request.task:
                    # Modified to capture and log the execution result
                    async def execute_and_log_task():
                        execution_result = await shiami.execute_task(request.task)
                        logger.info(f"One-time agent task completed")
                        if "logs" in execution_result:
                            logger.info(f"Captured {len(execution_result['logs'])} message logs")
                    
                    background_tasks.add_task(execute_and_log_task)
                    return {"status": "success", "message": f"One-time agent created and task started: {request.task[:50]}..."}
                else:
                    return {"status": "error", "message": "One-time agents require a task"}
            else:
                # For persistent agents, store in database and return the ID
                if not request.user_address or not request.name:
                    raise HTTPException(status_code=400, detail="User address and name are required for persistent agents")
                
                # If a task is provided, execute it and log it
                if request.task:
                    # Modified to capture and log the execution result with AI messages
                    async def execute_and_log_task(db_agent_id: int):
                        
                        db: Session = next(get_db())
                        try:
                            db_agent = db.query(Agent).filter(Agent.id == db_agent_id).first()
                            if not db_agent:
                                logger.error(f"Agent with ID {db_agent_id} not found")
                                return

                            logger.info("Executing task...")
                            execution_result = await shiami.execute_task(request.task)

                            if "logs" in execution_result:
                                for log_entry in execution_result["logs"]:
                                    db_agent.add_log(log_entry)

                                db.commit()
                                logger.info(f"Stored {len(execution_result['logs'])} message logs for agent {db_agent.id}")

                        finally:
                            db.close()
                    db_agent.add_log({
                        "timestamp": str(datetime.datetime.now()),
                        "task": request.task,
                        "type": "task_start"
                    })
                    db.commit()
                    
                    background_tasks.add_task(execute_and_log_task, db_agent.id)
                
                return {
                    "status": "success", 
                    "message": "Persistent agent created" + (f" and task started: {request.task[:50]}..." if request.task else ""),
                    "agent_id": db_agent.id
                }

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

        @self.app.websocket("/ws/agent/chat")
        async def websocket_agent_chat(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.add(websocket)
            
            try:
                data = await websocket.receive_json()
                connection = data.get("connection")
                action = data.get("action", "generate-text")
                params = data.get("params", [])
                
                if not self.state.cli.agent:
                    await websocket.send_json({"status": "error", "message": "No agent loaded"})
                    return

                try:
                    # Create a callback for the generate_text method
                    async def send_chunk(chunk):
                        if isinstance(chunk, str):
                            await websocket.send_text(chunk)
                        else:
                            await websocket.send_json(chunk)
                    
                    await self.state.cli.perform_action_websocket(
                        connection=connection,
                        action=action,
                        params=params,
                        websocket_callback=send_chunk
                    )
                    
                    # Send completion message
                    await websocket.send_json({"status": "complete"})
                
                except Exception as e:
                    logger.error(f"Error in websocket chat: {e}")
                    await websocket.send_json({"status": "error", "message": str(e)})
            
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.active_connections.remove(websocket)

        # Keep the old endpoint for backward compatibility
        @self.app.post("/agent/chat")
        async def agent_chat(action_request: ActionRequest):
            """Chat with the agent (Legacy endpoint)"""
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
        
        @self.app.post("/agent/{agent_id}/interact")
        async def interact_agent(agent_id: str, request: InteractAgentRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
            """Interact with a previously saved agent"""
            try:
                # Find the agent in the database
                db_agent = db.query(Agent).filter(Agent.id == agent_id).first()
                if not db_agent:
                    raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
                
                # Check if the user owns this agent
                if db_agent.user_address.lower() != request.user_address.lower():
                    raise HTTPException(status_code=403, detail="You do not have permission to interact with this agent")
                
                # Create a new Shiami instance using the stored configuration
                model_provider = self.state.cli.agent.model_provider
                llm_class: Type[LLMBaseConnection] = self.state.cli.agent.connection_manager._class_name_to_type(
                    model_provider)
                
                config = next(
                    (config for config in self.state.cli.agent.connection_manager.config if config["name"] == model_provider), None)
                
                llm = llm_class(config, self.state.cli.agent, False)._get_client()
                
                # Create the Shiami agent from the saved configuration
                shiami = Shiami(
                    agents=db_agent.agents_list,
                    llm=llm,
                    prompt="You are an AI assistant managing conversations between the agents. If you get ipfs hash = ba343ei.... then URL = https://ipfs.io/ipfs/ba343ei....",
                    prompts=db_agent.prompts,
                    data=db_agent.data,
                    agent=self.state.cli.agent,
                    agent_id=str(db_agent.id)  # Pass agent ID to associate scheduled jobs
                )
                
                # Add initial task log
                db_agent.add_log({
                    "timestamp": str(datetime.datetime.now()),
                    "task": request.task,
                    "type": "task_start"
                })
                db.commit()
                
                async def execute_and_log_task(db_agent_id: int):
                    db: Session = next(get_db())
                    try:
                        db_agent = db.query(Agent).filter(Agent.id == db_agent_id).first()
                        if not db_agent:
                            logger.error(f"Agent with ID {db_agent_id} not found")
                            return

                        logger.info("Executing task...")
                        execution_result = await shiami.execute_task(request.task)

                        if "logs" in execution_result:
                            for log_entry in execution_result["logs"]:
                                db_agent.add_log(log_entry)

                            db.commit()
                            logger.info(f"Stored {len(execution_result['logs'])} message logs for agent {db_agent.id}")

                    finally:
                        db.close()
                        
                background_tasks.add_task(execute_and_log_task, db_agent.id)
                
                return {
                    "status": "success", 
                    "message": f"Interaction started with agent {db_agent.name}: {request.task[:50]}...",
                    "agent_id": agent_id
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/agent/{agent_id}/logs")
        async def get_agent_logs(
            agent_id: str,
            db: Session = Depends(get_db)
        ):
            """Get logs for a specific agent"""
            try:
                # Find the agent in the database
                db_agent = db.query(Agent).filter(Agent.id == agent_id).first()
                if not db_agent:
                    raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
                logger.info(db_agent.logs)
                return {
                    "agent_id": agent_id,
                    "name": db_agent.name,
                    "logs": db_agent.logs
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.delete("/agent/{agent_id}")
        async def delete_agent(agent_id: str, request: DeleteAgentRequest, db: Session = Depends(get_db)):
            """Delete a previously saved agent"""
            try:
                # Find the agent in the database
                db_agent = db.query(Agent).filter(Agent.id == agent_id).first()
                if not db_agent:
                    raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
                
                # Verify ownership through signature
                expected_message = f"I authorize deleting agent with ID: {agent_id}"
                if not self.verify_signature(expected_message, request.signature, request.user_address):
                    raise HTTPException(status_code=401, detail="Invalid signature. Authorization failed.")
                
                # Check if the user owns this agent
                if db_agent.user_address.lower() != request.user_address.lower():
                    raise HTTPException(status_code=403, detail="You do not have permission to delete this agent")
                
                # Clean up any scheduled jobs for this agent
                cleaned_jobs = []
                try:
                    # Define a dummy async function for run_manager
                    async def dummy_run_manager(task: str):
                        logger.info(f"Dummy run manager called with: {task}")
                        return {}
                        
                    # Create temporary scheduler tool with proper run_manager function
                    temp_scheduler = SchedulerTool(run_manager=dummy_run_manager, agent_id=agent_id)

                    # Clean up the jobs
                    cleaned_jobs = temp_scheduler.cleanup_agent_jobs(agent_id)
                    logger.info(f"Cleaned up {len(cleaned_jobs)} scheduled jobs for agent {agent_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up scheduled jobs: {e}")
                    # Continue with agent deletion even if job cleanup fails
                
                # Delete the agent
                agent_name = db_agent.name
                db.delete(db_agent)
                db.commit()
                
                return {
                    "status": "success", 
                    "message": f"Agent '{agent_name}' with ID {agent_id} has been deleted successfully",
                    "cleaned_jobs": len(cleaned_jobs)
                }
                
            except Exception as e:
                logger.error(f"Error deleting agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))

def create_app():
    server = ZerePyServer()
    return server.app

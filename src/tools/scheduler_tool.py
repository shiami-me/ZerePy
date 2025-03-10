from typing import Dict, Any, Callable, Awaitable, Optional, List
from langchain_core.tools import BaseTool
from croniter import croniter
from datetime import datetime
import os
import json
import logging
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

logger = logging.getLogger(__name__)

TASK_REGISTRY = {}

class SchedulerTool(BaseTool):
    name: str = "task_scheduler"
    description: str = """Schedule tasks using cron expressions. This tool should only be used when explicitly asked to schedule recurring tasks.
    
    Inputs:
    - task: The task description to be scheduled (should not contain scheduling information)
    - cron: Cron expression for scheduling the task
    
    Cron expression examples:
    - Daily at midnight: "0 0 * * *"
    - Every 30 minutes: "*/30 * * * *"
    - Every Monday at 9am: "0 9 * * 1"
    - First day of month at 3am: "0 3 1 * *"
    
    Example Inputs:
    1. Schedule a daily Bitcoin price report:
       task: "generate a Bitcoin price report and email it to user@example.com"
       cron: "0 9 * * *"
    2. Schedule a weekly Ethereum analysis:
       task: "create an Ethereum market analysis and save it"
       cron: "0 10 * * 1"
    
    This tool will schedule the task and return the next execution time. It does not execute the task immediately.
    """
    
    # Define agent_id as a proper class attribute with type annotation
    agent_id: Optional[str] = None
    
    def __init__(self, run_manager: Callable[[str], Awaitable[Any]], agent_id: Optional[str] = None):
        super().__init__()
        self._scheduler = BackgroundScheduler(
            jobstores={'default': SQLAlchemyJobStore(url='sqlite:///jobs.sqlite')}
        )
        TASK_REGISTRY["run_manager"] = run_manager
        TASK_REGISTRY["logger"] = self._log_operation
        logger.info(f"Initializing scheduler with agent_id: {agent_id}")
        # Store the agent ID as an instance attribute
        self.agent_id = agent_id  # Now this will work since agent_id is properly defined
        self._scheduler.start()
        
        # Ensure logs directory exists
        os.makedirs("task_logs", exist_ok=True)

    def _validate_cron(self, cron_exp: str) -> bool:
        """Validate if the cron expression is valid"""
        try:
            return croniter.is_valid(cron_exp)
        except Exception:
            return False

    def _run(
        self, 
        task: str,
        cron: str,
    ) -> str:
        """Schedule a task using cron expression"""
        try:
            if not self._validate_cron(cron):
                return f"Invalid cron expression: {cron}"

            task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Store task details
            task_details = {
                "task": task,
                "cron": cron,
                "metadata": {},
                "created_at": datetime.now().isoformat(),
                "task_id": task_id,
                "agent_id": self.agent_id  # This now correctly accesses the Pydantic field
            }
            
            # Parse cron expression into kwargs
            cron_parts = cron.split()
            if len(cron_parts) != 5:
                return "Invalid cron expression format. Must have 5 parts: minute hour day month weekday"
                
            cron_kwargs = {
                'minute': cron_parts[0],
                'hour': cron_parts[1],
                'day': cron_parts[2],
                'month': cron_parts[3],
                'day_of_week': cron_parts[4]
            }
            
            # Add job to scheduler with proper kwargs
            self._scheduler.add_job(
                self._execute_task,
                'cron',
                args=[task, task_details],
                id=task_id,
                **cron_kwargs  # Pass as kwargs instead of tuple
            )
            
            self._log_operation("schedule", task_id, task_details)
            return f"Task scheduled successfully. Task ID: {task_id}"

        except Exception as e:
            logger.error(f"Error scheduling task: {e}")
            return f"Failed to schedule task: {str(e)}"
            
    def cleanup_agent_jobs(self, agent_id: str) -> List[str]:
        """Remove all scheduled jobs associated with a specific agent"""
        try:
            removed_jobs = []
            
            # Get all jobs
            jobs = self._scheduler.get_jobs()
            for job in jobs:
                try:
                    # Get job ID
                    job_id = job.id
                    
                    # Get job execution args
                    args = job.args
                    if len(args) >= 2 and isinstance(args[1], dict):
                        task_details = args[1]
                        if task_details.get("agent_id") == agent_id:
                            # This job belongs to the deleted agent - remove it
                            self._scheduler.remove_job(job_id)
                            removed_jobs.append(job_id)
                            self._log_operation("remove_agent_job", job_id, {
                                "agent_id": agent_id,
                                "removed_at": datetime.now().isoformat()
                            })
                except Exception as e:
                    logger.error(f"Error processing job during cleanup: {e}")
                    
            return removed_jobs
        except Exception as e:
            logger.error(f"Error cleaning up agent jobs: {e}")
            return []

    @staticmethod
    def _execute_task(task: str, task_details: Dict[str, Any]):
        """Execute the scheduled task"""
        try:
            logger.info(f"Executing task: {task}")
            # Create an event loop if there isn't one
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the coroutine in the event loop
            future = TASK_REGISTRY["run_manager"](task)
            if asyncio.iscoroutine(future):
                loop.run_until_complete(future)
            
            _log_operation = TASK_REGISTRY["logger"]
            
            # Log execution
            execution_log = {
                "task_id": task_details.get("task_id"),
                "executed_at": datetime.now().isoformat(),
                "status": "completed",
                "task": task,
                "agent_id": task_details.get("agent_id")  # Include agent ID in logs
            }
            _log_operation("execute", execution_log["task_id"], execution_log)
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            execution_log = {
                "task_id": task_details.get("task_id"),
                "executed_at": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e),
                "task": task,
                "agent_id": task_details.get("agent_id")  # Include agent ID in logs
            }
            _log_operation("execute_failed", execution_log["task_id"], execution_log)
    @staticmethod
    def _log_operation(operation: str, task_id: str, details: Dict[str, Any]):
        """Log scheduler operations"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "task_id": task_id,
            "details": details
        }
        
        with open("task_logs/scheduler_logs.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def __del__(self):
        """Cleanup scheduler on deletion"""
        if hasattr(self, '_scheduler'):
            self._scheduler.shutdown()

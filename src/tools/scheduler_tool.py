from typing import Dict, Any
from langchain_core.tools import BaseTool
from croniter import croniter
from datetime import datetime
import os
import json
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

logger = logging.getLogger(__name__)

TASK_REGISTRY = {}

class SchedulerTool(BaseTool):
    """Tool for scheduling and managing tasks using cron expressions"""
    name: str = "task_scheduler"
    description: str = """Schedule and manage tasks using cron expressions. Examples:
    - Daily at midnight: "0 0 * * *"
    - Every 30 minutes: "*/30 * * * *"
    - Every Monday at 9am: "0 9 * * 1"
    - First day of month at 3am: "0 3 1 * *"
    
    Inputs:
    - task: The task to be scheduled (shouldn't contain any information about schedule or time)
    - cron: Cron expression for scheduling the task
    
    Example Input - 
    1. Generate a research paper and save it in every 5 minutes. 
        task - "generate a research paper and save it"
        cron - */5 * * * *
    2. Write a research about blockchain and send it to example@gmail.com every day at 9:00 AM.
        task - "write a research about blockchain and send it to example@gmail.com"
        cron - 0 9 * * *
    """
    
    def __init__(self, run_manager):
        super().__init__()
        self._scheduler = BackgroundScheduler(
            jobstores={'default': SQLAlchemyJobStore(url='sqlite:///jobs.sqlite')}
        )
        TASK_REGISTRY["run_manager"] = run_manager
        TASK_REGISTRY["logger"] = self._log_operation
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
                "task_id": task_id
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
    @staticmethod
    def _execute_task(task: str, task_details: Dict[str, Any]):
        """Execute the scheduled task"""
        try:
            logger.info(f"Executing task: {task}")
            TASK_REGISTRY["run_manager"](task)
            _log_operation = TASK_REGISTRY["logger"]
            
            # Log execution
            execution_log = {
                "task_id": task_details.get("task_id"),
                "executed_at": datetime.now().isoformat(),
                "status": "completed",
                "task": task
            }
            _log_operation("execute", execution_log["task_id"], execution_log)
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            execution_log = {
                "task_id": task_details.get("task_id"),
                "executed_at": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e),
                "task": task
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
        if hasattr(self, 'scheduler'):
            self._scheduler.shutdown()

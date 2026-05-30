
"""
A2A (Agent-to-Agent) Protocol Implementation for CircularDrive AI.

Implements Google's Agent-to-Agent protocol specification, enabling
other AI agents to discover and interact with CircularDrive AI agents.

Protocol: https://google.github.io/A2A/

Key concepts:
- Agent Card: JSON metadata describing agent capabilities (/.well-known/agent.json)
- Tasks: Units of work sent between agents
- Streaming: SSE-based real-time task updates
"""
import json
import uuid
from datetime import datetime
from typing import Optional
from enum import Enum


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


# ─── Agent Card ───────────────────────────────────────────────────

AGENT_CARD = {
    "name": "CircularDrive AI",
    "description": "AI-powered Battery Second-Life & Material Passport Intelligence Platform. "
                   "Predicts EV battery State of Health, assigns second-life grades, generates "
                   "EU-compliant material passports, and optimizes recovery pathways.",
    "url": "http://localhost:8000",
    "provider": {
        "organization": "CircularDrive AI",
        "url": "http://localhost:8000",
    },
    "version": "1.0.0",
    "documentationUrl": "http://localhost:8000/docs",
    "capabilities": {
        "streaming": True,
        "pushNotifications": False,
        "stateTransitionHistory": True,
    },
    "authentication": {
        "schemes": ["none"],
    },
    "defaultInputModes": ["text/plain", "application/json"],
    "defaultOutputModes": ["text/plain", "application/json"],
    "skills": [
        {
            "id": "soh-prediction",
            "name": "Battery SOH Prediction",
            "description": "Predict battery State of Health (SOH) percentage and Remaining Useful Life (RUL) "
                           "using ML models trained on battery aging data.",
            "tags": ["battery", "SOH", "prediction", "ML", "health"],
            "examples": [
                "Predict the SOH for a battery with 1200 cycles",
                "What is the remaining useful life of this battery?",
                "Assess health of battery with 65mΩ internal resistance",
            ],
        },
        {
            "id": "second-life-grading",
            "name": "Second-Life Grading",
            "description": "Assign A/B/C/D grade to a battery with safety override rules. "
                           "Determines if battery should be reused, repurposed, refurbished, or recycled.",
            "tags": ["grading", "second-life", "safety", "reuse", "recycling"],
            "examples": [
                "Grade this battery for second-life use",
                "Is this battery safe for second-life applications?",
                "Should I recycle or reuse a battery with 72% SOH?",
            ],
        },
        {
            "id": "material-passport",
            "name": "Material Passport Generation",
            "description": "Generate EU Regulation 2023/1542 compliant digital battery passports with QR codes. "
                           "Includes identity, health, materials, lifecycle, and traceability data.",
            "tags": ["passport", "EU regulation", "QR code", "traceability", "compliance"],
            "examples": [
                "Generate a material passport for battery BAT-001",
                "Create an EU-compliant passport with QR code",
                "What data is in a battery passport?",
            ],
        },
        {
            "id": "recovery-optimization",
            "name": "Disassembly & Recovery Planning",
            "description": "Generate intelligent disassembly sequences and material recovery plans. "
                           "Calculates carbon impact, economic value, and recovery scores.",
            "tags": ["disassembly", "recovery", "recycling", "carbon", "materials"],
            "examples": [
                "Plan recovery for a Grade B battery",
                "What materials can be recovered from this battery?",
                "Estimate carbon savings from recycling this battery",
            ],
        },
        {
            "id": "design-recyclability",
            "name": "Design for Recyclability Analysis",
            "description": "Analyze component design attributes and suggest improvements "
                           "to increase recyclability score and reduce end-of-life processing costs.",
            "tags": ["design", "recyclability", "sustainability", "improvement"],
            "examples": [
                "Analyze the recyclability of this battery design",
                "How can I improve this battery pack's recyclability?",
                "Score the design for end-of-life processing",
            ],
        },
    ],
}


# ─── Task Management ──────────────────────────────────────────────

# In-memory task store (would be Redis/DB in production)
_task_store: dict = {}


def create_task(message_text: str, session_id: Optional[str] = None) -> dict:
    """Create a new A2A task."""
    task_id = str(uuid.uuid4())
    task = {
        "id": task_id,
        "sessionId": session_id or str(uuid.uuid4()),
        "status": {
            "state": TaskState.SUBMITTED,
            "timestamp": datetime.utcnow().isoformat(),
        },
        "history": [
            {
                "role": "user",
                "parts": [{"type": "text", "text": message_text}],
            }
        ],
        "artifacts": [],
        "metadata": {},
    }
    _task_store[task_id] = task
    return task


def update_task_status(task_id: str, state: TaskState, message: Optional[str] = None) -> dict:
    """Update task status."""
    task = _task_store.get(task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")

    task["status"] = {
        "state": state,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if message:
        task["status"]["message"] = {"role": "agent", "parts": [{"type": "text", "text": message}]}

    return task


def add_task_artifact(task_id: str, artifact_type: str, data: dict, name: str = "") -> dict:
    """Add an artifact to a task."""
    task = _task_store.get(task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")

    artifact = {
        "name": name or artifact_type,
        "parts": [{"type": "application/json", "data": data}],
        "metadata": {"created_at": datetime.utcnow().isoformat()},
    }
    task["artifacts"].append(artifact)
    return task


def complete_task(task_id: str, response_text: str, artifacts: list = None) -> dict:
    """Complete a task with response and optional artifacts."""
    task = _task_store.get(task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")

    task["status"] = {
        "state": TaskState.COMPLETED,
        "timestamp": datetime.utcnow().isoformat(),
        "message": {
            "role": "agent",
            "parts": [{"type": "text", "text": response_text}],
        },
    }

    task["history"].append({
        "role": "agent",
        "parts": [{"type": "text", "text": response_text}],
    })

    if artifacts:
        for a in artifacts:
            task["artifacts"].append(a)

    return task


def get_task(task_id: str) -> Optional[dict]:
    """Retrieve a task by ID."""
    return _task_store.get(task_id)


def cancel_task(task_id: str) -> dict:
    """Cancel a task."""
    return update_task_status(task_id, TaskState.CANCELED, "Task canceled by user")


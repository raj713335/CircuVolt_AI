# CircularDrive AI

## Battery Second-Life + Material Passport Intelligence Platform

An **agentic AI platform** that predicts EV battery State of Health, assigns a second-life/recycling grade, generates a QR-based material passport, and recommends the safest, highest-value circular-economy pathway — built with **LangGraph**, **MCP**, **A2A Protocol**, **AG-UI Protocol**, and **CopilotKit**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (React + CopilotKit + AG-UI Protocol)         │
│    ├── CopilotSidebar (conversational AI assistant)     │
│    ├── CopilotActions (frontend tool actions)           │
│    ├── Dashboard / SOH / Passport / Recovery / Design   │
│    └── Recharts visualizations                          │
└──────────────────────┬──────────────────────────────────┘
                       │ AG-UI Protocol (SSE streaming)
┌──────────────────────▼──────────────────────────────────┐
│  CopilotKit Runtime (FastAPI)                           │
│    ├── POST /copilotkit  → AG-UI streaming endpoint     │
│    ├── GET  /.well-known/agent.json  → A2A Agent Card   │
│    ├── POST /a2a/tasks/send  → A2A task processing      │
│    ├── POST /a2a/tasks/sendSubscribe → A2A streaming    │
│    └── REST API endpoints (/predict-soh, etc.)          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  LangGraph ReAct Agent                                  │
│    ├── System prompt (circularity domain expert)        │
│    └── LangChain @tool wrappers                         │
│         ├── predict_battery_soh                         │
│         ├── grade_battery                               │
│         ├── generate_material_passport                  │
│         ├── plan_recovery                               │
│         ├── calculate_circularity                       │
│         └── analyze_recyclability                       │
└──────────────────────┬──────────────────────────────────┘
                       │  (MCP-compatible tools)
┌──────────────────────▼──────────────────────────────────┐
│  MCP Tool Server (Model Context Protocol)               │
│    Standalone server: python mcp_server/server.py       │
│    Same tools exposed as MCP resources + tools          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  ML Models + SQLite Database                            │
│    ├── GradientBoosting SOH Predictor                   │
│    ├── Grading Engine (rule-based + safety overrides)   │
│    ├── Material Passport Generator                      │
│    ├── Recovery Optimizer (graph-based scoring)          │
│    ├── Carbon Calculator                                │
│    └── Recyclability Advisor                            │
└─────────────────────────────────────────────────────────┘
```

---

## Protocol Integration

### MCP (Model Context Protocol)
All circularity intelligence tools are exposed as MCP-compatible tools via `mcp_server/server.py`. Any MCP client can discover and invoke:
- `predict_battery_soh` — ML-powered SOH prediction
- `grade_battery` — Second-life grading with safety overrides
- `generate_material_passport` — EU Reg 2023/1542 compliant passport
- `plan_recovery` — Disassembly sequence optimization
- `calculate_circularity` — Circularity score calculation
- `analyze_recyclability` — Design improvement suggestions

MCP Resources provide context: grading rules, material composition data.

### A2A (Agent-to-Agent Protocol)
Implements Google's A2A specification for inter-agent communication:
- **Agent Card** at `/.well-known/agent.json` — discovery endpoint with skills
- **Task Send** at `/a2a/tasks/send` — synchronous task processing
- **Task Subscribe** at `/a2a/tasks/sendSubscribe` — streaming SSE responses
- **Task Status** at `/a2a/tasks/{id}` — check task progress

Other AI agents can discover and interact with CircularDrive AI agents autonomously.

### AG-UI Protocol + CopilotKit
The AG-UI (Agent-User Interaction) protocol streams agent state to the frontend:
- **TEXT_MESSAGE** events — streaming text responses
- **TOOL_CALL** events — real-time tool invocation visibility
- **STATE_SNAPSHOT/DELTA** — agent state synchronization
- **RUN lifecycle** events — start/finish tracking

CopilotKit implements AG-UI natively and provides:
- `CopilotSidebar` — conversational AI assistant panel
- `useCopilotAction` — frontend actions invokable by the agent
- `useCopilotReadable` — state shared with the agent

### LangGraph + LangChain
- **LangGraph ReAct Agent** — multi-step reasoning with tool use
- **LangChain tools** — `@tool` decorated functions wrapping ML models
- **Streaming** — real-time token streaming via CopilotKit runtime
- **Fallback** — works without OpenAI API key using direct tool execution

---

## Key Features

1. **Battery SOH Prediction** — XGBoost/GradientBoosting ML model with SHAP explainability
2. **Second-Life Grading** — A/B/C/D grades with safety override rules
3. **Material Passport** — EU Regulation 2023/1542 compliant with QR code
4. **Disassembly Optimizer** — Step-by-step recovery with carbon/economic analysis
5. **Circularity Scoring** — Multi-factor score (0-100)
6. **Design Recyclability** — Improvement suggestions with scoring
7. **AI Copilot** — Open-ended conversational interface via CopilotKit
8. **Agent Interoperability** — A2A protocol for agent-to-agent communication

---

## How to Run

### Prerequisites
- Python 3.10+
- Node.js 18+
- (Optional) OpenAI API key for LLM-powered agent

### Backend

```bash
cd backend
pip install -r requirements.txt

# Optional: set OpenAI key for full agent capabilities
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

python main.py
```

- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Agent Card: `http://localhost:8000/.well-known/agent.json`

### MCP Server (standalone)

```bash
cd backend
python mcp_server/server.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

- App: `http://localhost:5173`

---

## API Endpoints

| Method | Endpoint | Protocol | Description |
|--------|----------|----------|-------------|
| POST | `/copilotkit` | AG-UI | CopilotKit streaming endpoint |
| GET | `/.well-known/agent.json` | A2A | Agent discovery card |
| POST | `/a2a/tasks/send` | A2A | Send task to agent |
| POST | `/a2a/tasks/sendSubscribe` | A2A | Send task with SSE streaming |
| GET | `/a2a/tasks/{id}` | A2A | Get task status |
| POST | `/predict-soh` | REST | Predict battery SOH |
| POST | `/generate-passport` | REST | Generate material passport |
| GET | `/passport/{component_id}` | REST | Retrieve passport |
| POST | `/recommend-recovery` | REST | Get recovery plan |
| POST | `/calculate-circularity-score` | REST | Calculate circularity |
| POST | `/design-recyclability-suggestions` | REST | Design analysis |
| GET | `/sample-batteries` | REST | Demo battery data |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | LangGraph (ReAct pattern) |
| LLM Orchestration | LangChain + LangChain-OpenAI |
| Tool Protocol | MCP (Model Context Protocol) |
| Agent Communication | A2A (Google Agent-to-Agent Protocol) |
| UI Protocol | AG-UI (via CopilotKit) |
| Frontend | React, Vite, Tailwind CSS, CopilotKit, Recharts |
| Backend | FastAPI, Python |
| ML | Scikit-learn, GradientBoosting, NumPy, Pandas |
| Database | SQLite (SQLAlchemy ORM) |
| QR Generation | Python qrcode library |

---

## Demo Flow

1. Open the app → Dashboard with architecture overview
2. Click chat icon → AI Copilot opens (CopilotKit sidebar)
3. Ask: *"Predict SOH for a battery with 1200 cycles, 65mΩ resistance"*
4. Agent uses LangGraph → calls `predict_battery_soh` tool → streams result
5. Navigate to Passport tab → Generate QR-based material passport
6. Navigate to Recovery → View disassembly plan + carbon impact
7. Ask copilot: *"Why is this battery not suitable for second life?"*
8. Agent explains using grading rules from MCP resources




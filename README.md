# CloudEdgeAgent — 云边协同隐私保护 AI Agent 系统

Privacy-First Cloud-Edge Collaborative AI Agent System

## Architecture

```text
Frontend → API Gateway → ChatService → CollaborativeOrchestrator
                                            ├── PrivacyDetector (3-layer: Regex → NER → SLM)
                                            ├── ComplexityAnalyzer (edge SLM)
                                            ├── PolicyEngine (route matrix)
                                            └── Execute Mode:
                                                A: Direct Local
                                                B: Direct Cloud
                                                C: Sanitize → Cloud → Restore
                                                D: Sketch → Refine → Restore
```

## Project Structure

```text
project/
├── app/                          # Application source
│   ├── api/                      # FastAPI HTTP layer
│   │   ├── routers/              #   chat.py, health.py
│   │   ├── schemas/              #   Pydantic request/response models
│   │   └── dependencies/         #   Dependency injection wiring
│   ├── core/                     # Cross-cutting concerns
│   │   ├── config/               #   Pydantic Settings
│   │   ├── logger/               #   structlog setup
│   │   ├── exceptions/           #   Unified exception hierarchy
│   │   └── security/             #   API key validation
│   ├── domain/                   # Business abstractions (no external deps)
│   │   ├── agent/                #   BaseAgent, AgentResult
│   │   ├── llm/                  #   LLMClient interface
│   │   ├── memory/               #   MemoryStore interface
│   │   ├── privacy/              #   PrivacyDetector, Sanitizer, Policy
│   │   ├── rag/                  #   Chunker, Retriever, Reranker
│   │   └── tool/                 #   BaseTool, ToolRegistry
│   ├── infrastructure/           # Concrete implementations
│   │   ├── llm/                  #   OpenAI-compatible client + factory
│   │   ├── vectorstore/          #   QdrantMemoryStore
│   │   ├── database/             #   InMemorySessionStore
│   │   └── cache/                #   InMemoryCache
│   ├── services/                 # Business orchestration
│   │   ├── privacy_engine.py     #   3-layer detector + sanitizer + budget
│   │   ├── agent_orchestrator.py #   CollaborativeOrchestrator (4 modes)
│   │   └── chat_service.py       #   ChatService
│   └── main.py                   #   FastAPI app factory
├── tools/                        # Built-in tools
│   ├── search_tool.py            #   Web search (DuckDuckGo)
│   ├── calculator_tool.py        #   Safe math evaluation
│   └── time_tool.py              #   Current time
├── tests/
│   ├── unit/                     #   Unit tests
│   ├── integration/              #   Integration tests
│   └── e2e/                      #   End-to-end tests
├── configs/
│   └── .env.example              #   Environment variable template
├── scripts/
│   └── run.py                    #   Dev server runner
├── pyproject.toml                #   Project metadata + dependencies
└── CLAUDE.md                     #   Engineering guidelines
```

## Quick Start

### 0. Just start a agent cli...
后面的都还没实现。至少cli版（最小实现）现在可以交互——基于deepseek api。
```bash
python scripts/cli.py
```


### 1. Install dependencies

```bash
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp configs/.env.example .env
# Edit .env with your API keys
```

### 3. Start local LLM (Ollama)

```bash
ollama pull qwen2.5:7b-instruct
ollama serve
```

### 4. Run the server

```bash
python scripts/run.py
# or
uvicorn app.main:app --reload
```

### 5. Send a request

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "今天天气怎么样？"}'
```

### 6. Run tests

```bash
pytest tests/ -v
```

## Tech Stack

| Component | Choice | Purpose |
|-----------|--------|---------|
| Backend | FastAPI | Lightweight async API framework |
| Edge LLM | Ollama + Qwen2.5-7B | Local inference (privacy-sensitive) |
| Cloud LLM | DeepSeek API | Cloud inference (complex tasks) |
| Agent | ReAct loop | Think → Act → Observe → Reflect |
| Vector DB | Qdrant | Long-term memory + RAG |
| Privacy | 3-layer pipeline | Regex → NER → SLM judge |
| Logging | structlog | Structured JSON logging |

## Collaborate Modes

| Mode | Name | Flow | Use Case |
|------|------|------|----------|
| A | Direct Local | User → Edge → Answer | Low complexity, high privacy |
| B | Direct Cloud | User → Cloud → Answer | High complexity, no privacy concern |
| C | Sanitize-Cloud | User → Sanitize → Cloud → Restore → Answer | Sensitive + complex |
| D | Sketch-Refine | Edge sketch → Cloud refine → Edge restore | Confidential + complex (PBCR) |

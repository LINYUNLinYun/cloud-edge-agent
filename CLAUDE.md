# CLAUDE.md — AI Agent Project Engineering Guidelines

Version: 1.0

## 1. Mission

构建高内聚、低耦合、易测试、易扩展、易维护的 AI Agent 系统。任何新增代码必须满足：

- 单一职责原则（SRP）、开闭原则（OCP）、依赖倒置原则（DIP）
- 模块可独立开发与测试，禁止业务逻辑散落

## 2. Core Principles

**Architecture First** — 优先架构设计，禁止 `if provider == "openai": ...` 式分支，通过多态扩展。

```python
# BAD — 直接分支
if provider == "openai": ...
elif provider == "deepseek": ...

# GOOD — 抽象接口
class LLMClient(ABC): ...
```

**Dependency Inversion** — 高层不依赖具体实现。`Agent → LLMClient(Interface) → OpenAIClient / DeepSeekClient`，禁止 `Agent → OpenAI SDK`。

**Composition Over Inheritance** — 优先组合，避免继承树 `SearchAgent / RAGAgent / PlannerAgent`，优先 `Agent { Planner, ToolManager, MemoryManager }`。

**Explicit > Implicit** — 变量名必须表达业务含义：`processed_documents = process_documents(raw_documents)`，禁止 `data = process(data)`。

## 3. Project Structure & Layers

```text
project/
├── api/                # HTTP 接口、参数校验、返回格式（禁止业务逻辑）
│   ├── routers/
│   ├── schemas/
│   └── dependencies/
├── core/               # 配置、日志、异常、安全
│   ├── config/
│   ├── logger/
│   ├── exceptions/
│   └── security/
├── domain/             # 核心业务，不依赖 FastAPI / Redis / Milvus / OpenAI
│   ├── agent/
│   ├── memory/
│   ├── llm/
│   ├── tool/
│   ├── rag/
│   └── privacy/
├── infrastructure/     # 具体实现，允许依赖第三方 SDK
│   ├── llm/
│   ├── vectorstore/
│   ├── database/
│   ├── cache/
│   └── external/
├── services/           # 业务编排（Request → Agent → Memory → Response）
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── scripts/
├── docs/
├── configs/
├── pyproject.toml
└── README.md
```

**Layer 职责速查**：

| Layer | 职责 | 禁止 |
|-------|------|------|
| API | HTTP 接口、参数校验、格式转换 | Agent/RAG/数据库逻辑 |
| Service | 业务编排 | SQL、Prompt 细节、SDK 调用 |
| Domain | 核心业务（Agent/Memory/Tool/RAG/Privacy） | 依赖 FastAPI/Redis/Milvus/OpenAI |
| Infrastructure | 具体实现（OpenAIClient/MilvusStore/…） | — |

## 4. Component Design Rules

### 4.1 LLM — 统一接口

```python
class LLMClient(ABC):
    @abstractmethod
    async def invoke(self, messages: list[dict]) -> str: ...
```

实现：`OpenAIClient` / `DeepSeekClient` / `OllamaClient`。Agent 永远依赖 `LLMClient`，而非具体实现。

### 4.2 Tool — 插件化

```python
class BaseTool(ABC):
    name: str
    description: str
    @abstractmethod
    async def execute(self, **kwargs): ...
```

工具定义在 `domain/tool/`（base.py, registry.py），具体工具在 `tools/`。Agent 通过 `tool_manager.execute("search", query)` 调用，禁止直接调用 `search()`。

### 4.3 Memory — 抽象存储

```python
class MemoryStore(ABC): ...
# 实现：SQLiteMemoryStore / PostgresMemoryStore
```

替换数据库无需修改业务逻辑。

### 4.4 RAG — 模块化管道

```text
rag/
├── chunker/
├── embedder/
├── retriever/
├── reranker/
└── pipeline/
```

禁止一个 `rag.py` 实现全部功能。

## 5. Configuration & Infra Rules

- **配置**：用 `pydantic_settings.BaseSettings` 统一管理，禁止硬编码 `api_key="xxx"`。
- **日志**：使用 `logging` + `structlog`，结构化记录（`logger.info("tool_execute", tool="search", latency=1.25)`），禁止 `print()`。
- **异常**：统一继承 `BaseAppException` → `LLMException` / `ToolException` / `RAGException` / `MemoryException`，API 层统一捕获。

## 6. Testing & Code Quality

- 覆盖率 >= 80%
- 目录：`tests/{unit, integration, e2e}/`
- 必须通过：`ruff check .` / `ruff format .` / `mypy .` / `pytest`
- CI 失败禁止合并

## 7. Git & Documentation

- 分支：`main` / `develop` / `feature/*` / `bugfix/*`
- Commit 规范：`feat:` / `fix:` / `refactor:` / `docs:` / `test:` / `chore:`（例：`feat(agent): add tool routing`）
- 新增模块必须同步更新 `README.md` 和 `docs/`，包含 Purpose / Design / Dependency / Example

## 8. Coding Standards & DoD

**新增代码必须**：类型注解完整、Docstring 完整、测试完整。

```python
async def retrieve_documents(query: str, top_k: int = 5) -> list[Document]:
    """Retrieve relevant documents.

    Args:
        query: user query
        top_k: retrieval count
    Returns:
        retrieved documents
    """
```

**Agent 新增功能前**：阅读现有架构 → 阅读相关接口 → 优先复用 → 禁止复制粘贴。

**Definition of Done**：功能实现 + 单元测试 + 集成测试 + 类型检查 + Lint + 文档更新 + Code Review。否则视为未完成。

**Absolute Prohibitions**：God Class、超过 1000 行单文件、超过 200 行单函数、硬编码配置、print 调试、重复代码复制、API 层写业务逻辑、Service 层直接访问 SDK、跳过测试、跳过类型标注。发现后必须重构。

# 云边协同隐私保护 Agent 系统 — Architecture & Plan

**Version:** v1.0 | **Author:** XXX

---

## 1. 项目背景与核心思路

随着 DeepSeek、Qwen、GPT 等大模型能力提升，AI Agent 被广泛用于个人助理、代码开发、知识管理等场景。但现有系统存在两个核心痛点：

- **隐私风险**：用户的身份证号、手机号、银行流水、企业代码、私有文档等敏感数据可能直接上传至第三方云端。
- **本地模型能力不足**：本地部署的模型虽安全，但推理能力弱、长上下文有限、Tool Calling 效果差，无法独立完成复杂任务。

为此，本项目构建一个 **Privacy-First Cloud-Edge Collaborative Agent**，遵循 Local First、Privacy First、Cost Aware、Graceful Degradation 四项原则。核心思路是：简单任务本地执行，复杂任务云端执行；当面对敏感又复杂的数据时，先本地脱敏，再送云端推理，最后本地恢复结果。

---

## 2. 系统目标

**功能目标**：支持对话问答、Agent 推理、Tool 调用、RAG 检索、云边协同推理、隐私保护。

**非功能目标**：
- **隐私保护** — 敏感数据默认不出本地。
- **透明可控** — 用户可查看路由结果、脱敏内容及 Agent 执行全过程。
- **可扩展性** — 新模型、新工具、新知识库可快速接入。

---

## 3. 总体架构

系统采用分层架构，数据流向如下：

```text
Frontend → API Gateway → Agent Orchestrator
                              ├── Privacy Engine ──┐
                              └── Task Analyzer ───┤
                                                  ▼
                                           Policy Engine
                                          /            \
                                   Edge Agent        Cloud Agent
                                          \            /
                                    Tool Layer + Memory Layer
                                               │
                                         Final Response
```

---

## 4. 核心模块

### 4.1 LLM Client — 统一模型接口

定义统一的 `LLMClient` 抽象（`invoke`、`stream_invoke`、`think`、`embedding`），屏蔽底层差异。本地端接入 Ollama / vLLM，云端接入 DeepSeek、Qwen API 等 OpenAI 兼容接口。

### 4.2 Privacy Engine — 隐私分析（三级流水线）

- **Layer 1 — 规则引擎**：正则匹配手机号、身份证、邮箱、银行卡号等，延迟 < 1ms。
- **Layer 2 — NER 识别**：基于 Presidio 或 GLiNER 识别人名、地址、公司名等命名实体。
- **Layer 3 — SLM 隐私裁判**：使用 Qwen2.5-1.5B 判断文本是否敏感及其等级，输出 `{"level": "S1", "confidence": 0.92}`。

### 4.3 Task Analyzer — 任务复杂度评估

将任务分为五个等级并输出 `{"complexity": "simple"}` 等标签：

| 等级 | 描述 |
|------|------|
| L1 | FAQ 类应答 |
| L2 | 单步推理 |
| L3 | 多步推理 |
| L4 | Agent 任务 |
| L5 | 长链复杂任务 |

### 4.4 Policy Engine — 路由决策

综合隐私等级、复杂度等级和系统状态做决策，核心矩阵如下：

| Privacy | Complexity | Route           |
| ------- | ---------- | --------------- |
| S1      | L1-L2      | Edge            |
| S1      | L3-L5      | Cloud           |
| S2      | L1-L2      | Edge            |
| S2      | L3-L5      | Sanitized Cloud |
| S3      | Any        | Edge            |

### 4.5 Collaborative Orchestrator — 协同编排（核心创新）

- **Mode A — Direct Local**：用户 → Edge → 回答。适用于低复杂度、高隐私场景。
- **Mode B — Direct Cloud**：用户 → Cloud → 回答。适用于无敏感数据的高复杂度场景。
- **Mode C — Sanitize-Cloud-Restore**：用户 → 脱敏 → Cloud → 恢复 → 回答。适用于敏感复杂任务。
- **Mode D — Sketch-Refine**（扩展加分）：Edge 生成隐私摘要 → Cloud 细化 → Edge 恢复完整结果。

### 4.6 ReAct Agent 框架

Agent 按 Think → Action → Observation → Reflection 循环推理，内置 Search、Calculator、File Reader、RAG 等工具。

### 4.7 Tool Layer

首版提供四种工具：联网搜索（Search）、数学计算（Calculator）、时间查询（Time）、知识检索（RAG）。

### 4.8 Memory Layer — 双轨记忆

- **短期记忆**：最近对话上下文。
- **长期记忆**：基于 ChromaDB 的向量存储。
- **双轨记忆**（本项目特色）：本地保留 `MEMORY-FULL.md` 完整数据，云端仅存放脱敏后的 `MEMORY.md`。

---

## 5. 技术栈

| 层面 | 选型 | 说明 |
|------|------|------|
| Backend | FastAPI | 轻量高性能 |
| Agent 框架 | LangGraph | 状态机 + 条件路由，易于调试 |
| Local LLM | **Ollama + Qwen2.5-7B-Instruct（待定）** | 本地推理 |
| Cloud LLM | DeepSeek-V4 pro（兼容OpenAI） | 云端推理 |
| Vector DB | Qdrant | 长期记忆与RAG，免费&支持向量检索 |
| Frontend | React + Vite（或 Gradio MVP） | 对话界面与可视化 |

---

## 6. 开发路线图（6 周）

| Phase | 内容 | 交付物 |
|-------|------|--------|
| 1 (Day 1) | 基础设施 | LLM Client、Ollama/DeepSeek 接入、FastAPI 骨架 |
| 2 (Day 2) | 云边路由 | Complexity Analyzer、Privacy Analyzer、Policy Engine |
| 3 (Day 3) | 协同管线 | 脱敏模块、恢复模块、双模型协作 |
| 4 (Day 4) | Agent + Tools | ReAct Agent、Search Tool、Calculator Tool |
| 5 (Day 5) | RAG | ChromaDB 集成、本地知识库 |
| 6 (Day 6) | 前端 | Chat UI、路由可视化、隐私仪表盘 |

---

## 7. 评估方案

**对比基线**：Pure Cloud（全云端）、Pure Local（全本地）、Cloud-Edge Router（无脱敏路由）。

**评估指标**：
- **隐私** — 敏感信息泄露率
- **效率** — 端到端响应时间
- **成本** — Token 消耗、云端调用次数
- **质量** — GPT-as-Judge 评分、任务成功率

---

## 8. 预期创新点

1. **三级隐私路由**：规则引擎 + NER + SLM 组合判决，精度与延迟兼顾。
2. **云边协同 Agent 架构**：Mode A/B/C/D 四模式协同，按需在本地与云端间迁移。
3. **双轨记忆机制**：本地全量 + 云端脱敏，兼顾隐私与智能。
4. **Sketch-Refine 协同推理**（扩展）：边端生成摘要 → 云端补全 → 边端恢复。

---

## 9. MVP 范围

**必须完成**：LLM Client、Cloud/Edge Agent、Privacy Router、脱敏模块、ReAct Agent、Web UI。

**加分项**：RAG、Tool Calling、Dual Memory、Sketch-Refine、Privacy Dashboard。

---

## 10. 最终交付物

- **代码**：Backend、Frontend、部署脚本
- **文档**：Requirement.md、Architecture&Plan.md
- **演示**：Demo Video、PPT
- **报告**：设计报告、实验评估报告
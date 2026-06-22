"""Experiment 1: Latency Benchmark — 测量不同路由模式下的响应延迟指标.

Metrics:
  - TTFT (Time to First Token): 从发送请求到收到第一个 token 的时间
  - TPOT (Time Per Output Token): 每个输出 token 的平均耗时
  - Total Latency: 端到端总延迟
  - Prompt/Completion Tokens: 输入输出 token 数

Usage:
  1. 确保后端运行在 localhost:8000
  2. python experiments/exp1_latency.py
  3. 结果输出到 experiments/results/latency_results.json
"""

import asyncio
import json
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"

# --- 测试用例 ---
# 设计两组用例:
#   Mode A (本地): 简单知识问答，无敏感信息，低复杂度
#   Mode B/C (云端): 复杂推理任务 或 含敏感信息的复杂任务

MODE_A_CASES = [
    {
        "id": "a1",
        "query": "什么是马尔可夫链？请简要解释。",
        "expected_mode": "A",
        "description": "简单知识问答",
    },
    {
        "id": "a2",
        "query": "Python中list和tuple的区别是什么？",
        "expected_mode": "A",
        "description": "编程基础知识",
    },
    {
        "id": "a3",
        "query": "今天是星期几？",
        "expected_mode": "A",
        "description": "简单时间查询（调用工具）",
    },
    {
        "id": "a4",
        "query": "帮我算一下 2 的 10 次方是多少。",
        "expected_mode": "A",
        "description": "简单计算（调用工具）",
    },
    {
        "id": "a5",
        "query": "用一句话解释什么是量子计算。",
        "expected_mode": "A",
        "description": "简短知识问答",
    },
]

MODE_BC_CASES = [
    {
        "id": "b1",
        "query": "请详细分析 Transformer 架构中 Self-Attention 机制的计算复杂度，并讨论有哪些优化方法可以将其从 O(n²) 降低。",
        "expected_mode": "B",
        "description": "复杂技术分析",
    },
    {
        "id": "b2",
        "query": "请对比分析深度学习中 Batch Normalization 和 Layer Normalization 的原理、适用场景和优缺点，并给出代码示例。",
        "expected_mode": "B",
        "description": "复杂对比分析",
    },
    {
        "id": "b3",
        "query": "帮我分析一下我手机号13912345678的用户在哪个运营商，归属地是哪里，并给出该地区的经济发展概况。",
        "expected_mode": "C",
        "description": "含手机号的复杂查询（脱敏上云）",
    },
    {
        "id": "b4",
        "query": "我的身份证号是110101199001011234，请帮我分析这个号码对应的地区信息，并给出该地区的详细人口统计数据。",
        "expected_mode": "C",
        "description": "含身份证的复杂查询（脱敏上云）",
    },
    {
        "id": "b5",
        "query": "请帮我写一篇关于联邦学习在医疗数据隐私保护中应用的综述，要求包含技术原理、应用场景、挑战和未来方向，不少于800字。",
        "expected_mode": "B",
        "description": "长文本生成任务",
    },
]


async def measure_stream(query: str, client: httpx.AsyncClient) -> dict:
    """发送 SSE 流式请求并测量延迟指标.

    Returns:
        dict with ttft_ms, tpot_ms, total_ms, tokens, answer, mode, privacy_level
    """
    request_start = time.monotonic()
    ttft_ms = None
    token_times = []
    answer_chunks = []
    metadata = {}

    async with client.stream(
        "POST",
        f"{BASE_URL}/api/v1/chat/stream",
        json={"query": query},
        timeout=120.0,
    ) as response:
        buffer = ""
        async for raw_chunk in response.aiter_text():
            buffer += raw_chunk
            lines = buffer.split("\n")
            buffer = lines.pop()

            for line in lines:
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                json_str = line[6:]
                if not json_str:
                    continue
                try:
                    event = json.loads(json_str)
                except json.JSONDecodeError:
                    continue

                now = time.monotonic()

                if event.get("type") == "metadata":
                    metadata = event
                    metadata_recv_ms = (now - request_start) * 1000

                elif event.get("type") == "token":
                    if ttft_ms is None:
                        ttft_ms = (now - request_start) * 1000
                    token_times.append(now)
                    delta = event.get("delta", event.get("token", ""))
                    answer_chunks.append(delta)

                elif event.get("type") == "done":
                    done_latency = event.get("latency_ms", 0)

    total_ms = (time.monotonic() - request_start) * 1000

    # 计算 TPOT
    tpot_ms = None
    if len(token_times) >= 2:
        intervals = [
            (token_times[i + 1] - token_times[i]) * 1000
            for i in range(len(token_times) - 1)
        ]
        tpot_ms = sum(intervals) / len(intervals)

    return {
        "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
        "tpot_ms": round(tpot_ms, 2) if tpot_ms else None,
        "total_ms": round(total_ms, 1),
        "done_latency_ms": round(done_latency, 1) if done_latency else None,
        "num_chunks": len(token_times),
        "answer": "".join(answer_chunks),
        "mode": metadata.get("mode"),
        "privacy_level": metadata.get("privacy_level"),
        "complexity": metadata.get("complexity"),
        "session_id": metadata.get("session_id"),
    }


async def run_latency_tests():
    """运行延迟测试."""
    results = {
        "experiment": "latency_benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": BASE_URL,
        "mode_a_cases": [],
        "mode_bc_cases": [],
        "summary": {},
    }

    async with httpx.AsyncClient() as client:
        # 检查后端
        try:
            resp = await client.get(f"{BASE_URL}/health", timeout=5.0)
            resp.raise_for_status()
            print(f"[OK] 后端连接正常: {resp.json()}")
        except Exception as e:
            print(f"[FAIL] 无法连接后端 {BASE_URL}: {e}")
            print("请先启动后端: python -m uvicorn app.main:app --port 8000")
            return

        # 测试 Mode A (本地)
        print("\n" + "=" * 60)
        print("测试 Mode A (本地直答) — 预期路由到边缘 LLM")
        print("=" * 60)
        for case in MODE_A_CASES:
            print(f"\n  [{case['id']}] {case['description']}")
            print(f"  Query: {case['query'][:50]}...")
            result = await measure_stream(case["query"], client)
            result["case_id"] = case["id"]
            result["description"] = case["description"]
            result["expected_mode"] = case["expected_mode"]
            results["mode_a_cases"].append(result)
            print(f"  → Mode: {result['mode']}, Privacy: {result['privacy_level']}")
            print(f"  → TTFT: {result['ttft_ms']}ms, TPOT: {result['tpot_ms']}ms")
            print(f"  → Total: {result['total_ms']}ms, Chunks: {result['num_chunks']}")

        # 测试 Mode B/C (云端)
        print("\n" + "=" * 60)
        print("测试 Mode B/C (云端) — 预期路由到云端 LLM")
        print("=" * 60)
        for case in MODE_BC_CASES:
            print(f"\n  [{case['id']}] {case['description']}")
            print(f"  Query: {case['query'][:50]}...")
            result = await measure_stream(case["query"], client)
            result["case_id"] = case["id"]
            result["description"] = case["description"]
            result["expected_mode"] = case["expected_mode"]
            results["mode_bc_cases"].append(result)
            print(f"  → Mode: {result['mode']}, Privacy: {result['privacy_level']}")
            print(f"  → TTFT: {result['ttft_ms']}ms, TPOT: {result['tpot_ms']}ms")
            print(f"  → Total: {result['total_ms']}ms, Chunks: {result['num_chunks']}")

    # 计算汇总统计
    def calc_stats(cases: list[dict]) -> dict:
        valid = [c for c in cases if c["ttft_ms"] is not None]
        if not valid:
            return {}
        return {
            "count": len(valid),
            "avg_ttft_ms": round(sum(c["ttft_ms"] for c in valid) / len(valid), 1),
            "min_ttft_ms": round(min(c["ttft_ms"] for c in valid), 1),
            "max_ttft_ms": round(max(c["ttft_ms"] for c in valid), 1),
            "avg_tpot_ms": round(
                sum(c["tpot_ms"] for c in valid if c["tpot_ms"]) /
                max(1, len([c for c in valid if c["tpot_ms"]])), 2
            ),
            "avg_total_ms": round(
                sum(c["total_ms"] for c in valid) / len(valid), 1
            ),
        }

    results["summary"] = {
        "mode_a": calc_stats(results["mode_a_cases"]),
        "mode_bc": calc_stats(results["mode_bc_cases"]),
    }

    # 保存结果
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "latency_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印汇总
    print("\n" + "=" * 60)
    print("汇总统计")
    print("=" * 60)
    for mode_key, label in [("mode_a", "Mode A (本地)"), ("mode_bc", "Mode B/C (云端)")]:
        stats = results["summary"].get(mode_key, {})
        if not stats:
            print(f"\n{label}: 无有效数据")
            continue
        print(f"\n{label}:")
        print(f"  用例数:   {stats['count']}")
        print(f"  平均 TTFT: {stats['avg_ttft_ms']}ms")
        print(f"  平均 TPOT: {stats['avg_tpot_ms']}ms")
        print(f"  平均总延迟: {stats['avg_total_ms']}ms")
        print(f"  TTFT 范围: {stats['min_ttft_ms']}ms ~ {stats['max_ttft_ms']}ms")

    print(f"\n>> 详细结果已保存到: {out_path}")


if __name__ == "__main__":
    asyncio.run(run_latency_tests())

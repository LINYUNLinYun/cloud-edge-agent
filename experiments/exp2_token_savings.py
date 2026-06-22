"""Experiment 2: Token Savings — 对比云边协同路由 vs 纯云方案的 token 消耗和成本.

核心思路:
  - 我们的系统：简单任务走本地（消耗本地 token，不花钱），复杂任务走云端
  - 纯云基线：所有任务都走云端（全部消耗云端 token，全部花钱）
  - 对比两者的云端 token 消耗和成本差异

DeepSeek API 定价 (2026):
  - deepseek-chat: 输入 CNY 1/M tokens, 输出 CNY 2/M tokens
  - 本地 Ollama: 免费

Usage:
  1. 确保后端运行在 localhost:8000
  2. python experiments/exp2_token_savings.py
  3. 结果输出到 experiments/results/token_savings_results.json
"""

import asyncio
import json
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"

# DeepSeek 定价 (元/百万 token)
CLOUD_INPUT_PRICE = 1.0   # CNY 1 / M tokens
CLOUD_OUTPUT_PRICE = 2.0   # CNY 2 / M tokens

# --- 模拟日常使用场景 ---
# 设计 15 条覆盖日常使用场景的查询，混合简单/复杂、敏感/非敏感
DAILY_SCENARIOS = [
    # --- 简单问答（预期走本地 Mode A）---
    {
        "id": "s01",
        "query": "你好，今天过得怎么样？",
        "category": "闲聊",
        "expected_route": "local",
    },
    {
        "id": "s02",
        "query": "Python中for循环和while循环有什么区别？",
        "category": "简单知识",
        "expected_route": "local",
    },
    {
        "id": "s03",
        "query": "帮我算一下 15 * 23 + 47 等于多少。",
        "category": "简单计算",
        "expected_route": "local",
    },
    {
        "id": "s04",
        "query": "什么是机器学习？用一句话解释。",
        "category": "简单知识",
        "expected_route": "local",
    },
    {
        "id": "s05",
        "query": "现在几点了？",
        "category": "时间查询",
        "expected_route": "local",
    },
    {
        "id": "s06",
        "query": "HTTP和HTTPS的区别是什么？",
        "category": "简单知识",
        "expected_route": "local",
    },
    {
        "id": "s07",
        "query": "用Python写一个Hello World程序。",
        "category": "简单编程",
        "expected_route": "local",
    },
    # --- 复杂任务（预期走云端 Mode B）---
    {
        "id": "s08",
        "query": "请详细解释 Raft 共识算法的工作原理，包括 Leader Election、Log Replication 和 Safety 三个核心机制，并与 Paxos 进行对比分析。",
        "category": "复杂技术分析",
        "expected_route": "cloud",
    },
    {
        "id": "s09",
        "query": "请分析2024年全球AI芯片市场的竞争格局，包括NVIDIA、AMD、Intel、华为昇腾等主要玩家的技术路线和市场份额，并预测未来3年的发展趋势。",
        "category": "复杂分析",
        "expected_route": "cloud",
    },
    {
        "id": "s10",
        "query": "请设计一个分布式限流系统，要求支持每秒10万次请求，支持滑动窗口算法，给出架构设计、核心数据结构和伪代码。",
        "category": "系统设计",
        "expected_route": "cloud",
    },
    # --- 含敏感信息的复杂任务（预期走 Mode C 脱敏上云）---
    {
        "id": "s11",
        "query": "我的邮箱是zhangsan@company.com，请帮我分析这个邮箱域名对应的企业可能使用哪些云服务，并给出该企业数字化转型的建议方案。",
        "category": "含邮箱的复杂任务",
        "expected_route": "cloud_sanitize",
    },
    {
        "id": "s12",
        "query": "我手机号是13800138000，请帮我查一下这个号码的运营商和归属地，并分析该地区的通信基础设施发展情况。",
        "category": "含手机号的复杂任务",
        "expected_route": "cloud_sanitize",
    },
    # --- 含敏感信息的简单任务（预期走本地 Mode A）---
    {
        "id": "s13",
        "query": "我的银行卡号是6222021234567890123，请问这是哪个银行的卡？",
        "category": "含银行卡的简单任务",
        "expected_route": "local",
    },
    {
        "id": "s14",
        "query": "请帮我存一下我老板的电话：13912345678，邮箱：boss@company.com。",
        "category": "含联系方式的存储任务",
        "expected_route": "local",
    },
    # --- 中等复杂度（可能走本地或云端）---
    {
        "id": "s15",
        "query": "请解释什么是Docker容器，它和虚拟机有什么区别？各自的优缺点是什么？",
        "category": "中等知识问答",
        "expected_route": "local_or_cloud",
    },
]


async def send_query(query: str, client: httpx.AsyncClient) -> dict:
    """发送同步请求并获取路由信息."""
    resp = await client.post(
        f"{BASE_URL}/api/v1/chat",
        json={"query": query},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文约 1.5 字/token，英文约 0.75 词/token）.

    这是一个近似值，用于在无法获取实际 token 数时做估算。
    """
    chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


def calc_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """计算云端调用成本 (元)."""
    return (
        prompt_tokens * CLOUD_INPUT_PRICE / 1_000_000
        + completion_tokens * CLOUD_OUTPUT_PRICE / 1_000_000
    )


async def run_token_savings():
    """运行 token 节省量测试."""
    results = {
        "experiment": "token_savings",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": BASE_URL,
        "pricing": {
            "cloud_input_price_per_m": CLOUD_INPUT_PRICE,
            "cloud_output_price_per_m": CLOUD_OUTPUT_PRICE,
            "local_price": "免费 (Ollama)",
        },
        "scenarios": [],
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
            return

        print("\n" + "=" * 70)
        print("Token 节省量实验 — 模拟日常使用场景")
        print("=" * 70)

        for scenario in DAILY_SCENARIOS:
            print(f"\n  [{scenario['id']}] {scenario['category']}")
            print(f"  Query: {scenario['query'][:60]}...")

            try:
                result = await send_query(scenario["query"], client)
            except Exception as e:
                print(f"  [FAIL] 请求失败: {e}")
                results["scenarios"].append({
                    **scenario,
                    "error": str(e),
                })
                continue

            mode = result.get("mode", "unknown")
            privacy = result.get("privacy_level", "unknown")
            answer = result.get("answer", "")

            # 判断实际路由
            if mode in ("direct_local", "A"):
                actual_route = "local"
            elif mode in ("sanitize_cloud", "C"):
                actual_route = "cloud_sanitize"
            else:
                actual_route = "cloud"

            # 估算 token
            prompt_tokens = estimate_tokens(scenario["query"])
            completion_tokens = estimate_tokens(answer)

            # 计算纯云方案的成本（所有请求都走云端）
            pure_cloud_cost = calc_cost(prompt_tokens, completion_tokens)

            # 我们的方案成本（只有走云端的才花钱）
            our_cost = pure_cloud_cost if actual_route != "local" else 0.0

            entry = {
                "id": scenario["id"],
                "category": scenario["category"],
                "query": scenario["query"],
                "mode": mode,
                "privacy_level": privacy,
                "actual_route": actual_route,
                "expected_route": scenario["expected_route"],
                "answer_length": len(answer),
                "est_prompt_tokens": prompt_tokens,
                "est_completion_tokens": completion_tokens,
                "pure_cloud_cost_yuan": round(pure_cloud_cost, 6),
                "our_cost_yuan": round(our_cost, 6),
                "saved_yuan": round(pure_cloud_cost - our_cost, 6),
            }
            results["scenarios"].append(entry)

            route_label = "本地" if actual_route == "local" else "云端" if actual_route == "cloud" else "脱敏上云"
            saved = pure_cloud_cost - our_cost
            print(f"  → Mode: {mode}, 实际路由: {route_label}")
            print(f"  → 估算 tokens: 输入~{prompt_tokens}, 输出~{completion_tokens}")
            print(f"  → 纯云成本: CNY {pure_cloud_cost:.4f}, 我们: CNY {our_cost:.4f}, 节省: CNY {saved:.4f}")

    # 汇总统计
    valid = [s for s in results["scenarios"] if "error" not in s]
    local_cases = [s for s in valid if s["actual_route"] == "local"]
    cloud_cases = [s for s in valid if s["actual_route"] != "local"]

    total_pure_cloud = sum(s["pure_cloud_cost_yuan"] for s in valid)
    total_ours = sum(s["our_cost_yuan"] for s in valid)
    total_saved = total_pure_cloud - total_ours
    savings_pct = (total_saved / total_pure_cloud * 100) if total_pure_cloud > 0 else 0

    results["summary"] = {
        "total_scenarios": len(valid),
        "local_routed": len(local_cases),
        "cloud_routed": len(cloud_cases),
        "total_pure_cloud_cost_yuan": round(total_pure_cloud, 6),
        "total_our_cost_yuan": round(total_ours, 6),
        "total_saved_yuan": round(total_saved, 6),
        "savings_percentage": round(savings_pct, 1),
    }

    # 保存结果
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "token_savings_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印汇总
    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)
    print(f"  总场景数:     {len(valid)}")
    print(f"  路由到本地:   {len(local_cases)} 个")
    print(f"  路由到云端:   {len(cloud_cases)} 个")
    print(f"  纯云方案总成本: CNY {total_pure_cloud:.4f}")
    print(f"  我们的方案成本: CNY {total_ours:.4f}")
    print(f"  节省金额:      CNY {total_saved:.4f}")
    print(f"  节省比例:      {savings_pct:.1f}%")
    print(f"\n>> 详细结果已保存到: {out_path}")


if __name__ == "__main__":
    asyncio.run(run_token_savings())

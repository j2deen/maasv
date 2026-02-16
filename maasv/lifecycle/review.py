"""
Conversation Review - Sleep-Time Compute

Second-pass analysis of recent conversations with more time/budget.
Uses the injected LLMProvider — no direct API calls.
"""

import logging
import json
from typing import Callable
from datetime import datetime

logger = logging.getLogger("maasv.lifecycle.review")


def run_review_job(data: dict, cancel_check: Callable[[], bool]) -> dict:
    """
    Run a review job on recent conversation history.

    Args:
        data: {"messages": [...], "recent_memories": [...], "last_extraction_time": float}
        cancel_check: Function to check if job should stop
    """
    import time

    messages = data.get("messages", [])
    recent_memories = data.get("recent_memories", [])
    last_extraction = data.get("last_extraction_time", 0)

    if last_extraction and (time.time() - last_extraction) < 300:
        logger.info("[Review] Skipping - recent extraction from compaction (within 5 min)")
        return {"insights": [], "stored": 0, "skipped": "recent_extraction"}

    if not messages or len(messages) < 4:
        return {"insights": [], "stored": 0}
    if cancel_check():
        return {"insights": [], "stored": 0, "cancelled": True}

    conversation_text = _format_conversation(messages)
    memories_text = _format_memories(recent_memories)

    if cancel_check():
        return {"insights": [], "stored": 0, "cancelled": True}

    insights = _extract_insights(conversation_text, memories_text)

    if cancel_check():
        return {"insights": insights, "stored": 0, "cancelled": True}

    stored = _store_insights(insights)

    return {"insights": insights, "stored": stored}


def _format_conversation(messages: list) -> str:
    """Format messages for review."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        if role == "user":
            lines.append(f"[{timestamp}] User: {content}")
        elif role == "assistant":
            lines.append(f"[{timestamp}] Assistant: {content}")
        elif role == "system":
            lines.append(f"[{timestamp}] [System: {content}]")
    return "\n".join(lines)


def _format_memories(memories: list) -> str:
    """Format recent memories for context."""
    if not memories:
        return "No recent memories."
    lines = []
    for mem in memories:
        content = mem.get("content", "")
        category = mem.get("category", "")
        lines.append(f"- [{category}] {content}")
    return "\n".join(lines)


def _extract_insights(conversation: str, memories: str) -> list[dict]:
    """Use LLM to extract insights from conversation review."""
    import maasv

    prompt = f"""You are reviewing a conversation between a user and their AI assistant.
Your job is to find insights that weren't explicitly stated - patterns, preferences, connections.

Recent memories already stored:
{memories}

Conversation to review:
{conversation}

Look for:
1. **Patterns**: Repeated behaviors, consistent preferences, habits
2. **Implicit preferences**: Things the user likes/dislikes without saying directly
3. **Connections**: Links between topics, people, or projects not explicitly stated
4. **Follow-ups**: Things that should be remembered for later

For each insight, provide:
- type: pattern, preference, connection, or follow_up
- insight: Clear description
- evidence: Quote or reference from conversation
- actionable: Whether the assistant should do something with this
- confidence: 0.0-1.0

Return as JSON array. Only include genuinely valuable insights, not obvious ones.
If nothing notable found, return empty array.

JSON output:"""

    try:
        config = maasv.get_config()
        llm = maasv.get_llm()

        text = llm.call(
            messages=[{"role": "user", "content": prompt}],
            model=config.review_model,
            max_tokens=2000,
            source="sleep-review",
        )

        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        insights = json.loads(text)
        logger.info(f"[Review] Extracted {len(insights)} insights")
        return insights

    except json.JSONDecodeError as e:
        logger.warning(f"[Review] Failed to parse response: {e}")
        return []
    except Exception as e:
        logger.error(f"[Review] API call failed: {e}")
        return []


def _store_insights(insights: list[dict]) -> int:
    """Store insights as memories."""
    if not insights:
        return 0

    from maasv.core.store import store_memory

    stored = 0
    for insight in insights:
        try:
            description = insight.get("insight", "")
            evidence = insight.get("evidence", "")
            confidence = insight.get("confidence", 0.5)
            insight_type = insight.get("type", "unknown")

            if not description or confidence < 0.6:
                continue

            category_map = {
                "pattern": "behavior",
                "preference": "preference",
                "connection": "relationship",
                "follow_up": "reminder"
            }
            category = category_map.get(insight_type, "insight")

            store_memory(
                content=description,
                category=category,
                source="sleep_review",
                confidence=confidence * 0.9,
                metadata={
                    "evidence": evidence,
                    "insight_type": insight_type,
                    "reviewed_at": datetime.now().isoformat()
                }
            )

            stored += 1
            logger.debug(f"[Review] Stored insight: {description[:50]}...")

        except Exception as e:
            logger.warning(f"[Review] Failed to store insight: {e}")

    return stored

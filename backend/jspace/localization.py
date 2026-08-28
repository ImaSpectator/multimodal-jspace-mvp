from __future__ import annotations

import re


_CONFLICT_VALUE_ZH = {
    "appears successful": "看起来已成功",
    "delivered": "已送达",
    "wifi visible": "Wi-Fi 可见",
    "partial access": "部分访问",
    "cancellation requested": "已请求取消",
    "new itinerary visible": "新行程已显示",
    "return completed": "退货已完成",
    "100% progress": "进度显示 100%",
    "connected": "已连接",
    "workspace visible": "工作区可见",
    "adjusted badge": "已显示调整标记",
    "confirmed": "已确认",
    "active": "有效",
    "upgrade visible": "升级信息可见",
    "cancelled": "已取消",
    "ticket visible": "票券可见",
    "upgrade complete": "升级已完成",
    "completed": "已完成",
}

_CONFLICT_EMOTION_ZH = {
    "uncertain": "不确定",
    "confused": "困惑",
    "anxious": "焦虑",
    "disappointed": "失望",
    "frustrated": "沮丧",
    "angry": "生气",
    "impatient": "不耐烦",
    "skeptical": "怀疑",
    "distressed": "难受",
    "embarrassed": "尴尬",
    "neutral": "中性",
    "calm": "平静",
    "curious": "好奇",
    "hopeful": "有希望",
    "appreciative": "感谢",
    "satisfied": "满意",
    "relieved": "安心",
}


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", str(text or "")))


def conflict_description_zh(description: str, description_zh: str | None = None) -> str:
    """Return a fully Chinese conflict explanation, including for stale sessions.

    Older Streamlit sessions may contain Conflict objects serialized before the
    `description_zh` field was added.  The UI therefore cannot rely only on the
    field being present.  This function reconstructs the Chinese explanation
    from the stable English conflict templates used by the engine and falls
    back to a Chinese-only generic explanation for any future/unknown template.
    """
    localized = str(description_zh or "").strip()
    if localized and _contains_chinese(localized):
        return localized

    raw = str(description or "").strip()
    lowered = raw.lower()

    if (
        "customer believes the issue is resolved" in lowered
        and "authoritative system" in lowered
        and "unresolved" in lowered
    ):
        return "客户认为问题已经解决，但权威系统仍显示问题尚未解决。"

    visible_match = re.search(
        r"customer-facing evidence suggests\s+['\"“‘]?(.+?)['\"”’]?,?\s+while the authoritative system remains unresolved\. ?$",
        raw,
        flags=re.IGNORECASE,
    )
    if visible_match:
        value = visible_match.group(1).strip()
        value_zh = _CONFLICT_VALUE_ZH.get(value.lower())
        if value_zh:
            return f"面向客户的证据显示“{value_zh}”，但权威系统仍显示问题尚未解决。"
        # Keep the fallback completely Chinese instead of leaking an unknown
        # English status into Chinese mode.
        return "面向客户的界面显示问题似乎已经解决，但权威系统仍显示问题尚未解决。"

    emotion_match = re.search(
        r"the customer says the issue is resolved, but their\s+(.+?)\s+affect remains strong\s*\((\d+)%\)\. ?$",
        raw,
        flags=re.IGNORECASE,
    )
    if emotion_match:
        emotion = emotion_match.group(1).strip().lower()
        percentage = emotion_match.group(2)
        emotion_zh = _CONFLICT_EMOTION_ZH.get(emotion, "负面")
        return f"客户表示问题已经解决，但其“{emotion_zh}”情绪仍然较强（{percentage}%）。"

    # Last-resort UI safety: Chinese mode should never display an English-only
    # conflict paragraph in the yellow conflict card.
    return "检测到客户侧信息与权威系统状态之间存在冲突；当前应以权威系统状态为准，并继续核实后再确认结果。"

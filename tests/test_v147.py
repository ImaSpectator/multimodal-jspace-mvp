from backend.jspace.localization import conflict_description_zh


def test_v147_stale_high_connected_conflict_is_fully_chinese():
    text = conflict_description_zh(
        "Customer-facing evidence suggests 'connected', while the authoritative system remains unresolved.",
        None,
    )
    assert text == "面向客户的证据显示“已连接”，但权威系统仍显示问题尚未解决。"
    assert "Customer" not in text
    assert "connected" not in text
    assert "authoritative" not in text


def test_v147_stale_high_adjusted_badge_conflict_is_fully_chinese():
    text = conflict_description_zh(
        "Customer-facing evidence suggests 'adjusted badge', while the authoritative system remains unresolved.",
        None,
    )
    assert text == "面向客户的证据显示“已显示调整标记”，但权威系统仍显示问题尚未解决。"
    assert "adjusted badge" not in text


def test_v147_stale_medium_emotion_conflict_is_fully_chinese():
    text = conflict_description_zh(
        "The customer says the issue is resolved, but their frustrated affect remains strong (80%).",
        None,
    )
    assert text == "客户表示问题已经解决，但其“沮丧”情绪仍然较强（80%）。"
    assert "frustrated" not in text


def test_v147_current_localized_description_is_preserved():
    text = conflict_description_zh(
        "Customer-facing evidence suggests 'connected', while the authoritative system remains unresolved.",
        "面向客户的证据显示“已连接”，但权威系统仍显示问题尚未解决。",
    )
    assert text == "面向客户的证据显示“已连接”，但权威系统仍显示问题尚未解决。"


def test_v147_unknown_conflict_never_leaks_english_in_chinese_mode():
    text = conflict_description_zh("Some future conflict template written only in English.", None)
    assert text == "检测到客户侧信息与权威系统状态之间存在冲突；当前应以权威系统状态为准，并继续核实后再确认结果。"
    assert "English" not in text
    assert "future" not in text

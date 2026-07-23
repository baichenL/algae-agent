# frontend/approval_history.py 负责渲染审批历史内容
from frontend.http import get_pending_list

STATUS_OPTIONS = {
    "全部": "all",
    "已批准": "approved",
    "已拒绝": "denied",
    "待审批": "pending",
}


def _load_approval_items(backend_url: str, status: str) -> list[dict]:
    response = get_pending_list(backend_url, timeout=10, status=status)
    if response.status_code != 200:
        raise RuntimeError(response.text)
    return response.json().get("pending", [])


def _format_table_rows(items: list[dict]) -> list[dict]:
    return [
        {
            "id": item.get("id"),
            "action_type": item.get("action_type"),
            "status": item.get("status"),
            "requester": item.get("requester"),
            "created_at": item.get("created_at"),
            "reviewed_at": item.get("reviewed_at"),
            "reviewed_by": item.get("reviewed_by"),
            "summary": item.get("summary"),
        }
        for item in items
    ]


def render_approval_history(st, backend_url: str):
    st.title("📋 审批历史")
    st.caption("查看 Pending Action 的审批生命周期记录")

    if st.button("返回实验控制台"):
        st.session_state["active_view"] = "chat"
        st.rerun()

    status_label = st.selectbox("状态筛选", list(STATUS_OPTIONS.keys()))
    status_value = STATUS_OPTIONS[status_label]

    try:
        items = _load_approval_items(backend_url, status_value)
    except Exception as exc:
        st.error(f"加载审批历史失败: {exc}")
        return

    if not items:
        st.info("当前筛选条件下没有审批记录。")
        return

    st.dataframe(_format_table_rows(items), use_container_width=True, hide_index=True)

    st.markdown("### 记录明细")
    for item in items:
        title = f"pending {item.get('id')} | {item.get('status')} | {item.get('summary')}"
        with st.expander(title):
            st.write(f"action_type: {item.get('action_type')}")
            st.write(f"requester: {item.get('requester')}")
            st.write(f"source: {item.get('source')}")
            st.write(f"risk_level: {item.get('risk_level')}")
            st.write(f"created_at: {item.get('created_at')}")
            st.write(f"reviewed_at: {item.get('reviewed_at')}")
            st.write(f"reviewed_by: {item.get('reviewed_by')}")
            if item.get("review_reason"):
                st.write(f"review_reason: {item.get('review_reason')}")
            st.markdown("**object**")
            st.json(item.get("object") or {})
            st.markdown("**payload**")
            st.json(item.get("payload") or {})

import os
import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = Path(__file__).resolve().parent
if str(FRONTEND_DIR) in sys.path:
    sys.path.remove(str(FRONTEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

from frontend.approval_history import render_approval_history
from frontend.render import render_chat_history, render_chat_input, render_pending_email_draft
from frontend.sidebar import render_sidebar
from frontend.simulation import render_simulation_panel
from frontend.scientific import render_scientific_workbench
from frontend.state import init_session_state


st.set_page_config(
    page_title="微藻智能体实验控制台",
    page_icon="🧫",
    layout="wide",
)
init_session_state(st)


def render_page_navigation() -> None:
    active_view = st.session_state.get("active_view", "chat")
    st.sidebar.markdown("## 页面导航")
    destinations = [
        ("chat", "💬 Agent 对话"),
        ("simulation", "🧪 仿真控制台"),
        ("approval_history", "📋 审批历史"),
        ("scientific", "🔬 Scientific Workbench"),
    ]
    for view, label in destinations:
        if st.sidebar.button(
            label,
            key=f"navigate_{view}",
            type="primary" if view == active_view else "secondary",
            use_container_width=True,
        ):
            if active_view != view:
                st.session_state["active_view"] = view
                st.rerun()
    st.sidebar.divider()


render_page_navigation()
active_view = st.session_state.get("active_view", "chat")

if active_view == "scientific":
    render_scientific_workbench(st, BACKEND_URL)
elif active_view == "simulation":
    st.title("🧪 自动传代设备仿真控制台")
    st.caption("LangGraph 工作流、设备状态与故障恢复的实时可视化")
    render_simulation_panel(st, BACKEND_URL, standalone=True)
elif active_view == "approval_history":
    render_approval_history(st, BACKEND_URL)
else:
    st.title("🧫 微藻智能体实验控制台")
    st.caption("基于 DeepSeek 与 LangGraph 的实验室 Agent 对话和审批中心")
    render_sidebar(st, BACKEND_URL)
    render_chat_history(st)
    render_pending_email_draft(st, BACKEND_URL)
    render_chat_input(st, BACKEND_URL)

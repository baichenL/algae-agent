# frontend/state.py 负责初始化和管理前端会话状态
def init_session_state(st):
    if "last_operation" not in st.session_state:
        st.session_state["last_operation"] = None
    if "fetched_strains" not in st.session_state:
        st.session_state["fetched_strains"] = []
    if "fetched_pending" not in st.session_state:
        st.session_state["fetched_pending"] = []
    # 3. 初始化网页的本地聊天记忆
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_email_draft" not in st.session_state:
        st.session_state["pending_email_draft"] = None
    if "email_send_status" not in st.session_state:
        st.session_state["email_send_status"] = None
    if "active_view" not in st.session_state:
        st.session_state["active_view"] = "chat"
    if "simulation_run_id" not in st.session_state:
        st.session_state["simulation_run_id"] = None
    if "simulation_error" not in st.session_state:
        st.session_state["simulation_error"] = None

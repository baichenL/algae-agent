# frontend/render.py 负责将后端返回的结果智能渲染成用户友好的界面元素
from frontend.http import send_email

def extract_email_error(response):
    try:
        payload = response.json()
    except Exception:
        return response.text

    detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
    if isinstance(detail, dict):
        return detail.get("message") or detail.get("detail") or str(detail)
    return str(detail)


def send_pending_email_draft(st, backend_url, draft):
    recipients = [
        item.strip()
        for item in st.session_state.get("email_draft_recipients", "").split(",")
        if item.strip()
    ]
    payload = {
        "subject": st.session_state.get("email_draft_subject", ""),
        "body": st.session_state.get("email_draft_body", ""),
        "recipients": recipients,
        "source": "frontend_confirm",
        "metadata": {
            **draft.get("metadata", {}),
            "template_type": draft.get("template_type", "manual_check"),
        },
    }
    try:
        send_resp = send_email(backend_url, payload, timeout=15)
        if send_resp.status_code == 200:
            result = send_resp.json()
            message = result.get("message") or "已发送"
            st.session_state["email_send_status"] = {"ok": True, "message": message}
            st.session_state["pending_email_draft"] = None
            st.session_state.messages.append({"role": "assistant", "content": f"邮件{message}。"})
            st.rerun()
        else:
            st.session_state["email_send_status"] = {
                "ok": False,
                "message": extract_email_error(send_resp),
            }
            st.rerun()
    except Exception as e:
        st.session_state["email_send_status"] = {
            "ok": False,
            "message": f"邮件发送接口调用失败: {str(e)}",
        }
        st.rerun()


def render_pending_email_draft(st, backend_url):
    status = st.session_state.get("email_send_status")
    if status:
        if status.get("ok"):
            st.success(status.get("message") or "已发送")
        else:
            st.error(f"邮件发送失败: {status.get('message')}")

    draft = st.session_state.get("pending_email_draft")
    if not draft:
        return

    st.markdown("### 邮件草稿")
    st.text_input("收件人", value=", ".join(draft.get("recipients", [])), key="email_draft_recipients")
    st.text_input("主题", value=draft.get("subject", ""), key="email_draft_subject")
    st.text_area("正文", value=draft.get("body", ""), key="email_draft_body", height=220)
    col_send, col_cancel = st.columns(2)
    with col_send:
        if st.button("确认发送邮件", key="confirm_email_send"):
            send_pending_email_draft(st, backend_url, draft)
    with col_cancel:
        if st.button("取消发送", key="cancel_email_send"):
            st.session_state["pending_email_draft"] = None
            st.session_state["email_send_status"] = None
            st.rerun()


from frontend.http import confirm_pending, get_pending_list, send_chat_message, submit_strain_tool


def sync_pending_cache(st, backend_url, timeout: int = 5) -> None:
    """同步前端 pending 缓存，避免聊天创建的审批单刷新后不可见。"""
    if not backend_url:
        return
    try:
        response = get_pending_list(backend_url, timeout=timeout)
        if response.status_code == 200:
            st.session_state["fetched_pending"] = response.json().get("pending", [])
    except Exception:
        pass


def render_chat_history(st):
    # 在网页上智能渲染历史聊天气泡
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "🤖" in msg["content"] and "【" in msg["content"]:
                parts = msg["content"].split("---")
                st.markdown(parts[0])
                if len(parts) > 1:
                    with st.expander("🔍 展开历史硬件流水线回执"):
                        st.code(parts[1].strip())
            else:
                st.markdown(msg["content"])



def render_user_message(st, user_input):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})


def render_agent_response(st, backend_url, result):
    agent_output = result.get("agent_output", {})
    natural_reply = result.get("natural_reply", "")
    action = agent_output.get("action")

    if action == "force_subculture":
        render_force_subculture(st, agent_output, natural_reply)
    elif action == "chat":
        render_chat(st, agent_output, natural_reply)
    elif action == "list_strains":
        render_list_strains(st, agent_output, natural_reply)
    elif action == "list_pending":
        render_list_pending(st, backend_url, agent_output, natural_reply)
    elif action == "email_draft":
        render_email_draft(st, agent_output, natural_reply)
    elif action == "email_sent":
        render_email_sent(st, agent_output, natural_reply)
    elif action == "email_send_failed":
        render_email_send_failed(st, agent_output, natural_reply)
    elif action == "rag_answer":
        render_rag_answer(st, agent_output, natural_reply)
    elif action == "composite_result":
        render_composite_result(st, agent_output, natural_reply)
    elif action == "workflow_subculture":
        render_workflow_subculture(st, backend_url, agent_output, natural_reply)
    else:
        render_generic_tool_response(st, backend_url, agent_output, natural_reply)


def render_force_subculture(st, agent_output, natural_reply):
    display_msg = natural_reply or agent_output.get("msg", "🤖 自动化流水线启动成功！")
    st.success(display_msg)

    st.markdown("### 📊 底层 LangGraph 实时硬件流水线日志")
    logs_str = ""
    for log in agent_output.get("execution_logs", []):
        st.code(log, language="bash")
        logs_str += f"{log}\n"

    st.session_state.messages.append({
        "role": "assistant",
        "content": f"{display_msg}\n\n---\n{logs_str}"
    })


def render_chat(st, agent_output, natural_reply):
    display_content = natural_reply or agent_output.get("content", "未能获取到有效答复。")
    st.markdown(display_content)
    st.session_state.messages.append({"role": "assistant", "content": display_content})


def render_rag_answer(st, agent_output, natural_reply):
    if agent_output.get("blocked"):
        st.warning(natural_reply or "该请求超出 RAG 知识层边界。")
        st.session_state.messages.append({"role": "assistant", "content": natural_reply})
        return

    answer = agent_output.get("answer") or {}
    citations = agent_output.get("citations") or []
    conclusion = answer.get("conclusion") or natural_reply or "根据当前知识库，未生成可用回答。"

    st.markdown("**结论**")
    st.markdown(conclusion)

    evidence = answer.get("evidence") or []
    st.markdown("**依据**")
    if evidence:
        for item in evidence:
            st.markdown(
                f"- 来源 {item.get('source_id')}: "
                f"{item.get('file_name')} | {item.get('doc_type')} | "
                f"{item.get('location')} | {item.get('quote_summary')}"
            )
    else:
        st.markdown("- 当前没有可引用依据。")

    uncertainty = answer.get("uncertainty") or []
    st.markdown("**不确定性**")
    if uncertainty:
        for item in uncertainty:
            st.markdown(f"- {item}")
    else:
        st.markdown("- 当前知识库未给出额外限制。")

    if citations:
        st.markdown("**引用来源**")
        for item in citations:
            location = _format_rag_location(item)
            st.markdown(
                f"- 来源 {item.get('source_id')}: `{item.get('file_name')}` "
                f"({item.get('doc_type')}, {location})"
            )

    display_content = natural_reply or _build_rag_history_text(answer, citations)
    st.session_state.messages.append({"role": "assistant", "content": display_content})


def _format_rag_location(citation):
    if citation.get("page_number"):
        return f"page {citation.get('page_number')}"
    if citation.get("sheet_name"):
        if citation.get("row_start") and citation.get("row_end"):
            return f"sheet {citation.get('sheet_name')}, rows {citation.get('row_start')}-{citation.get('row_end')}"
        return f"sheet {citation.get('sheet_name')}"
    return citation.get("section") or citation.get("chunk_id") or "chunk"


def _build_rag_history_text(answer, citations):
    lines = [
        "结论：",
        answer.get("conclusion") or "",
        "",
        "依据：",
    ]
    for item in answer.get("evidence") or []:
        lines.append(
            f"- 来源 {item.get('source_id')}: {item.get('file_name')} | "
            f"{item.get('doc_type')} | {item.get('location')} | {item.get('quote_summary')}"
        )
    lines.extend(["", "不确定性："])
    for item in answer.get("uncertainty") or []:
        lines.append(f"- {item}")
    if citations:
        lines.extend(["", "引用来源："])
        for item in citations:
            lines.append(f"- 来源 {item.get('source_id')}: {item.get('file_name')} ({_format_rag_location(item)})")
    return "\n".join(lines)


def render_list_strains(st, agent_output, natural_reply):
    strains = agent_output.get("strains", [])
    if strains:
        st.markdown("**当前数据库中的品系：**")
        lines = []
        for s in strains:
            line = f"- {s.get('strain_id')} | {s.get('name_cn')} ({s.get('name_en')}) 代数={s.get('generation_number')} 天数={s.get('days_since_last_subculture')}"
            st.markdown(line)
            lines.append(line)
        # 记录到对话历史
        st.session_state.messages.append({"role": "assistant", "content": natural_reply + "\n" + "\n".join(lines)})
    else:
        st.markdown(natural_reply or "未找到任何品系")
        st.session_state.messages.append({"role": "assistant", "content": natural_reply})


def render_list_pending(st, backend_url, agent_output, natural_reply):
    pend = agent_output.get("pending", [])
    st.session_state["fetched_pending"] = pend
    if pend:
        st.markdown("**当前待确认请求：**")
        lines = []
        for it in pend:
            pending_id = it.get("id")
            summary = it.get('summary') or f"pending {it.get('id')}"
            st.markdown(f"- {summary}")
            col_confirm, col_deny = st.columns(2)
            with col_confirm:
                if st.button(f"确认 pending {pending_id}", key=f"chat_confirm_pending_{pending_id}"):
                    try:
                        resp = confirm_pending(backend_url, pending_id, True, timeout=10)
                        if resp.status_code in {200, 202}:
                            sync_pending_cache(st, backend_url)
                            st.success(f"已确认 pending {pending_id}。")
                            st.rerun()
                        else:
                            st.error(f"确认失败: {resp.text}")
                    except Exception as e:
                        st.error(f"调用确认接口失败: {str(e)}")
            with col_deny:
                if st.button(f"拒绝 pending {pending_id}", key=f"chat_deny_pending_{pending_id}"):
                    try:
                        resp = confirm_pending(backend_url, pending_id, False, timeout=10)
                        if resp.status_code == 200:
                            sync_pending_cache(st, backend_url)
                            st.info(f"已拒绝 pending {pending_id}。")
                            st.rerun()
                        else:
                            st.error(f"拒绝失败: {resp.text}")
                    except Exception as e:
                        st.error(f"调用拒绝接口失败: {str(e)}")
            lines.append(summary)
        st.session_state.messages.append({"role": "assistant", "content": natural_reply + "\n" + "\n".join(lines)})
    else:
        st.markdown(natural_reply or "暂无待确认请求")
        st.session_state.messages.append({"role": "assistant", "content": natural_reply})


def render_email_draft(st, agent_output, natural_reply):
    draft = agent_output.get("draft", {})
    display_content = natural_reply or "已生成邮件草稿，请预览后确认发送。"
    st.session_state["pending_email_draft"] = draft
    st.session_state["email_send_status"] = None
    st.markdown(display_content)
    st.session_state.messages.append({"role": "assistant", "content": display_content})
    st.rerun()


def render_email_sent(st, agent_output, natural_reply):
    st.success(natural_reply or "邮件已发送。")
    st.session_state.messages.append({"role": "assistant", "content": natural_reply or "邮件已发送。"})


def render_email_send_failed(st, agent_output, natural_reply):
    error_message = natural_reply or agent_output.get("message") or "邮件发送失败。"
    st.error(error_message)
    st.session_state["email_send_status"] = {"ok": False, "message": error_message}
    st.session_state.messages.append({"role": "assistant", "content": f"邮件发送失败: {error_message}"})


def render_composite_result(st, agent_output, natural_reply):
    """Render an ordered multi-intent result without hiding child actions."""

    steps = agent_output.get("steps") or []
    st.markdown(natural_reply or "已完成组合请求。")
    for index, step in enumerate(steps, start=1):
        child = step.get("agent_output") or {}
        action = child.get("action") or step.get("action") or "unknown"
        with st.expander(f"步骤 {index}: {action}"):
            st.markdown(step.get("natural_reply") or "已处理。")
            if child.get("pending_id"):
                st.code(f"pending_id={child.get('pending_id')}")
                sync_pending_cache(st, st.session_state.get("backend_url"))
        if action == "email_draft" and child.get("draft"):
            st.session_state["pending_email_draft"] = child["draft"]
            st.session_state["email_send_status"] = None
    st.session_state.messages.append({
        "role": "assistant",
        "content": natural_reply or "已完成组合请求。",
    })


def render_generic_tool_response(st, backend_url, agent_output, natural_reply):
    # 处理一般工具回复（包含待确认或需补充信息的请求）
    display_content = natural_reply or agent_output.get("content", "操作完成")
    # 如果工具要求补充信息，展示补充字段表单并提交到后端
    if agent_output.get("require_more_info"):
        render_require_more_info(st, backend_url, agent_output, display_content)
    # 如果工具要求确认（例如新增/更新/删除），展示确认对话
    elif agent_output.get("require_confirmation"):
        render_require_confirmation(st, backend_url, agent_output, display_content)
    else:
        st.markdown(display_content)
        st.session_state.messages.append({"role": "assistant", "content": display_content})


def render_require_more_info(st, backend_url, agent_output, display_content):
    missing = agent_output.get("missing_fields") or agent_output.get("response_payload", {}).get("missing_fields") or []
    questions = agent_output.get("questions") or agent_output.get("response_payload", {}).get("questions") or []
    st.info("需要补充以下信息以继续：")
    with st.form(key="fill_missing_fields"):
        filled = {}
        for field in missing:
            filled[field] = st.text_input(field)
        submit_more = st.form_submit_button("提交信息并继续")
    if submit_more and any(v for v in filled.values()):
        tool_name = agent_output.get("response_payload", {}).get("action") or agent_output.get("action") or "add_algae_strain"
        try:
            resp = submit_strain_tool(backend_url, tool_name, filled, timeout=10)
            if resp.status_code in {200, 202}:
                result = resp.json()
                st.success("已提交补充信息。")
                st.json(result)
            else:
                st.error(f"提交失败: {resp.text}")
        except Exception as e:
            st.error(f"提交补充信息失败: {str(e)}")
    st.session_state.messages.append({"role": "assistant", "content": display_content})


def render_require_confirmation(st, backend_url, agent_output, display_content):
    sid = agent_output.get("response_payload", {}).get("confirmation_target") or agent_output.get("response_payload", {}).get("strain_id") or "该品系"
    pending_id = agent_output.get("pending_id") or agent_output.get("response_payload", {}).get("pending_id")
    if pending_id:
        sync_pending_cache(st, backend_url)
    prompt = agent_output.get("confirmation_prompt") or agent_output.get("response_payload", {}).get("confirmation_prompt") or f"确认要执行该操作吗？"
    if not pending_id:
        st.error("该操作需要审批，但后端没有返回 pending_id。请刷新 pending 列表核对后再操作。")
        return

    with st.form(key=f"confirm_form_{pending_id}"):
        st.warning(f"已创建审批单 pending {pending_id}。是否批准执行？\n\n{prompt}")
        confirm = st.form_submit_button("批准并执行")
        deny = st.form_submit_button("拒绝此审批单")
    if confirm:
        # 调用确认接口批准
        try:
            resp = confirm_pending(backend_url, pending_id, True, timeout=10)
            if resp.status_code in {200, 202}:
                result = resp.json()
                sync_pending_cache(st, backend_url)
                message = result.get("msg") or "审批已通过；请根据 workflow run 状态确认是否执行完成。"
                st.session_state.messages.append({"role": "assistant", "content": message})
                st.success(message)
                st.rerun()
            else:
                st.error(f"批准失败: {resp.text}")
        except Exception as e:
            st.error(f"调用确认接口失败: {str(e)}")
    elif deny:
        try:
            resp = confirm_pending(backend_url, pending_id, False, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                sync_pending_cache(st, backend_url)
                message = result.get("msg") or "已拒绝该请求，未执行任何数据修改。"
                st.session_state.messages.append({"role": "assistant", "content": message})
                st.info(message)
                st.rerun()
            else:
                st.error(f"拒绝失败: {resp.text}")
        except Exception as e:
            st.error(f"调用拒绝接口失败: {str(e)}")
    rendered_key = f"rendered_confirmation_message_{pending_id}"
    if not st.session_state.get(rendered_key):
        st.session_state.messages.append({"role": "assistant", "content": display_content})
        st.session_state[rendered_key] = True


def render_backend_error(st, backend_url, response):
    # 专门识别未找到品系的错误，提供即时添加并确认的交互
    err_text = response.text
    if "数据库中未找到该藻种记录" in err_text or "未找到该藻种" in err_text:
        render_quick_add_strain(st, backend_url, response)
    else:
        st.error(f"❌ 后端中枢报错: {response.text}")


def render_quick_add_strain(st, backend_url, response):
    st.error("❌ 后端中枢报错: 数据库中未找到该藻种记录")
    st.info("检测到目标品系未录入。您可以在此处直接添加并当场确认，确认后将立即写入数据库。")
    with st.form(key="quick_add_strain"):
        qa_strain_id = st.text_input("strain_id（例如 Spirulina_01）")
        qa_name_cn = st.text_input("中文名（name_cn）", value="螺旋藻")
        qa_name_en = st.text_input("英文名（name_en）", value="Spirulina platensis")
        qa_generation = st.number_input("generation_number", min_value=0, value=1)
        qa_days = st.number_input("days_since_last_subculture", min_value=0, value=0)
        qa_submit = st.form_submit_button("添加并确认")
    if qa_submit:
        try:
            # 提交工具创建 pending
            resp = submit_strain_tool(backend_url, "add_algae_strain", {"strain_id": qa_strain_id, "name_cn": qa_name_cn, "name_en": qa_name_en, "generation_number": int(qa_generation), "days_since_last_subculture": int(qa_days)}, timeout=10)
            if resp.status_code == 200:
                resj = resp.json()
                pending_id = resj.get("response_payload", {}).get("pending_id") or resj.get("pending_id")
                # 立即批准（用户的确认即为审批通过）
                conf = confirm_pending(backend_url, pending_id, True, timeout=10)
                if conf.status_code in {200, 202}:
                    st.success(f"已添加并批准品系 {qa_strain_id}，现在可以继续传代请求。")
                else:
                    st.error(f"添加已提交但批准失败: {conf.text}")
            else:
                st.error(f"提交添加请求失败: {resp.text}")
        except Exception as e:
            st.error(f"操作失败: {str(e)}")


def render_connection_error(st, error):
    st.error(f"📡 无法建立连接！请确认后台 Docker 容器是否开启。错误详情: {str(error)}")


def render_chat_input(st, backend_url):
    # 捕捉用户在网页底部的输入框输入 st：Streamlit 模块对象;backend_url：后端 FastAPI 地址
    if user_input := st.chat_input("请输入微藻实验指令（如：'立刻帮我执行传代' 或 '设计一组小球藻氮限制配方'）"):
        render_user_message(st, user_input)

        with st.chat_message("assistant"):
            with st.spinner("智能体大脑正在研判意图并调度系统..."):
                try:
                    response = send_chat_message(
                        backend_url,
                        {"message": user_input, "session_id": "streamlit_user_01"},
                        timeout=30
                    )

                    if response.status_code == 200:
                        render_agent_response(st, backend_url, response.json())
                    else:
                        render_backend_error(st, backend_url, response)

                except Exception as e:
                    render_connection_error(st, e)


def render_workflow_subculture(st, backend_url, agent_output, natural_reply):
    # 展示协议闭环证据：协议摘要、验证结果、模拟预览和待审批入口。
    pending_id = agent_output.get("pending_id")
    protocol = agent_output.get("protocol_summary") or {}
    validation = agent_output.get("validation_summary") or {}
    simulation = agent_output.get("simulation_summary") or {}
    run_preview = agent_output.get("run_preview") or []

    sync_pending_cache(st, backend_url)
    st.success(natural_reply or f"已创建传代 Workflow 待审批请求 pending {pending_id}。")
    if protocol:
        st.markdown("**协议摘要**")
        st.json({
            "protocol_id": protocol.get("protocol_id"),
            "protocol_hash": protocol.get("protocol_hash"),
            "template_id": protocol.get("template_id"),
            "target_strain_id": protocol.get("target_strain_id"),
            "expected_generation": protocol.get("expected_generation"),
        })
    if validation:
        st.markdown("**确定性验证**")
        if validation.get("valid"):
            st.success(
                f"验证通过：{validation.get('passed_count', 0)}/"
                f"{validation.get('check_count', 0)} checks passed"
            )
        else:
            st.error(f"验证失败：{validation.get('error_codes') or []}")
    if simulation:
        st.markdown("**模拟执行**")
        if simulation.get("success"):
            st.success(
                f"模拟通过：{simulation.get('step_count', 0)} steps, "
                f"simulation_id={simulation.get('simulation_id')}"
            )
        else:
            st.error(f"模拟失败：{simulation.get('failure_code')}")
    if run_preview:
        st.markdown("**Run Preview**")
        st.dataframe(run_preview, hide_index=True)
    if pending_id:
        st.info(f"pending {pending_id} 正在等待人工审批。")
    st.session_state.messages.append({
        "role": "assistant",
        "content": natural_reply or f"workflow_subculture pending {pending_id}",
    })

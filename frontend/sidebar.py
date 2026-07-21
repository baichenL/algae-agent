# frontend/sidebar.py 负责渲染侧边栏内容
from frontend.http import (
    confirm_pending,
    get_last_operation,
    get_pending_list,
    get_strain_list,
    submit_strain_tool,
)


def refresh_pending_cache(st, backend_url, timeout: int = 5) -> bool:
    try:
        pending_response = get_pending_list(backend_url, timeout=timeout)
        if pending_response.status_code == 200:
            st.session_state["fetched_pending"] = pending_response.json().get("pending", [])
            return True
        return False
    except Exception:
        return False


def refresh_strain_cache(st, backend_url, timeout: int = 5) -> bool:
    try:
        strain_response = get_strain_list(backend_url, timeout=timeout)
        if strain_response.status_code == 200:
            st.session_state["fetched_strains"] = strain_response.json().get("strains", [])
            return True
        return False
    except Exception:
        return False


def render_sidebar(st, backend_url):
    # 2. 侧边栏：微藻实时状态（显示全局最后数据库操作时间）
    st.sidebar.header("🧫 微藻实时状态")

    if st.session_state.get("last_confirm_result"):
        result = st.session_state["last_confirm_result"]
        st.sidebar.success(result.get("msg") or "审批已处理。")
        workflow_result = result.get("workflow_result")
        if workflow_result:
            st.sidebar.write(
                f"workflow={workflow_result.get('status')} | "
                f"generation={workflow_result.get('current_generation')} | "
                f"days_since_last_subculture={workflow_result.get('days_since_last_subculture')}"
            )
        elif result.get("action_type") or result.get("strain_id"):
            st.sidebar.write(
                f"{result.get('action_type') or ''} {result.get('strain_id') or ''} | "
                f"executed={result.get('executed')}"
            )

    # 定义一个获取全局最后数据库操作时间的函数（从后端审计表读取）
    def fetch_algae_status():
        try:
            r = get_last_operation(backend_url, timeout=3)
            if r.status_code == 200:
                st.session_state["last_operation"] = r.json().get("last_operation")
                return True
        except Exception:
            pass
        return False


    # 显示全局最后操作（仅显示时间及简短摘要）
    if st.session_state.get("last_operation"):
        lo = st.session_state.get("last_operation") or {}
        performed_at = lo.get("performed_at")
        st.sidebar.markdown("**最后数据库操作时间**")
        if performed_at:
            st.sidebar.text(f"{performed_at[:19]}")
        else:
            st.sidebar.text("未知")
        # 简短摘要：动作与对应的 strain
        action = lo.get("action") or lo.get("action")
        sid = lo.get("strain_id") or lo.get("strain_id")
        if action or sid:
            st.sidebar.write(f"{action or ''} {sid or ''}")
    else:
        st.sidebar.info("点击按钮获取最后的数据库操作时间")

    # 管理品系：在前端直接发起增/改/删请求并即时显示审批按钮
    st.sidebar.header("🔧 管理品系")
    with st.sidebar.expander("新增 / 更新 / 删除 品系", expanded=False):
        st.subheader("添加/更新品系")
        with st.form(key="add_update_form"):
            form_tool = st.selectbox("操作", ["add_algae_strain", "update_algae_strain"])
            f_strain_id = st.text_input("strain_id")
            f_name_cn = st.text_input("中文名 (name_cn)")
            f_name_en = st.text_input("英文名 (name_en)")
            f_generation = st.number_input("generation_number", min_value=0, value=1)
            f_days = st.number_input("days_since_last_subculture", min_value=0, value=0)
            submit_tool = st.form_submit_button("提交")
        if submit_tool:
            args = {"strain_id": f_strain_id, "name_cn": f_name_cn, "name_en": f_name_en, "generation_number": int(f_generation), "days_since_last_subculture": int(f_days)}
            try:
                resp = submit_strain_tool(backend_url, form_tool, args, timeout=10)
                if resp.status_code == 200:
                    result = resp.json()
                    st.success("已提交。")
                    st.json(result)
                    # 如果需要确认，立刻在当前位置显示确认按钮，点击即为审批通过
                    if result.get("response_payload", {}).get("require_confirmation") or result.get("require_confirmation"):
                        pending_id = result.get("response_payload", {}).get("pending_id") or result.get("pending_id")
                        prompt = result.get("response_payload", {}).get("confirmation_prompt") or result.get("confirmation_prompt") or "请确认该操作："
                        st.warning(prompt)
                        if st.button(f"确认并执行（pending {pending_id}）", key=f"confirm_add_{pending_id}"):
                            try:
                                r = confirm_pending(backend_url, pending_id, True, timeout=10)
                                if r.status_code in {200, 202}:
                                    result_payload = r.json()
                                    st.session_state["last_confirm_result"] = result_payload
                                    st.success(result_payload.get("msg") or "审批已通过；请检查执行状态。")
                                    st.json(result_payload)
                                else:
                                    st.error(f"批准失败: {r.text}")
                            except Exception as e:
                                st.error(f"批准接口调用失败: {str(e)}")
                        if st.button(f"取消（拒绝 pending {pending_id}）", key=f"deny_add_{pending_id}"):
                            try:
                                r = confirm_pending(backend_url, pending_id, False, timeout=10)
                                if r.status_code == 200:
                                    st.info("已拒绝该请求。")
                                else:
                                    st.error(f"拒绝失败: {r.text}")
                            except Exception as e:
                                st.error(f"调用拒绝接口失败: {str(e)}")
                else:
                    st.error(f"提交失败: {resp.text}")
            except Exception as e:
                st.error(f"提交工具失败: {str(e)}")

        st.markdown("---")
        st.subheader("删除品系")
        with st.form(key="delete_form"):
            d_strain_id = st.text_input("要删除的 strain_id", key="del_strain")
            del_submit = st.form_submit_button("提交删除请求")
        if del_submit:
            try:
                resp = submit_strain_tool(backend_url, "delete_algae_strain", {"strain_id": d_strain_id}, timeout=10)
                if resp.status_code == 200:
                    result = resp.json()
                    st.success("删除请求已创建。")
                    st.json(result)
                    if result.get("response_payload", {}).get("require_confirmation") or result.get("require_confirmation"):
                        pending_id = result.get("response_payload", {}).get("pending_id") or result.get("pending_id")
                        prompt = result.get("response_payload", {}).get("confirmation_prompt") or result.get("confirmation_prompt") or f"确认要删除 {d_strain_id} 吗？"
                        st.warning(f"已创建审批单 pending {pending_id}。是否批准执行？\n\n{prompt}")
                        if st.button(f"批准并执行删除（pending {pending_id}）", key=f"confirm_del_{pending_id}"):
                            try:
                                r = confirm_pending(backend_url, pending_id, True, timeout=10)
                                if r.status_code in {200, 202}:
                                    result_payload = r.json()
                                    st.session_state["last_confirm_result"] = result_payload
                                    st.success(result_payload.get("msg") or "审批已通过；请检查执行结果。")
                                    st.json(result_payload)
                                    st.rerun()
                                else:
                                    st.error(f"批准失败: {r.text}")
                            except Exception as e:
                                st.error(f"批准接口调用失败: {str(e)}")
                        if st.button(f"拒绝此审批单（pending {pending_id}）", key=f"deny_del_{pending_id}"):
                            try:
                                r = confirm_pending(backend_url, pending_id, False, timeout=10)
                                if r.status_code == 200:
                                    result_payload = r.json()
                                    st.info(result_payload.get("msg") or "已拒绝该删除请求，未执行任何数据修改。")
                                    st.rerun()
                                else:
                                    st.error(f"拒绝失败: {r.text}")
                            except Exception as e:
                                st.error(f"调用拒绝接口失败: {str(e)}")
                else:
                    st.error(f"删除请求失败: {resp.text}")
            except Exception as e:
                st.error(f"提交删除请求失败: {str(e)}")

        st.markdown("---")
        # 新增：显示所有品系与待确认请求，支持本地乐观确认（改为将结果保存到 session_state，按钮持续存在）
        # 初次加载时自动从后端获取，避免刷新页面后 UI 丢失已知数据
        if not st.session_state["fetched_strains"]:
            refresh_strain_cache(st, backend_url, timeout=5)
        refresh_pending_cache(st, backend_url, timeout=5)
        if not st.session_state["fetched_strains"]:
            try:
                # 同步获取最后 DB 操作时间（侧边栏显示）
                try:
                    last = get_last_operation(backend_url, timeout=3)
                    if last.status_code == 200:
                        st.session_state["last_operation"] = last.json().get("last_operation")
                except Exception:
                    pass
            except Exception:
                pass

        if st.button("刷新并显示所有品系和 pending"):
            try:
                if not refresh_strain_cache(st, backend_url, timeout=5):
                    st.error("获取品系失败")
                if not refresh_pending_cache(st, backend_url, timeout=5):
                    st.error("获取 pending 失败")
            except Exception as e:
                st.error(f"刷新失败: {str(e)}")

        # 总是渲染缓存的品系与 pending（保证确认按钮在页面重载后仍存在）
        strains = st.session_state.get("fetched_strains", [])
        pendings = st.session_state.get("fetched_pending", [])

        if strains:
            st.subheader("当前品系列表")
            for s in strains:
                st.write(f"- {s.get('strain_id')} | {s.get('name_cn')} ({s.get('name_en')}) 代数={s.get('generation_number')} 天数={s.get('days_since_last_subculture')}")
        else:
            st.info("未加载品系列表，点击刷新以获取")

        st.subheader("待确认列表")
        if not pendings:
            st.write("暂无待确认请求")
        for it in list(pendings):
            pid = it.get('id')
            st.write(f"pending {pid}: {it.get('summary')}")
            col_confirm, col_deny = st.columns(2)
            with col_confirm:
                if st.button(f"批准并执行 pending {pid}", key=f"local_confirm_{pid}"):
                    st.success(f"正在批准并执行 pending {pid}...")
                    try:
                        rr = confirm_pending(backend_url, pid, True, timeout=5)
                        if rr.status_code in {200, 202}:
                            st.session_state["last_confirm_result"] = rr.json()
                            st.session_state["fetched_pending"] = [x for x in st.session_state["fetched_pending"] if x.get('id') != pid]
                            try:
                                rr2 = get_strain_list(backend_url, timeout=5)
                                if rr2.status_code == 200:
                                    st.session_state["fetched_strains"] = rr2.json().get("strains", [])
                            except Exception:
                                pass
                            st.rerun()
                        else:
                            st.error(f"后端批准失败: {rr.text}")
                    except Exception as e:
                        st.error(f"调用后端确认失败: {str(e)}")
            with col_deny:
                if st.button(f"拒绝 pending {pid}", key=f"local_deny_{pid}"):
                    try:
                        rr = confirm_pending(backend_url, pid, False, timeout=5)
                        if rr.status_code == 200:
                            st.session_state["fetched_pending"] = [x for x in st.session_state["fetched_pending"] if x.get('id') != pid]
                            st.rerun()
                        else:
                            st.error(f"后端拒绝失败: {rr.text}")
                    except Exception as e:
                        st.error(f"调用后端拒绝失败: {str(e)}")

    # 之前的独立待确认侧栏已移除；确认现在在提交后就地完成（点击确认即为批准）
    # 按钮触发状态查询（非阻塞）
    if st.sidebar.button("🔄 刷新状态"):
        with st.sidebar:
            with st.spinner("获取状态..."):
                if fetch_algae_status():
                    st.success("状态已更新")
                    st.rerun()  # 刷新页面显示新状态
                else:
                    st.error("获取状态失败")

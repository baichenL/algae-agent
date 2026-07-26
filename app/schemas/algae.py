# schemas/algae.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class ChatRequest(BaseModel):
    """客户端请求数据约束"""
    message: str = Field(..., description="实验员输入的指令或数据描述", example="帮我设计一组微藻培养基氮源优化方案")
    session_id: str = Field("default_session", description="会话ID，用于做多用户或多实验之间的记忆隔离")

class ChatResponse(BaseModel):
    """服务器统一返回数据约束"""
    status: str = Field("success", description="状态码")
    session_id: str = Field(..., description="当前会话ID")
    agent_output: Dict[str, Any] = Field(..., description="大模型返回的结构化 JSON 实验方案")
    natural_reply: str = Field(..., description="智能体最终对实验员说的自然语言交互文本")

    agent_status: Optional[str] = None
    trace_id: Optional[str] = None
    pending_id: Optional[int] = None
    state_observation_refs: List[str] = Field(default_factory=list)


class SubcultureRequest(BaseModel):
    """请求传代时前端需要提供的最核心元数据"""
    days_since_last_subculture: int = Field(..., description="距离上次传代过去的天数", example=6)
    generation_number: int = Field(..., description="当前微藻代数", example=14)
    source_reactor_id: str = Field(..., description="当前老藻液所在的瓶号/反应器ID", example="Reactor_A")
    strain_id: str = Field("Chlorella_01", description="藻种品系名称", example="Chlorella_01")

class SubcultureResponse(BaseModel):
    """工作流完毕后，返回给前端的最终工业级回执"""
    status: str = Field("success", description="状态码")
    workflow_status: str = Field(..., description="工作流最终状态 (SUCCESS 或 FAILED)")
    current_generation: int = Field(..., description="完成传代后的最新代数（若成功则已自动+1）")
    days_counter: int = Field(..., description="最新的周期计数器（若成功则已自动清零）")
    inoculation_time: str = Field(..., description="本次接种的真实物理时间戳")
    execution_logs: List[str] = Field(..., description="各 SOP 节点顺次累加的硬件操作回执链")

class ManualSubcultureRequest(BaseModel):
    strain_id: str = Field("Chlorella_01", description="藻种名称", example="Chlorella_01")

class StatusQueryRequest(BaseModel):
    strain_id: str = Field("Chlorella_01", description="藻种名称")


class AddStrainRequest(BaseModel):
    strain_id: str = Field(..., description="品系编号，例如 Chlamy_01 或 CC-125")
    name_cn: str = Field(..., description="中文名，例如 莱茵衣藻")
    name_en: str = Field(..., description="拉丁学名，例如 Chlamydomonas reinhardtii")
    generation_number: int = Field(1, description="初始代数，默认 1")
    days_since_last_subculture: int = Field(0, description="距上次传代天数，默认 0")


class ConfirmPendingRequest(BaseModel):
    """前端请求确认待办审批的结构化数据"""
    pending_id: int
    approve: bool = Field(..., description="True 表示批准，False 表示拒绝")

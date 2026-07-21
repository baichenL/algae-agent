# app/api/chat.py 聊天接口入口
# API 层本身不处理 Agent 逻辑，只负责 HTTP 接入、异常转换和统一响应
# 接收ChatRequest，把请求转交给Service层，调用 chat_service.handle_chat(),返回ChatResponse
from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.algae import ChatRequest, ChatResponse
from app.services.chat.chat_service import handle_chat
from app.core.security import ApiPrincipal, require_scientist

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(
    payload: ChatRequest,
    _: ApiPrincipal = Depends(require_scientist),
) -> ChatResponse:
    try:
        return await handle_chat(payload)
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"服务器内部异常: {str(e)}"
        )

from typing import Callable
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:

    tool_call = getattr(request, "tool_call", None)

    logger.info(f"[tool monitor]原始 tool_call: {tool_call}")

    # 兼容 dict / object 两种情况
    if isinstance(tool_call, dict):
        tool_name = tool_call.get("name") or tool_call.get("tool") or tool_call.get("tool_name") or "unknown_tool"
        tool_args = tool_call.get("args", {})
    else:
        tool_name = getattr(tool_call, "name", None) or getattr(tool_call, "tool", None) or getattr(tool_call, "tool_name", None) or "unknown_tool"
        tool_args = getattr(tool_call, "args", {})

    logger.info(f"[tool monitor]执行工具：{tool_name}")
    logger.info(f"[tool monitor]传入参数：{tool_args}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具 {tool_name} 调用成功")

        if tool_name == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"[tool monitor]工具 {tool_name} 调用失败，原因：{str(e)}", exc_info=True)
        raise


@before_model
def log_before_model(
        state: AgentState,
        runtime: Runtime,
):
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return load_report_prompts()
    return load_system_prompts()
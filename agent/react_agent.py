from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (
    rag_summarize,
    get_target_service,
    get_time_range,
    fetch_alert_data,
    fetch_metric_data,
    fetch_log_summary,
    fetch_service_topology,
    fetch_report_data,
    fill_context_for_report,
)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
import time


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[
                rag_summarize,
                get_target_service,
                get_time_range,
                fetch_alert_data,
                fetch_metric_data,
                fetch_log_summary,
                fetch_service_topology,
                fetch_report_data,
                fill_context_for_report,
            ],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def _extract_text(self, result) -> str:
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last = messages[-1]
                content = getattr(last, "content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("text"):
                            texts.append(block["text"])
                    return "".join(texts)
        return str(result)

    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # 非流式调用，先拿完整结果
        result = self.agent.invoke(
            input_dict,
            context={"report": False}
        )

        final_text = self._extract_text(result)

        # 伪流式：按固定长度切块输出
        chunk_size = 20
        for i in range(0, len(final_text), chunk_size):
            yield final_text[i:i + chunk_size]
            time.sleep(0.02)


if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("分析一下 order-service 最近1小时的异常情况"):
        print(chunk, end="", flush=True)
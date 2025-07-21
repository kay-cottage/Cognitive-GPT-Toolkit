# -*- coding: utf-8 -*-
"""
Created on May.12 2025
@author: gw.kayak


类O系列GPT v1: 工具调用版

核心特性 (v1更新):
- 新增 'call_tool' 动作: 引入了一个专用的工具调用动作，使AI的意图更清晰，也更便于管理和扩展工具集。
- 注册工具集: 在 `shared_state` 中引入了 'tools' 字典，用于注册可供AI调用的Python函数。


"""
import json
import copy
import traceback
import os
import requests
import io
import contextlib
import sys
from openai import OpenAI

# ----------------- 初始化OpenAI客户端 -----------------
client = OpenAI(api_key='')



# ----------------- 全局配置与提示词 -----------------
CONFIG = {
    "default_model": "gpt-4o",
    "subtask_model": "gpt-4o-mini",
    "max_depth": 3,
    "max_rounds_per_run": 10,
}

PROMPTS = {
    "MAIN_SYSTEM_PROMPT": """
你是'协调器GPT'，一个拥有高级自主规划、执行和反思能力的AI代理。你的最终目标是完成 <OVERARCHING_GOAL>。
你必须始终以一个合法的、严格遵循规范的JSON对象进行响应。

**JSON响应结构:**
{
  "thought": "(string) 你的详细思考过程、对过往步骤的分析、以及下一步的计划。",
  "action": "(string) 你决定执行的动作。必须是以下之一: "call_tool", "execute_code", "generate_handler", "invoke_gpt", "summarize_memory", "final_answer"。",
  "params": "(object) 执行该动作所需的参数。"
}

**可用动作 (action) 和对应的参数 (params):**

1.  `"action": "call_tool"`: 调用一个在 `shared_state['tools']` 中预先注册的工具。
    `"params": {"id": "<tool_call_id>", "tool_name": "<name_of_the_tool>", "tool_params": {"arg1": "value1", ...}}`
    - `tool_name`: 要调用的工具函数名。
    - `tool_params`: 一个包含工具所需参数的字典。

2.  `"action": "execute_code"`: 执行一段隔离的Python代码。用于无法通过工具完成的自定义逻辑。
    `"params": {"id": "<step_id>", "code": "<python_code_string>"}`

3.  `"action": "generate_handler"`: 动态创建一个Python函数，用于后续的数据处理。
    `"params": {"id": "<handler_id>", "code": "def <handler_id>(data, state): ..."}`

4.  `"action": "invoke_gpt"`: 委派一个复杂的、需要创造性或深度分析的子任务。
    `"params": {"id": "<subtask_id>", "messages": [{"role": "system", "content": "..."}, ...]}`

5.  `"action": "summarize_memory"`: 对话历史过长或需要整理思路时，调用此功能进行总结。
    `"params": {"id": "<summary_id>", "summary_instruction": "..."}`

6.  `"action": "final_answer"`: 任务完成，输出最终答案。
    `"params": {"answer_data": {"key": "value", ...}}`

**你的工作流程:**
1.  **分析**: 仔细分析 `shared_state` (特别是 `tools`, `step_history`)，了解可用工具和任务进展。
2.  **思考**: 在 `thought` 中阐述你的计划。优先考虑使用 `call_tool`。
3.  **行动**: 选择一个 `action` 并提供 `params`。
4.  **接收反馈**: 系统会执行你的指令，并将结构化的结果返回给你。
5.  **循环**: 重复以上步骤，直到你输出 `final_answer`。
"""
}

# ----------------- 核心执行逻辑 -----------------

def run(messages, shared_state, depth=0):
    """主协调循环，负责与GPT交互、解析并执行动作。"""
    if depth > CONFIG["max_depth"]:
        return {"error": "Maximum recursion depth reached"}, messages, shared_state

    history = copy.deepcopy(messages)
    
    for i in range(CONFIG["max_rounds_per_run"]):
        print(f"\n[深度:{depth}, 回合:{i+1}] --- 正在发送请求至GPT...")
        
        if i > 0 or depth > 0:
            state_summary = f"""
<CURRENT_STATE_SUMMARY>
- **Available Tools**: {list(shared_state.get('tools', {}).keys())}
- **Memory Summary**: {shared_state.get('memory_summary', 'Not yet summarized.')}
- **Variables**: {list(shared_state.get('variables', {}).keys())}
- **Handlers**: {list(shared_state.get('handlers', {}).keys())}
- **Step History Count**: {len(shared_state.get('step_history', []))}
</CURRENT_STATE_SUMMARY>
"""
            history.insert(-1, {"role": "user", "content": state_summary})

        response = client.chat.completions.create(
            model=CONFIG["default_model"],
            messages=history,
            response_format={"type": "json_object"}
        )
        msg = response.choices[0].message
        print(msg)
        history.append(msg.model_dump())

        try:
            response_json = json.loads(msg.content)
            print(f"[深度:{depth}] --- AI动作: {response_json.get('action')}")
            print(f"  思考: {response_json.get('thought')}")
        except (json.JSONDecodeError, AttributeError):
            error_feedback = f"错误: 上一次的响应不是一个有效的JSON对象。请严格按照规范生成JSON。收到的内容: {msg.content}"
            print(f"  错误: {error_feedback}")
            history.append({"role": "user", "content": error_feedback})
            continue

        final_answer, should_continue = process_ai_response(response_json, history, shared_state, depth)

        if final_answer is not None:
            return final_answer, history, shared_state
        if not should_continue:
            return {"error": "AI响应处理失败或动作无效。"}, history, shared_state

    return {"error": f"已达到最大回合数 ({CONFIG['max_rounds_per_run']})。"}, history, shared_state

def process_ai_response(response_json, history, shared_state, depth):
    """解析并执行AI响应中定义的动作，提供结构化反馈。"""
    action = response_json.get("action")
    params = response_json.get("params", {})
    step_id = params.get("id", f"step_{len(shared_state['step_history'])}")
    status = "success"
    result_data = None

    try:
        if action == "call_tool":
            tool_name = params.get("tool_name")
            tool_params = params.get("tool_params", {})
            if not tool_name or tool_name not in shared_state.get('tools', {}):
                raise ValueError(f"工具 '{tool_name}' 未找到或未注册。")
            
            print(f"  正在调用工具 (id={step_id}): {tool_name} with params {tool_params}")
            tool_function = shared_state['tools'][tool_name]
            result_data = tool_function(**tool_params)
            shared_state['variables'][step_id] = result_data
            print(f"  工具返回: {result_data}")

        elif action == "execute_code":
            code_to_run = params.get("code", "")
            print(f"  正在执行代码 (id={step_id})...")
            # ... (代码与v6版本相同)
            stdout_io = io.StringIO()
            local_scope = {'shared_state': shared_state}
            with contextlib.redirect_stdout(stdout_io):
                exec(code_to_run, globals(), local_scope)
            lines = [line for line in code_to_run.strip().split('\n') if line.strip()]
            return_value = None
            if lines:
                try:
                    return_value = eval(lines[-1], globals(), local_scope)
                except Exception: pass
            stdout_val = stdout_io.getvalue()
            shared_state['variables'][step_id] = return_value if return_value is not None else stdout_val
            result_data = {"stdout": stdout_val, "return_value": return_value}
            print(f"  标准输出: {stdout_val.strip()}\n  返回值: {return_value}")

        elif action == "generate_handler":
            # ... (代码与v6版本相同)
            code_to_run = params.get("code", "")
            handler_id = params.get("id", "unnamed_handler")
            print(f"  正在生成处理器 (id={handler_id})...")
            scope = {}
            exec(code_to_run, globals(), scope)
            shared_state['handlers'][handler_id] = scope[handler_id]
            result_data = f"处理器 '{handler_id}' 已成功创建并可用。"
            print(f"  结果: {result_data}")

        elif action == "invoke_gpt":
            # ... (代码与v6版本相同)
            messages = params.get("messages")
            if not messages: raise ValueError("'messages' is required for invoke_gpt.")
            print("  正在调用子GPT...")
            sub_model = params.get("model", CONFIG["subtask_model"])
            sub_result, _, _ = run(messages, {}, depth + 1)
            shared_state['variables'][step_id] = sub_result
            result_data = sub_result
            print(f"  子任务结果: {json.dumps(sub_result, ensure_ascii=False, indent=2)}")

        elif action == "summarize_memory":
            # ... (代码与v6版本相同)
            instruction = params.get("summary_instruction", "请总结对话历史。")
            print("  正在进行记忆压缩...")
            compression_messages = [
                {"role": "system", "content": f"你是一个高效的记忆压缩器。请根据以下指示总结提供的对话历史：'{instruction}'。"},
                {"role": "user", "content": f"<HISTORY_TO_SUMMARIZE>\n{json.dumps(history, indent=2, ensure_ascii=False)}\n</HISTORY_TO_SUMMARIZE>"}
            ]
            summary = client.chat.completions.create(model=CONFIG["subtask_model"], messages=compression_messages).choices[0].message.content
            shared_state['memory_summary'] = summary
            shared_state['variables'][step_id] = summary
            result_data = summary
            print(f"  新的记忆摘要: {summary}")

        elif action == "final_answer":
            print("  收到最终答案。")
            answer_data = params.get("answer_data", {})
            answer_data['final_shared_state'] = shared_state
            return answer_data, False
            
        else:
            raise ValueError(f"未知的动作: {action}")

    except Exception:
        status = "error"
        error_message = traceback.format_exc()
        result_data = error_message
        print(f"  错误: 执行动作 '{action}' 时发生错误:\n{error_message}")

    shared_state['step_history'].append({
        "id": step_id, "action": action, "params": params,
        "status": status, "result": result_data
    })
    
    feedback = f"""
<SYSTEM_FEEDBACK>
  <action_receipt>
    <id>{step_id}</id>
    <action>{action}</action>
    <status>{status}</status>
    <result>{json.dumps(result_data, ensure_ascii=False, default=str)}</result>
  </action_receipt>
</SYSTEM_FEEDBACK>
"""
    history.append({"role": "user", "content": feedback})
    return None, True


# ----------------- 命令行入口 -----------------
def get_website_content(url: str):
    """一个简单的工具，用于获取网站的文本内容。"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"Error fetching URL {url}: {e}"

def main():
    """主函数，从命令行运行代理。"""
    print("欢迎GPT类O系列模型 v1！")
    task = input("请输入你的复杂任务 (例如: '帮我ocr识别xxx文档然后翻译'): ")
    
    shared_state = {
        'variables': {},
        'handlers': {},
        'step_history': [],
        'memory_summary': None,
        'tools': {
            'http_get': get_website_content,
        }
    }
    
    task_overview = {"role": "user", "content": f"<OVERARCHING_GOAL>{task}</OVERARCHING_GOAL>"}
    system_prompt = {"role": "system", "content": PROMPTS["MAIN_SYSTEM_PROMPT"]}
    initial_messages = [system_prompt, task_overview]

    final_result, _, _ = run(initial_messages, shared_state)

    print("\n\n==================== 任务完成 ====================")
    
    class CustomEncoder(json.JSONEncoder):
        def default(self, o):
            if callable(o): return f"<function {o.__name__}>"
            try: return super().default(o)
            except TypeError: return str(o)

    print(json.dumps(final_result, indent=2, ensure_ascii=False, cls=CustomEncoder))
    print("====================================================")

if __name__ == "__main__":
    main()

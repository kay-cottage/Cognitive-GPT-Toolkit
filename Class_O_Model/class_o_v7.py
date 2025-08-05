#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import json
import copy
import traceback
import os
import requests
import io
import contextlib
import sys
import argparse
from datetime import datetime
import zipfile
from bs4 import BeautifulSoup   # Requires `pip install bs4` for HTML parsing
import inspect # 引入 inspect 模块

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Initialize OpenAI client
try:
    client = OpenAI(
        api_key='s7',  # 如果未配置环境变量，这里可以替换为你的 API Key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    client.models.list()  # Test connection
except Exception as e:
    print("\033[91mError: OpenAI API key is invalid or not set. Please set the OPENAI_API_KEY environment variable.\033[0m")
    print(f"Specific Error: {e}")
    client = None

CONFIG = {
    "default_model": "qwen3-coder-plus",
    "subtask_model": "gpt-4o-mini",
    "max_depth": 5,
    "max_rounds_per_run": 25,
    "session_dir": "sessions",
    "max_output_length": 15000,
    "memory_threshold": 20,             # 自动摘要阈值（消息条数）
    "recent_steps_to_include": 5,       # 修复 KeyError
    "char_threshold": 20000,            # 历史总字符数阈值
    "max_msg_char": 1200,                # 单条消息最大字符
    "debug_mode": True               # 开启后会打印发送给模型的完整提示
}


MAIN_SYSTEM_PROMPT = """你是 'CoordinatorGPT'，一个具备自主规划、执行和自我修正能力的智能体。你的最终目标是完成 <OVERARCHING_GOAL>。

你必须始终输出一个合法的 JSON 对象，结构固定如下：

{
  "thought": "(string) 你的推理：包含对当前状态的理解、下一步计划和理由。",
  "action": "(string) 你要执行的行动名称",
  "params": { ... 根据所选动作所需的参数 ... }
}

**可用动作 (action) 及其 params 规范：**

1. "execute_code"
   - 说明: 执行一段新的 Python 代码。代码将以给定 id 名称保存至 shared_state['variables']。
   - params 示例: {"id": "<代码变量名>", "code": "<python代码字符串>"}

2. "edit_code"
   - 说明: 修改 shared_state['variables'] 中已存在的一段代码。
   - params 示例: {"id": "<步骤ID>", "target_variable": "<已有代码变量名>", "edit_instructions": "<清晰的修改说明或diff>"}
   - **重要**: 使用此动作前，请确认 target_variable 已存在。

3. "generate_handler"
   - 说明: 动态创建一个可复用的 Python 函数 (处理器)。
   - params 示例: {"id": "<handler名>", "code": "def <handler名>(data, state): ..."}

4. "invoke_gpt"
   - 说明: 将一个子任务委派给较轻量的 GPT 代理执行。
   - params 示例: {"id": "<子任务ID>", "messages": [{"role": "...", "content": "..."}, ...]}

5. "summarize_memory"
   - 说明: 当对话历史过长需要压缩时，使用此动作总结记忆。
   - params 示例: {"id": "<步骤ID>", "summary_instruction": "总结的指示"}

6. "seek_human_assistance"
   - 说明: 当需要人类输入或授权时使用此动作请求帮助。
   - params 示例: {"id": "<请求ID>", "question": "<向人类提出的问题>"}

7. "call_tool"
   - 说明: 当你需要外部工具协助时，选择该动作。
   - params 示例: {"id": "<工具调用ID>", "tool_name": null, "tool_params": null}
   - **特殊规则**: 
       - 你只需要在本轮明确表达“需要调用工具”即可，不必直接指定工具名称和参数。
       - 系统会自动弹出一份最新的可用工具列表（包括描述和参数），你需要在收到该列表后，再输出你最终选择的工具名和参数。
       - 工具池会随上下文动态变化，请始终以弹出的工具列表为准。
       - 严格按照工具描述填写参数，避免遗漏或错误。

8. "final_answer"
   - 说明: 任务已完成，输出最终结果并结束流程。
   - params 示例: {"answer_data": { ... 最终答案结构 ... }}

**工作流程：**

1. 在生成响应之前，先检查 shared_state（尤其是 'tools', 'variables', 'step_history'），确保所需的资源存在且可用。
2. 在 "thought" 中：
   - 回顾目标 (<OVERARCHING_GOAL>) 和当前进展。
   - 列出可能的方案或行动，并选择其中最合理的一步。
   - 如果是错误恢复场景，首先分析失败原因并提出修复思路。
3. 选择一个最适合当前情况的动作，并给出符合参数规范的 "params"。
4. 等待系统反馈：
   - **<SYSTEM_FEEDBACK>**: 表示上一步执行成功，你可以继续按计划进行。
   - **<ERROR_FEEDBACK>**: 表示上一步执行失败，你进入调试模式，按照下述“错误处理”规则进行恢复。

**错误处理 (<ERROR_FEEDBACK>) 规则：**

- 收到错误反馈后，你必须在下一次响应的 "thought" 中首先针对 `offending_code` 和 `error_details` 进行深入分析，找出错误的根本原因，并形成一个明确的修复假设。
- 然后选择一个合适的后续动作（通常使用 `edit_code` 修改现有代码；如有必要也可使用 `execute_code` 重写实现）来尝试修复问题。
- 避免无意义的重复；如果同样的错误多次出现，请在 "thought" 中明确表示将尝试不同的方案来解决问题。

**约束：**

- 只输出 JSON，不要包含额外的解释性文本或代码块标记。
- "action" 字段的值必须是上述列表中的一个。
- "params" 必须包含所选动作要求的所有必要字段，不要缺漏。
- 在最终目标完成之前，不要使用 "final_answer" 动作。

开始执行吧。
"""

ALLOWED_ACTIONS = {"call_tool", "execute_code", "edit_code", "generate_handler", "invoke_gpt", "summarize_memory", "seek_human_assistance", "final_answer"}

# Google Custom Search API configuration (replace with your API key and CX or set env variables)
GOOGLE_API_KEY = "As"
GOOGLE_CX = "5a2"
GOOGLE_SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

def get_website_content(url: str):
    """HTTP GET request to retrieve the content of the given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text  
    except requests.RequestException as e:
        return f"获取URL {url} 时出错: {e}"


def memorize_tool(state: dict, content: str):
    """
    将一段文本内容存入永久记忆中，供后续步骤参考。
    :param state: 当前的共享状态字典。
    :param content: 需要记忆的文本内容。
    :return: 包含状态和已记忆内容确认的字典。
    """
    state.setdefault('permanent_memory', []).append(content)
    print(f"\033[35m[Memory] 新增记忆: {content[:100]}...\033[0m")
    return {"status": "memorized", "content_preview": content[:100] + "..."}

# --- 格式化与辅助函数 ---

def format_tool_list(tools: dict) -> str:
    """将工具列表格式化为 <AVAILABLE_TOOLS> XML 格式。"""
    xml = "<AVAILABLE_TOOLS>\n"
    for name, func in tools.items():
        sig_str = "N/A"
        doc_str = "No description available."
        try:
            sig = inspect.signature(func)
            sig_str = str(sig)
        except (TypeError, ValueError):
            pass # 无法获取签名（例如，对于某些内置函数）
        
        doc = inspect.getdoc(func)
        if doc:
            doc_str = doc.strip()

        xml += (
            f"  <tool>\n"
            f"    <name>{name}</name>\n"
            f"    <description>{doc_str}</description>\n"
            f"    <params>{sig_str}</params>\n"
            f"  </tool>\n"
        )
    xml += "</AVAILABLE_TOOLS>"
    return xml

def format_step_history(step_history: list, n: int) -> str:
    """
    最近 n 步思维链历史XML，成功的代码执行类action只保留精简内容。
    """
    if not step_history:
        return ""
    recent = step_history[-n:]
    xml = "<PAST_STEPS_SUMMARY>\n"
    code_actions = {"execute_code", "edit_code", "generate_handler"}
    for step in recent:
        status = step.get('status')
        action = step.get('action')
        if status == "success" and action in code_actions:
            xml += (
                f"  <step>\n"
                f"    <id>{step.get('id')}</id>\n"
                f"    <thought>{step.get('thought')}</thought>\n"
                f"    <action>{action}</action>\n"
                f"    <status>{status}</status>\n"
                f"    <feedback>(代码及执行结果已省略)</feedback>\n"
                f"  </step>\n"
            )
        else:
            result_str = json.dumps(step.get('result'), ensure_ascii=False)
            if len(result_str) > 300:
                result_str = result_str[:300] + "...(截断)"
            xml += (
                f"  <step>\n"
                f"    <id>{step.get('id')}</id>\n"
                f"    <thought>{step.get('thought')}</thought>\n"
                f"    <action>{action}</action>\n"
                f"    <status>{status}</status>\n"
                f"    <feedback>{result_str}</feedback>\n"
                f"  </step>\n"
            )
    xml += "</PAST_STEPS_SUMMARY>"
    return xml


def get_human_input(question: str) -> str:
    """请求人类输入。"""
    print(f"\n\033[93m{'='*20} 需要人类协助 {'='*20}\033[0m")
    print(f"\033[93mAI 请求: {question}\033[0m")
    ans = input("请输入您的回应 > ")
    print(f"\033[93m{'='*52}\033[0m\n")
    return ans.strip()

def create_error_feedback(step_id, action, error_details, offending_code=None) -> str:
    """标准化的错误反馈块。"""
    code_block = ""
    if offending_code:
        code_block = (
            f"\n  <offending_code>\n"
            f"    ```python\n"
            f"{offending_code}\n"
            f"    ```\n"
            f"  </offending_code>"
        )
    return (
        f"\n<ERROR_FEEDBACK>\n"
        f"  <failed_step>\n"
        f"    <id>{step_id}</id>\n"
        f"    <action>{action}</action>\n"
        f"  </failed_step>"
        f"{code_block}\n"
        f"  <error_details>{error_details}</error_details>\n"
        f"  <instruction>你上一步失败了。请仔细分析错误原因，并在下一次响应中修复它。</instruction>\n"
        f"</ERROR_FEEDBACK>\n"
    )

def validate_ai_response(response_json):
    """校验 AI 返回格式。"""
    if not isinstance(response_json, dict):
        raise ValueError("AI 返回内容不是一个合法的 JSON 对象。")
    if "action" not in response_json or "params" not in response_json or "thought" not in response_json:
        raise ValueError("AI 返回的 JSON 中缺少必要字段 (action, params, thought)。")
    if response_json["action"] not in ALLOWED_ACTIONS:
        raise ValueError(f"检测到非法的 action: '{response_json['action']}'。允许的 actions 包括: {ALLOWED_ACTIONS}")
    return True

# -------------------- 核心执行逻辑 --------------------

def process_ai_response(response_json, history, shared_state, depth, session_id):
    thought = response_json["thought"]
    action = response_json["action"]
    params = response_json.get("params", {})
    step_id = params.get("id", f"step_{len(shared_state.get('step_history', []))}")
    offending_code = None

    print(f"\033[34m[Thought] {thought}\033[0m")
    print(f"\033[32m[Action] {action}\033[0m")
    print(f"\033[32m[Params] {json.dumps(params, ensure_ascii=False, indent=2)}\033[0m")

    try:
        if action == "call_tool":
            call_id = params.get("id")
            if call_id is None:
                raise ValueError("'id' 是 call_tool 的必需字段。")
            tool_name = params.get("tool_name")
            tool_params = params.get("tool_params", {})
            
            # 第一阶段：如果模型不知道用什么工具，它会请求工具列表
            if tool_name is None:
                print("\033[36m[System] AI 请求工具列表...\033[0m")
                return {"type": "REQUEST_TOOL_LIST", "id": call_id}, True
            
            # 第二阶段：执行指定的工具
            print(f"\033[36m[System] 正在执行工具: {tool_name}...\033[0m")
            func = shared_state['tools'].get(tool_name)
            if not func:
                raise ValueError(f"未找到名为 '{tool_name}' 的工具。")
            
            result = func(**tool_params)
            shared_state.setdefault('variables', {})[step_id] = result

        elif action == "execute_code":
            code = params.get("code", "")
            offending_code = code
            print(f"\033[36m[System] 正在执行代码...\n---\n{code}\n---\033[0m")
            
            stdout_buf = io.StringIO()
            local_vars = {'shared_state': shared_state}
            with contextlib.redirect_stdout(stdout_buf):
                exec(code, globals(), local_vars)
            stdout_val = stdout_buf.getvalue()
            
            result = {
                "status": "success",
                "stdout": stdout_val,
                "locals": {k: str(v)[:200] for k, v in local_vars.items() if k != 'shared_state'}
            }
            shared_state.setdefault('variables', {})[step_id] = code
            shared_state['variables'][f"{step_id}_result"] = result

        elif action == "edit_code":
            target = params.get("target_variable")
            instr = params.get("edit_instructions")
            if not target or not instr:
                raise ValueError("edit_code 需同时提供 target_variable 和 edit_instructions。")
            
            orig = shared_state['variables'].get(target)
            if orig is None:
                raise ValueError(f"在 shared_state 中未找到要编辑的变量 '{target}'。")
            
            offending_code = orig
            print(f"\033[36m[System] 正在调用子任务模型以编辑代码: {target}...\033[0m")
            
            editor_msgs = [
                {"role":"system","content":"你是一个专业的 Python 代码编辑器。请根据用户指令修改代码，并只返回修改后的、完整的、不含任何解释的 Python 代码块。"},
                {"role":"user","content":f"修改指令:\n{instr}\n\n原始代码:\n```python\n{orig}\n```"}
            ]
            resp = client.chat.completions.create(
                model=CONFIG["subtask_model"], messages=editor_msgs
            )
            edited = resp.choices[0].message.content.strip()
            
            if edited.startswith("```python"):
                edited = edited[len("```python"):].strip()
            if edited.startswith("```"):
                edited = edited[3:].strip()
            if edited.endswith("```"):
                edited = edited[:-3].strip()
                
            shared_state['variables'][target] = edited
            result = {"status":"edited", "new_code_preview":edited[:200]+"..."}

        elif action == "seek_human_assistance":
            q = params.get("question","")
            ans = get_human_input(q)
            result = {"human_response": ans}
            shared_state.setdefault('variables', {})[step_id] = result

        elif action == "final_answer":
            return params.get("answer_data", {}), False

        else:
            raise ValueError(f"接收到未知的 action: {action}")

        # 成功反馈
        print(f"\033[92m[Success] 动作 '{action}' 执行成功。\033[0m")
        result_str = json.dumps(result, ensure_ascii=False)
        if len(result_str) > 500:
            result_str = result_str[:500] + "...(结果过长已截断)"
        print(f"\033[92m[Result] {result_str}\033[0m")
        
        shared_state.setdefault('step_history', []).append({
            "id": step_id, "thought": thought, "action": action,
            "params": params, "status": "success", "result": result
        })
        
        fb = json.dumps(result, ensure_ascii=False)
        history.append({"role":"user","content":
                        f"<SYSTEM_FEEDBACK><id>{step_id}</id><action>{action}</action>"
                        f"<status>success</status><result>{fb}</result></SYSTEM_FEEDBACK>"})
        return None, True

    except Exception as e:
        err = traceback.format_exc()
        print(f"\033[91m[Error] 动作 '{action}' 执行失败: {e}\033[0m")
        print(f"\033[91m[Traceback]\n{err}\033[0m")
        
        shared_state.setdefault('step_history', []).append({
            "id": step_id, "thought": thought, "action": action,
            "params": params, "status": "error", "result": str(e)
        })
        
        history.append({"role":"user","content":
                        create_error_feedback(step_id, action, err, offending_code)})
        return None, True

def trim_history(history):
    """按字符总量裁剪历史记录。"""
    char_threshold = CONFIG['char_threshold']
    total_chars = sum(len(m.get('content','')) for m in history)
    if total_chars <= char_threshold:
        return history
    
    print(f"\033[93m[History] 历史记录过长 ({total_chars} > {char_threshold} chars)，进行裁剪...\033[0m")
    keep_front = 3
    keep_back = CONFIG['recent_steps_to_include'] + 5
    return (
        history[:keep_front]
        + [{"role": "system", "content": "...(中间历史记录已省略)..."}]
        + history[-keep_back:]
    )

def build_prompt_messages(history, shared_state):
    """构造发送给模型的完整消息列表。"""
    msgs = []
    
    # 1. 系统主提示
    msgs.append(history[0])
    
    # 2. 工具列表
    msgs.append({"role": "system", "content": format_tool_list(shared_state['tools'])})
    
    # 3. 初始用户提示
    if len(history) > 1:
        msgs.append(history[1])
        
    # 4. 当前状态摘要
    state_sum = {
        "variables": list(shared_state.get('variables', {}).keys()),
        "permanent_memory_count": len(shared_state.get('permanent_memory', []))
    }
    msgs.append({"role": "system", "content":
                 f"<CURRENT_STATE_SUMMARY>\n{json.dumps(state_sum, indent=2)}\n</CURRENT_STATE_SUMMARY>"})

    # 5. 永久记忆
    if shared_state.get('permanent_memory'):
        mem_text = "\n- " + "\n- ".join(shared_state['permanent_memory'])
        msgs.append({"role": "system", "content": f"<PERMANENT_MEMORY>{mem_text}\n</PERMANENT_MEMORY>"})

    # 6. 对话摘要 (如果存在)
    if shared_state.get('memory_summary'):
        msgs.append({"role": "system", "content":
                     f"<CONVERSATION_SUMMARY>\n{shared_state['memory_summary']}\n</CONVERSATION_SUMMARY>"})

    # 7. 最近步骤的详细历史
    n = CONFIG['recent_steps_to_include']
    step_xml = format_step_history(shared_state.get('step_history', []), n)
    if step_xml:
        msgs.append({"role": "system", "content": step_xml})

    # 8. 剩余的对话历史
    trimmed = trim_history(history[2:])
    for m in trimmed:
        content = m.get('content', '')
        if len(content) > CONFIG['max_msg_char']:
            content = content[:CONFIG['max_msg_char']] + "...(单条消息过长已截断)"
        msgs.append({"role": m['role'], "content": content})

    return msgs

def run(session_id, messages, shared_state, depth=0):
    """核心循环。"""
    if client is None:
        return {"error":"OpenAI client 未初始化。"}, messages, shared_state
    if depth > CONFIG["max_depth"]:
        return {"error":"已达到最大递归深度。"}, messages, shared_state

    history = copy.deepcopy(messages)
    for round_idx in range(1, CONFIG["max_rounds_per_run"] + 1):
        print(f"\n\033[1;95m{'='*25} Round {round_idx} {'='*25}\033[0m")
        
        # 自动记忆摘要
        if len(history) > CONFIG["memory_threshold"] and shared_state.get('memory_summary') is None:
            print("\033[93m[System] 对话历史较长，正在尝试生成摘要...\033[0m")
            try:
                
                instr = "总结对话历史的关键点，包括用户目标与工具调用结果。"
                seg = history[2:]

                comp_msgs = [
                {"role": "system", "content": "你是记忆压缩助手，请根据指示总结对话。"},
                {"role": "user", "content": f"指示:{instr}\n\n<HISTORY>\n{json.dumps(seg, ensure_ascii=False, indent=2)}\n</HISTORY>"}
                ]
                resp = client.chat.completions.create(
                model=CONFIG["subtask_model"], messages=comp_msgs)
                shared_state['memory_summary'] = resp.choices[0].message.content.strip()

            except Exception as e:
                print(f"\033[91m[Error] 自动摘要失败: {e}\033[0m")

        # 构造本轮 prompt
        to_model = build_prompt_messages(history, shared_state)
        
        if CONFIG["debug_mode"]:
            print("\033[37m--- PROMPT TO MODEL ---\n"
                  f"{json.dumps(to_model, ensure_ascii=False, indent=2)}\n"
                  "--- END PROMPT ---\033[0m")

        # 调用 LLM
        try:
            resp = client.chat.completions.create(
                model=CONFIG["default_model"],
                messages=to_model,
                response_format={"type":"json_object"},
                #temperature=0.5,
            )
            msg = resp.choices[0].message
            history.append(msg.model_dump())
            response_json = json.loads(msg.content)
            validate_ai_response(response_json)
        except Exception as e:
            err_msg = f"API 调用或 JSON 解析失败: {e}\n原始响应: {msg.content if 'msg' in locals() else 'N/A'}"
            print(f"\033[91m[Fatal Error] {err_msg}\033[0m")
            history.append({"role":"user","content":create_error_feedback("api_error","system",err_msg)})
            continue

        # 处理响应
        result, cont = process_ai_response(response_json, history, shared_state, depth, session_id)
        save_session(session_id, history, shared_state)

        if isinstance(result, dict) and result.get("type")=="REQUEST_TOOL_LIST":
            call_id = result["id"]
            instr = (
                f"<SYSTEM_TOOL_PROMPT>\n"
                f"  <instruction>请从以下列表中选择一个工具，并在下一次响应中通过 'call_tool' 动作并填写 'tool_name' 和 'tool_params' 来调用它。</instruction>\n"
                f"  <call_id>{call_id}</call_id>\n"
                f"{format_tool_list(shared_state['tools'])}\n"
                f"</SYSTEM_TOOL_PROMPT>"
            )
            history.append({"role":"system","content":instr})
            continue

        if not cont:
            return result, history, shared_state

    return {"error":f"已达到最大运行轮次 ({CONFIG['max_rounds_per_run']})"}, history, shared_state

# --------------- 会话持久化 ----------------

def save_session(session_id, messages, shared_state):
    os.makedirs(CONFIG['session_dir'], exist_ok=True)
    msg_path = os.path.join(CONFIG['session_dir'], f"{session_id}_messages.json")
    state_path = os.path.join(CONFIG['session_dir'], f"{session_id}_state.json")
    try:
        with open(msg_path, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        
        state_to_save = {k:v for k,v in shared_state.items() if k not in ['tools','handlers']}
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state_to_save, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"\033[91m[Error] 保存会话失败: {e}\033[0m")

def load_session(session_id):
    msg_file = os.path.join(CONFIG['session_dir'], f"{session_id}_messages.json")
    state_file = os.path.join(CONFIG['session_dir'], f"{session_id}_state.json")
    if not os.path.exists(msg_file) or not os.path.exists(state_file):
        print(f"\033[91m[Error] 会话 {session_id} 不存在或不完整。\033[0m")
        return None, None
    
    with open(msg_file, 'r', encoding='utf-8') as f:
        messages = json.load(f)
    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    # 重新绑定工具函数
    shared_state = {**state, 'handlers': {}}
    tools = {
        #'google_search': google_search,
        'http_get': get_website_content,
        'memorize': lambda content, _state=shared_state: memorize_tool(_state, content)
    }
    shared_state['tools'] = tools
    
    print(f"\033[92m会话 {session_id} 加载成功。\033[0m")
    return messages, shared_state

# -------------------- 主入口 --------------------

def main():
    parser = argparse.ArgumentParser(description="CoordinatorGPT - 自主 AI 代理框架")
    parser.add_argument("--task", type=str, help="要执行的新任务目标。")
    parser.add_argument("--resume", type=str, help="要恢复的会话 ID。")
    parser.add_argument("--file", type=str, help="加载一个文件作为初始上下文。")
    args = parser.parse_args()

    if not args.task and not args.resume and not args.file:
        print("请选择操作:")
        print("1. 开始一个新任务")
        print("2. 恢复一个旧会话")
        choice = input("请输入选项 (1/2) > ")
        if choice == '1':
            args.task = input("请输入任务目标 > ")
        elif choice == '2':
            args.resume = input("请输入会话 ID > ")
        else:
            print("无效输入，程序退出。")
            return

    if args.resume:
        session_id = args.resume
        messages, shared_state = load_session(session_id)
        if messages is None: return
        messages.append({"role":"user","content":"<SYSTEM_MESSAGE>会话已成功恢复。请继续你的任务。</SYSTEM_MESSAGE>"})
    else:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        shared_state = {
            'variables': {}, 'handlers': {},
            'step_history': [], 'memory_summary': None,
            'permanent_memory': [],
            'tools': {}
        }
        # 绑定工具，确保 state 引用正确
        tools = {
            #'google_search': google_search,
            'http_get': get_website_content,
            'memorize': lambda content: memorize_tool(shared_state, content)
        }
        shared_state['tools'] = tools

        goal = args.task or f"处理文件 {args.file}"
        init_content = f"<OVERARCHING_GOAL>{goal}</OVERARCHING_GOAL>"
        
        if args.file:
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                var_name = os.path.basename(args.file).replace('.', '_')
                shared_state['variables'][var_name] = file_content
                init_content += f"\n<FILE_CONTEXT name='{var_name}'>文件内容已加载到变量 '{var_name}' 中。</FILE_CONTEXT>"
                print(f"文件 '{args.file}' 已加载到变量 '{var_name}'。")
            except Exception as e:
                print(f"\033[91m[Error] 加载文件 '{args.file}' 失败: {e}\033[0m")
                return

        messages = [
            {"role":"system","content": MAIN_SYSTEM_PROMPT},
            {"role":"user","content": init_content}
        ]

    result, history, final_state = run(session_id, messages, shared_state)
    
    save_session(session_id, history, final_state)
    
    print(f"\n\033[1;92m{'='*25} 任务结束 {'='*25}\033[0m")
    print(f"\033[92m会话 ID: {session_id}\033[0m")
    print("\033[92m最终结果:\033[0m")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\033[1;92m{'='*62}\033[0m")

if __name__ == "__main__":
    if client is None and len(sys.argv) > 1 and sys.argv[1] not in ['-h', '--help']:
        print("\033[91m由于 OpenAI 客户端未成功初始化，程序无法继续运行。\033[0m")
        sys.exit(1)
    main()

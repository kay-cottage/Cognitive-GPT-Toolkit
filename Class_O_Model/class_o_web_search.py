#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CoordinatorGPT (Unified Prompt Version) - Enhanced with Web Search and Memory Management.
=======================================
This version integrates web search tools (Google and DuckDuckGo) and improves long-term memory handling.
- New Tools: google_search and ddg_search for web queries, and memorize for saving important info.
- Memory Management: Automatic conversation summarization and persistent memory injection to manage context length.
"""

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
    "memory_threshold": 20  # Threshold for auto-summarization of conversation
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
   - 说明: 调用一个预先注册在 shared_state['tools'] 中的工具函数。
   - params 示例: {"id": "<工具调用ID>", "tool_name": "<工具名称>", "tool_params": { ... }}
   - **提示**: 当前可用的工具包括:
       - "http_get": 执行 HTTP GET 请求从 URL 获取网页内容。
       - "file_tool": 进行文件读写操作。
       - "google_search": 使用 Google 自定义搜索执行网络查询 (提供 query 等参数)。
       - "ddg_search": 使用 DuckDuckGo 搜索网络信息。
       - "memorize": 将重要信息存入长期记忆。

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
GOOGLE_API_KEY = "s"
GOOGLE_CX = "52"
GOOGLE_SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

def get_website_content(url: str):
    """HTTP GET request to retrieve the content of the given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"获取URL {url} 时出错: {e}"

def get_human_input(question: str) -> str:
    """Pause the agent and ask for human assistance/input."""
    print("\n\033[93m==================== 需要人类协助 ====================\033[0m")
    print(f"\033[93mAI 请求: {question}\033[0m")
    user_input = input("请输入您的回应 > ")
    print("\033[93m==========================================================\033[0m\n")
    return user_input.strip()

def file_tool(operation: str, path: str, content: str = None, start_line: int = 1, max_lines: int = None, max_chars: int = None, encoding: str = 'utf-8'):
    """
    文件处理工具:
    支持多种类型文件的读取、写入、追加等操作。
    - 对于读取操作: 自动截断过长内容，提供文件行数、大小等概要信息。可使用 start_line、max_lines、max_chars 参数分段读取长文件。
    - 对于写入/追加操作: 在写入前自动创建必要的目录，成功后返回写入的字节数。
    """
    try:
        # Ensure directory exists for write/append
        if operation in ('write', 'append'):
            if path:
                dir_path = os.path.dirname(path)
                if dir_path and not os.path.exists(dir_path):
                    try:
                        os.makedirs(dir_path, exist_ok=True)
                    except Exception as e:
                        return f"无法创建目录 {dir_path}: {e}"
        # Handle directory
        if os.path.isdir(path):
            if operation == 'read':
                try:
                    entries = os.listdir(path)
                except Exception as e:
                    return f"读取目录 {path} 时出错: {e}"
                max_list = 100
                listing = entries[:max_list]
                result = {
                    "operation": "read",
                    "file_path": path,
                    "is_directory": True,
                    "entries": listing,
                    "total_entries": len(entries)
                }
                if len(entries) > max_list:
                    result["entries_truncated"] = True
                return result
            elif operation == 'info':
                try:
                    entries = os.listdir(path)
                except Exception as e:
                    return f"读取目录 {path} 时出错: {e}"
                total_entries = len(entries)
                size_bytes = 0
                for entry in entries:
                    full_path = os.path.join(path, entry)
                    if os.path.isfile(full_path):
                        size_bytes += os.path.getsize(full_path)
                result = {
                    "operation": "info",
                    "file_path": path,
                    "is_directory": True,
                    "total_entries": total_entries,
                    "total_size_bytes": size_bytes
                }
                return result
            else:
                return f"路径 {path} 是一个目录，无法执行 {operation} 操作。"
        # For read/info, file must exist
        if operation in ('read', 'info') and not os.path.exists(path):
            return f"文件 {path} 不存在。"
        if operation == 'info':
            info = {
                "operation": "info",
                "file_path": path,
                "exists": True,
                "is_directory": False
            }
            info["size_bytes"] = os.path.getsize(path)
            text_mode = True
            try:
                with open(path, 'r', encoding=encoding) as f:
                    data = f.read()
                    info["line_count"] = data.count('\n') + (1 if data and data[-1] != '\n' else 0)
                    info["word_count"] = len(data.split())
            except Exception as e:
                text_mode = False
                info["line_count"] = None
                info["word_count"] = None
            try:
                info["created_time"] = datetime.fromtimestamp(os.path.getctime(path)).strftime("%Y-%m-%d %H:%M:%S")
                info["modified_time"] = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                info["created_time"] = None
                info["modified_time"] = None
            info["file_type"] = "text" if text_mode else "binary"
            return info
        elif operation == 'read':
            file_size = os.path.getsize(path)
            ext = os.path.splitext(path)[1].lower()
            if ext == '.zip':
                try:
                    with zipfile.ZipFile(path, 'r') as zf:
                        file_list = []
                        for info in zf.infolist():
                            file_list.append({
                                "filename": info.filename,
                                "file_size": info.file_size,
                                "compressed_size": info.compress_size,
                                "is_dir": info.is_dir()
                            })
                    return {
                        "operation": "read",
                        "file_path": path,
                        "file_type": "zip",
                        "contents": file_list,
                        "note": "ZIP 压缩文件内容列表。",
                        "size_bytes": file_size
                    }
                except Exception as e:
                    return f"读取 ZIP 文件 {path} 时出错: {e}"
            binary_exts = {'.docx', '.pptx', '.xlsx', '.pth', '.pdf'}
            if ext in binary_exts:
                return {
                    "operation": "read",
                    "file_path": path,
                    "file_type": "binary",
                    "size_bytes": file_size,
                    "note": "二进制文件，无法直接读取内容。"
                }
            try:
                f = open(path, 'r', encoding=encoding)
            except Exception as e:
                try:
                    f = open(path, 'rb')
                    f.close()
                    return {
                        "operation": "read",
                        "file_path": path,
                        "file_type": "binary",
                        "size_bytes": file_size,
                        "note": "二进制文件或非UTF-8编码，内容无法解码。"
                    }
                except Exception as e2:
                    return f"读取文件 {path} 时出错: {e2}"
            start = start_line if start_line and start_line > 0 else 1
            safe_char_limit = max_chars if max_chars is not None else 10000
            global_safe_cap = 15000
            if safe_char_limit > global_safe_cap:
                safe_char_limit = global_safe_cap
            line_limit = max_lines if max_lines is not None else None
            collected_lines = []
            char_count = 0
            current_line_num = 0
            truncated = False
            # Skip lines until start_line
            while current_line_num < start - 1:
                line = f.readline()
                if not line:
                    break
                current_line_num += 1
            if current_line_num < start - 1:
                f.close()
                return {
                    "operation": "read",
                    "file_path": path,
                    "file_type": "text",
                    "start_line": start,
                    "content": "",
                    "lines_read": 0,
                    "total_lines": current_line_num,
                    "truncated": False,
                    "note": "起始行超过文件总行数。"
                }
            total_lines_counted = current_line_num
            try:
                for line in f:
                    total_lines_counted += 1
                    if not truncated:
                        if (line_limit is not None and len(collected_lines) >= line_limit) or (char_count + len(line) > safe_char_limit):
                            if not collected_lines and char_count < safe_char_limit:
                                part = line[:safe_char_limit - char_count]
                                collected_lines.append(part.rstrip('\n'))
                                char_count += len(part)
                            truncated = True
                        else:
                            collected_lines.append(line.rstrip('\n'))
                            char_count += len(line)
                            continue
                    # 已截断: 仅计数后续行，不再存储
                f.close()
            except UnicodeDecodeError as e:
                f.close()
                return {
                    "operation": "read",
                    "file_path": path,
                    "file_type": "binary",
                    "size_bytes": file_size,
                    "note": "文件包含无法解码的字符，读取中断。"
                }
            content_text = "\n".join(collected_lines)
            result = {
                "operation": "read",
                "file_path": path,
                "file_type": "text",
                "start_line": start,
                "content": content_text,
                "lines_read": len(collected_lines),
                "total_lines": total_lines_counted,
                "truncated": truncated
            }
            if truncated:
                next_start = start + len(collected_lines)
                if next_start <= total_lines_counted:
                    result["note"] = f"内容已截断。使用 'start_line': {next_start} 获取后续内容。"
                else:
                    result["note"] = "内容已截断。"
            else:
                result["note"] = "内容已完整读取。"
            return result
        elif operation == 'write':
            try:
                with open(path, 'w', encoding=encoding) as f:
                    written = f.write(content if content is not None else "")
                return {
                    "operation": "write",
                    "file_path": path,
                    "status": "success",
                    "bytes_written": written
                }
            except Exception as e:
                return f"写入文件 {path} 时出错: {e}"
        elif operation == 'append':
            try:
                with open(path, 'a', encoding=encoding) as f:
                    written = f.write(content if content is not None else "")
                return {
                    "operation": "append",
                    "file_path": path,
                    "status": "success",
                    "bytes_written": written
                }
            except Exception as e:
                return f"追加写入文件 {path} 时出错: {e}"
        else:
            return f"未知的操作: {operation}"
    except Exception as e:
        return f"执行文件操作时出现错误: {e}"

def google_search(query: str, start: int = 1, safe: str = "off", num_results: int = 5, exclude_sites: list = None) -> dict:
    """使用 Google 自定义搜索 API 搜索网络内容，返回指定数量的结果。"""
    if exclude_sites:
        # Append "-site:xyz.com" filters to query to exclude certain domains
        for site in exclude_sites:
            query += f" -site:{site}"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "q": query,
        "start": start,
        "safe": safe,
        "num": num_results
    }
    try:
        resp = requests.get(GOOGLE_SEARCH_ENDPOINT, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": f"调用 Google 搜索 API 出错: {e}"}
    results = []
    for item in data.get("items", []):
        results.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet")
        })
    return {
        "query": query,
        "engine": "google",
        "results": results
    }

def ddg_search(query: str, num_results: int = 5, exclude_sites: list = None) -> dict:
    """使用 DuckDuckGo HTML 接口搜索网络内容，返回前若干结果。"""
    base_url = "https://html.duckduckgo.com/html/?q="
    if exclude_sites:
        for site in exclude_sites:
            query += f" -site:{site}"
    url = base_url + requests.utils.requote_uri(query)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        html_text = resp.text
    except Exception as e:
        return {"error": f"DuckDuckGo 搜索请求失败: {e}"}
    # Parse HTML to extract results
    soup = BeautifulSoup(html_text, "html.parser")
    links_div = soup.find(id="links")
    if not links_div:
        return {"query": query, "engine": "duckduckgo", "results": []}
    results = []
    for result in links_div.find_all(class_="results_links", limit=num_results):
        # Find title and URL
        main_link = result.find(class_="links_main")
        if not main_link:
            continue
        a_tag = main_link.find("a")
        if not a_tag:
            continue
        title = a_tag.get_text()
        link = a_tag.get("href", "")
        # Find snippet
        snippet_tag = result.find(class_="result__snippet")
        snippet = snippet_tag.get_text() if snippet_tag else ""
        results.append({
            "title": title,
            "link": link,
            "snippet": snippet
        })
    return {
        "query": query,
        "engine": "duckduckgo",
        "results": results
    }

def memorize_tool(content: str):
    """将提供的内容存入永久记忆列表。"""
    shared_state['permanent_memory'].append(content)
    return {"status": "memorized", "content": content}

def validate_ai_response(response_json):
    if not isinstance(response_json, dict):
        raise ValueError("AI返回内容不是JSON对象。")
    if "action" not in response_json:
        raise ValueError("AI响应缺少 action 字段。")
    if response_json["action"] not in ALLOWED_ACTIONS:
        raise ValueError(f"非法的 action: {response_json['action']}。允许值: {list(ALLOWED_ACTIONS)}")
    if "thought" not in response_json:
        raise ValueError("AI响应缺少 thought 字段。")
    return True

def create_error_feedback(step_id, action, error_details, offending_code=None):
    code_block = ""
    if offending_code:
        code_block = f"\n  <offending_code>\n    ```python\n    {offending_code}\n    ```\n  </offending_code>"
    return f"\n<ERROR_FEEDBACK>\n  <failed_step>\n    <id>{step_id}</id>\n    <action>{action}</action>\n  </failed_step>{code_block}\n  <error_details>\n    {error_details}\n  </error_details>\n  <instruction>\n    你上一步的操作失败了。你必须进入调试模式。\n    在你的下一次响应中，请首先在 \"thought\" 中分析 `offending_code` 和 `error_details`，以推断错误的根本原因并形成清晰的修复假设。\n    然后请选择一个合适的动作 (例如 \"edit_code\" 或必要时 \"execute_code\") 来尝试修复这个问题。\n    如果类似错误反复出现，请考虑改变策略以避免无效的重复尝试。\n  </instruction>\n</ERROR_FEEDBACK>\n"

def process_ai_response(response_json, history, shared_state, depth, session_id):
    action = response_json.get("action")
    params = response_json.get("params", {})
    step_id = params.get("id", f"step_{len(shared_state['step_history'])}")
    offending_code = None

    try:
        if not action:
            raise ValueError("AI响应中缺少 action 字段。")
        status = "success"
        result_data = None

        if action == "execute_code":
            code_to_run = params.get("code", "")
            offending_code = code_to_run
            print(f"  正在执行代码并存入变量 '{step_id}'...")
            stdout_io = io.StringIO()
            local_scope = {'shared_state': shared_state}
            with contextlib.redirect_stdout(stdout_io):
                exec(code_to_run, globals(), local_scope)
            stdout_val = stdout_io.getvalue()
            shared_state['variables'][step_id] = code_to_run
            result_data = {"status": "success", "stdout": stdout_val, "locals": {k: str(v) for k, v in local_scope.items() if k not in ('shared_state',)}}
            shared_state['variables'][f"{step_id}_result"] = result_data
            print(f"  STDOUT: {stdout_val.strip()}")

        elif action == "edit_code":
            target_var = params.get("target_variable")
            instructions = params.get("edit_instructions")
            if not target_var or not instructions:
                raise ValueError("'target_variable' 和 'edit_instructions' 是必需的。")
            original_code = shared_state['variables'].get(target_var)
            if original_code is None:
                raise ValueError(f"在 shared_state 中未找到代码变量 '{target_var}'。")
            offending_code = original_code
            print(f"  正在编辑变量 '{target_var}' 中的代码...")
            editor_messages = [
                {"role": "system", "content": "你是一个专业的代码编辑器。根据指令修改代码。只输出修改后的完整代码，不要添加解释或markdown。"},
                {"role": "user", "content": f"**编辑指令:**\n{instructions}\n\n**原始代码:**\n```python\n{original_code}\n```"}
            ]
            edit_response = client.chat.completions.create(model=CONFIG["subtask_model"], messages=editor_messages)
            edited_code = edit_response.choices[0].message.content.strip()
            if edited_code.startswith("```"):
                edited_code = edited_code.strip('`')
                edited_code = edited_code.replace('python', '', 1).strip()
            shared_state['variables'][target_var] = edited_code
            result_data = {"status": "代码编辑成功", "new_code_preview": edited_code[:200] + ("..." if len(edited_code) > 200 else "")}
            print("  代码编辑成功。")

        elif action == "generate_handler":
            code_to_run = params.get("code", "")
            offending_code = code_to_run
            handler_id = params.get("id", "unnamed_handler")
            print(f"  正在生成处理器 (id={handler_id})...")
            scope = {}
            exec(code_to_run, globals(), scope)
            shared_state['handlers'][handler_id] = scope[handler_id]
            result_data = f"处理器 '{handler_id}' 创建成功。"
            print(f"  结果: {result_data}")

        elif action == "invoke_gpt":
            original_messages = params.get("messages")
            if not original_messages:
                raise ValueError("'messages' 是 invoke_gpt 的必需参数。")
            print("  正在调用隔离的子GPT...")
            sub_system_prompt = {"role": "system", "content": MAIN_SYSTEM_PROMPT}
            sub_messages = [sub_system_prompt] + original_messages
            sub_shared_state = {
                'variables': {},
                'handlers': {},
                'step_history': [],
                'memory_summary': None,
                'permanent_memory': [],
                'tools': shared_state.get('tools', {})
            }
            sub_session_id = f"{session_id}_sub_{len(shared_state['step_history'])}"
            sub_result, _, _ = run(sub_session_id, sub_messages, sub_shared_state, depth + 1)
            shared_state['variables'][step_id] = sub_result
            result_data = sub_result
            print(f"  子任务结果: {json.dumps(sub_result, ensure_ascii=False, indent=2)}")

        elif action == "summarize_memory":
            instruction = params.get("summary_instruction", "总结对话历史。")
            print("  正在压缩记忆...")
            history_to_summarize = history[:-1]  # exclude the latest message (likely the request to summarize)
            compression_messages = [
                {"role": "system", "content": "你是一个高效的记忆压缩助手。请根据以下指示总结提供的对话历史。"},
                {"role": "user", "content": f"指示: {instruction}\n\n<HISTORY_TO_SUMMARIZE>\n{json.dumps(history_to_summarize, indent=2, ensure_ascii=False)}\n</HISTORY_TO_SUMMARIZE>"}
            ]
            summary_resp = client.chat.completions.create(model=CONFIG["subtask_model"], messages=compression_messages)
            summary = summary_resp.choices[0].message.content
            shared_state['memory_summary'] = summary
            result_data = {"summary": summary}
            print(f"  新的记忆摘要: {summary}")

        elif action == "seek_human_assistance":
            result_data = {"human_response": get_human_input(params.get("question", ""))}
            shared_state['variables'][step_id] = result_data

        elif action == "call_tool":
            tool_name = params.get("tool_name")
            tool_func = shared_state.get('tools', {}).get(tool_name)
            if not tool_func:
                raise ValueError(f"未找到工具 '{tool_name}'。")
            result_data = tool_func(**params.get("tool_params", {}))
            shared_state['variables'][step_id] = result_data

        elif action == "final_answer":
            print("  \033[92m收到最终答案。\033[0m")
            return params.get("answer_data", {}), False

        else:
            raise ValueError(f"未知的行动: {action}")

    except Exception:
        status = "error"
        error_message = traceback.format_exc()
        print(f"  \033[91m错误: 执行行动 '{action}' 时发生错误:\n{error_message}\033[0m")
        feedback = create_error_feedback(step_id, str(action), error_message, offending_code)
        history.append({"role": "user", "content": feedback})
        return None, True

    # Log the successful step
    shared_state['step_history'].append({
        "id": step_id, "action": action, "params": params,
        "status": status, "result": result_data
    })

    # Prepare system feedback to agent
    result_str = json.dumps(result_data, ensure_ascii=False, default=str)
    if len(result_str) > CONFIG['max_output_length']:
        truncated_result = result_str[:CONFIG['max_output_length']]
        result_str = truncated_result + f"\n\n[...因长度超限，输出被截断。完整输出已存储在会话状态中...]"
        print(f"  \033[93m[注意] 行动结果在添加到AI历史前被截断。\033[0m")

    feedback = (f"<SYSTEM_FEEDBACK><action_receipt><id>{step_id}</id><action>{action}</action><status>{status}</status>"
                f"<result>{result_str}</result></action_receipt></SYSTEM_FEEDBACK>")
    history.append({"role": "user", "content": feedback})
    return None, True

def run(session_id, messages, shared_state, depth=0):
    if client is None:
        return {"error": "OpenAI client not available."}, messages, shared_state
    if depth > CONFIG["max_depth"]:
        return {"error": "已达到最大递归深度"}, messages, shared_state

    history = copy.deepcopy(messages)

    for i in range(CONFIG["max_rounds_per_run"]):
        # Before each model call, manage memory if needed
        if len(history) > CONFIG["memory_threshold"] and shared_state.get('memory_summary') is None:
            # Auto-summarize conversation (except initial and recent few messages) to manage context length
            try:
                initial_keep = 2  # keep the first 2 messages (system + initial user)
                final_keep = 5    # keep the last 5 messages in detail
                if len(history) > initial_keep + final_keep:
                    # Prepare content to summarize (skip first 2 and last 5 messages)
                    history_to_summarize = history[initial_keep: len(history) - final_keep]
                else:
                    history_to_summarize = history[initial_keep:]  # (in case total < initial+final)
                summary_instruction = "总结对话历史的关键点。"
                compression_messages = [
                    {"role": "system", "content": "你是一个高效的记忆压缩助手。请根据以下对话内容总结要点。"},
                    {"role": "user", "content": f"指示: {summary_instruction}\n\n<HISTORY_TO_SUMMARIZE>\n{json.dumps(history_to_summarize, ensure_ascii=False, indent=2)}\n</HISTORY_TO_SUMMARIZE>"}
                ]
                summary_resp = client.chat.completions.create(model=CONFIG["subtask_model"], messages=compression_messages)
                summary_text = summary_resp.choices[0].message.content.strip()
                shared_state['memory_summary'] = summary_text
                # (We do not insert the summary into history here; it will be injected in prompt assembly)
                print(f"\033[94m[Memory] 对话已摘要以节省上下文长度。\033[0m 摘要内容: {summary_text[:100]}...")
            except Exception as e:
                print(f"\033[91m[Memory] 自动摘要失败: {e}\033[0m")
                # If summarization fails, we proceed without it (might risk context length issues)

        # Build the prompt messages, injecting memory summary or notes if present
        messages_to_model = []
        # Always include initial system prompt and first user message
        if history:
            messages_to_model.append(history[0])
        if len(history) > 1:
            messages_to_model.append(history[1])
        # Inject permanent memory notes (if any)
        if shared_state.get('permanent_memory'):
            notes_text = "\n- ".join([""] + shared_state['permanent_memory'])  # prepend a blank to format list
            memory_msg = {"role": "user", "content": f"<MEMORY_NOTES>{notes_text}\n</MEMORY_NOTES>"}
            messages_to_model.append(memory_msg)
        # Inject summarized memory (if available) and then recent messages
        if shared_state.get('memory_summary'):
            summary_msg = {"role": "user", "content": f"<MEMORY_SNIPPET>\n{shared_state['memory_summary']}\n</MEMORY_SNIPPET>"}
            messages_to_model.append(summary_msg)
            # Include only the last few detailed messages after the summary
            recent_count = 5
            if len(history) > 2:
                recent_msgs = history[-recent_count:]
                messages_to_model.extend(recent_msgs)
        else:
            # If no summary, include all messages from index 2 onward (full history)
            if len(history) > 2:
                messages_to_model.extend(history[2:])

        # Debug: print the prompt assembly for verification (optional)
        # print(f"Messages to model (roles): {[m['role'] for m in messages_to_model]}")

        print(f"\n\033[94m[深度:{depth}, 回合:{i+1}/{CONFIG['max_rounds_per_run']}]\033[0m --- 调用GPT...")
        try:
            response = client.chat.completions.create(
                model=CONFIG["default_model"],
                messages=messages_to_model,
                response_format={"type": "json_object"}
            )
            msg = response.choices[0].message
            history.append(msg.model_dump())
            response_json = json.loads(msg.content)
            validate_ai_response(response_json)
            print(f"\033[92m[AI 行动]\033[0m {response_json.get('action')}")
            print(f"\033[90m[思考]\033[0m {response_json.get('thought')}")
        except Exception as e:
            error_details = f"API调用或JSON解析失败: {e}。收到的内容: {msg.content if 'msg' in locals() else 'N/A'}"
            print(f"  \033[91m{error_details}\033[0m")
            history.append({"role": "user", "content": create_error_feedback("api_or_json_error", "system", error_details)})
            continue

        final_answer, should_continue = process_ai_response(response_json, history, shared_state, depth, session_id)
        save_session(session_id, history, shared_state)
        if final_answer is not None:
            return final_answer, history, shared_state
        if not should_continue:
            return None, history, shared_state

    return {"error": f"已达到最大回合数 ({CONFIG['max_rounds_per_run']})。"}, history, shared_state

def save_session(session_id, messages, shared_state):
    session_dir = CONFIG['session_dir']
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    msg_path = os.path.join(session_dir, f"{session_id}_messages.json")
    state_path = os.path.join(session_dir, f"{session_id}_state.json")
    try:
        with open(msg_path + ".tmp", 'w', encoding='utf-8') as f_out:
            json.dump(messages, f_out, indent=2, ensure_ascii=False)
        os.replace(msg_path + ".tmp", msg_path)
        # Exclude non-serializable or large objects from state before saving
        state_to_save = {k: v for k, v in shared_state.items() if k not in ['tools', 'handlers', 'prompt_version']}
        with open(state_path + ".tmp", 'w', encoding='utf-8') as f_out:
            json.dump(state_to_save, f_out, indent=2, ensure_ascii=False, default=str)
        os.replace(state_path + ".tmp", state_path)
    except Exception as e:
        print(f"\n\033[91m[会话管理] 错误: 保存会话 '{session_id}' 失败: {e}\033[0m")

def load_session(session_id):
    session_dir = CONFIG['session_dir']
    messages_file = os.path.join(session_dir, f"{session_id}_messages.json")
    state_file = os.path.join(session_dir, f"{session_id}_state.json")
    if not os.path.exists(messages_file) or not os.path.exists(state_file):
        print(f"\033[91m[错误] 未找到会话 '{session_id}' 的文件。\033[0m")
        return None, None
    with open(messages_file, 'r', encoding='utf-8') as f_in:
        messages = json.load(f_in)
    with open(state_file, 'r', encoding='utf-8') as f_in:
        loaded_state = json.load(f_in)
    # Remove deprecated fields if present
    if 'prompt_version' in loaded_state:
        loaded_state.pop('prompt_version', None)
    # Reconstruct shared_state with fresh tools and handlers
    shared_state = {
        **loaded_state,
        'handlers': {},
        'tools': {
            'http_get': get_website_content,
            'file_tool': file_tool,
            'google_search': google_search,
            'ddg_search': ddg_search,
            'memorize': memorize_tool
        }
    }
    print(f"[会话管理] 成功加载会话 '{session_id}'。")
    return messages, shared_state

def main():
    parser = argparse.ArgumentParser(description="CoordinatorGPT: 一个具有高级上下文理解和调试能力的AI代理。")
    parser.add_argument("--task", type=str, help="直接提供任务目标。")
    parser.add_argument("--resume", type=str, help="通过会话ID恢复先前的会话。")
    parser.add_argument("--file", type=str, help="加载文件内容作为初始上下文。")
    args = parser.parse_args()
    task_str = args.task
    session_id_to_resume = args.resume
    file_path = args.file

    if not task_str and not session_id_to_resume and not file_path:
        print("欢迎使用 CoordinatorGPT!")
        print("1: 开始新任务")
        print("2: 恢复先前的会话")
        choice = input("请选择一个选项 [1/2] > ")
        if choice == '1':
            task_str = input("请输入您的任务目标 > ")
        elif choice == '2':
            session_id_to_resume = input("请输入要恢复的会话ID > ")
        else:
            print("无效选择。正在退出。")
            return

    initial_messages, shared_state, session_id = [], {}, ""
    if session_id_to_resume:
        session_id = session_id_to_resume
        initial_messages, shared_state = load_session(session_id)
        if initial_messages is None:
            return
        print("\n\033[93m[系统] 会话已恢复。请检查历史和状态以继续。\033[0m\n")
        initial_messages.append({"role": "user", "content": "<SYSTEM_MESSAGE>会话已恢复。请检查历史和状态以继续。</SYSTEM_MESSAGE>"})
    else:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"新任务已开始！会话ID: {session_id}")
        # Initialize shared state for new session
        shared_state = {
            'variables': {},
            'handlers': {},
            'step_history': [],
            'memory_summary': None,
            'permanent_memory': [],   # Added permanent memory store
            'tools': {
                'http_get': get_website_content,
                'file_tool': file_tool,
                'google_search': google_search,
                'ddg_search': ddg_search,
                'memorize': memorize_tool
            }
        }
        goal = task_str if task_str else f"处理文件: {file_path}"
        initial_content = f"<OVERARCHING_GOAL>{goal}</OVERARCHING_GOAL>"
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f_in:
                    file_content = f_in.read()
                file_var_name = os.path.basename(file_path).replace('.', '_')
                shared_state['variables'][file_var_name] = file_content
                initial_content += (f"\n\n<FILE_CONTEXT name='{file_var_name}'>\n文件 '{file_path}' 的内容已加载到 `shared_state['variables']['{file_var_name}']`。\n</FILE_CONTEXT>")
                print(f"已从 '{file_path}' 加载内容到变量 '{file_var_name}'。")
            except FileNotFoundError:
                print(f"\033[91m错误: 未找到文件 '{file_path}'。将在没有文件上下文的情况下启动。\033[0m")
            except Exception as e:
                print(f"\033[91m错误: 读取文件 '{file_path}' 失败: {e}\033[0m")
        initial_messages = [
            {"role": "system", "content": MAIN_SYSTEM_PROMPT},
            {"role": "user", "content": initial_content}
        ]

    final_result, final_history, final_state = run(session_id, initial_messages, shared_state)
    save_session(session_id, final_history, final_state)
    print(f"\n[会话管理] 会话 '{session_id}' 已保存。")

    class CustomEncoder(json.JSONEncoder):
        def default(self, o):
            if callable(o):
                return f"<function {o.__name__}>"
            try:
                return super().default(o)
            except TypeError:
                return str(o)

    print("\n\033[92m==================== 任务完成 ====================\033[0m")
    if isinstance(final_result, dict):
        final_result['final_shared_state'] = final_state
    print(json.dumps(final_result, indent=2, ensure_ascii=False, cls=CustomEncoder))
    print("\033[92m=====================================================\033[0m")

if __name__ == "__main__":
    if client is None and len(sys.argv) > 1 and sys.argv[1] not in ['-h', '--help']:
        sys.exit(1)
    main()

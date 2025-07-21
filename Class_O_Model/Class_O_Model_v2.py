import json
import copy
import traceback
import os
import requests
import io
import contextlib
import sys
import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- 初始化OpenAI客户端 -----------------
# 使用环境变量或直接在此处设置
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY"))

# ----------------- 全局配置与提示词 -----------------
CONFIG = {
    "default_model": "gpt-4o",
    "subtask_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small", # V8: 用于工具检索
    "max_depth": 3,
    "max_rounds_per_run": 15, # 增加回合数以支持更复杂的思考链
    "working_memory_size": 6, # V8: 工作记忆保留最近6个步骤
}

PROMPTS = {
    "MAIN_SYSTEM_PROMPT": """
你是'协调器GPT v8'，一个具备高级元认知能力的AI代理。你的目标是完成 <OVERARCHING_GOAL>。
你必须通过一系列结构化的JSON动作来驱动任务。反思和调整是你成功的关键。

**JSON响应结构:**
{
  "thought": "(string) 详细思考过程。分析当前状态、可用知识(learnings)、历史步骤，并规划下一步。优先使用工具，当遇到困难或完成阶段性任务时，考虑'reflect'。",
  "action": "(string) 必须是以下之一: 'search_tools', 'call_tool', 'execute_code', 'generate_handler', 'invoke_gpt', 'reflect', 'update_goal', 'summarize_memory', 'final_answer'。",
  "params": "(object) 执行动作所需的参数。"
}

**核心动作 (action) 和参数 (params):**

1.  `"action": "search_tools"`: 当你需要一个工具但不确定其确切名称时，用自然语言描述你的需求。
    `"params": {"id": "<search_id>", "query": "<natural_language_description_of_need>"}`

2.  `"action": "call_tool"`: 从可用工具列表中调用一个已知的工具。
    `"params": {"id": "<tool_call_id>", "tool_name": "<tool_name>", "tool_params": {"arg1": "value1"}}`

3.  `"action": "execute_code"`: 执行临时的、一次性的Python代码。
    `"params": {"id": "<code_exec_id>", "code": "<python_code>"}`

4.  `"action": "reflect"`: **(元认知)** 审视历史和当前状态，以提炼知识或识别问题。
    `"params": {"id": "<reflect_id>", "reflection_prompt": "我应该从过去的成功/失败中学到什么？"}`

5.  `"action": "update_goal"`: **(任务调整)** 根据反思结果，更新或分解顶层目标。
    `"params": {"id": "<goal_update_id>", "new_goal": "<new_or_refined_overarching_goal>"}`
    
6.  `"action": "create_tool"`: **(自我成长)** 将一段有用的代码封装成一个可重用的新工具。
    `"params": {"id": "<tool_create_id>", "tool_name": "...", "code": "def ...", "docstring": "详细描述..."}`

7.  `"action": "final_answer"`: 任务完成，输出最终答案。
    `"params": {"answer_data": {...}}`

**你的工作流程:**
1.  **分析**: 仔细分析 `shared_state`，特别是 `overarching_goal`, `learnings`, `available_tools` 和 `step_history`。
2.  **思考**: 在 `thought` 中阐述你的计划。如果遇到阻碍或任务阶段完成，优先考虑使用 `reflect` 来评估和学习。
3.  **行动**: 选择最合适的 `action`。如果不确定用哪个工具，先用 `search_tools`。
4.  **循环**: 接收系统反馈，不断迭代，直到 `overarching_goal` 完成。
"""
}

# ----------------- V8: 工具管理器 -----------------
class ToolManager:
    """管理工具的注册、检索和执行。"""
    def __init__(self):
        self.tools = {}
        self.vectorizer = TfidfVectorizer()
        self.tool_vectors = None
        self.tool_names = []

    def register_tool(self, name, func):
        """注册一个新工具并更新索引。"""
        self.tools[name] = func
        print(f"[ToolManager] 注册工具: {name}")
        self._update_index()

    def _update_index(self):
        """(重新)构建工具的文本描述和TF-IDF向量索引。"""
        self.tool_names = list(self.tools.keys())
        # 使用函数名和文档字符串作为描述
        descriptions = [f"{name}: {self.tools[name].__doc__ or 'No description available.'}" for name in self.tool_names]
        if descriptions:
            self.tool_vectors = self.vectorizer.fit_transform(descriptions)

    def search(self, query: str, top_k: int = 3):
        """使用TF-IDF进行简单的文本相似度搜索来查找工具。"""
        if not self.tool_names:
            return []
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tool_vectors).flatten()
        # 获取top_k个最相似的工具的索引
        top_k_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_k_indices:
            if similarities[idx] > 0.1: # 设定一个阈值，避免不相关的结果
                tool_name = self.tool_names[idx]
                results.append({
                    "name": tool_name,
                    "description": self.tools[tool_name].__doc__,
                    "similarity": similarities[idx]
                })
        print(f"[ToolManager] 搜索 '{query}', 找到: {[r['name'] for r in results]}")
        return results

    def call(self, name, params):
        if name not in self.tools:
            raise ValueError(f"工具 '{name}' 未找到。")
        return self.tools[name](**params)

# ----------------- 核心执行逻辑 -----------------

def manage_memory(history, shared_state):
    """V8: 动态管理上下文，压缩旧历史。"""
    if len(history) < CONFIG["working_memory_size"] * 2: # 每个步骤有请求和反馈，所以*2
        return history, shared_state

    print("[MemoryManager] 上下文过长，正在压缩旧的情景记忆...")
    
    # 保留系统提示和初始任务
    preserved_history = history[:2]
    # 保留最近的N个步骤 (请求+反馈)
    working_memory = history[-CONFIG["working_memory_size"]:]
    
    # 要压缩的旧历史
    history_to_compress = history[2:-CONFIG["working_memory_size"]]

    # 生成旧历史的摘要
    summary_prompt = f"""
    请将以下对话历史记录压缩成一段简洁的摘要，保留关键操作和结果。
    <HISTORY_TO_COMPRESS>
    {json.dumps(history_to_compress, indent=2, ensure_ascii=False)}
    </HISTORY_TO_COMPRESS>
    """
    compression_messages = [
        {"role": "system", "content": "你是一个高效的记忆压缩器。"},
        {"role": "user", "content": summary_prompt}
    ]
    summary = client.chat.completions.create(model=CONFIG["subtask_model"], messages=compression_messages).choices[0].message.content
    
    # 更新共享状态中的摘要
    shared_state['memory_summary'] = shared_state.get('memory_summary', '') + "\n" + summary
    print(f"[MemoryManager] 新的摘要部分: {summary}")

    # 构建新的、更短的历史记录
    new_history = preserved_history
    new_history.append({"role": "user", "content": f"<PAST_SUMMARY>{shared_state['memory_summary']}</PAST_SUMMARY>"})
    new_history.extend(working_memory)
    
    return new_history, shared_state


def run(messages, shared_state, tool_manager, depth=0):
    """主协调循环，负责与GPT交互、解析并执行动作。"""
    if depth > CONFIG["max_depth"]:
        return {"error": "Maximum recursion depth reached"}, messages, shared_state

    history = copy.deepcopy(messages)
    
    for i in range(CONFIG["max_rounds_per_run"]):
        print(f"\n[深度:{depth}, 回合:{i+1}] --- 正在准备发送请求...")
        
        # V8: 记忆管理
        history, shared_state = manage_memory(history, shared_state)

        # V8: 构造动态的系统状态摘要
        state_summary_content = f"""
<CURRENT_STATE_SUMMARY>
- **Overarching Goal**: {shared_state.get('overarching_goal')}
- **Available Tools (use 'search_tools' to find more)**: {list(tool_manager.tools.keys())}
- **Learnings (Key insights from 'reflect')**: {json.dumps(shared_state.get('learnings'), indent=2, ensure_ascii=False)}
- **Variables**: {list(shared_state.get('variables', {}).keys())}
- **Handlers**: {list(shared_state.get('handlers', {}).keys())}
- **Step History Count**: {len(shared_state.get('step_history', []))}
- **Memory Summary**: {shared_state.get('memory_summary', 'Not yet summarized.')}
</CURRENT_STATE_SUMMARY>
"""
        # 将状态摘要作为用户消息插入，确保AI能看到
        current_turn_messages = history + [{"role": "user", "content": state_summary_content}]

        response = client.chat.completions.create(
            model=CONFIG["default_model"],
            messages=current_turn_messages,
            response_format={"type": "json_object"}
        )
        msg = response.choices[0].message
        print(f"[深度:{depth}] --- AI响应:\n{msg.content}")
        history.append(msg.model_dump())

        try:
            response_json = json.loads(msg.content)
            print(f"[深度:{depth}] --- AI动作: {response_json.get('action')}")
        except (json.JSONDecodeError, AttributeError):
            error_feedback = f"错误: 上一次的响应不是一个有效的JSON对象。请严格按照规范生成JSON。收到的内容: {msg.content}"
            print(f"  错误: {error_feedback}")
            history.append({"role": "user", "content": error_feedback})
            continue

        final_answer, should_continue = process_ai_response(response_json, history, shared_state, tool_manager, depth)

        if final_answer is not None:
            return final_answer, history, shared_state
        if not should_continue:
            # e.g. update_goal might stop the current loop to restart with a new goal
            return {"status": "restarting_with_new_goal"}, history, shared_state

    return {"error": f"已达到最大回合数 ({CONFIG['max_rounds_per_run']})。"}, history, shared_state

def process_ai_response(response_json, history, shared_state, tool_manager, depth):
    """解析并执行AI响应中定义的动作，提供结构化反馈。"""
    action = response_json.get("action")
    params = response_json.get("params", {})
    step_id = params.get("id", f"step_{len(shared_state['step_history'])}")
    status = "success"
    result_data = None
    should_continue = True

    try:
        # --- V8: 新增和修改的动作 ---
        if action == "search_tools":
            query = params.get("query")
            if not query: raise ValueError("'query' is required for search_tools.")
            search_results = tool_manager.search(query)
            result_data = f"Found {len(search_results)} relevant tools: {json.dumps(search_results, ensure_ascii=False)}"
            print(f"  工具搜索结果: {result_data}")

        elif action == "reflect":
            reflection_prompt = params.get("reflection_prompt", "从过去的步骤中总结关键的成功经验和失败教训。")
            print(f"  正在进行元认知反思 (id={step_id})...")
            reflection_messages = [
                {"role": "system", "content": "你是一个深刻的反思者。根据用户提供的上下文和提示，进行深入分析并输出结构化的洞察。"},
                {"role": "user", "content": f"""
                **Reflection Prompt**: {reflection_prompt}
                **Full History**: {json.dumps(shared_state['step_history'], indent=2, ensure_ascii=False)}
                **Current Variables**: {json.dumps(list(shared_state['variables'].keys()))}
                
                请输出一个JSON对象，包含你的洞察，例如：{{"learnings": ["..."], "identified_problems": ["..."]}}
                """}
            ]
            reflection_response = client.chat.completions.create(
                model=CONFIG["subtask_model"], 
                messages=reflection_messages,
                response_format={"type": "json_object"}
            ).choices[0].message.content
            
            reflection_json = json.loads(reflection_response)
            # 将学习到的内容添加到共享状态
            if 'learnings' in reflection_json and isinstance(reflection_json['learnings'], list):
                shared_state['learnings'].extend(reflection_json['learnings'])
            result_data = reflection_json
            print(f"  反思结果: {result_data}")

        elif action == "update_goal":
            new_goal = params.get("new_goal")
            if not new_goal: raise ValueError("'new_goal' is required for update_goal.")
            print(f"  正在更新目标 (id={step_id})...")
            old_goal = shared_state['overarching_goal']
            shared_state['overarching_goal'] = new_goal
            result_data = f"目标已从 '{old_goal}' 更新为 '{new_goal}'。"
            print(f"  结果: {result_data}")
            # 通常更新目标后，我们可能希望从头开始循环，让AI用新目标重新规划
            # 这里我们通过返回 should_continue=False 来示意主循环可能需要重置
            # should_continue = False 
            # For simplicity in this example, we continue the loop. A more robust system might restart.

        elif action == "create_tool":
            # V8 设计蓝图: 实现工具自进化
            # 在真实系统中，这里需要严格的安全沙箱和代码验证
            tool_name = params.get("tool_name")
            code = params.get("code")
            docstring = params.get("docstring", "No docstring provided.")
            if not all([tool_name, code]):
                raise ValueError("'tool_name' and 'code' are required for create_tool.")
            
            print(f"  正在尝试创建新工具 (id={step_id}): {tool_name}")
            # 危险: 在生产环境中，exec是极不安全的。需要沙箱！
            # For demonstration purposes only:
            try:
                scope = {}
                full_code = f"{code}\n\nnew_tool_func = {tool_name}"
                exec(full_code, globals(), scope)
                new_func = scope['new_tool_func']
                new_func.__doc__ = docstring
                tool_manager.register_tool(tool_name, new_func)
                result_data = f"工具 '{tool_name}' 已成功创建并注册。"
            except Exception as e:
                raise ValueError(f"创建工具失败: {e}")

        # --- v7 已有动作的适配 ---
        elif action == "call_tool":
            tool_name = params.get("tool_name")
            tool_params = params.get("tool_params", {})
            print(f"  正在调用工具 (id={step_id}): {tool_name} with params {tool_params}")
            result_data = tool_manager.call(tool_name, tool_params)
            shared_state['variables'][step_id] = result_data
            print(f"  工具返回: {result_data}")

        elif action == "execute_code":
            # ... (代码与v7版本相同, 但现在应鼓励AI用 create_tool 代替重复的 execute_code)
            code_to_run = params.get("code", "")
            print(f"  正在执行代码 (id={step_id})...")
            stdout_io = io.StringIO()
            local_scope = {'shared_state': shared_state, 'tool_manager': tool_manager}
            with contextlib.redirect_stdout(stdout_io):
                exec(code_to_run, globals(), local_scope)
            # ... (后续逻辑与v7相同)
            lines = [line for line in code_to_run.strip().split('\n') if line.strip()]
            return_value = local_scope.get('result', None) # 推荐在代码中设置一个`result`变量
            if not return_value and lines:
                try: return_value = eval(lines[-1], globals(), local_scope)
                except Exception: pass
            stdout_val = stdout_io.getvalue()
            shared_state['variables'][step_id] = return_value if return_value is not None else stdout_val
            result_data = {"stdout": stdout_val, "return_value": return_value}
            print(f"  标准输出: {stdout_val.strip()}\n  返回值: {return_value}")

        elif action == "invoke_gpt":
            # ... (代码与v7版本相同)
            messages = params.get("messages")
            if not messages: raise ValueError("'messages' is required for invoke_gpt.")
            print("  正在调用子GPT...")
            sub_model = params.get("model", CONFIG["subtask_model"])
            # 子任务也应该能使用工具
            sub_result, _, _ = run(messages, {}, tool_manager, depth + 1)
            shared_state['variables'][step_id] = sub_result
            result_data = sub_result
            print(f"  子任务结果: {json.dumps(sub_result, ensure_ascii=False, indent=2)}")

        elif action == "final_answer":
            print("  收到最终答案。")
            answer_data = params.get("answer_data", {})
            # 不再将整个shared_state附加到最终答案中，因为它可能非常大
            # answer_data['final_shared_state_summary'] = {k: v for k, v in shared_state.items() if k != 'step_history'}
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
        "status": status, "result": result_data if isinstance(result_data, (dict, list, str, int, float, bool)) else str(result_data)
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
    return None, should_continue


# ----------------- 命令行入口 -----------------
def get_website_content(url: str):
    """
    一个简单的工具，用于获取网站的文本内容。
    Args:
        url (str): 要获取的网站URL。
    Returns:
        str: 网站的HTML文本内容。
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"Error fetching URL {url}: {e}"

def main():
    """主函数，从命令行运行代理。"""
    print("欢迎使用协调器GPT v8 (认知版)！")
    task = input("请输入你的复杂任务 (例如: '分析'https://example.com'的HTML结构，然后告诉我它的标题是什么。'): ")
    
    # V8: 初始化 ToolManager
    tool_manager = ToolManager()
    tool_manager.register_tool('http_get', get_website_content)
    
    # V8: 初始化 Shared State
    shared_state = {
        'overarching_goal': task,
        'variables': {},
        'handlers': {},
        'step_history': [],
        'memory_summary': None,
        'learnings': [], # V8: 新增学习/知识库
    }
     
    # 准备初始消息
    system_prompt = PROMPTS["MAIN_SYSTEM_PROMPT"].replace("<OVERARCHING_GOAL>", task)
    initial_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"任务开始。我的目标是: {task}"}
    ]

    final_result, _, _ = run(initial_messages, shared_state, tool_manager)

    print("\n\n==================== 任务完成 ====================")
     
    class CustomEncoder(json.JSONEncoder):
        def default(self, o):
            if callable(o): return f"<function {o.__name__}>"
            try: return super().default(o)
            except TypeError: return str(o)

    print(json.dumps(final_result, indent=2, ensure_ascii=False, cls=CustomEncoder))
    print("====================================================")
    print("\n最终知识库(Learnings):")
    print(json.dumps(shared_state.get('learnings'), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoordinatorGPT v13: Robust, Diff-Based Editing & Full Functionality

Core Features (v13 Update):
- **Bug Fix for `ValueError`**: Resolved the critical error where `edit_code` would fail if the code wasn't pre-loaded into `shared_state`. The script now has a more logical flow to prevent this.
- **Token-Efficient Diff Editing**: The `edit_code` action has been re-engineered. It now uses Python's `difflib` library to generate and apply text diffs. Instead of sending the entire code back and forth, the AI now only needs to process the changes, significantly reducing token consumption, especially for large codebases. The AI can still opt to provide a full code block if the changes are substantial.
- **Restored Full Functionality**: This version ensures all previously designed features (`invoke_gpt`, `summarize_memory`, `generate_handler`, etc.) are present and correctly integrated.
- **Structured Debugging Loop**: Retains the powerful "analyze-then-fix" debugging cycle from v12, forcing a logical approach to error resolution.

Inherited Features:
- Stateful error context injection, real-time session persistence, interactive task input, human-in-the-loop collaboration.

Dependencies:
- pip install openai>=1.3.0 requests

Environment Variables:
- OPENAI_API_KEY=sk-...
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
import subprocess
import difflib
from datetime import datetime
from openai import OpenAI

# ----------------- Initialize OpenAI Client -----------------
try:
    # It's recommended to use environment variables for API keys.
    # The key 'sk-A' is a placeholder and will not work.
    client = OpenAI(api_key='skPEA')
# 模型配置
    client.models.list()
except Exception as e:
    print("\033[91mError: OpenAI API key is invalid or not set. Please set the OPENAI_API_KEY environment variable.\033[0m")
    print(f"Specific Error: {e}")
    client = None

# ----------------- Global Configuration & Prompts -----------------
CONFIG = {
    "default_model": "gpt-4o",
    "subtask_model": "gpt-4o-mini",
    "max_depth": 5,
    "max_rounds_per_run": 25,
    "session_dir": "sessions"
}

PROMPTS = {
    "MAIN_SYSTEM_PROMPT": """
You are 'CoordinatorGPT v13', an advanced AI agent with capabilities for planning, execution, and robust debugging. Your ultimate goal is to accomplish the <OVERARCHING_GOAL>.
You must always respond in a strict JSON format.

**Core Workflow & Structured Debugging:**
1.  **Analyze & Plan**: Think based on the history and `shared_state` to formulate a plan.
2.  **Act**: Choose an action and execute it.
3.  **Receive Feedback**:
    - **Success**: You will receive a `<SYSTEM_FEEDBACK>` message. Continue with your plan.
    - **Failure**: You will receive a special `<ERROR_FEEDBACK>` containing the full error code and logs.
4.  **Structured Debugging (Key Capability)**:
    - Upon receiving `<ERROR_FEEDBACK>`, you **must stop** your original plan and enter debugging mode.
    - **Step 1: Analyze (`analyze_error`)**: Your first action **must** be to call `analyze_error`. In the `thought`, carefully study the `<offending_code>` and `<error_details>` to diagnose the root cause and form a clear repair hypothesis.
    - **Step 2: Repair (`edit_code` or other actions)**: After your analysis is confirmed, your next action will be to fix the issue. This is often `edit_code`, but could be another action depending on your analysis.

**Available Actions (`action`) and their Parameters (`params`):**

1.  `"action": "analyze_error"`: **(Debug Step 1)** The mandatory first action after an error to record your analysis.
    `"params": {"id": "<analysis_id>", "analysis_thought": "<your_detailed_analysis_and_hypothesis>"}`

2.  `"action": "execute_code"`: Executes a new Python code block and stores it in `shared_state['variables']`.
    `"params": {"id": "<variable_name_for_code>", "code": "<python_code_string>"}`

3.  `"action": "edit_code"`: **(Debug Step 2 / Refinement)** Modifies existing code in `shared_state['variables']` using diff-based instructions.
    `"params": {"id": "<edit_id>", "target_variable": "<variable_name_holding_code>", "edit_instructions": "<clear_instructions_on_what_to_change_or_a_unified_diff_string>"}`

4.  `"action": "generate_handler"`: Dynamically creates a Python function for later use.
    `"params": {"id": "<handler_id>", "code": "def <handler_id>(data, state): ..."}`

5.  `"action": "execute_background_process"`: Runs an independent background task or system command.
    `"params": {"id": "<task_id>", "command": "<shell_command_to_execute>", "description": "<description>"}`

6.  `"action": "invoke_gpt"`: Delegates a smaller, quick-feedback subtask.
    `"params": {"id": "<subtask_id>", "messages": [...]}`

7.  `"action": "summarize_memory"`: Compresses and summarizes the conversation history.
    `"params": {"id": "<summary_id>", "summary_instruction": "..."}`

8.  `"action": "seek_human_assistance"`: Call this when you need human intelligence or authorization.
    `"params": {"id": "<request_id>", "question": "<specific_question_for_the_human>"}`

9.  `"action": "call_tool"`: Invokes a pre-registered tool.
    `"params": {"id": "<tool_call_id>", "tool_name": "<name_of_the_tool>", "tool_params": {}}`

10. `"action": "final_answer"`: The task is complete. Output the final answer.
    `"params": {"answer_data": {}}`
"""
}

# ----------------- Core Execution Logic -----------------

def run_shell_command(command: str) -> dict:
    print(f"  [Background Process] Executing command: {command}")
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            encoding='utf-8', timeout=300
        )
        output = {
            "status": "success" if result.returncode == 0 else "error",
            "return_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()
        }
        if result.returncode == 0:
            print(f"  [Background Process] Success.")
            if result.stdout: print(f"    [stdout]: {result.stdout[:500].strip()}...")
        else:
            print(f"  \033[91m[Background Process] Failed. Return Code: {result.returncode}\033[0m")
            if result.stderr: print(f"    \033[91m[stderr]: {result.stderr.strip()}\033[0m")
        return output
    except Exception as e:
        error_message = f"Exception during background process execution: {traceback.format_exc()}"
        print(f"  \033[91m[Background Process] {error_message}\033[0m")
        return {"status": "error", "return_code": -1, "stdout": "", "stderr": str(e)}

def get_human_input(question: str) -> str:
    print("\n\033[93m==================== ASSISTANCE REQUIRED ====================\033[0m")
    print(f"\033[93mAI Request: {question}\033[0m")
    user_input = input("Please provide your response > ")
    print("\033[93m==========================================================\033[0m\n")
    return user_input.strip()

def run(session_id, messages, shared_state, depth=0):
    if client is None:
        return {"error": "OpenAI client not available."}, messages, shared_state
    if depth > CONFIG["max_depth"]:
        return {"error": "Maximum recursion depth reached"}, messages, shared_state

    history = copy.deepcopy(messages)
    
    for i in range(CONFIG["max_rounds_per_run"]):
        print(f"\n\033[94m[Depth:{depth}, Round:{i+1}/{CONFIG['max_rounds_per_run']}]\033[0m --- Invoking GPT...")
        
        try:
            response = client.chat.completions.create(
                model=CONFIG["default_model"],
                messages=history,
                response_format={"type": "json_object"}
            )
            msg = response.choices[0].message
            history.append(msg.model_dump())
            response_json = json.loads(msg.content)
            
            print(f"\033[92m[AI Action]\033[0m {response_json.get('action')}")
            print(f"\033[90m[Thought]\033[0m {response_json.get('thought')}")
        
        except Exception as e:
            error_details = f"API call or JSON parsing failed: {e}. Received content: {msg.content if 'msg' in locals() else 'N/A'}"
            print(f"  \033[91m{error_details}\033[0m")
            history.append({"role": "user", "content": create_error_feedback("api_or_json_error", "system", error_details)})
            continue

        final_answer, should_continue = process_ai_response(response_json, history, shared_state, depth, session_id)
        
        save_session(session_id, history, shared_state)

        if final_answer is not None:
            return final_answer, history, shared_state
        if not should_continue:
            return {"error": "AI response processing failed or action was invalid."}, history, shared_state

    return {"error": f"Max rounds ({CONFIG['max_rounds_per_run']}) reached."}, history, shared_state

def create_error_feedback(step_id, action, error_details, offending_code=None):
    code_block = ""
    if offending_code:
        code_block = f"""
  <offending_code>
    ```python
    {offending_code}
    ```
  </offending_code>"""

    return f"""
<ERROR_FEEDBACK>
  <failed_step>
    <id>{step_id}</id>
    <action>{action}</action>
  </failed_step>{code_block}
  <error_details>
    {error_details}
  </error_details>
  <instruction>
    Your last action failed. You MUST enter debugging mode.
    Your first step now is to use the `analyze_error` action.
    In your thought for that action, study the `offending_code` and `error_details` to form a clear hypothesis about the root cause.
  </instruction>
</ERROR_FEEDBACK>
"""

def process_ai_response(response_json, history, shared_state, depth, session_id):
    action = response_json.get("action")
    params = response_json.get("params", {})
    step_id = params.get("id", f"step_{len(shared_state['step_history'])}")
    
    try:
        status = "success"
        result_data = None
        offending_code = None

        if action == "analyze_error":
            analysis = params.get("analysis_thought")
            print(f"  [Debug Analysis] AI's analysis: {analysis}")
            result_data = {"status": "analysis_acknowledged", "message": "Your analysis has been noted. Please proceed with a fix in your next step."}
        
        elif action == "execute_code":
            code_to_run = params.get("code", "")
            offending_code = code_to_run
            print(f"  Executing code and storing in variable '{step_id}'...")
            stdout_io = io.StringIO()
            with contextlib.redirect_stdout(stdout_io):
                exec(code_to_run, globals(), {'shared_state': shared_state})
            stdout_val = stdout_io.getvalue()
            shared_state['variables'][step_id] = code_to_run
            result_data = {"status": "success", "stdout": stdout_val}
            shared_state['variables'][f"{step_id}_result"] = result_data
            print(f"  STDOUT: {stdout_val.strip()}")

        elif action == "edit_code":
            target_var = params.get("target_variable")
            instructions = params.get("edit_instructions")
            if not target_var or not instructions: raise ValueError("'target_variable' and 'edit_instructions' are required.")
            original_code = shared_state['variables'].get(target_var)
            if original_code is None: raise ValueError(f"Code variable '{target_var}' not found in shared_state.")
            
            print(f"  Editing code in variable '{target_var}' via instructions...")

            # Check if instructions are a diff
            if instructions.strip().startswith("---") or instructions.strip().startswith("+++"):
                 print("  Applying diff-based edit...")
                 diff = instructions
                 lines = original_code.splitlines(True)
                 patched_lines = list(difflib.patch_hunk(diff.splitlines(True), lines))
                 edited_code = "".join(patched_lines)
            else: # Natural language instructions
                print("  Applying edit via natural language instructions...")
                editor_messages = [
                    {"role": "system", "content": "You are an expert code editor. Apply the user's instructions to the provided code. Output ONLY the complete, raw, modified code. Do not add any explanations, comments, or markdown formatting."},
                    {"role": "user", "content": f"**EDIT INSTRUCTIONS:**\n{instructions}\n\n**ORIGINAL CODE:**\n```python\n{original_code}\n```"}
                ]
                edit_response = client.chat.completions.create(model=CONFIG["subtask_model"], messages=editor_messages)
                edited_code = edit_response.choices[0].message.content.strip().removeprefix("```python").removesuffix("```").strip()

            shared_state['variables'][target_var] = edited_code
            result_data = {"status": "code edited successfully", "new_code_preview": edited_code[:200] + "..."}
            print(f"  Code edit successful.")

        elif action == "generate_handler":
            code_to_run = params.get("code", "")
            handler_id = params.get("id", "unnamed_handler")
            print(f"  Generating handler (id={handler_id})...")
            scope = {}
            exec(code_to_run, globals(), scope)
            shared_state['handlers'][handler_id] = scope[handler_id]
            result_data = f"Handler '{handler_id}' created successfully."
            print(f"  Result: {result_data}")

        elif action == "invoke_gpt":
            messages = params.get("messages")
            if not messages: raise ValueError("'messages' is required for invoke_gpt.")
            print("  Invoking sub-GPT...")
            sub_model = params.get("model", CONFIG["subtask_model"])
            sub_session_id = f"{session_id}_sub_{len(shared_state['step_history'])}"
            sub_result, _, _ = run(sub_session_id, messages, {}, depth + 1)
            shared_state['variables'][step_id] = sub_result
            result_data = sub_result
            print(f"  Sub-task result: {json.dumps(sub_result, ensure_ascii=False, indent=2)}")

        elif action == "summarize_memory":
            instruction = params.get("summary_instruction", "Summarize the dialogue history.")
            print("  Compressing memory...")
            history_to_summarize = history[:-1] # Exclude the summarize command itself
            compression_messages = [
                {"role": "system", "content": f"You are an efficient memory compressor. Summarize the provided dialogue history according to the instruction: '{instruction}'."},
                {"role": "user", "content": f"<HISTORY_TO_SUMMARIZE>\n{json.dumps(history_to_summarize, indent=2, ensure_ascii=False)}\n</HISTORY_TO_SUMMARIZE>"}
            ]
            summary = client.chat.completions.create(model=CONFIG["subtask_model"], messages=compression_messages).choices[0].message.content
            shared_state['memory_summary'] = summary
            result_data = {"summary": summary}
            print(f"  New memory summary: {summary}")

        elif action == "execute_background_process":
            command = params.get("command", "")
            result_data = run_shell_command(command)
            if result_data.get("status") == "error":
                if command.startswith("python"):
                    script_path = command.split()[1]
                    if os.path.exists(script_path):
                        with open(script_path, 'r', encoding='utf-8') as f: offending_code = f.read()
                raise Exception(result_data.get('stderr'))
            shared_state['variables'][step_id] = result_data

        elif action == "seek_human_assistance":
            result_data = {"human_response": get_human_input(params.get("question", ""))}
            shared_state['variables'][step_id] = result_data
        
        elif action == "final_answer":
            print("  \033[92mFinal answer received.\033[0m")
            return params.get("answer_data", {}), False
        
        elif action == "call_tool":
            tool_name = params.get("tool_name")
            tool_func = shared_state.get('tools', {}).get(tool_name)
            if not tool_func: raise ValueError(f"Tool '{tool_name}' not found.")
            result_data = tool_func(**params.get("tool_params", {}))
            shared_state['variables'][step_id] = result_data
        
        else:
             raise ValueError(f"Unknown action: {action}")

    except Exception as e:
        status = "error"
        error_message = traceback.format_exc()
        print(f"  \033[91mERROR: An error occurred while executing action '{action}':\n{error_message}\033[0m")
        feedback = create_error_feedback(step_id, action, error_message, offending_code)
        history.append({"role": "user", "content": feedback})
        return None, True

    # Success Feedback
    shared_state['step_history'].append({
        "id": step_id, "action": action, "params": params,
        "status": status, "result": result_data
    })
    feedback = f"<SYSTEM_FEEDBACK><action_receipt><id>{step_id}</id><action>{action}</action><status>{status}</status><result>{json.dumps(result_data, ensure_ascii=False, default=str)}</result></action_receipt></SYSTEM_FEEDBACK>"
    history.append({"role": "user", "content": feedback})
    return None, True

# ----------------- Utility Functions & Main Entry Point -----------------
def get_website_content(url: str):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"Error fetching URL {url}: {e}"

def save_session(session_id, messages, shared_state):
    session_dir = CONFIG['session_dir']
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    msg_path = os.path.join(session_dir, f"{session_id}_messages.json")
    state_path = os.path.join(session_dir, f"{session_id}_state.json")
    try:
        with open(msg_path + ".tmp", 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        os.replace(msg_path + ".tmp", msg_path)
        
        state_to_save = {k: v for k, v in shared_state.items() if k not in ['tools', 'handlers']}
        with open(state_path + ".tmp", 'w', encoding='utf-8') as f:
            json.dump(state_to_save, f, indent=2, ensure_ascii=False, default=str)
        os.replace(state_path + ".tmp", state_path)
    except Exception as e:
        print(f"\n\033[91m[Session Management] Error: Failed to save session '{session_id}': {e}\033[0m")

def load_session(session_id):
    session_dir = CONFIG['session_dir']
    messages_file = os.path.join(session_dir, f"{session_id}_messages.json")
    state_file = os.path.join(session_dir, f"{session_id}_state.json")
    if not os.path.exists(messages_file) or not os.path.exists(state_file):
        print(f"\033[91m[Error] Session files for '{session_id}' not found.\033[0m")
        return None, None
    with open(messages_file, 'r', encoding='utf-8') as f: messages = json.load(f)
    with open(state_file, 'r', encoding='utf-8') as f: loaded_state = json.load(f)
    shared_state = {**loaded_state, 'handlers': {}, 'tools': {'http_get': get_website_content}}
    print(f"[Session Management] Successfully loaded session '{session_id}'.")
    return messages, shared_state

def main():
    parser = argparse.ArgumentParser(description="CoordinatorGPT v13: A robust AI agent with diff-based editing.")
    parser.add_argument("--task", type=str, help="Provide the task directly via command line.")
    parser.add_argument("--resume", type=str, help="Resume a session via its ID from the command line.")
    parser.add_argument("--file", type=str, help="Load a file's content as the initial task context.")
    
    args = parser.parse_args()
    task_str = args.task
    session_id_to_resume = args.resume
    file_path = args.file

    if not task_str and not session_id_to_resume:
        print("Welcome to CoordinatorGPT v13!")
        print("1: Start a new task")
        print("2: Resume a previous session")
        choice = input("Select an option [1/2] > ")
        if choice == '1':
            task_str = input("Please enter your task goal > ")
        elif choice == '2':
            session_id_to_resume = input("Please enter the session ID to resume > ")
        else:
            print("Invalid choice. Exiting.")
            return

    initial_messages, shared_state, session_id = [], {}, ""
    if session_id_to_resume:
        session_id = session_id_to_resume
        initial_messages, shared_state = load_session(session_id)
        if initial_messages is None: return
        initial_messages.append({"role": "user", "content": "<SYSTEM_MESSAGE>Session has been resumed. Review history and state to continue.</SYSTEM_MESSAGE>"})
    
    elif task_str or file_path:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"New task started! Session ID: {session_id}")
        shared_state = {'variables': {}, 'handlers': {}, 'step_history': [], 'memory_summary': None, 'tools': {'http_get': get_website_content}}
        
        goal = task_str if task_str else f"Process the file: {file_path}"
        
        initial_content = f"<OVERARCHING_GOAL>{goal}</OVERARCHING_GOAL>"

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                # Store the file content in a variable for the AI to access
                file_var_name = os.path.basename(file_path).replace('.', '_')
                shared_state['variables'][file_var_name] = file_content
                initial_content += f"\n\n<FILE_CONTEXT name='{file_var_name}'>\nThe content of '{file_path}' has been loaded into the `shared_state['variables']['{file_var_name}']` variable.\n</FILE_CONTEXT>"
                print(f"Loaded content from '{file_path}' into variable '{file_var_name}'.")
            except FileNotFoundError:
                print(f"\033[91mError: File not found at '{file_path}'. Starting without file content.\033[0m")
            except Exception as e:
                print(f"\033[91mError: Failed to read file '{file_path}': {e}\033[0m")

        initial_messages = [{"role": "system", "content": PROMPTS["MAIN_SYSTEM_PROMPT"]}, {"role": "user", "content": initial_content}]

    if not initial_messages:
        print("No task to execute. Exiting.")
        return

    final_result, final_history, final_state = run(session_id, initial_messages, shared_state)
    
    save_session(session_id, final_history, final_state)
    print(f"\n[Session Management] Session '{session_id}' has been finalized and saved.")
    
    print("\n\n\033[92m==================== TASK COMPLETE ====================\033[0m")
    class CustomEncoder(json.JSONEncoder):
        def default(self, o):
            if callable(o): return f"<function {o.__name__}>"
            try: return super().default(o)
            except TypeError: return str(o)

    if isinstance(final_result, dict):
        final_result['final_shared_state'] = final_state
    
    print(json.dumps(final_result, indent=2, ensure_ascii=False, cls=CustomEncoder))
    print("\033[92m=====================================================\033[0m")

if __name__ == "__main__":
    if client is None and len(sys.argv) > 1 and sys.argv[1] not in ['-h', '--help']:
        sys.exit(1)
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSDecipher-GPT: AI-Powered JavaScript Reverse Engineering Assistant

Description:
This script combines a powerful autonomous AI agent (`CoordinatorGPT` architecture)
with specialized JavaScript analysis tools (`tree-sitter` and a Node.js executor).
It is designed to tackle complex JavaScript reverse engineering tasks by
autonomously analyzing code, extracting functions, executing snippets,
managing deobfuscation maps, and intelligently asking for human assistance
when necessary.

Key Features:
- Specialized JS Toolset: Integrates tree-sitter for AST parsing and a Node.js
  runner for dynamic code execution.
- Reverse Engineering Workflow: The AI is guided by a prompt that encourages a
  systematic approach: overview, drill-down, execute, comprehend, and iterate.
- Human-in-the-Loop: A new `ask_human_for_help` action allows the AI to pause
  and request user input for insurmountable challenges.
- Advanced Context Management: Manages a `deobfuscation_map` and runtime
  variable states to handle complex, obfuscated codebases.
- Robust & Extensible: Built on the proven CoordinatorGPT v7 pattern, allowing
  for easy addition of new tools and capabilities.

Dependencies:
- pip install openai>=1.3.0 requests tree-sitter tree-sitter-javascript

Environment Variables:
- OPENAI_API_KEY=your_openai_api_key

Setup:
1. Make sure Node.js is installed and accessible in your system's PATH.
2. Run `pip install -r requirements.txt` (see dependencies above).
3. Set the `OPENAI_API_KEY` environment variable.
"""
import json
import copy
import traceback
import os
import requests
import io
import contextlib
import sys
import pathlib
import re
import subprocess
from openai import OpenAI
from tree_sitter import Language, Parser
import importlib

# --- OpenAI Client Initialization ---
# Ensure your API key is set as an environment variable `OPENAI_API_KEY`
client = OpenAI()

# ----------------- JS Analysis Tools -----------------

class JSAnalyzer:
    """A dedicated tool for analyzing JavaScript files."""
    def __init__(self, file_path: str):
        if not pathlib.Path(file_path).exists():
            raise FileNotFoundError(f"The file {file_path} was not found.")
        self.file_path = file_path
        self.code_bytes = pathlib.Path(file_path).read_bytes()
        self.code_text = self.code_bytes.decode('utf-8', errors='ignore')

        # Initialize tree-sitter for JavaScript
        try:
            lang_mod = importlib.import_module("tree_sitter_javascript")
            js_lang = Language(lang_mod.language())
            self.parser = Parser(js_lang)
            self.tree = self.parser.parse(self.code_bytes)
            self.node_types = ['function_declaration', 'method_definition', 'function_expression', 'arrow_function']
        except ImportError:
            print("❌ tree-sitter-javascript not found. Please run: pip install tree-sitter-javascript", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to initialize tree-sitter parser: {e}", file=sys.stderr)
            sys.exit(1)

    def list_functions(self) -> list:
        """
        Lists all top-level and nested functions in the JavaScript file.
        Returns a list of dictionaries with function info.
        """
        funcs = []
        def recurse(node):
            if node.type in self.node_types:
                name = '<anonymous>'
                # Heuristics to find the function name
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self.code_bytes[name_node.start_byte:name_node.end_byte].decode(errors='ignore')
                # Check for variable declarator assignment: const myFunc = () => {}
                elif node.parent and node.parent.type == 'variable_declarator':
                    name_node = node.parent.child_by_field_name('name')
                    if name_node:
                        name = self.code_bytes[name_node.start_byte:name_node.end_byte].decode(errors='ignore')

                sr, sc = node.start_point[0] + 1, node.start_point[1] + 1
                er, ec = node.end_point[0] + 1, node.end_point[1] + 1
                funcs.append({'name': name, 'start_line': sr, 'start_col': sc, 'end_line': er, 'end_col': ec})

            for child in node.children:
                recurse(child)

        recurse(self.tree.root_node)
        return funcs

    def extract_code_snippet(self, start_line: int, end_line: int) -> str:
        """
        Extracts a specific code snippet from the file by line numbers.
        Line numbers are 1-based.
        """
        if start_line < 1 or end_line > len(self.code_text.splitlines()):
            return "Error: Line numbers are out of bounds."
        
        lines = self.code_text.splitlines()
        # Adjust for 0-based indexing
        snippet = "\n".join(lines[start_line - 1 : end_line])
        return snippet

    def execute_js_snippet(self, code_snippet: str, context_vars: dict = None) -> dict:
        """
        Executes a JavaScript snippet using Node.js and returns the output.
        `context_vars` can be used to set up the environment before execution.
        """
        context_setup = ""
        if context_vars:
            for key, value in context_vars.items():
                # JSON encode strings to handle quotes and special characters
                val_str = json.dumps(value)
                context_setup += f"const {key} = {val_str};\n"

        full_script = context_setup + code_snippet
        
        try:
            # We use '-p' to print the result of the last expression
            process = subprocess.run(
                ['node', '-p', full_script],
                capture_output=True,
                text=True,
                timeout=10, # 10-second timeout to prevent infinite loops
                encoding='utf-8'
            )
            
            if process.returncode == 0:
                return {"status": "success", "stdout": process.stdout.strip(), "stderr": process.stderr.strip()}
            else:
                return {"status": "error", "stdout": process.stdout.strip(), "stderr": process.stderr.strip()}

        except FileNotFoundError:
            return {"status": "error", "stderr": "Node.js not found. Please ensure it is installed and in your PATH."}
        except subprocess.TimeoutExpired:
            return {"status": "error", "stderr": "Execution timed out after 10 seconds."}
        except Exception as e:
            return {"status": "error", "stderr": f"An unexpected error occurred: {str(e)}"}


# ----------------- Global Config & Prompts -----------------
CONFIG = {
    "default_model": "gpt-4o",
    "subtask_model": "gpt-4o-mini",
    "max_depth": 5,
    "max_rounds_per_run": 15,
}

PROMPTS = {
    "MAIN_SYSTEM_PROMPT": """
You are 'JSDecipher-GPT', an AI assistant specialized in JavaScript reverse engineering. Your goal is to analyze, understand, and deobfuscate the provided JavaScript file to fulfill the user's <OVERARCHING_GOAL>.

You must always respond in a valid JSON format, adhering strictly to the specified structure.

**JSON Response Structure:**
{
  "thought": "(string) Your detailed thought process. Analyze the previous steps, evaluate the current state, and formulate a plan for the next action. You should maintain a 'deobfuscation_map' in your thoughts to track renamed variables.",
  "action": "(string) The action you will take. Must be one of: 'list_functions', 'extract_code_snippet', 'execute_js_snippet', 'ask_human_for_help', 'summarize_memory', 'final_answer'.",
  "params": "(object) The parameters for the chosen action."
}

**Your Reverse Engineering Workflow:**
1.  **Overview**: Start by using `list_functions` to get a map of the entire script. Identify potentially interesting functions based on their names or proximity to other interesting code.
2.  **Drill Down**: Use `extract_code_snippet` to isolate a function or a block of code for detailed analysis.
3.  **Execute & Understand**: Use `execute_js_snippet` to run the extracted code. This is your primary method for understanding what the code *does*. You can run small, self-contained parts of a larger function. If a snippet depends on other variables, you can provide them via the `context_vars` parameter.
4.  **Deobfuscate & Map**: As you figure out what a variable or function does (e.g., `_0x54f3a` is actually `decodeBase64`), document this in your `thought` process. Maintain and expand a mental `deobfuscation_map`.
5.  **Iterate**: Repeat this process, building your understanding of the code piece by piece. Use the results from one step to inform the next.
6.  **Seek Help**: If you encounter a complex cryptographic algorithm, a browser-specific API that Node.js cannot run (like `window.btoa`), or logic you cannot decipher, use `ask_human_for_help`. Provide the human with the code snippet and a clear question.

**Available Actions & Parameters:**

1.  `"action": "list_functions"`: Get an overview of all functions in the file.
    `"params": {}`

2.  `"action": "extract_code_snippet"`: Get the source code for a specific region.
    `"params": {"start_line": <int>, "end_line": <int>}`

3.  `"action": "execute_js_snippet"`: Run a piece of JS code in a sandboxed Node.js environment.
    `"params": {"code_snippet": "<js_code_string>", "context_vars": {"var1": "value1", ...}}`

4.  `"action": "ask_human_for_help"`: Pause execution and ask the user for help.
    `"params": {"question": "(string) Your specific question for the human.", "code_context": "(string) The relevant code snippet."}`

5.  `"action": "summarize_memory"`: Summarize the history when it gets too long.
    `"params": {"summary_instruction": "e.g., 'Summarize the findings about the encryption routine.'"}`

6.  `"action": "final_answer"`: You have completed the task and are providing the final result.
    `"params": {"answer_data": {"summary": "...", "deobfuscated_code": "...", "key_findings": [...]}}`

Begin your analysis now.
"""
}

# ----------------- Core Execution Logic -----------------

def run(messages, shared_state, depth=0):
    """Main coordinator loop."""
    if depth > CONFIG["max_depth"]:
        return {"error": "Maximum recursion depth reached"}, messages, shared_state

    history = copy.deepcopy(messages)
    
    for i in range(CONFIG["max_rounds_per_run"]):
        print(f"\n[Depth:{depth}, Round:{i+1}] --- Sending request to GPT...")
        
        # Add state summary to every request after the first one
        if i > 0 or depth > 0:
            state_summary = f"""
<CURRENT_STATE_SUMMARY>
- **File Under Analysis**: {shared_state.get('file_path', 'Not set')}
- **Memory Summary**: {shared_state.get('memory_summary', 'Not yet summarized.')}
- **Known Variables Count**: {len(shared_state.get('variables', {}))}
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
        history.append(msg.model_dump())

        try:
            response_json = json.loads(msg.content)
            print(f"[Depth:{depth}] --- AI Action: {response_json.get('action')}")
            # Limit printing of long thoughts
            thought = response_json.get('thought', '')
            print(f"  Thought: {thought[:250]}{'...' if len(thought) > 250 else ''}")
        except (json.JSONDecodeError, AttributeError):
            error_feedback = f"Error: Your last response was not valid JSON. Please strictly follow the format. Received: {msg.content}"
            print(f"  Error: {error_feedback}")
            history.append({"role": "user", "content": error_feedback})
            continue

        final_answer, should_continue = process_ai_response(response_json, history, shared_state, depth)

        if final_answer is not None:
            return final_answer, history, shared_state
        if not should_continue:
            return {"error": "AI response processing failed or action was invalid."}, history, shared_state

    return {"error": f"Max rounds reached ({CONFIG['max_rounds_per_run']})."}, history, shared_state

def process_ai_response(response_json, history, shared_state, depth):
    """Parses and executes the AI's chosen action."""
    action = response_json.get("action")
    params = response_json.get("params", {})
    step_id = f"step_{len(shared_state['step_history'])}"
    status = "success"
    result_data = None
    analyzer = shared_state.get('analyzer')

    try:
        if not action:
            raise ValueError("Action is missing in the response.")

        if action == "list_functions":
            if not analyzer: raise ValueError("JSAnalyzer not initialized.")
            result_data = analyzer.list_functions()
            print(f"  Tool Result: Found {len(result_data)} functions.")

        elif action == "extract_code_snippet":
            if not analyzer: raise ValueError("JSAnalyzer not initialized.")
            start, end = params.get("start_line"), params.get("end_line")
            if not all([start, end]): raise ValueError("`start_line` and `end_line` are required.")
            result_data = analyzer.extract_code_snippet(int(start), int(end))
            print(f"  Tool Result: Extracted {len(result_data.splitlines())} lines of code.")

        elif action == "execute_js_snippet":
            if not analyzer: raise ValueError("JSAnalyzer not initialized.")
            code = params.get("code_snippet")
            context = params.get("context_vars")
            if not code: raise ValueError("`code_snippet` is required.")
            result_data = analyzer.execute_js_snippet(code, context)
            print(f"  Tool Result: {result_data}")

        elif action == "ask_human_for_help":
            question = params.get("question", "I'm stuck. Can you help?")
            code_context = params.get("code_context", "")
            print("\n" + "="*20 + " HUMAN ASSISTANCE REQUIRED " + "="*20)
            print(f"AI's Question: {question}")
            if code_context:
                print("\n--- Code Context ---\n" + code_context + "\n--------------------")
            human_response = input("Your Answer > ")
            result_data = {"human_feedback": human_response}
            print("="*63 + "\n")

        elif action == "summarize_memory":
            instruction = params.get("summary_instruction", "Summarize the dialogue history.")
            print("  Compressing memory...")
            compression_messages = [
                {"role": "system", "content": f"You are an efficient memory compressor. Summarize the provided dialogue history according to the instruction: '{instruction}'."},
                {"role": "user", "content": f"<HISTORY_TO_SUMMARIZE>\n{json.dumps(history, indent=2, ensure_ascii=False)}\n</HISTORY_TO_SUMMARIZE>"}
            ]
            summary = client.chat.completions.create(model=CONFIG["subtask_model"], messages=compression_messages).choices[0].message.content
            shared_state['memory_summary'] = summary
            result_data = summary
            print(f"  New Memory Summary: {summary[:150]}...")

        elif action == "final_answer":
            print("  Received final answer.")
            answer_data = params.get("answer_data", {})
            return answer_data, False
        
        else:
            raise ValueError(f"Unknown action: {action}")

        shared_state['variables'][step_id] = result_data

    except Exception as e:
        status = "error"
        error_message = traceback.format_exc()
        result_data = {"error_details": error_message}
        print(f"  Error executing action '{action}':\n{error_message}")

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


# ----------------- Command-Line Entry Point -----------------

def main():
    """Main function to run the agent from the command line."""
    print("Welcome to JSDecipher-GPT!")
    print("="*30)
    
    file_path = input("Enter the path to the JavaScript file you want to analyze: ").strip()
    try:
        analyzer = JSAnalyzer(file_path)
    except (FileNotFoundError, Exception) as e:
        print(f"Error initializing analyzer: {e}", file=sys.stderr)
        sys.exit(1)

    task = input("What is your goal for this analysis? (e.g., 'Find the encryption key', 'Understand the login logic'): ")
    
    shared_state = {
        'file_path': file_path,
        'analyzer': analyzer,
        'variables': {},
        'step_history': [],
        'memory_summary': None,
    }
    
    task_overview = {"role": "user", "content": f"<OVERARCHING_GOAL>{task}</OVERARCHING_GOAL>"}
    system_prompt = {"role": "system", "content": PROMPTS["MAIN_SYSTEM_PROMPT"]}
    initial_messages = [system_prompt, task_overview]

    final_result, _, _ = run(initial_messages, shared_state)

    print("\n\n==================== TASK COMPLETE ====================")
    
    # Custom JSON encoder to handle non-serializable objects like the analyzer
    class CustomEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, JSAnalyzer):
                return f"<JSAnalyzer for {o.file_path}>"
            try:
                return super().default(o)
            except TypeError:
                return str(o)

    print(json.dumps(final_result, indent=2, ensure_ascii=False, cls=CustomEncoder))
    print("=====================================================")

if __name__ == "__main__":
    main()

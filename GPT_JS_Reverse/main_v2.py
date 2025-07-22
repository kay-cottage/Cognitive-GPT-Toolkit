#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
!!!!!!!!!!!!!!!!!!!!!!!!!!LOGGING DEMO!!!!!!!!!!!!!!!!!!!!!!!!


Welcome to JSDecipher-GPT (True Context Shielding)!
============================================================
Enter the path to the JavaScript file to analyze: E:\code_new\David_Test\David_2025\David_new\tool_html\html\xhs.js
2025-07-22 12:55:23,434 - [INFO] - JSAnalyzer initialized for E:\code_new\David_Test\David_2025\David_new\tool_html\html\xhs.js (28609 lines).
What is your goal for this analysis? (e.g., 'Find the encryption key'): 研究一下X-S是如何生成的，我要完整的实现代码
2025-07-22 12:55:31,665 - [INFO] - [Depth:0, Round:1] --- Preparing request...
2025-07-22 12:55:31,674 - [INFO] - Estimated token count: 767
2025-07-22 12:55:31,674 - [INFO] - Sending request to GPT... 
2025-07-22 12:55:35,140 - [INFO] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-07-22 12:55:35,175 - [INFO] - AI Action: find_code_by_keyword
2025-07-22 12:55:35,176 - [INFO] - AI Thought: To investigate how the 'X-S' value is generated, I must locate its generation logic within the codebase. A strategic point to start would be to search for occurrences of 'X-S', which could lead us to its generation function or usage context.
2025-07-22 12:55:35,186 - [INFO] - Search for 'X-S' found 4 results.
2025-07-22 12:55:35,186 - [INFO] - [Depth:0, Round:2] --- Preparing request...
2025-07-22 12:55:35,188 - [INFO] - Estimated token count: 1362
2025-07-22 12:55:35,189 - [INFO] - Sending request to GPT...
2025-07-22 12:55:37,636 - [INFO] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-07-22 12:55:37,638 - [INFO] - AI Action: extract_code_snippet
2025-07-22 12:55:37,638 - [INFO] - AI Thought: The search results show multiple occurrences of 'X-Sign' and 'X-S-Common', suggesting they might be part of the same hashing or signature generation process. A function `encryptToken` is notably close to one of these mentions, suggesting a relationship. To confirm and understand how 'X-S' is specifi...
2025-07-22 12:55:37,639 - [INFO] - [Depth:0, Round:3] --- Preparing request...
2025-07-22 12:55:37,641 - [INFO] - Estimated token count: 1663
2025-07-22 12:55:37,641 - [INFO] - Sending request to GPT...
2025-07-22 12:55:41,429 - [INFO] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-07-22 12:55:41,431 - [INFO] - AI Action: find_code_by_keyword
2025-07-22 12:55:41,431 - [INFO] - AI Thought: The `encryptToken` function appears to be involved in generating some kind of token, using a combination of URL parts and parameters, coupled with a constant value `s = "WSUDD"`. The outcome seems to involve a transformation such as a hash or signature with `f()` being called on concatenated string ...
2025-07-22 12:55:41,448 - [WARNING] - Search for 'function f' yielded 35 results, exceeding the limit of 25. Returning a truncated list.
2025-07-22 12:55:41,449 - [INFO] - Result was truncated. Generating a concise feedback summary for the AI to prevent context overflow.
2025-07-22 12:55:41,450 - [INFO] - [Depth:0, Round:4] --- Preparing request...
2025-07-22 12:55:41,453 - [INFO] - Estimated token count: 1915
2025-07-22 12:55:41,454 - [INFO] - Sending request to GPT...
2025-07-22 12:55:44,295 - [INFO] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-07-22 12:55:44,299 - [INFO] - AI Action: find_code_by_keyword
2025-07-22 12:55:44,300 - [INFO] - AI Thought: The search for 'function f' was too broad, likely because 'f' could be an obfuscated or commonly used function. To refine my search, I should search for assignments to 'f', possibly in the form of 'f =' followed by a function declaration or function assignment.
2025-07-22 12:55:44,316 - [INFO] - Search for 'f\s*=\s*function' found 9 results.
2025-07-22 12:55:44,316 - [INFO] - [Depth:0, Round:5] --- Preparing request...
2025-07-22 12:55:44,322 - [INFO] - Estimated token count: 2929
2025-07-22 12:55:44,322 - [INFO] - Sending request to GPT...
2025-07-22 12:55:46,715 - [INFO] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-07-22 12:55:46,720 - [INFO] - AI Action: extract_code_snippet
2025-07-22 12:55:46,720 - [INFO] - AI Thought: The function `f` is defined at line 6735. This function might be a utility for generating signatures or hashes, which is 
why it was used in `encryptToken`. To confirm its functionality, I'll extract and review the code for this function definition and its context.
2025-07-22 12:55:46,720 - [INFO] - [Depth:0, Round:6] --- Preparing request...
2025-07-22 12:55:46,731 - [INFO] - Estimated token count: 3225
2025-07-22 12:55:46,731 - [INFO] - Sending request to GPT...
2025-07-22 12:55:52,241 - [INFO] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-07-22 12:55:52,243 - [INFO] - AI Action: find_code_by_keyword
2025-07-22 12:55:52,244 - [INFO] - AI Thought: The extracted code shows that `f` is defined as an anonymous function at line 6735, and it seems to involve device and browser capability checks rather than directly related to hashing or signature generation. Thus, this might not be the `f` function used in `encryptToken`. Given the obfuscated natu...
2025-07-22 12:55:52,260 - [INFO] - Search for 'encryptToken\s*\([^)]*f\(' found 0 results.
2025-07-22 12:55:52,260 - [INFO] - [Depth:0, Round:7] --- Preparing request...
2025-07-22 12:55:52,273 - [INFO] - Estimated token count: 3446
2025-07-22 12:55:52,274 - [INFO] - Sending request to GPT...
2025-07-22 12:55:58,164 - [INFO] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-07-22 12:55:58,166 - [INFO] - AI Action: find_code_by_keyword
2025-07-22 12:55:58,166 - [INFO] - AI Thought: The last search was designed to find invocations of `encryptToken` where `f` is explicitly utilized for some transformation. However, it yielded no results, suggesting that `f` may be an imported or globally defined function, or its usage in `encryptToken` is more indirect. I'll explore other possib...
2025-07-22 12:55:58,184 - [INFO] - Search for 'var f\s*=\s*r\(' found 6 results.
2025-07-22 12:55:58,186 - [INFO] - [Depth:0, Round:8] --- Preparing request...
2025-07-22 12:55:58,196 - [INFO] - Estimated token count: 4313
2025-07-22 12:55:58,197 - [INFO] - Sending request to GPT...
2025-07-22 12:56:10,019 - [INFO] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-07-22 12:56:10,066 - [INFO] - AI Action: extract_code_snippet
2025-07-22 12:56:10,067 - [INFO] - AI Thought: The search results indicate several instances where `f` is defined as `r(<module_id>)`, suggesting it is imported from modules. The assignments to `f` at different locations are potentially the utility or cryptographic function used in `encryptToken`. Line 18073 (`var f = r(31547)`) and its context ...
2025-07-22 12:56:10,069 - [INFO] - [Depth:0, Round:9] --- Preparing request...
2025-07-22 12:56:10,077 - [INFO] - Estimated token count: 4615
2025-07-22 12:56:10,077 - [INFO] - Sending request to GPT...
2025-07-22 12:56:23,311 - [INFO] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-07-22 12:56:23,314 - [INFO] - AI Action: find_code_by_keyword
2025-07-22 12:56:23,314 - [INFO] - AI Thought: The code snippet indicates that `f` is indeed a module `r(31547)` and is used here with `f._(e)`, likely pointing to a method or utility function within that module. This may be related to type-checking or conversion functions typically used in payloads, such as `isError`. However, this doesn't dire...
2025-07-22 12:56:23,333 - [WARNING] - Search for 'f\._' yielded 34 results, exceeding the limit of 25. Returning a truncated list.
2025-07-22 12:56:23,334 - [INFO] - Result was truncated. Generating a concise feedback summary for the AI to prevent context overflow.
2025-07-22 12:56:23,335 - [INFO] - [Depth:0, Round:10] --- Preparing request...
2025-07-22 12:56:23,352 - [INFO] - Estimated token count: 4886
2025-07-22 12:56:23,353 - [INFO] - Sending request to GPT...
2025-07-22 12:56:27,026 - [INFO] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-07-22 12:56:27,028 - [INFO] - AI Action: get_object_keys
2025-07-22 12:56:27,029 - [INFO] - AI Thought: The search for `f._` yielded too many results, suggesting it is widely used. This means `f` likely provides multiple utilities or methods. To narrow this down and find specifically what methods are offered by `f`, I'll inspect the keys or methods under `f` by examining the object structure. This wil...
2025-07-22 12:56:27,031 - [INFO] - [Depth:0, Round:11] --- Preparing request...
2025-07-22 12:56:27,045 - [INFO] - Estimated token count: 5072
2025-07-22 12:56:27,047 - [INFO] - Sending request to GPT...
2025-07-22 12:56:33,460 - [INFO] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-07-22 12:56:33,464 - [INFO] - AI Action: find_code_by_keyword
2025-07-22 12:56:33,465 - [INFO] - AI Thought: The attempt to directly inspect `f` failed, possibly due to the variable being dynamically assigned or out of the current analysis scope. However, considering the previous explorations and that we have explored all connections to `f`, re-evaluate assumption: Since the generation logic is 
not apparen...
2025-07-22 12:56:33,480 - [INFO] - Search for 'f\.(hash|sign|encode|generate)' found 0 results.
2025-07-22 12:56:33,483 - [INFO] - [Depth:0, Round:12] --- Preparing request...
2025-07-22 12:56:33,496 - [INFO] - Estimated token count: 5303
2025-07-22 12:56:33,496 - [INFO] - Sending request to GPT...
2025-07-22 12:56:38,503 - [INFO] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-07-22 12:56:38,505 - [INFO] - AI Action: ask_human_for_help
2025-07-22 12:56:38,505 - [INFO] - AI Thought: Finding no results for typical cryptographic or string manipulation methods within `f`, it appears I've exhausted direct 
keyword searches using partial identifiers. It may be the case that instead of 'f' directly, another alias or function used within `encryptToken` encapsulates the logic necessary ...

==================== HUMAN ASSISTANCE REQUIRED ====================
AI's Question: Could you provide deeper details on code dynamics or specific X or S related features in `encryptToken` function, such as key variables derived obscure dynamically, further insights appreciated for closer procedural accuracy.

--- Code Context ---
The current utility `f` leverages obfuscated procedures and lacks clear entry point related transformations consistent generation 'X-S'. Includes indirect pathways or dependencies not fully enumerated.
--------------------
Your Answer >





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
import logging
from openai import OpenAI
from tree_sitter import Language, Parser
import importlib
import tiktoken

# ----------------- Structured Logging Setup -----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    stream=sys.stdout
)

# ----------------- 初始化OpenAI客户端 -----------------
# It's recommended to use environment variables for API keys.
client = OpenAI(api_key='skPEA')
# 模型配置


# ----------------- JS Analysis Tools -----------------

class JSAnalyzer:
    """A dedicated tool for analyzing JavaScript files with surgical precision."""
    def __init__(self, file_path: str):
        if not pathlib.Path(file_path).exists():
            raise FileNotFoundError(f"The file {file_path} was not found.")
        self.file_path = file_path
        self.code_bytes = pathlib.Path(file_path).read_bytes()
        self.code_text = self.code_bytes.decode('utf-8', errors='ignore')
        self.lines = self.code_text.splitlines()
        logging.info(f"JSAnalyzer initialized for {file_path} ({len(self.lines)} lines).")

        try:
            lang_mod = importlib.import_module("tree_sitter_javascript")
            js_lang = Language(lang_mod.language())
            self.parser = Parser(js_lang)
            self.tree = self.parser.parse(self.code_bytes)
            self.func_node_types = ['function_declaration', 'method_definition', 'function_expression', 'arrow_function']
        except ImportError:
            logging.error("tree-sitter-javascript not found. Please run: pip install tree-sitter-javascript", exc_info=True)
            sys.exit(1)
        except Exception as e:
            logging.error(f"Failed to initialize tree-sitter parser: {e}", exc_info=True)
            sys.exit(1)

    def _get_node_text(self, node):
        return self.code_bytes[node.start_byte:node.end_byte].decode(errors='ignore')

    def find_code_by_keyword(self, keyword: str, is_regex: bool = False, context_lines: int = 2, max_results: int = 25) -> dict:
        """
        Finds code snippets by keyword or regex. If matches exceed max_results,
        it returns a summary to prevent context overflow.
        """
        results = []
        total_matches = 0
        try:
            pattern = re.compile(keyword) if is_regex else None
        except re.error as e:
            logging.warning(f"Invalid regex provided: '{keyword}'. Error: {e}")
            return {"error": f"Invalid regex: {e}"}

        for i, line in enumerate(self.lines):
            match_found = False
            if pattern:
                if pattern.search(line):
                    match_found = True
            elif keyword in line:
                match_found = True
            
            if match_found:
                total_matches += 1
                if len(results) < max_results:
                    start = max(0, i - context_lines)
                    end = min(len(self.lines), i + context_lines + 1)
                    snippet = "\n".join(f"{j+1: >4}: {self.lines[j]}" for j in range(start, end))
                    results.append({
                        'match_line': i + 1,
                        'context_snippet': snippet
                    })
        
        if total_matches > max_results:
            logging.warning(f"Search for '{keyword}' yielded {total_matches} results, exceeding the limit of {max_results}. Returning a truncated list.")
            return {
                "status": "truncated",
                "total_matches": total_matches,
                "displayed_matches": len(results),
                "results": results,
                "suggestion": "Your search keyword was too broad and returned too many results. Please refine your keyword to be more specific. For example, instead of just 'encrypt', try searching for 'function encrypt' or a specific variable name associated with the encryption logic."
            }
        
        logging.info(f"Search for '{keyword}' found {total_matches} results.")
        return {
            "status": "complete",
            "total_matches": total_matches,
            "results": results
        }

    def get_object_keys(self, variable_name: str, line_number: int) -> dict:
        """
        Finds an object declaration near a line and returns its top-level keys.
        This is useful for inspecting module-like objects.
        """
        try:
            target_node = self.tree.root_node.descendant_for_point_range((line_number - 1, 0), (line_number, 0))
            if not target_node:
                return {"status": "error", "error_details": f"Could not find a node at line {line_number}"}

            search_node = target_node
            declarator_node = None
            # Search upwards from the target line for the variable declaration
            while search_node:
                if search_node.type == 'variable_declarator':
                    name_node = search_node.child_by_field_name('name')
                    if name_node and self._get_node_text(name_node) == variable_name:
                        declarator_node = search_node
                        break
                search_node = search_node.parent
            
            if not declarator_node:
                return {"status": "error", "error_details": f"Could not find variable declaration for '{variable_name}' near line {line_number}"}

            object_node = declarator_node.child_by_field_name('value')
            if not object_node or object_node.type != 'object':
                return {"status": "error", "error_details": f"Variable '{variable_name}' is not assigned an object literal."}

            keys = []
            for child in object_node.children:
                if child.type == 'pair':
                    key_node = child.child_by_field_name('key')
                    if key_node:
                        keys.append(self._get_node_text(key_node).strip('"\''))
            
            logging.info(f"Found {len(keys)} keys for object '{variable_name}' near line {line_number}.")
            return {"status": "success", "keys": keys}
        except Exception as e:
            logging.error(f"Error in get_object_keys for '{variable_name}': {e}", exc_info=True)
            return {"status": "error", "error_details": f"An unexpected error occurred: {str(e)}"}


    def extract_code_snippet(self, start_line: int, end_line: int) -> str:
        """Extracts a code snippet by line numbers (1-based)."""
        if start_line < 1 or end_line > len(self.lines) or start_line > end_line:
            return "Error: Line numbers are out of bounds."
        return "\n".join(self.lines[start_line - 1 : end_line])

    def execute_js_snippet(self, code_snippet: str, context_vars: dict = None) -> dict:
        """Executes a JavaScript snippet using Node.js."""
        context_setup = ""
        if context_vars:
            for key, value in context_vars.items():
                val_str = json.dumps(value)
                context_setup += f"const {key} = {val_str};\n"
        full_script = context_setup + code_snippet
        try:
            process = subprocess.run(
                ['node', '-p', full_script], 
                capture_output=True, 
                text=True, 
                timeout=10, 
                encoding='utf-8'
            )
            if process.returncode == 0:
                return {"status": "success", "stdout": process.stdout.strip(), "stderr": process.stderr.strip()}
            else:
                return {"status": "error", "stdout": process.stdout.strip(), "stderr": process.stderr.strip()}
        except FileNotFoundError:
            return {"status": "error", "stderr": "Node.js not found. Please ensure Node.js is installed and in your PATH."}
        except subprocess.TimeoutExpired:
            return {"status": "error", "stderr": "Execution timed out after 10 seconds."}
        except Exception as e:
            logging.error(f"Unexpected error executing JS snippet: {e}", exc_info=True)
            return {"status": "error", "stderr": f"An unexpected error occurred: {str(e)}"}

# ----------------- Global Config & Prompts -----------------
CONFIG = {
    "default_model": "gpt-4o",
    "summarizer_model": "gpt-4o-mini",
    "max_depth": 5,
    "max_rounds_per_run": 30,
    "context_token_threshold": 20000,
}

PROMPTS = {
    "MAIN_SYSTEM_PROMPT": """
You are 'JSDecipher-GPT v8', an AI assistant for JavaScript reverse engineering. Your goal is to fulfill the user's <OVERARCHING_GOAL> with surgical precision.

**Core Directive: AVOID CONTEXT OVERFLOW. BE STRATEGIC.**

**Token Efficiency Strategy: Search -> Inspect -> Refine**
1.  **Broad Search:** Start with `find_code_by_keyword` using a simple string (e.g., 'encrypt', 'login') to find initial points of interest.
2.  **Handling Truncated Results:** If the tool feedback shows `{"status": "truncated", ...}`, your search was too broad. The full list of results will NOT be provided to save context space. You MUST refine your search using the provided `suggestion`. Do not ask for the full list. Instead, create a more specific keyword or regex. For example, if you searched for `login` and it was truncated, try `function login` or `LoginRequest`.
3.  **Inspect Objects:** If your search reveals an important-looking object (e.g., a utility object `p`), use `get_object_keys` to see what functions it contains. This is a cheap way to discover function names.
4.  **Refine Search:** Use the keys you discovered to perform a more targeted search with `find_code_by_keyword`. Use the `is_regex` parameter if you need to find obfuscated code (e.g., `keyword: "p\\\\.lz\\s*=\\s*function"`, `is_regex: true`).
5.  **Extract & Analyze:** Once you have high confidence you've located the correct function, use `extract_code_snippet` to get its full body for analysis or execution.

**JSON Response Structure:**
{
  "thought": "(string) Your detailed thought process following the 'Search -> Inspect -> Refine' strategy. State your hypothesis, your next surgical action, and justify why you are taking it. If you received a truncated result, explain how you are refining your search.",
  "action": "(string) Must be one of: 'find_code_by_keyword', 'get_object_keys', 'extract_code_snippet', 'execute_js_snippet', 'ask_human_for_help', 'final_answer'.",
  "params": "(object) Parameters for the action."
}

**Available Actions & Parameters:**
1.  `"action": "find_code_by_keyword"`: Search code.
    `"params": {"keyword": "<string_or_regex>", "is_regex": <bool>, "context_lines": <int>}`
2.  `"action": "get_object_keys"`: Inspect an object's keys.
    `"params": {"variable_name": "<var_name>", "line_number": <int>}`
3.  `"action": "extract_code_snippet"`: Get source code for a region.
    `"params": {"start_line": <int>, "end_line": <int>}`
4.  `"action": "execute_js_snippet"`: Run a small piece of JS code.
    `"params": {"code_snippet": "<js_code_string>", "context_vars": {...}}`
5.  `"action": "ask_human_for_help"`: Ask for help on complex logic.
    `"params": {"question": "...", "code_context": "..."}`
6.  `"action": "final_answer"`: Provide the final result.
    `"params": {"answer_data": {...}}`

Begin your strategic analysis now.
"""
}

# ----------------- Core Execution Logic -----------------

def get_tokenizer(model: str = "gpt-4"):
    try: return tiktoken.encoding_for_model(model)
    except KeyError: return tiktoken.get_encoding("cl100k_base")

def estimate_tokens(messages: list, tokenizer) -> int:
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(tokenizer.encode(str(value)))
            if key == "name": num_tokens -= 1
    num_tokens += 2
    return num_tokens

def summarize_history_in_chunks(history: list, tokenizer, chunk_size: int = 12000) -> str:
    """Summarizes a long history by processing it in uniform, iterative chunks."""
    logging.info(f"Starting robust rolling summarization for {len(history)} messages.")
    user_goal = history[1]['content']
    history_to_summarize = history[2:]
    current_chunk, rolling_summary, chunk_num = [], "", 1
    
    def process_chunk(chunk_to_process, prev_summary):
        nonlocal chunk_num
        logging.info(f"Summarizing chunk #{chunk_num} ({len(chunk_to_process)} messages)...")
        summary_instruction = f"""You are an expert memory compressor. The user's AI is analyzing a large JS file. The AI's overarching goal is: {user_goal}. A summary of the preceding conversation is: <PREVIOUS_SUMMARY>{prev_summary or "This is the first chunk."}</PREVIOUS_SUMMARY>. Now, summarize the following chunk, integrating key findings into the previous summary to create an updated summary. Focus on the deobfuscation map, variables, and tested hypotheses. <CHUNK_TO_SUMMARIZE>{json.dumps(chunk_to_process, ensure_ascii=False, indent=2)}</CHUNK_TO_SUMMARIZE>"""
        compression_messages = [{"role": "system", "content": summary_instruction}]
        summary_response = client.chat.completions.create(model=CONFIG["summarizer_model"], messages=compression_messages)
        new_summary = summary_response.choices[0].message.content
        logging.info(f"Rolling summary updated. (First 100 chars: {new_summary[:100].replace(chr(10), ' ')}...)")
        chunk_num += 1
        return new_summary

    for message in history_to_summarize:
        current_chunk.append(message)
        # Using a simple length check on the JSON string as a proxy for token count
        if len(json.dumps(current_chunk)) > chunk_size:
            rolling_summary = process_chunk(current_chunk, rolling_summary)
            current_chunk = []
    if current_chunk:
        rolling_summary = process_chunk(current_chunk, rolling_summary)
    logging.info("Rolling summarization complete.")
    return rolling_summary

def run(messages, shared_state, depth=0):
    if depth > CONFIG["max_depth"]: 
        logging.error("Maximum recursion depth reached.")
        return {"error": "Maximum recursion depth reached"}, messages, shared_state
    
    history, tokenizer = copy.deepcopy(messages), get_tokenizer(CONFIG["default_model"])
    
    for i in range(CONFIG["max_rounds_per_run"]):
        logging.info(f"[Depth:{depth}, Round:{i+1}] --- Preparing request...")
        estimated_size = estimate_tokens(history, tokenizer)
        logging.info(f"Estimated token count: {estimated_size}")

        if estimated_size > CONFIG["context_token_threshold"]:
            logging.warning(f"Token count {estimated_size} exceeds threshold {CONFIG['context_token_threshold']}. Initiating memory compression.")
            try:
                final_summary = summarize_history_in_chunks(history, tokenizer)
                shared_state['memory_summary'] = final_summary
                history = [history[0], history[1], {"role": "user", "content": f"<CONTEXT_SUMMARY>\n{final_summary}\n</CONTEXT_SUMMARY>"}]
                logging.info("History successfully compressed.")
            except Exception as e:
                logging.critical(f"FATAL ERROR during memory summarization: {e}", exc_info=True)
                error_log_path = "error_context_log.json"
                logging.info(f"Dumping full context to {error_log_path} for debugging.")
                with open(error_log_path, 'w', encoding='utf-8') as f: json.dump(history, f, indent=2, ensure_ascii=False)
                return {"error": "Failed to summarize history. Context saved to error_context_log.json."}, history, shared_state

        logging.info("Sending request to GPT...")
        response = client.chat.completions.create(model=CONFIG["default_model"], messages=history, response_format={"type": "json_object"})
        msg = response.choices[0].message
        history.append(msg.model_dump())

        try:
            response_json = json.loads(msg.content)
            logging.info(f"AI Action: {response_json.get('action')}")
            thought = response_json.get('thought', '')
            logging.info(f"AI Thought: {thought[:300]}{'...' if len(thought) > 300 else ''}")
        except (json.JSONDecodeError, AttributeError) as e:
            error_feedback = f"Error: Invalid JSON response. You must follow the specified JSON format. Received: {msg.content}"
            logging.warning(f"Invalid JSON response from AI: {msg.content}")
            history.append({"role": "user", "content": error_feedback})
            continue

        final_answer, should_continue = process_ai_response(response_json, history, shared_state, depth)
        if final_answer is not None: return final_answer, history, shared_state
        if not should_continue: return {"error": "Processing failed or action was invalid."}, history, shared_state
        
    logging.warning(f"Max rounds reached ({CONFIG['max_rounds_per_run']}).")
    return {"error": f"Max rounds reached ({CONFIG['max_rounds_per_run']})."}, history, shared_state

def process_ai_response(response_json, history, shared_state, depth):
    action, params = response_json.get("action"), response_json.get("params", {})
    step_id = f"step_{len(shared_state['step_history'])}"
    status, result_data, analyzer = "success", None, shared_state.get('analyzer')

    try:
        if not action: raise ValueError("Action is missing from the AI response.")
        if not analyzer: raise ValueError("JSAnalyzer not initialized in shared state.")

        if action == "find_code_by_keyword":
            keyword = params.get("keyword")
            if not keyword: raise ValueError("`keyword` is required for find_code_by_keyword.")
            result_data = analyzer.find_code_by_keyword(
                keyword, 
                params.get("is_regex", False), 
                params.get("context_lines", 2)
            )
        
        elif action == "get_object_keys":
            var_name, line_num = params.get("variable_name"), params.get("line_number")
            if not all([var_name, line_num]): raise ValueError("`variable_name` and `line_number` are required.")
            result_data = analyzer.get_object_keys(var_name, int(line_num))

        elif action == "extract_code_snippet":
            start, end = params.get("start_line"), params.get("end_line")
            if not all([start, end]): raise ValueError("`start_line` and `end_line` are required.")
            result_data = analyzer.extract_code_snippet(int(start), int(end))

        elif action == "execute_js_snippet":
            code, context = params.get("code_snippet"), params.get("context_vars")
            if not code: raise ValueError("`code_snippet` is required.")
            result_data = analyzer.execute_js_snippet(code, context)

        elif action == "ask_human_for_help":
            question = params.get("question", "I'm stuck. Can you help?")
            code_context = params.get("code_context", "")
            print("\n" + "="*20 + " HUMAN ASSISTANCE REQUIRED " + "="*20)
            print(f"AI's Question: {question}")
            if code_context: print("\n--- Code Context ---\n" + code_context + "\n--------------------")
            result_data = {"human_feedback": input("Your Answer > ")}
            print("="*63 + "\n")

        elif action == "final_answer":
            logging.info("Received final answer from AI.")
            return params.get("answer_data", {}), False
        else: 
            raise ValueError(f"Unknown action: {action}")

    except Exception as e:
        status = "error"
        result_data = {"error_details": str(e)}
        logging.error(f"Error executing action '{action}': {e}", exc_info=True)

    # --- CONTEXT SHIELDING LOGIC ---
    # The full result is always saved in the step history for logging/debugging.
    shared_state['step_history'].append({"id": step_id, "action": action, "params": params, "status": status, "result": result_data})

    feedback_result = result_data
    # If the result from the tool was truncated, create a concise summary for the AI's context.
    if isinstance(result_data, dict) and result_data.get("status") == "truncated":
        logging.info("Result was truncated. Generating a concise feedback summary for the AI to prevent context overflow.")
        # This new object *omits* the verbose 'results' list.
        feedback_result = {
            "status": "truncated",
            "total_matches": result_data.get("total_matches"),
            "suggestion": result_data.get("suggestion", "Results were truncated. Please refine your search.")
        }
    
    # The feedback to the AI now uses the potentially summarized 'feedback_result'.
    feedback = f"""<SYSTEM_FEEDBACK><action_receipt><id>{step_id}</id><action>{action}</action><status>{status}</status><result>{json.dumps(feedback_result, ensure_ascii=False, default=str)}</result></action_receipt></SYSTEM_FEEDBACK>"""
    history.append({"role": "user", "content": feedback})
    return None, True

# ----------------- Command-Line Entry Point -----------------

def main():
    """Main function to run the agent from the command line."""
    print("Welcome to JSDecipher-GPT v8 (True Context Shielding)!")
    print("="*60)
    
    file_path = input("Enter the path to the JavaScript file to analyze: ").strip()
    try:
        analyzer = JSAnalyzer(file_path)
    except (FileNotFoundError, Exception) as e:
        logging.critical(f"Could not initialize analyzer: {e}", exc_info=True)
        sys.exit(1)

    task = input("What is your goal for this analysis? (e.g., 'Find the encryption key'): ")
    shared_state = {'file_path': file_path, 'analyzer': analyzer, 'variables': {}, 'step_history': [], 'memory_summary': None}
    initial_messages = [{"role": "system", "content": PROMPTS["MAIN_SYSTEM_PROMPT"]}, {"role": "user", "content": f"<OVERARCHING_GOAL>{task}</OVERARCHING_GOAL>"}]
    final_result, _, _ = run(initial_messages, shared_state)

    print("\n\n==================== TASK COMPLETE ====================")
    class CustomEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, JSAnalyzer): return f"<JSAnalyzer for {o.file_path}>"
            try: return super().default(o)
            except TypeError: return str(o)
            
    print(json.dumps(final_result, indent=2, ensure_ascii=False, cls=CustomEncoder))
    print("=====================================================")
    # Optionally save the full session log
    session_log_path = "session_log.json"
    logging.info(f"Saving full session state to {session_log_path}")
    with open(session_log_path, 'w', encoding='utf-8') as f:
        json.dump(shared_state, f, indent=2, ensure_ascii=False, cls=CustomEncoder)


if __name__ == "__main__":
    main()

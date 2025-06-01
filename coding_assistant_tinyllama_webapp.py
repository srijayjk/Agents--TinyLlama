import subprocess
import re
from flask import Flask, render_template_string, request, redirect, url_for
from datetime import datetime

app = Flask(__name__)
LOG_FILE = "assistant_log.txt"
history = []  # In-memory interaction memory

# HTML template
HTML_TEMPLATE = """
<!doctype html>
<title>TinyLLaMA Coding Agent</title>
<h2>🤖 TinyLLaMA Web Assistant</h2>
<form method=post>
  <textarea name=prompt rows=4 cols=80 placeholder="Ask me to code something...">{{ prompt or '' }}</textarea><br>
  <input type=submit value="Send">
</form>
{% if response %}
<h3>📥 Response:</h3><pre>{{ response }}</pre>
{% endif %}
{% if code %}
<h3>📦 Code Extracted:</h3><pre>{{ code }}</pre>
{% endif %}
{% if output %}
<h3>▶️ Execution Output:</h3><pre>{{ output }}</pre>
{% endif %}
<hr>
<h3>🧠 Memory</h3>
{% for entry in history %}
  <b>Prompt:</b> {{ entry.prompt }}<br>
  <b>Response:</b><pre>{{ entry.response[:500] }}...</pre>
  {% if entry.execution_result %}<b>Result:</b> {{ entry.execution_result }}<br>{% endif %}
  <hr>
{% endfor %}
"""

def log(msg: str):
    with open(LOG_FILE, "a") as f:
        f.write(f"\n[{datetime.now()}] {msg}\n")

def query_tinyllama(prompt: str) -> str:
    context = summarize_memory(history[-3:])  # Include last 3 turns
    full_prompt = context + "\n\n" + prompt if context else prompt
    result = subprocess.run(
        ['ollama', 'run', 'tinyllama', full_prompt],
        capture_output=True, text=True
    )
    return result.stdout.strip()

def summarize_memory(mem):
    """Simple summarization of past interactions."""
    return "\n".join([f"User: {h['prompt']}\nAssistant: {h['response'][:200]}" for h in mem])

def extract_code_blocks(text: str):
    return re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)

def run_python_code(code: str):
    try:
        exec_globals = {}
        exec(code, exec_globals)
        log("Code executed successfully.")
        return "✅ Code executed successfully."
    except Exception as e:
        error_msg = f"❌ Error: {e}"
        log(error_msg)
        return error_msg

@app.route('/', methods=['GET', 'POST'])
def index():
    prompt = response = code = output = ""
    if request.method == 'POST':
        prompt = request.form['prompt']
        response = query_tinyllama(prompt)
        log(f"Prompt: {prompt}\nResponse: {response}")
        code_blocks = extract_code_blocks(response)
        if code_blocks:
            code = code_blocks[0]
            output = run_python_code(code)
        else:
            output = "⚠️ No valid Python code found."
        history.append({
            "prompt": prompt,
            "response": response,
            "code": code if code_blocks else "",
            "execution_result": output
        })
    return render_template_string(HTML_TEMPLATE,
                                  prompt=prompt,
                                  response=response,
                                  code=code,
                                  output=output,
                                  history=history)

if __name__ == '__main__':
    print("🌐 Running at http://localhost:5050")
    app.run(debug=True, port=5050)


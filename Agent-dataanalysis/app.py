from flask import Flask, request, render_template, send_from_directory
import pandas as pd
import os
import re
import traceback
import matplotlib.pyplot as plt
from langchain_ollama import OllamaLLM

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

llm = OllamaLLM(model="tinyllama")

def extract_code_blocks(text):
    """Extract code between triple backticks"""
    matches = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return matches

def run_code(code, df):
    """Run code and capture stdout/stderr, exposing `df`, `pd`, and `plt`."""
    import io, contextlib, traceback
    output = io.StringIO()
    # Anything you put in this dict will be visible to the exec-ed code
    globals_dict = {
        "df": df,          # ← the DataFrame from the CSV upload
        "pd": pd,          # ← so the LLM can still import if it wants
        "plt": plt         # ← for plots
    }
    try:
        with contextlib.redirect_stdout(output):
            exec(code, globals_dict)
    except Exception:
        output.write(traceback.format_exc())
    return output.getvalue()

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    code_result = ""
    if request.method == "POST":
        prompt = request.form.get("prompt", "")
        file = request.files.get("file")

        if file:
            df = pd.read_csv(file)
            summary = df.describe(include="all").to_string()
            preview = df.head(5).to_string()
            csv_prompt = (
             f"Here is a preview of the dataset (in a DataFrame named `df`):\n\n{preview}"
            f"\n\nSummary stats:\n{summary}"
            "\n\nThe dataset is already loaded into a variable named `df`."
            "\nAvoid reading and using `pd.read_csv()` or referencing files like 'dataset.csv'."
            f"\n\n{prompt}"
            )
            
        else:
            csv_prompt = prompt

        # Ask LLaMA
        llm_response = llm.invoke(csv_prompt)

        # Extract and run code
        code_blocks = extract_code_blocks(llm_response)
        if code_blocks:
            code_result = run_code(code_blocks[0], df)
        else:
            code_result = "(No code found in LLM response)"

        response = llm_response

    return render_template("index.html", response=response, result=code_result)

@app.route("/static/<filename>")
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)


if __name__ == '__main__':
    print("🌐 Running at http://localhost:5050")
    app.run(debug=True, port=5050)
# ==============================================================================
# CASE STUDY: Credit Approval Committee (CAC) Server with Dynamic Scraper & Live Progress
# Converted from Bank_Board_MCP1-3.ipynb (runs locally; requires: pip install gradio groq)
# ==============================================================================
import os
import re
import time
import urllib.request

import gradio as gr
from groq import Groq

# 1. System Prompt & Context Definition
SYSTEM_PROMPT = """
You act as the Senior Credit Officer on the Credit Approval Committee (CAC) for Corporate and Syndicated Lending.
Your role is to assess massive debt facility requests based on macro trends, leverage ratios, and underwriting policy metrics.

You have access to text pulled directly from target web resources provided by the user. Group, read, analyze, and synthesize this content to formulate structural credit mitigants.
"""


# 2. Lightweight Raw HTML Text Scraper Tool
def extract_clean_text_from_url(url):
    try:
        # Define enterprise headers to avoid basic bot blocking
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) CreditCommitteeAgent/1.0'}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=7) as response:
            html = response.read().decode('utf-8', errors='ignore')

            # Remove scripts, style sheets, and grab body text elements
            text = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', html)
            text = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', text)
            text = re.sub(r'<[^>]+>', ' ', text)
            clean_text = re.sub(r'\s+', ' ', text).strip()

            # Return a manageable chunk for the model context window
            return clean_text[:4000]
    except Exception as e:
        return f"[Failed to pull from {url}: {str(e)}]"


# 3. Underwriting Arbitration Engine with Live UI Progress Tracking
def run_cac_session(api_key, model_choice, user_query, url_1, url_2, url_3, progress=gr.Progress()):
    api_key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "ERROR: Please provide a valid Groq API Key."
    if not url_1 and not url_2 and not url_3:
        return "ERROR: Please provide at least one target URL to extract context from."

    try:
        scraped_payload = ""

        # Step 1: Initial Handshake
        progress(0.0, desc="🔄 Establishing Model Context Protocol (MCP) framework...")
        time.sleep(0.5)

        # Step 2: Target URL 1 Extraction
        if url_1:
            progress(0.2, desc="🌐 Requesting target data stream from Source 1...")
            res_1 = extract_clean_text_from_url(url_1)
            scraped_payload += f"\n\n--- Source Context [URL 1: {url_1}] ---\n{res_1}"
            time.sleep(0.5)

        # Step 3: Target URL 2 Extraction
        if url_2:
            progress(0.4, desc="🌐 Processing content matrix from Source 2...")
            res_2 = extract_clean_text_from_url(url_2)
            scraped_payload += f"\n\n--- Source Context [URL 2: {url_2}] ---\n{res_2}"
            time.sleep(0.5)

        # Step 4: Target URL 3 Extraction
        if url_3:
            progress(0.6, desc="🌐 Aggregating metrics from optional Source 3...")
            res_3 = extract_clean_text_from_url(url_3)
            scraped_payload += f"\n\n--- Source Context [URL 3: {url_3}] ---\n{res_3}"
            time.sleep(0.5)

        # Step 5: Packaging Data context Vectors
        progress(0.8, desc="📊 Tokenizing gathered text data arrays into LLM context...")
        full_user_input = f"{user_query}\n\n[Ingested External Material Data]{scraped_payload}"
        time.sleep(0.3)

        # Step 6: Querying the Groq-hosted model
        progress(0.9, desc=f"🧠 Running Wholesale Underwriting Analysis via {model_choice}...")
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_user_input},
            ],
            temperature=0.15,
        )

        progress(1.0, desc="✅ Credit Decision Formulation Complete.")
        return completion.choices[0].message.content

    except Exception as e:
        return f"Error executing committee analysis: {str(e)}"


# 4. Streamlined Gradio UI Frontend
with gr.Blocks() as demo:
    gr.Markdown("## 📈 MCP Notebook 3: Credit Approval Committee (CAC) Intelligence Hub")
    gr.Markdown("**Case Study**: Live data mining and asset verification workspace for multi-million dollar banking operations.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔑 Gateway Configurations")
            api_key_input = gr.Textbox(
                label="Groq API Key", type="password", placeholder="gsk_... (or set GROQ_API_KEY env var)",
                value=os.environ.get("GROQ_API_KEY", ""),
            )
            model_dropdown = gr.Dropdown(choices=["openai/gpt-oss-120b"], value="openai/gpt-oss-120b", label="Model Selection")

            gr.Markdown("### 🌐 Dynamic Research Targets")
            url_input_1 = gr.Textbox(label="Target URL 1 (Required)", placeholder="https://example.com/financials", value="https://wikipedia.org/wiki/Basel_III")
            url_input_2 = gr.Textbox(label="Target URL 2 (Optional)", placeholder="https://example.com/filings")
            url_input_3 = gr.Textbox(label="Target URL 3 (Optional)", placeholder="https://example.com/macro-trends")

        with gr.Column(scale=2):
            gr.Markdown("### 🏛️ Committee Deal Desk")
            query_input = gr.Textbox(
                label="Deal Pitch / Policy Variance Request",
                value="Infrastructure Corp X is requesting a $1.2B facility. Evaluate how their current 1.32x DSCR performance stacks against the regulatory framework rules scraped from our live text sources below. Offer direct remediation suggestions.",
                lines=5,
            )

            submit_btn = gr.Button("Analyze Live URLs & Formulate Decision", variant="primary")

            gr.Markdown("### 📋 Executive Memorandum Output")
            output_display = gr.Markdown(label="Generated Committee Decision")

    # Map click actions and inputs to target function blocks
    submit_btn.click(
        fn=run_cac_session,
        inputs=[api_key_input, model_dropdown, query_input, url_input_1, url_input_2, url_input_3],
        outputs=output_display,
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), share=False, server_port=int(os.environ.get("PORT", 8090)))

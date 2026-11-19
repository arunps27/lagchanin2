# ==============================================================================
# CASE STUDY: Global Executive Compliance & Audit Multi-Framework Registry Engine
# ==============================================================================

import gradio as gr
from groq import Groq
import urllib.request
import re
import time

# 1. Structured System Prompt using the CO-STAR Framework
SYSTEM_PROMPT = """
# CONTEXT
You are the Chief Compliance Officer (CCO) presiding over the Global Compliance and Audit Committee. The committee is auditing organizational technical architectures against various international regulatory frameworks (HIPAA, GDPR, DPDI/DUAA, EU AI Act).

# OBJECTIVE
Evaluate systemic cross-border gaps, compliance liabilities, and technical data exposures based on the provided business topology and live web-scraped framework data snippets.

# STYLE & TONE
Institutional-grade, precise, analytical, and authoritative. Avoid generic advice; refer directly to the scraped registry text assets provided.

# AUDIENCE
Board of Directors, internal auditors, legal counsel, and technical compliance engineers.

# RESPONSE LAYER (DYNAMIC CONDITIONAL FORMATTING CODE)
Depending on the data payload provided below, you must strictly follow one of these two output schemas:

### SCHEMA A: SINGLE SOURCE SELECTED
If only ONE live framework source is provided in the payload:
1. **Target Framework Deep-Dive**: Comprehensive breakdown of compliance vectors for that specific regulation.
2. **Citations & References**: Explicitly list sections or definitions extracted from the source material.
3. **Operational Gaps & Remedies**: Concrete actions to close vulnerabilities under this single mandate.

### SCHEMA B: MULTIPLE SOURCES SELECTED
If TWO OR MORE live framework sources are provided in the payload:
1. **Independent Regulatory Reports**: Generate a completely distinct, standalone compliance report for EACH selected source independently (including specific citations for each).
2. **Cross-Framework Comparison Table**: A clear Markdown comparison table detailing key regulatory overlaps, conflicts, enforcement penalty tiers, and data boundaries.
3. **Unified Conclusion & Executive Recommendation**: A single synthesized strategic direction to reconcile all selected frameworks concurrently.
"""

# 2. Automated Web Scraping Sub-System
def extract_clean_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ComplianceAuditAgent/1.0'}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=8) as response:
            html = response.read().decode('utf-8', errors='ignore')

            # Fast Regex Strip for HTML elements
            text = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', html)
            text = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', text)
            text = re.sub(r'<[^>]+>', ' ', text)
            clean_text = re.sub(r'\s+', ' ', text).strip()

            return clean_text[:3500]
    except Exception as e:
        return f"[Registry Scrape Failure for {url}: {str(e)}]"

# 3. Dynamic Multi-Source Compliance Auditor Engine
def run_compliance_audit(api_key, model_choice, business_case, frameworks, progress=gr.Progress()):
    if not api_key:
        return "ERROR: Please provide a valid Groq API Key."
    if not frameworks:
        return "ERROR: Please select at least one regulatory framework source from the checkbox matrix."

    try:
        registry_map = {
            "HIPAA (Healthcare Data Privacy)": "https://en.wikipedia.org/wiki/Health_Insurance_Portability_and_Accountability_Act",
            "GDPR (European Union Privacy Standard)": "https://en.wikipedia.org/wiki/General_Data_Protection_Regulation",
            "DPDI / DUAA (UK Regulatory Shift)": "https://ico.org.uk/about-the-ico/what-we-do/legislation-we-cover/data-use-and-access-act-2025/",
            "EU AI Act (Risk-Based AI Governance)": "https://en.wikipedia.org/wiki/Artificial_Intelligence_Act"
        }

        compiled_scraped_context = ""
        total_steps = len(frameworks)

        progress(0.0, desc="🔄 Initializing MCP Compliance Auditor Handshake...")
        time.sleep(0.4)

        # Scrape checked frameworks
        for i, target_framework in enumerate(frameworks):
            step_ratio = (i + 1) / (total_steps + 2)
            url_target = registry_map[target_framework]

            progress(step_ratio, desc=f"🌐 Scraping Registry: {target_framework}...")
            scraped_text = extract_clean_text_from_url(url_target)

            compiled_scraped_context += f"\n\n--- LIVE FRAMEWORK REFERENCE MATERIAL: {target_framework} ---\nSource URL: {url_target}\nContent Snippet:\n{scraped_text}\n"
            time.sleep(0.5)

        progress((total_steps + 1) / (total_steps + 2), desc="📊 Building global cross-reference matrix vectors...")

        # Explicit notification block to instruct the LLM on which Schema track to follow
        schema_instruction = "[METADATA CONTROL: SINGLE SOURCE SELECTION PROCESSED. APPLY SCHEMA A.]" if total_steps == 1 else f"[METADATA CONTROL: MULTIPLE SOURCES ({total_steps}) DETECTED. APPLY SCHEMA B.]"

        full_llm_payload = f"{schema_instruction}\n\nUser Business Topology Memo:\n{business_case}\n\n[Scraped Material Assets]\n{compiled_scraped_context}"
        time.sleep(0.4)

        progress(0.95, desc="🧠 Compiling Comprehensive Report via Llama-3.3-70b...")
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_llm_payload}
            ],
            temperature=0.1
        )

        progress(1.0, desc="✅ Comprehensive Regulatory Governance Audit Compiled.")
        return completion.choices[0].message.content

    except Exception as e:
        return f"Error executing comprehensive audit: {str(e)}"

# 4. Gradio Frontend Interface
with gr.Blocks() as demo:
    gr.Markdown("## ⚖️ MCP Notebook 4: Executive Compliance & Audit Committee Server")
    gr.Markdown("**Case Study**: Global multi-framework audit engine tracking cross-border data constraints and AI system restrictions.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔑 Gateway Credentials")
            api_key_input = gr.Textbox(label="Groq API Key", type="password", placeholder="gsk_...")
            model_dropdown = gr.Dropdown(choices=["openai/gpt-oss-120b"], value="openai/gpt-oss-120b", label="Model Routing")

            gr.Markdown("### 🗂️ Registry Source Selection")
            framework_selector = gr.CheckboxGroup(
                choices=[
                    "HIPAA (Healthcare Data Privacy)",
                    "GDPR (European Union Privacy Standard)",
                    "DPDI / DUAA (UK Regulatory Shift)",
                    "EU AI Act (Risk-Based AI Governance)"
                ],
                label="Target Regulations (Select 1 for Schema A, Multiple for Schema B)",
                value=["GDPR (European Union Privacy Standard)", "EU AI Act (Risk-Based AI Governance)"]
            )

        with gr.Column(scale=2):
            gr.Markdown("### 🏛️ Audit Investigation Workspace")
            case_input = gr.Textbox(
                label="Organizational Operations & Systems Topology Memo",
                value="An international fintech institution is processing medical payment profiles using automated predictive models. They handle data across both London and Frankfurt offices. Assess the data protection exposures, cookie/ADM changes under the UK Data Use and Access Act, and evaluate whether the system crosses into High-Risk boundaries under the EU AI Act framework rules.",
                lines=6
            )

            submit_btn = gr.Button("Fetch Frameworks & Generate Report", variant="primary")

            gr.Markdown("### 📋 Executive Memorandum Output")
            output_display = gr.Markdown(label="Generated Governance Audit Report")

    submit_btn.click(
        fn=run_compliance_audit,
        inputs=[api_key_input, model_dropdown, case_input, framework_selector],
        outputs=output_display
    )

if __name__ == "__main__":
    demo.launch(share=False, server_port=None, theme=gr.themes.Soft())
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Load Model & Tokenizer
# -------------------------
model_name = "ibm-granite/granite-3.2-2b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# Response Generator
# -------------------------
def generate_response(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    return response

# -------------------------
# Task 1: City Analysis
# -------------------------
def city_analysis(city_name):
    prompt = f"""
Provide a detailed analysis of {city_name} including:
1. Crime Index and safety statistics
2. Accident rates and traffic safety information
3. Overall safety assessment

City: {city_name}
Analysis:
"""
    return generate_response(prompt, max_length=1000)

# -------------------------
# Task 2: Citizen Interaction
# -------------------------
def citizen_interaction(query):
    prompt = f"""
As a government assistant, provide accurate and helpful information about the following citizen query related to public services, government policies, or civic issues:

Query: {query}
Response:
"""
    return generate_response(prompt, max_length=1000)

# -------------------------
# Login Function
# -------------------------
def login_user(name, city, mobile):
    if not name or not city or not mobile:
        return gr.update(visible=True), gr.update(visible=False), "⚠️ Please fill all details!"
    else:
        welcome_msg = f"✅ Welcome {name} from {city}! (Mobile: {mobile})"
        return gr.update(visible=False), gr.update(visible=True), welcome_msg

# -------------------------
# Gradio UI with Login
# -------------------------
with gr.Blocks() as app:
    gr.Markdown("# 🔐 Citizen-AI Login Page")

    # --- Login Page ---
    with gr.Group(visible=True) as login_page:
        name_in = gr.Textbox(label="Name", placeholder="Enter your name")
        city_in = gr.Textbox(label="City", placeholder="Enter your city")
        mobile_in = gr.Textbox(label="Mobile Number", placeholder="Enter your mobile number")
        login_btn = gr.Button("Login")
        login_status = gr.Markdown("")

    # --- Main App Page (Initially Hidden) ---
    with gr.Group(visible=False) as main_page:
        gr.Markdown("## 🏙️ Citizen-AI: City Analysis & Public Services Assistant")

        with gr.Tabs():
            with gr.TabItem("City Analysis"):
                city_input = gr.Textbox(label="Enter City Name", placeholder="e.g., New York, London, Mumbai...")
                city_output = gr.Textbox(label="City Analysis Result", lines=15)
                gr.Button("Analyze City").click(city_analysis, inputs=city_input, outputs=city_output)

            with gr.TabItem("Citizen Services"):
                citizen_query = gr.Textbox(label="Your Query", placeholder="Ask about public services, government policies, civic issues...", lines=4)
                citizen_output = gr.Textbox(label="Government Response", lines=15)
                gr.Button("Get Info").click(citizen_interaction, inputs=citizen_query, outputs=citizen_output)

    # Button Action → Switch Pages
    login_btn.click(
        fn=login_user,
        inputs=[name_in, city_in, mobile_in],
        outputs=[login_page, main_page, login_status]
    )

app.launch()

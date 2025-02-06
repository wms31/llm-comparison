import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Streamlit UI
st.title("Optimize Prompts for Mistral/GPT-2 LLMs and Compare Results")


# Define Hugging Face API
API_URL = "https://api-inference.huggingface.co/models/"
HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

# Define LLMs to test
models = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "gpt-2": "openai-community/gpt2"
}

# Define Prompts
base_prompt = st.text_area("Enter your baseline prompt:")
few_shot_prompt = f"{base_prompt} Here are some examples: ..."
cot_prompt = f"Think step by step: {base_prompt}"
role_based_prompt = f"Act as an expert: {base_prompt}"

# Query Hugging Face API
def query_llm(model, prompt):
    payload = {"inputs": prompt}
    response = requests.post(API_URL + model, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    return "Error fetching response"

# Display Outputs and Comparison
results = []
if st.button("Run Comparison"):
    for model_name, model_id in models.items():
        for prompt_type, prompt in zip(["Baseline", "Few-shot", "CoT", "Role-based"], [base_prompt, few_shot_prompt, cot_prompt, role_based_prompt]):
            output = query_llm(model_id, prompt)
            results.append([model_name, prompt_type, len(output), output])  # Store output text
    df = pd.DataFrame(results, columns=["Model", "Prompt Type", "Output Length", "Generated Output"])
    st.write(df)
    
    # Visualization
    st.subheader("Performance Comparison")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="Model", y="Output Length", hue="Prompt Type", ax=ax)
    ax.set_ylabel("Generated Output Length")
    ax.set_title("Comparison of Different LLMs Based on Output Length")
    st.pyplot(fig)
    
    st.markdown("""
    ### Explanation of the Graph
    - The bar chart compares different LLMs based on the length of the generated text.
    - Different prompt strategies (Baseline, Few-shot, CoT, Role-based) may affect output size.
    - Longer outputs may indicate richer responses, but not necessarily better accuracy.
                
    """)

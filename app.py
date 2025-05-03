import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import requests
st.set_option('client.showErrorDetails', False)


genai.configure(api_key="api key")#api key....i have to put this in env

st.set_page_config(page_title="Gemma Fine-Tuner", layout="wide")


st.title("🤖🛠️Gemma Fine-Tuning Interface")
st.markdown("Prototype UI for fine-tuning Gemma models with PEFT (LoRA, QLoRA) on Google Colab")


st.header("🗂️  Dataset")
dataset_source = st.radio("Choose Dataset Source", ["Upload File", "From URL"])
df = None  # placeholder for the dataframe

if dataset_source == "Upload File":
    uploaded_file = st.file_uploader("Upload your dataset (CSV, JSON, etc.)", type=["csv", "json", "txt"])
    if uploaded_file:
        try:
            file_content = uploaded_file.read()
            data_list = json.loads(file_content) 

            df = pd.DataFrame(data_list)
            df.columns = df.columns.astype(str)
            df = df.astype(str)  
            data_dict = dict(zip(df["input"], df["output"]))
            gem_model = genai.GenerativeModel("gemini-2.0-flash")
            response= gem_model.generate_content(f'this is a data set for fine tuning gemma please remove impurities and improve data for gemma fine tuning and return only data as json, if nothing to fix then return same data:{data_dict}')
            final_data=response.text[7:-4]
            try:
                final_dict = json.loads(final_data)  # parse JSON

                df_final = pd.DataFrame([{"input": k, "output": v} for k, v in final_dict.items()])
            except Exception as e:
                st.error(f'error occured:{e}')




        except Exception as e:
            st.error(f"Error reading file: {e}")
else:
    dataset_url = st.text_input("Enter dataset URL")
    if dataset_url:
        try:
            df = pd.read_csv(dataset_url)
        except Exception as e:
            st.error(f"Error loading from URL: {e}")

if df is not None or 'df_final' in locals():
    tab1, tab2 = st.tabs(["📊 Preview", "🧪 Info"])
    
    with tab1:
        if df is not None:
            st.subheader("Preview of Uploaded Data")
            df = df.astype(str)
            st.dataframe(df.head(50), use_container_width=True)
        
        if 'df_final' in locals():
            df = df.reset_index(drop=True)
            df_final = df_final.reset_index(drop=True)

            def highlight_changes(x):
                df1 = df.copy()
                df2 = df_final.copy()
                styles = pd.DataFrame('', index=x.index, columns=x.columns)
                
                for col in df1.columns:
                    if col in df2.columns:
                        mask = df1[col] != df2[col]
                        styles.loc[mask, col] = 'background-color: red'  
                return styles

            # highlighting
            st.subheader('suggested changes from LLM')
            styled_df_final = df_final.style.apply(highlight_changes, axis=None)
            st.dataframe(
    styled_df_final,
    use_container_width=True,
)
    
    with tab2:
        if df is not None:
            st.subheader("Dataset Summary (Uploaded Data)")
            st.write("**Shape:**", df.shape)
            st.write("**Columns:**", df.columns.tolist())
            st.write(df.describe())
        
        if 'df_final' in locals():
            st.subheader("Dataset Summary (LLM Cleaned Data)")
            st.write("**Shape:**", df_final.shape)
            st.write("**Columns:**", df_final.columns.tolist())

st.divider()


st.header("🔧 Fine-Tuning Configuration")

col1, col2 = st.columns(2)

with col1:
    base_model = st.selectbox("Base Model", ["Gemma 2B"])
    method = st.selectbox("Fine-Tuning Method", ["LoRA", "QLoRA"])
    epochs = st.slider("Epochs", 1, 10, 3)
    batch_size = st.number_input("Batch Size", 1, 128, 2)
    max_steps = st.number_input("Max Steps (0 = auto)", 0, 100000, 0)

with col2:
    learning_rate = st.number_input("Learning Rate", 0.00001, 0.01, 0.0002, format="%.5f")
    weight_decay = st.number_input("Weight Decay", 0.0, 0.5, 0.01)
    dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.1)
    output_dir = st.text_input("Output Directory Name", "gemma_finetuned")

st.divider()

#connect to backend
st.header("⚙️ Backend Settings")
backend_url = st.text_input("Enter your backend endpoint", placeholder="https://xxxx.ngrok.xyz")






if st.button("🚀 Start Fine-Tuning"):
    if backend_url.strip() == "":
        st.error("Please enter the backend URL.")
    else:
        # build the payload
        config_payload = {
            "dataset": df_final.to_dict(orient="records") if 'df_final' in locals() else df.to_dict(orient="records"),
            "params": {
                "base_model": base_model,
                "method": method,
                "epochs": epochs,
                "batch_size": batch_size,
                "max_steps": max_steps,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "output_dir": output_dir,
            }
        }
        with st.spinner("⏳ Fine tuning in progress... this will take time according to how big is dataset"):
            try:
                response = requests.post(f"{backend_url}/train", json=config_payload)
                if response.status_code == 200:
                    st.success("✅ fine tuning finished!")
                    st.header("📜 Training Logs / Status")
                    st.code("Logs will appear here once training starts...", language="bash")


                    with st.expander("📊 Final Metrics"):
                        st.metric("Loss", "working on it")
                        st.metric("Accuracy", "working")
                    st.json(response.json())
                    download_url = response.json().get("download_url")
                    st.markdown(f"[⬇️ Download Model]({backend_url}/download)")
                else:
                    st.error(f"🚫 Backend error: {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"⚠️ Failed to connect to backend: {e}")

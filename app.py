import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image
from streamlit_ketcher import st_ketcher

# Page Configuration
st.set_page_config(
    page_title="ScreenBetes",
    page_icon="🧬",
    layout="centered"
)

# ============================================
# Function to generate Morgan Fingerprint
# ============================================
def fp_morgan(smiles, radius=2, nBits=1024):
    """
    Generates Morgan fingerprint (ECFP4 like) with radius 2 and 1024 bits.
    Returns a numpy array.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        
        # Generate bit vector
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        
        # Convert to numpy array
        arr = np.zeros((1,), dtype=int)
        DataStructs = AllChem.DataStructs
        DataStructs.ConvertToNumpyArray(fp_bit, arr)
        
        return arr, mol
    except Exception as e:
        # st.error(f"Error processing SMILES: {e}")
        return None, None

# ============================================
# Load Models
# ============================================
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "PTP1B": "xgboost_ptp1b_final.pkl",
        "SLTG2": "qsar_model_SLTG2.pkl",
        "DPP4": "model_ddp4_rf.pkl"
    }
    
    for name, filename in model_files.items():
        try:
            # Try loading local model from qsar_app/ folder (dev env)
            path = f"qsar_app/{filename}"
            models[name] = joblib.load(path)
        except FileNotFoundError:
            try:
                # Fallback (deployed env)
                 models[name] = joblib.load(filename)
            except:
                 models[name] = None
    return models

models = load_models()

# ============================================
# Custom CSS for Sophisticated UI
# ============================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    /* Custom Button Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #ce93d8 0%, #ab47bc 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #ba68c8 0%, #9c27b0 100%);
    }

    /* Result Card Styling */
    .result-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #5d4037;
    }
    
    .metric-label {
        color: #8d6e63;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #fff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        padding: 0 20px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #f3e5f5;
        color: #8e24aa;
    }

</style>
""", unsafe_allow_html=True)


# ============================================
# Interface
# ============================================

# Layout: Image (Left/Center) - Spacer - Team (Right)
col_title, col_spacer, col_team = st.columns([6, 3, 2])

with col_title:
    # Use absolute path to ensure image is found
    img_path = os.path.join(os.path.dirname(__file__), 'header.png')
    st.image(img_path, use_container_width=True)

with col_team:
    # Popover discreto
    with st.popover("👥 Team"):
        st.markdown("""
        **Authors & Contributors:**
        
        *   Carlos S. H. Shiraishi
        *   Gabriel Grechuk
        *   Eduardo Carvalho Nunes
        *   Marcus T. Scotti
        *   Eugene Muratov
        *   Miguel A. Prieto
        *   Sandrina A. Heleno
        *   Rui M. V. Abreu
        """)

st.markdown("""
Predicting biological activity against two targets:

1.  **PTP1B (Protein Tyrosine Phosphatase 1B)**
    *   A negative regulator of insulin signaling. Inhibition of PTP1B improves insulin sensitivity, making it a target for Diabetes (Type 2) and Obesity.
    
2.  **SLTG2 (Sodium-Glucose Transport Protein 2)**
    *   Responsible for reabsorbing glucose in the kidneys. Inhibiting SGLT2 allows glucose to be excreted in urine, lowering blood sugar levels.

3.  **DPP4 (Dipeptidyl peptidase-4)**
    *   An enzyme that degrades incretins (GLP-1). DPP4 inhibitors prolong the activity of incretins, stimulating insulin secretion and regulating blood glucose.

**Input**: Molecule SMILES.
""")

# Check generic model status
if all(m is None for m in models.values()):
    st.error("❌ CRITICAL: No models found!")
else:
    loaded_names = [name for name, m in models.items() if m is not None]
    # Custom styled message (Light Purple)
    st.markdown(f"""
    <div style="
        background-color: #f3e5f5;
        color: #4a148c;
        padding: 12px;
        border-radius: 12px;
        font-weight: 600;
        border: 1px solid #e1bee7;
        margin-bottom: 20px;
    ">
        ✅ Loaded models: {', '.join(loaded_names)}
    </div>
    """, unsafe_allow_html=True)

# Tabs for Prediction Modes
tab1, tab2 = st.tabs(["🧪 Single Prediction", "📦 Batch Prediction"])

# --- TAB 1: SINGLE ---
with tab1:
    st.header("Single Prediction")
    
    # Checkbox to enable drawing
    use_drawer = st.checkbox("🎨 Draw Molecule")
    
    smiles_input = ""
    
    if use_drawer:
        # Columns layout to control width (Horizontal)
        # c1 and c3 are margins, c2 is content
        c1, c2, c3 = st.columns([1, 6, 1]) 
        with c2:
            # Default initial value can be empty or a simple molecule
            smiles_input = st_ketcher(key="ketcher_input", height=400)
            st.info(f"SMILES generated from drawing: `{smiles_input}`")
    else:
        smiles_input = st.text_input("Enter Molecule SMILES:", placeholder="Ex: CC(=O)Oc1ccccc1C(=O)O")

    if st.button("🔍 Run Prediction"):
        if not smiles_input:
            st.warning("Please enter a valid SMILES code or draw a molecule.")
        else:
            with st.spinner("Processing molecule..."):
                # Generate features
                features, mol = fp_morgan(smiles_input, radius=2, nBits=1024)
                
                if features is None:
                    st.error("❌ Invalid SMILES! Could not generate molecular structure.")
                else:
                    # Show Structure
                    c_img, _ = st.columns([1, 2])
                    with c_img:
                        st.image(Draw.MolToImage(mol, size=(300, 300)), caption="Structure")

                    st.divider()
                    st.subheader("Results")

                    # Conversion
                    fp_list = [int(x) for x in features]
                    X_input = np.array([fp_list])

                    # Display results for each model
                    # Create generic columns dynamically
                    cols = st.columns(len(models))
                    
                    for idx, (target_name, model) in enumerate(models.items()):
                        with cols[idx]:
                            if model is None:
                                st.error(f"{target_name}: Model not loaded.")
                            else:
                                try:
                                    proba = model.predict_proba(X_input)[:, 1][0]
                                    classe = model.predict(X_input)[0]
                                    
                                    # Logic for color/label
                                    label = "ACTIVE" if classe == 1 else "INACTIVE"
                                    # Choose color for the badge/text
                                    color_style = "color: #2e7d32;" if classe == 1 else "color: #c62828;" # Green/Red matches
                                    bg_badge = "background-color: #e8f5e9;" if classe == 1 else "background-color: #ffebee;"
                                    
                                    # HTML Card
                                    st.markdown(f"""
                                    <div class="result-card">
                                        <h3 style="margin:0; color: #4a148c;">{target_name}</h3>
                                        <hr style="margin: 10px 0; border: 0; border-top: 1px solid #eee;">
                                        <div class="metric-label">Probability</div>
                                        <div class="metric-value">{proba:.1%}</div>
                                        <div style="margin-top: 15px;">
                                            <span style="
                                                {bg_badge}
                                                {color_style}
                                                padding: 5px 12px;
                                                border-radius: 20px;
                                                font-weight: bold;
                                                font-size: 0.9rem;
                                            ">
                                                {label}
                                            </span>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                except Exception as e:
                                    st.error(f"Error: {e}")


# --- TAB 2: BATCH ---
with tab2:
    st.header("Batch Prediction")
    st.markdown("Enter a list of SMILES (one per line) or upload a CSV/TXT file.")
    
    input_method = st.radio("Input Method:", ["Text (List)", "File Upload"])
    
    smiles_list = []
    
    if input_method == "Text (List)":
        text_input = st.text_area("Paste SMILES here (one per line):", height=200)
        if text_input:
            smiles_list = [s.strip() for s in text_input.split('\n') if s.strip()]
            
    else:
        uploaded_file = st.file_uploader("Choose a CSV or TXT file", type=["csv", "txt"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_upload = pd.read_csv(uploaded_file)
                    # Try to find SMILES column
                    potential_cols = [c for c in df_upload.columns if "smile" in c.lower()]
                    if potential_cols:
                        st.info(f"Using column: {potential_cols[0]}")
                        smiles_list = df_upload[potential_cols[0]].dropna().astype(str).tolist()
                    else:
                        st.warning("Could not identify a 'SMILES' column. Using the first column.")
                        smiles_list = df_upload.iloc[:, 0].dropna().astype(str).tolist()
                else: # TXT
                    stringio = uploaded_file.getvalue().decode("utf-8")
                    smiles_list = [s.strip() for s in stringio.split('\n') if s.strip()]
            except Exception as e:
                st.error(f"Error reading file: {e}")

    if st.button("🚀 Process Batch"):
        if not smiles_list:
            st.warning("The SMILES list is empty.")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total = len(smiles_list)
            
            for i, smi in enumerate(smiles_list):
                # Update progress
                progress_bar.progress((i + 1) / total)
                status_text.text(f"Processing {i+1}/{total}...")
                
                # Predict
                features, _ = fp_morgan(smi, radius=2, nBits=1024)
                
                row_result = {"SMILES": smi}
                
                if features is not None:
                    fp_list = [int(x) for x in features]
                    X_input = np.array([fp_list])
                    
                    for target_name, model in models.items():
                        if model:
                            try:
                                prob = model.predict_proba(X_input)[:, 1][0]
                                cls = model.predict(X_input)[0]
                                row_result[f"{target_name}_Prob"] = prob
                                row_result[f"{target_name}_Class"] = "ACTIVE" if cls == 1 else "INACTIVE"
                            except:
                                row_result[f"{target_name}_Prob"] = None
                                row_result[f"{target_name}_Class"] = "ERROR"
                        else:
                             row_result[f"{target_name}_Class"] = "MODEL NOT LOADED"
                else:
                    for target_name in models.keys():
                        row_result[f"{target_name}_Prob"] = None
                        row_result[f"{target_name}_Class"] = "INVALID SMILES"
                
                results.append(row_result)
            
            status_text.text("Done!")
            
            if results:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results)
                
                # Download
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="multi_target_predictions.csv",
                    mime="text/csv",
                )



# ============================================
# Footer: Visitor Analytics (Discrete)
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin-top: 20px; opacity: 0.7;">
    <p style="font-size: 0.8rem; color: #aaa; margin-bottom: 5px;">Visitor Stats</p>
    <a href="https://info.flagcounter.com/vcM0">
        <img src="https://s11.flagcounter.com/count2/vcM0/bg_ffffff/txt_000000/border_CCCCCC/columns_2/maxflags_10/viewers_0/labels_0/pageviews_0/flags_0/percent_0/" 
             alt="Flag Counter" border="0" style="height: 40px;">
    </a>
</div>
""", unsafe_allow_html=True)



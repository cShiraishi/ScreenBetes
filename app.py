import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image

# Configuração da Página
st.set_page_config(
    page_title="PTP1B QSAR Predictor",
    page_icon="🧬",
    layout="centered"
)

# ============================================
# Função para gerar fingerprint Morgan
# ============================================
def fp_morgan(smiles, radius=2, nBits=1024):
    """
    Gera o fingerprint Morgan (ECFP4 like) com raio 2 e 1024 bits.
    Retorna um numpy array.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        
        # Gerar bit vector
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        
        # Converter para numpy array
        arr = np.zeros((1,), dtype=int)
        DataStructs = AllChem.DataStructs
        DataStructs.ConvertToNumpyArray(fp_bit, arr)
        
        return arr, mol
    except Exception as e:
        st.error(f"Erro ao processar SMILES: {e}")
        return None, None

# ============================================
# Carregar Modelo
# ============================================
@st.cache_resource
def load_model():
    try:
        # Tenta carregar o modelo local
        model = joblib.load("qsar_app/xgboost_ptp1b_final.pkl")
        return model
    except FileNotFoundError:
        # Fallback se estiver rodando dentro da pasta qsar_app
        try:
             model = joblib.load("xgboost_ptp1b_final.pkl")
             return model
        except:
             return None

model = load_model()

# ============================================
# Interface
# ============================================
st.title("🧬 Preditor de Bioatividade PTP1B")
st.markdown("""
Este aplicativo utiliza um modelo **XGBoost** para prever a atividade biológica de compostos químicos 
contra a enzima **PTP1B** (Protein Tyrosine Phosphatase 1B), um alvo importante para Diabetes e Obesidade.

**Entrada**: Código SMILES da molécula.
""")

if model is None:
    st.error("❌ Erro CRÍTICO: Modelo `xgboost_ptp1b_final.pkl` não encontrado!")
else:
    st.success(f"✅ Modelo carregado com sucesso!")


# ============================================
# Interface
# ============================================
st.title("🧬 Preditor de Bioatividade PTP1B")
st.markdown("""
Este aplicativo utiliza um modelo **XGBoost** para prever a atividade biológica de compostos químicos 
contra a enzima **PTP1B**.
""")

if model is None:
    st.error("❌ Erro CRÍTICO: Modelo `xgboost_ptp1b_final.pkl` não encontrado!")
else:
    st.success(f"✅ Modelo carregado com sucesso!")

# Tabs para Modos de Predição
tab1, tab2 = st.tabs(["🧪 Predição Individual", "📦 Predição em Lote"])

# --- TAB 1: INDIVIDUAL ---
with tab1:
    st.header("Predição Individual")
    smiles_input = st.text_input("Insira o SMILES da molécula:", placeholder="Ex: CC(=O)Oc1ccccc1C(=O)O")

    if st.button("🔍 Realizar Predição Individual"):
        if not smiles_input:
            st.warning("Por favor, insira um código SMILES válido.")
        else:
            with st.spinner("Processando molécula..."):
                # Gerar features
                features, mol = fp_morgan(smiles_input, radius=2, nBits=1024)
                
                if features is None:
                    st.error("❌ SMILES inválido! Não foi possível gerar a estrutura molecular.")
                else:
                    col_img, col_res = st.columns([1, 2])
                    
                    with col_img:
                        st.subheader("Estrutura")
                        img = Draw.MolToImage(mol, size=(300, 300))
                        st.image(img)
                    
                    with col_res:
                        st.subheader("Resultado")
                        
                        # Conversão e Predição
                        fp_list = [int(x) for x in features]
                        X_input = np.array([fp_list])
                        
                        try:
                            proba = model.predict_proba(X_input)[:, 1][0]
                            classe = model.predict(X_input)[0]
                            
                            st.metric("Probabilidade de Atividade", f"{proba:.2%}")
                            
                            label = "ATIVO" if classe == 1 else "INATIVO"
                            color = "green" if classe == 1 else "red"
                            st.markdown(f"### Classificação: **:{color}[{label}]**")
                            
                            if classe == 1:
                                st.balloons()
                                st.success("🎉 Molécula Promissora!")
                            else:
                                st.info("Baixa probabilidade de atividade.")
                                
                        except Exception as e:
                            st.error(f"Erro na predição: {e}")

# --- TAB 2: LOTE ---
with tab2:
    st.header("Predição em Lote")
    st.markdown("Insira uma lista de SMILES (um por linha) ou faça upload de um arquivo CSV/TXT.")
    
    input_method = st.radio("Método de Entrada:", ["Texto (Lista)", "Upload de Arquivo"])
    
    smiles_list = []
    
    if input_method == "Texto (Lista)":
        text_input = st.text_area("Cole os SMILES aqui (um por linha):", height=200)
        if text_input:
            smiles_list = [s.strip() for s in text_input.split('\n') if s.strip()]
            
    else:
        uploaded_file = st.file_uploader("Escolha um arquivo CSV ou TXT", type=["csv", "txt"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_upload = pd.read_csv(uploaded_file)
                    # Tenta encontrar coluna de SMILES
                    potential_cols = [c for c in df_upload.columns if "smile" in c.lower()]
                    if potential_cols:
                        st.info(f"Usando coluna: {potential_cols[0]}")
                        smiles_list = df_upload[potential_cols[0]].dropna().astype(str).tolist()
                    else:
                        st.warning("Não foi possível identificar uma coluna de 'SMILES'. Usando a primeira coluna.")
                        smiles_list = df_upload.iloc[:, 0].dropna().astype(str).tolist()
                else: # TXT
                    stringio = uploaded_file.getvalue().decode("utf-8")
                    smiles_list = [s.strip() for s in stringio.split('\n') if s.strip()]
            except Exception as e:
                st.error(f"Erro ao ler arquivo: {e}")

    if st.button("🚀 Processar Lote"):
        if not smiles_list:
            st.warning("A lista de SMILES está vazia.")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total = len(smiles_list)
            
            for i, smi in enumerate(smiles_list):
                # Update progress
                progress_bar.progress((i + 1) / total)
                status_text.text(f"Processando {i+1}/{total}...")
                
                # Predict
                features, _ = fp_morgan(smi, radius=2, nBits=1024)
                if features is not None:
                    fp_list = [int(x) for x in features]
                    X_input = np.array([fp_list])
                    
                    try:
                        prob = model.predict_proba(X_input)[:, 1][0]
                        cls = model.predict(X_input)[0]
                        results.append({
                            "SMILES": smi,
                            "Probabilidade": prob,
                            "Classe": "ATIVO" if cls == 1 else "INATIVO",
                            "Classe_Int": int(cls)
                        })
                    except:
                        results.append({"SMILES": smi, "Probabilidade": None, "Classe": "ERRO"})
                else:
                    results.append({"SMILES": smi, "Probabilidade": None, "Classe": "SMILES INVÁLIDO"})
            
            status_text.text("Concluído!")
            
            if results:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results)
                
                # Download
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Baixar Resultados (CSV)",
                    data=csv,
                    file_name="predicoes_qsar.csv",
                    mime="text/csv",
                )
                
                # Summary metrics
                n_ativos = df_results[df_results["Classe"] == "ATIVO"].shape[0]
                st.metric("Total de Ativos Encontrados", n_ativos)

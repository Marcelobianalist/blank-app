import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time # Importamos time para ver los tiempos

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Asistente de Codificación CIE-10",
    page_icon="🩺",
    layout="wide"
)

# --- Funciones de Carga y Procesamiento ---

# --- PUNTO DE CONTROL 1: El script ha comenzado ---
st.info("Iniciando aplicación... Por favor espere.")

@st.cache_resource
def load_model():
    st.info("Paso 1/4: Cargando modelo de IA (puede tardar si es la primera vez)...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    st.success("Paso 1/4: ¡Modelo de IA cargado en memoria!")
    return model

@st.cache_data
def load_and_prepare_data():
    st.info("Paso 2/4: Descargando catálogo CIE-10 desde la web...")
    DATA_URL = "https://raw.githubusercontent.com/gpalacin/misc/main/datasets/cie10_codigos_diagnosticos.csv"
    try:
        df = pd.read_csv(DATA_URL)
    except Exception as e:
        st.error(f"Error al cargar los datos desde la fuente online: {e}")
        st.stop()
    
    df.dropna(subset=['code', 'description'], inplace=True)
    df = df[df['description'].str.strip() != '']
    df['code_4d'] = df['code'].str.replace('.', '', regex=False)
    df['code_4d'] = df['code_4d'].apply(lambda x: x.ljust(4, 'X') if len(x) == 3 else x).str.slice(0, 4)
    
    st.success("Paso 2/4: ¡Catálogo CIE-10 descargado y procesado!")
    return df

@st.cache_data
def create_embeddings(_model, descriptions):
    st.info("Paso 3/4: Creando embeddings de IA (este es el paso más largo, puede tardar varios minutos)...")
    start_time = time.time()
    
    # Usaremos un placeholder para mostrar el progreso en la app
    progress_bar = st.progress(0, text="Procesando descripciones...")
    
    embeddings = _model.encode(descriptions, convert_to_tensor=True, show_progress_bar=False) # show_progress_bar=False para no duplicar en terminal
    
    # Actualizamos la barra de progreso manualmente para dar feedback visual
    progress_bar.progress(100, text="Embeddings creados.")
    
    end_time = time.time()
    st.success(f"Paso 3/4: ¡Embeddings creados en {end_time - start_time:.2f} segundos!")
    time.sleep(2) # Pausa para que el usuario pueda leer el mensaje
    progress_bar.empty() # Limpiar la barra de progreso
    return embeddings.cpu().numpy()

def get_coding_guidance(code):
    # ... (el resto de la función es igual, no es necesario copiarla de nuevo) ...
    chapter = code[0].upper()
    guidance = []
    if chapter in ['A', 'B']: guidance.append("**Guía:** Para enfermedades infecciosas, considere codificar también el organismo causal si la CIE-10 lo indica.")
    elif chapter == 'C' or (chapter == 'D' and int(code[1:3]) <= 48): guidance.append("**Guía:** Para neoplasias, especifique el comportamiento (maligno, benigno, in situ) y la localización. Use códigos de la sección Z para historial personal de neoplasia.")
    elif chapter == 'F': guidance.append("**Guía:** Para trastornos mentales, sea lo más específico posible. Considere el estado (ej. en remisión), la severidad y si es un episodio único o recurrente.")
    elif chapter == 'I': guidance.append("**Guía:** Para enfermedades circulatorias, especifique la cronicidad (agudo vs. crónico). Para hipertensión, considere si hay relación causal con enfermedades renales o cardíacas.")
    elif chapter == 'J': guidance.append("**Guía:** Para enfermedades respiratorias, distinga entre agudo y crónico. Si hay una infección, codifique el organismo si es conocido.")
    elif chapter == 'M': guidance.append("**Guía:** Para enfermedades musculoesqueléticas, especifique la lateralidad (derecho/izquierdo) y la articulación o zona exacta si es posible.")
    elif chapter == 'R': guidance.append("**Guía:** Los códigos 'R' (síntomas y signos) son para casos no diagnosticados. Si se llega a un diagnóstico definitivo, debe reemplazarse por el código de esa enfermedad.")
    elif chapter in ['S', 'T']: guidance.append("**Guía:** Para lesiones y traumatismos, es crucial incluir la causa externa (códigos V, W, X, Y). Especifique si es un encuentro inicial, subsecuente o una secuela.")
    elif chapter == 'Z': guidance.append("**Guía:** Los códigos 'Z' no son enfermedades, sino factores que influyen en el estado de salud (ej. controles, historial). Úselos como código principal o secundario según corresponda.")
    else: guidance.append("**Guía General:** Revise la documentación clínica para asegurar que el código seleccionado refleje con la máxima precisión el diagnóstico. Considere si se necesitan códigos adicionales para manifestaciones o comorbildades.")
    return "\n".join(guidance)
    
# --- Ejecución y Carga ---
model = load_model()
df = load_and_prepare_data()
embeddings = create_embeddings(model, df['description'].tolist())

# --- PUNTO DE CONTROL 2: La carga ha finalizado, ahora se dibuja la interfaz ---
st.info("Paso 4/4: Inicialización completa. ¡La aplicación está lista!")
time.sleep(1) # Pequeña pausa
st.experimental_rerun() # Esto recargará la app para limpiar los mensajes de carga

# --- Interfaz Principal de la Aplicación ---
st.title("🩺 Asistente Inteligente de Codificación CIE-10")
st.markdown("""
Esta herramienta utiliza un modelo de lenguaje para recomendar los códigos de diagnóstico CIE-10 más probables basados en una descripción clínica.
**Nota:** Es una ayuda y no reemplaza el juicio clínico profesional ni las normativas locales de codificación.
""")

with st.container(border=True):
    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_area("Ingrese la descripción clínica, síntoma o diagnóstico:", height=100, placeholder="Ej: Paciente con dolor de cabeza intenso y fiebre, sospecha de cefalea tensional...")
    with col2:
        num_results = st.slider("Resultados a mostrar:", min_value=1, max_value=10, value=5)
        search_button = st.button("Buscar Diagnósticos", type="primary", use_container_width=True)

if search_button and user_query:
    with st.spinner("Buscando diagnósticos relevantes..."):
        query_embedding = model.encode([user_query], convert_to_tensor=True).cpu().numpy()
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(similarities)[-num_results:][::-1]
    st.subheader(f"Top {num_results} diagnósticos recomendados:")
    for i, idx in enumerate(top_indices):
        code = df.iloc[idx]['code_4d']
        description = df.iloc[idx]['description']
        similarity = similarities[idx]
        with st.container(border=True):
            st.markdown(f"#### **{i+1}. {code}** - {description}")
            st.progress(float(similarity), text=f"Similitud: {similarity:.2%}")
            guidance = get_coding_guidance(code)
            st.info(guidance)
elif search_button and not user_query:
    st.warning("Por favor, ingrese una descripción clínica para buscar.")

st.markdown("---")
st.markdown("Desarrollado como una herramienta de apoyo para profesionales de la salud.")

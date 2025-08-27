import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Asistente de Codificación CIE-10",
    page_icon="🩺",
    layout="wide"
)

# --- Funciones de Carga y Procesamiento (con caché para eficiencia) ---

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_and_prepare_data():
    DATA_URL = "https://raw.githubusercontent.com/verasativa/CIE-10/refs/heads/master/codes.json"
    with st.spinner("Cargando catálogo CIE-10 (JSON) desde la web..."):
        try:
            df = pd.read_json(DATA_URL)
        except Exception as e:
            st.error(f"Error al cargar los datos desde la fuente online: {e}")
            st.stop()
    
    df.dropna(subset=['code', 'description'], inplace=True)
    df = df[df['description'].str.strip() != '']
    df['code_4d'] = df['code'].str.replace('.', '', regex=False)
    df['code_4d'] = df['code_4d'].apply(lambda x: x.ljust(4, 'X') if len(x) == 3 else x).str.slice(0, 4)
    return df

@st.cache_data
def create_embeddings(_model, descriptions):
    with st.spinner("Inicializando el motor de IA... (esto puede tardar un momento la primera vez)"):
        embeddings = _model.encode(descriptions, convert_to_tensor=True, show_progress_bar=True)
    return embeddings.cpu().numpy()

# --- NUEVA FUNCIÓN: Alertas de Codificación Compleja ---
def show_complex_coding_alert(query):
    """
    Detecta palabras clave en la consulta del usuario y muestra alertas específicas
    para escenarios de codificación que requieren múltiples códigos.
    """
    query_lower = query.lower()
    alert_triggered = False

    # Diccionario de alertas
    alerts = {
        ("suicidio", "autoinflingida", "autolesión", "auto inflingida"): """
        ### ⚠️ Alerta de Codificación Compleja: Lesión Autoinfligida
        Un **intento de suicidio** o **lesión autoinfligida** requiere al menos **DOS** códigos:
        1.  **Código de Lesión (Capítulo XIX: S00-T98):** Describe el daño físico (ej: `S610` para herida de muñeca, `T149` para traumatismo no especificado).
        2.  **Código de Causa Externa (Capítulo XX: X60-X84):** Describe la intencionalidad y el método (ej: `X78X` para objeto cortante, `X70X` para ahorcamiento).
        
        *Opcionalmente, añada un código de salud mental (Capítulo V: F00-F99) si está diagnosticado.*
        """,
        ("accidente", "caída", "golpe", "atropello", "quemadura", "mordedura"): """
        ### ⚠️ Alerta de Codificación Compleja: Traumatismo y Lesiones
        Un **traumatismo** o **lesión accidental** requiere al menos **DOS** códigos:
        1.  **Código de Lesión (Capítulo XIX: S00-T98):** Describe el daño físico (ej: `S826` para fractura de peroné).
        2.  **Código de Causa Externa (Capítulo XX: V01-Y98):** Describe el evento que causó la lesión (ej: `W19X` para caída no especificada, `V031` para peatón atropellado).
        """
    }

    for keywords, message in alerts.items():
        if any(keyword in query_lower for keyword in keywords):
            st.warning(message, icon="⚠️")
            alert_triggered = True
            break # Muestra solo la primera alerta relevante
    
    return alert_triggered

# --- Función de Orientación para Codificación (sin cambios) ---
def get_coding_guidance(code):
    # ... (el resto de la función es igual, no es necesario cambiarla) ...
    if not code: return ""
    chapter = code[0].upper()
    guidance = []
    is_neoplasia_range = (chapter == 'D' and len(code) > 2 and code[1:3].isdigit() and 0 <= int(code[1:3]) <= 48)
    if chapter in ['A', 'B']: guidance.append("**Guía:** Para enfermedades infecciosas, considere codificar también el organismo causal si la CIE-10 lo indica.")
    elif chapter == 'C' or is_neoplasia_range: guidance.append("**Guía:** Para neoplasias, especifique el comportamiento (maligno, benigno, in situ) y la localización.")
    elif chapter == 'F': guidance.append("**Guía:** Para trastornos mentales, sea lo más específico posible. Considere el estado (ej. en remisión), la severidad y si es un episodio único o recurrente.")
    elif chapter == 'I': guidance.append("**Guía:** Para enfermedades circulatorias, especifique la cronicidad (agudo vs. crónico).")
    elif chapter == 'J': guidance.append("**Guía:** Para enfermedades respiratorias, distinga entre agudo y crónico.")
    elif chapter == 'M': guidance.append("**Guía:** Para enfermedades musculoesqueléticas, especifique la lateralidad (derecho/izquierdo).")
    elif chapter == 'R': guidance.append("**Guía:** Los códigos 'R' (síntomas y signos) son para casos no diagnosticados. Si se llega a un diagnóstico definitivo, debe reemplazarse.")
    elif chapter in ['S', 'T']: guidance.append("**Guía de Lesión:** Este es un código de lesión. **Recuerde añadir un código de Causa Externa (V01-Y98)** para describir cómo y por qué ocurrió la lesión.")
    elif chapter in ['V', 'W', 'X', 'Y']: guidance.append("**Guía de Causa Externa:** Este es un código de causa externa. **Asegúrese de tener también un código de Lesión (S00-T98)** que describa el daño físico.")
    elif chapter == 'Z': guidance.append("**Guía:** Los códigos 'Z' no son enfermedades, sino factores que influyen en el estado de salud.")
    else: guidance.append("**Guía General:** Revise la documentación clínica para asegurar que el código seleccionado refleje con la máxima precisión el diagnóstico.")
    return "\n".join(guidance)
    
# --- Interfaz Principal de la Aplicación ---
st.title("🩺 Asistente Inteligente de Codificación CIE-10")
st.markdown("...") # ... (sin cambios aquí) ...

model = load_model()
df = load_and_prepare_data()
embeddings = create_embeddings(model, df['description'].tolist())

with st.container(border=True):
    # ... (sin cambios en la definición de la interfaz) ...
    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_area("Ingrese la descripción clínica, síntoma o diagnóstico:", height=100, placeholder="Ej: Paciente con dolor de cabeza intenso y fiebre, sospecha de cefalea tensional...")
    with col2:
        num_results = st.slider("Resultados a mostrar:", min_value=1, max_value=10, value=5)
        search_button = st.button("Buscar Diagnósticos", type="primary", use_container_width=True)

if search_button and user_query:
    # --- LLAMADA A LA NUEVA FUNCIÓN DE ALERTA ---
    show_complex_coding_alert(user_query)

    with st.spinner("Buscando diagnósticos relevantes..."):
        # ... (la lógica de búsqueda es la misma) ...
        query_embedding = model.encode([user_query], convert_to_tensor=True).cpu().numpy()
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(similarities)[-num_results:][::-1]

    st.subheader(f"Top {num_results} diagnósticos recomendados:")
    st.info("Recuerde que esta búsqueda puede ser solo un punto de partida. Use las alertas y guías para una codificación más precisa.")
    
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

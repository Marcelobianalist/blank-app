import streamlit as st
import pandas as pd
import unicodedata
from thefuzz import process

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Workbench CIE-10",
    page_icon="🧠",
    layout="centered" # El diseño centrado es mejor para este flujo
)

# --- Funciones de Lógica y Carga de Datos (con Caché) ---
def normalize_text(text: str) -> str:
    """Limpia y normaliza texto para búsquedas robustas."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    return "".join([c for c in text if not unicodedata.combining(c)])

@st.cache_data
def load_and_prepare_data():
    """Carga, procesa y optimiza el catálogo CIE-10 una sola vez."""
    DATA_URL = "https://raw.githubusercontent.com/verasativa/CIE-10/refs/heads/master/codes.json"
    with st.spinner("Cargando y optimizando el catálogo CIE-10..."):
        df = pd.read_json(DATA_URL).dropna(subset=['code', 'description'])
        df['code_4d'] = df['code'].str.replace('.', '', regex=False).apply(
            lambda x: x.ljust(4, 'X') if len(x) == 3 else x
        ).str.slice(0, 4)
        df['desc_norm'] = df['description'].apply(normalize_text)
        df['block_code'] = df['code'].str[:3]

        chapters_dict = {
            'A': "I. Infecciosas y parasitarias", 'B': "I. Infecciosas y parasitarias",
            'C': "II. Neoplasias (tumores)", 'D': "III-IV. Sangre / Endocrinas", 
            'E': "III-IV. Sangre / Endocrinas", 'F': "V. Trastornos mentales",
            'G': "VI. Sistema nervioso", 'H': "VII-VIII. Ojo y oído", 'I': "IX. Circulatorio",
            'J': "X. Respiratorio", 'K': "XI. Digestivo", 'L': "XII. Piel",
            'M': "XIII. Musculoesquelético", 'N': "XIV. Genitourinario",
            'O': "XV. Embarazo y parto", 'P': "XVI. Perinatales",
            'Q': "XVII. Malformaciones congénitas", 'R': "XVIII. Síntomas y signos",
            'S': "XIX. Traumatismos y envenenamientos", 'T': "XIX. Traumatismos y envenenamientos",
            'V': "XX. Causas externas", 'W': "XX. Causas externas", 'X': "XX. Causas externas", 'Y': "XX. Causas externas",
            'Z': "XXI. Factores que influyen en la salud", 'U': "XXII. Códigos para propósitos especiales"
        }
        df['chapter_letter'] = df['code'].str[0]
        df['chapter_desc'] = df['chapter_letter'].map(chapters_dict).fillna("Capítulo no especificado")
        
        # Obtener descripción del bloque para contexto
        block_descriptions = df.groupby('block_code')['description'].first()
        df['block_desc'] = df['block_code'].map(block_descriptions)
    return df

def get_coding_guidance(code):
    """Genera consejos de codificación contextuales y visuales."""
    if not code: return "", "info"
    chapter = code[0].upper()
    if chapter in ['S', 'T']:
        return "⚠️ **¡Atención!** Este es un código de **Lesión**. Es obligatorio añadir un código de **Causa Externa (V-Y)** que describa cómo ocurrió.", "error"
    if chapter in ['V', 'W', 'X', 'Y']:
        return "⚠️ **¡Atención!** Este es un código de **Causa Externa**. Debe usarse como código secundario junto a un código de **Lesión (S, T)**.", "error"
    if chapter == 'R':
        return "💡 **Consejo:** Los códigos 'R' son para síntomas o hallazgos sin un diagnóstico definitivo. Si se confirma una enfermedad, este código debe ser reemplazado.", "warning"
    if chapter == 'Z':
        return "💡 **Consejo:** Los códigos 'Z' no son enfermedades. Describen situaciones como controles, seguimientos o factores de riesgo.", "info"
    if chapter == 'C' or (chapter == 'D' and len(code) > 2 and code[1:3].isdigit() and int(code[1:3]) <= 48):
        return "💡 **Consejo:** Para neoplasias, es crucial especificar la localización y el comportamiento (maligno, benigno, etc.).", "info"
    return "✅ **Guía General:** Asegúrese de que este código sea el más específico posible según la documentación clínica.", "success"

# --- Funciones de Renderizado de la UI ---

def render_search_view():
    """Muestra la interfaz de búsqueda principal."""
    st.header("🔍 Búsqueda Inteligente CIE-10")
    search_query = st.text_input("Escriba un término o código para iniciar el análisis:", placeholder="Ej: diabetes, infarto, F322...")

    if search_query:
        sq_norm = normalize_text(search_query)
        
        # Búsqueda por código tiene prioridad
        code_matches = df[df['code_4d'].str.startswith(sq_norm.upper())]
        
        if not code_matches.empty:
            result_df = code_matches.head(20)
        else:
            # Búsqueda por descripción si no hay match de código
            results = process.extract(sq_norm, df['desc_norm'], limit=20)
            matched_descs = [r[0] for r in results]
            result_df = df[df['desc_norm'].isin(matched_descs)]

        if not result_df.empty:
            st.write("---")
            st.subheader("Resultados encontrados:")
            st.caption("Haga clic en un resultado para un análisis profundo.")
            for _, row in result_df.iterrows():
                if st.button(f"**{row['code_4d']}** – {row['description']}", key=row['code_4d'], use_container_width=True):
                    st.session_state.selected_code_4d = row['code_4d']
                    st.rerun() # Recarga la app para mostrar la vista de análisis
        else:
            st.warning("No se encontraron coincidencias. Intente con otros términos.")

def render_focus_view(code_4d):
    """Muestra el panel de análisis detallado para un código seleccionado."""
    row = df[df['code_4d'] == code_4d].iloc[0]

    # Botón para volver a la búsqueda
    if st.button("⬅️ Volver a la búsqueda"):
        st.session_state.selected_code_4d = None
        st.rerun()

    st.title(f"Análisis del Código: {row['code_4d']}")
    st.header(row['description'])
    st.divider()

    # Jerarquía / "Migas de Pan"
    st.markdown(f"**Ruta Jerárquica:** `{row['chapter_desc']}` ➡️ `{row['block_code']} - {row['block_desc']}`")
    
    # Guía de Codificación
    with st.container(border=True):
        st.subheader("🧠 Guía de Codificación Inteligente")
        guidance, level = get_coding_guidance(row['code_4d'])
        getattr(st, level)(guidance)

    # Códigos Relacionados para mayor precisión
    related_codes_df = df[df['block_code'] == row['block_code']]
    if len(related_codes_df) > 1:
        with st.expander(f"Ver otros códigos en el bloque '{row['block_code']}' para mejorar la precisión"):
            for _, related_row in related_codes_df.iterrows():
                if related_row['code_4d'] == code_4d:
                    st.markdown(f"🔹 **{related_row['code_4d']} – {related_row['description']} (actualmente seleccionado)**")
                else:
                    st.markdown(f"🔸 `{related_row['code_4d']}` – {related_row['description']}")
            st.caption("Considere usar uno de estos códigos si describe mejor la condición del paciente.")

# --- Flujo Principal de la Aplicación ---
st.title("🧠 Workbench de Codificación CIE-10")
st.markdown("Una herramienta intuitiva para analizar y entender la codificación clínica.")

df = load_and_prepare_data()

# Decidir qué vista mostrar
if 'selected_code_4d' not in st.session_state:
    st.session_state.selected_code_4d = None

if st.session_state.selected_code_4d:
    render_focus_view(st.session_state.selected_code_4d)
else:
    render_search_view()

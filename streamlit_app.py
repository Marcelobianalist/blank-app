import streamlit as st
import pandas as pd
import unicodedata
from thefuzz import process

# --- Configuración ---
st.set_page_config(
    page_title="Portal Interactivo CIE-10",
    page_icon="📖",
    layout="wide"
)

# --- Funciones de utilidad ---
def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    return "".join([c for c in text if not unicodedata.combining(c)])

@st.cache_data
def load_and_prepare_data():
    DATA_URL = "https://raw.githubusercontent.com/verasativa/CIE-10/refs/heads/master/codes.json"
    df = pd.read_json(DATA_URL).dropna(subset=['code', 'description'])

    df['code_4d'] = df['code'].str.replace('.', '', regex=False).apply(
        lambda x: x.ljust(4, 'X') if len(x) == 3 else x
    ).str.slice(0, 4)
    df['desc_norm'] = df['description'].apply(normalize_text)

    chapters_dict = {
        'A': "I. Infecciosas y parasitarias", 'B': "I. Infecciosas y parasitarias",
        'C': "II. Neoplasias (tumores)",
        'D': "III-IV. Sangre / Endocrinas", 'E': "III-IV. Sangre / Endocrinas",
        'F': "V. Trastornos mentales", 'G': "VI. Sistema nervioso",
        'H': "VII-VIII. Ojo y oído", 'I': "IX. Circulatorio",
        'J': "X. Respiratorio", 'K': "XI. Digestivo", 'L': "XII. Piel",
        'M': "XIII. Musculoesquelético", 'N': "XIV. Genitourinario",
        'O': "XV. Embarazo y parto", 'P': "XVI. Perinatales",
        'Q': "XVII. Malformaciones congénitas", 'R': "XVIII. Síntomas",
        'S': "XIX. Traumatismos", 'T': "XIX. Traumatismos",
        'V': "XX. Causas externas", 'W': "XX. Causas externas",
        'X': "XX. Causas externas", 'Y': "XX. Causas externas",
        'Z': "XXI. Factores de salud", 'U': "XXII. Especiales"
    }
    df['chapter_letter'] = df['code'].str[0]
    df['chapter_desc'] = df['chapter_letter'].map(chapters_dict).fillna("Capítulo no especificado")

    return df

def get_coding_guidance(code):
    if not code: return "", "info"
    chapter = code[0].upper()
    if chapter in ['S', 'T']:
        return "⚠️ Código de **Lesión** → requiere además una **Causa Externa (V-Y)**.", "error"
    if chapter in ['V', 'W', 'X', 'Y']:
        return "⚠️ Código de **Causa Externa** → úselo con un código de **Lesión (S, T)**.", "error"
    if chapter == 'R':
        return "💡 Códigos 'R' son síntomas → reemplace si hay diagnóstico definitivo.", "warning"
    if chapter == 'Z':
        return "💡 Códigos 'Z' describen controles, riesgos o historia clínica.", "info"
    if chapter == 'C':
        return "💡 En neoplasias, especifique localización y comportamiento.", "info"
    return "✅ Use el código más específico posible.", "info"

# --- Cargar datos ---
df = load_and_prepare_data()

if 'selected_code' not in st.session_state:
    st.session_state.selected_code = None

tab1, tab2 = st.tabs(["🔍 Búsqueda en Vivo", "🗺️ Explorador"])

# --- TAB 1: AUTOCOMPLETE ---
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Búsqueda tipo Google")
        search_query = st.text_input("Escriba un código o palabra clave:", placeholder="Ej: R458, fractura, cefalea...")

        result_df = pd.DataFrame()
        if search_query:
            sq_norm = normalize_text(search_query)

            # Primero coincidencias directas por código
            code_matches = df[df['code_4d'].str.startswith(sq_norm.upper())]

            if not code_matches.empty:
                result_df = code_matches
            else:
                # Fuzzy en descripción
                results = process.extract(sq_norm, df['desc_norm'].tolist(), limit=15)
                matched_descs = [r[0] for r in results]
                result_df = df[df['desc_norm'].isin(matched_descs)]

        if not result_df.empty:
            st.dataframe(
                result_df[['code_4d', 'description']],
                use_container_width=True, hide_index=True,
                column_config={"code_4d": "Código", "description": "Descripción"}
            )
            selected = st.selectbox(
                "Seleccione un código para ver detalles:",
                [f"{row['code_4d']} - {row['description']}" for _, row in result_df.iterrows()],
                key="code_selector"
            )
            if selected:
                st.session_state.selected_code = selected.split(" - ")[0]
        else:
            if search_query:
                st.warning("No se encontraron coincidencias.")

    with col2:
        st.header("📋 Detalles")
        if st.session_state.selected_code:
            row = df[df['code_4d'] == st.session_state.selected_code].iloc[0]
            st.subheader(f"Código: {row['code_4d']}")
            st.markdown(f"**Descripción:** {row['description']}")
            st.markdown(f"**Capítulo:** {row['chapter_desc']} ({row['chapter_letter']})")
            st.divider()
            guidance, level = get_coding_guidance(row['code_4d'])
            getattr(st, level)(guidance)  # muestra con st.error / st.warning / st.info
        else:
            st.info("Seleccione un código para ver sus detalles aquí.")

# --- TAB 2: EXPLORADOR GUIADO ---
with tab2:
    st.header("Explorador por Capítulos")
    chapter = st.selectbox("Capítulo:", sorted(df['chapter_desc'].unique()), key="chapter_selector")
    df_chapter = df[df['chapter_desc'] == chapter]

    block = st.selectbox("Bloque (primeros 3 dígitos):", sorted(df_chapter['code'].str[:3].unique()), key="block_selector")
    df_block = df_chapter[df_chapter['code'].str.startswith(block)]

    st.dataframe(df_block[['code_4d', 'description']], hide_index=True, use_container_width=True)

    if not df_block.empty:
        st.markdown("### Detalles de los códigos seleccionados:")
        for _, row in df_block.iterrows():
            st.markdown(f"- **Código:** {row['code_4d']} - **Descripción:** {row['description']}")
    else:
        st.warning("No hay códigos disponibles para el bloque seleccionado.")

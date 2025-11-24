import re
import os
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq  # usando Groq Llama 3

def inject_dark_cyan_blade_theme():
    st.markdown(
        """
        <style>
        :root {
            --petrol: #0F3B4D;
            --petrol-light: #1E5A70;
            --petrol-soft: #C6D9DF;
            --bg-main: #F2F4F7;   /* cinza prata */
            --bg-card: #FFFFFF;
            --border-subtle: #D5DCE1;
            --text-main: #1E293B;
            --text-muted: #64748B;
        }

        /* ====== APP GERAL ====== */
        .stApp {
            background: var(--bg-main) !important;
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
        }

        /* Container principal */
        .block-container {
            padding-top: 1.3rem;
            max-width: 1250px;
        }

        /* Markdown padrão */
        [data-testid="stMarkdown"] {
            color: var(--text-main) !important;
        }

        h1, h2, h3 {
            color: var(--petrol) !important;
        }

        /* ====== HEADER DO STREAMLIT ====== */
        header[data-testid="stHeader"] {
            background: #ffffffd9 !important;
            border-bottom: 1px solid var(--border-subtle);
            backdrop-filter: blur(8px);
        }
        header[data-testid="stHeader"] * {
            color: var(--text-main) !important;
        }

        /* ====== SIDEBAR — Azul petróleo + texto bem legível ====== */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d2538 0%, #091b28 100%);
            border-right: 1px solid rgba(51, 123, 172, 0.35);
            color: #f0f6fc !important;
        }

        /* Texto da sidebar */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #f0f6fc !important; /* azul gelo — ótimo contraste no fundo escuro */
        }
        /* === CORREÇÃO DO BOTÃO "Browse files" DO FILE UPLOADER === */
        [data-testid="stFileUploader"] button {
            background: #1e293b !important;        /* azul petróleo escuro */
            color: #f1f5f9 !important;              /* texto cinza muito claro */
            border-radius: 10px !important;
            border: 1px solid rgba(255,255,255,0.25) !important;
            padding: 0.45rem 1.2rem !important;
            font-weight: 600 !important;
        }

        /* hover */
        [data-testid="stFileUploader"] button:hover {
            background: #334155 !important;         /* azul petróleo mais claro */
            color: #ffffff !important;
            border-color: rgba(255,255,255,0.4) !important;
        }
        /* Dropzone */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            border: 1px dashed rgba(130, 180, 220, 0.8);
            color: #f0f6fc;
        }
        /* Texto dentro da caixa de upload */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] p {
            color: #e2e8f0 !important;   /* cinza claro */
        }

        /* Título e texto do dropzone */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] * {
            color: #e2e8f0 !important;
        }
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] p {
            color: #e8eef6 !important;
        }

        /* Inputs */
        [data-testid="stSidebar"] input, 
        [data-testid="stSidebar"] textarea {
            background: rgba(255, 255, 255, 0.12) !important;
            color: #f8fafc !important;
            border-radius: 8px;
            border: 1px solid rgba(180, 220, 255, 0.45);
        }

        [data-testid="stSidebar"] input:focus,
        [data-testid="stSidebar"] textarea:focus {
            border-color: #4aa8ff !important;
            box-shadow: 0 0 0 1px rgba(74, 168, 255, 0.5);
        }

        /* Botão */
        [data-testid="stSidebar"] .stButton > button {
            background: #2a5273;
            color: #ffffff;
            border-radius: 8px;
            padding: 0.55rem 1.3rem;
            border: 1px solid #3d6f93;
            font-weight: 500;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            filter: brightness(1.08);
            border-color: #5ea9d6;
        }

        /* Upload */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
            background: #0D1A22;
            border: 1px dashed #80AFC2;
            border-radius: 10px;
        }

        /* Inputs na sidebar */
        [data-testid="stSidebar"] .stTextInput > div > div > input {
            background-color: #0F1F27 !important;
            border: 1px solid #4B7486 !important;
            color: #E2ECF2 !important;
            border-radius: 6px;
        }

        /* Botões da sidebar */
        [data-testid="stSidebar"] .stButton > button {
            background: var(--petrol-light);
            border-radius: 6px;
            color: white;
            border: none;
            font-weight: 500;
            padding: 0.55rem 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.35);
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            filter: brightness(1.08);
        }

        /* ====== TÍTULO E SUBTÍTULO DO APP ====== */
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: var(--petrol);
        }
        .main-subtitle {
            font-size: 0.95rem;
            color: var(--text-muted);
        }
        .header-divider {
            border-bottom: 1px solid var(--border-subtle);
            margin-top: 0.6rem;
            margin-bottom: 1rem;
        }

        /* ====== TABS (modernas e quadradas) ====== */
        [data-testid="stTabs"] button[role="tab"] {
            background: #e4e8eb;
            color: var(--text-muted);
            border: 1px solid transparent;
            border-radius: 6px;
            padding: 0.4rem 1.1rem;
            font-weight: 500;
            margin-right: 0.35rem;
        }

        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            background: var(--petrol);
            color: white !important;
            border-radius: 6px;
            box-shadow: 0 3px 8px rgba(15, 59, 77, 0.35);
            border-color: var(--petrol);
        }

        [data-testid="stTabs"] button[role="tab"]:hover {
            background: #d8e0e4;
        }

        /* ====== CARDS ====== */
        .datacard {
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border-subtle);
            padding: 1.1rem 1.3rem;
            box-shadow: 0 4px 14px rgba(0,0,0,0.04);
        }

        .datacard h3 {
            color: var(--petrol);
            margin-bottom: 0.55rem;
        }

        /* ====== BOTÕES NO CONTEÚDO ====== */
        .stButton > button {
            background: var(--petrol);
            color: white;
            border-radius: 6px;
            padding: 0.45rem 1.3rem;
            border: none;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(15, 59, 77, 0.35);
        }
        .stButton > button:hover {
            filter: brightness(1.05);
        }

        /* ====== CAMPOS DE TEXTO NO CONTEÚDO ====== */
        textarea, input, .stTextInput > div > div > input {
            background-color: #ffffff !important;
            color: var(--text-main);
            border-radius: 6px !important;
            border: 1px solid var(--border-subtle) !important;
        }
        textarea:focus, input:focus,
        .stTextInput > div > div > input:focus {
            border-color: var(--petrol) !important;
            box-shadow: 0 0 0 1px rgba(15, 59, 77, 0.4);
        }

        /* ====== DATAFRAME ====== */
        .stDataFrame {
            border-radius: 6px !important;
            border: 1px solid var(--border-subtle);
            background: white;
        }

        /* ====== HEADER NATIVO DO STREAMLIT — IGUAL À SIDEBAR ====== */
        header[data-testid="stHeader"] {
            background: linear-gradient(180deg, #102A36 0%, #0A1D26 100%) !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            color: #E5ECF0 !important;
            backdrop-filter: none !important; /* mantém a aparência sólida */
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
        }

        header[data-testid="stHeader"] * {
            color: #E5ECF0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Carrega variáveis de ambiente
# ---------------------------------------------------------------------------
load_dotenv()

def get_groq_api_key() -> str:
    """
    Prioridade:
    1) st.secrets["GROQ_API_KEY"]  (quando existir secrets.toml)
    2) Variável de ambiente / .env (GROQ_API_KEY)
    """

    # Caminhos onde o Streamlit procura o secrets.toml
    possible_paths = [
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
    ]

    has_secrets_file = any(os.path.exists(p) for p in possible_paths)

    # Só tenta acessar st.secrets se o arquivo realmente existir
    if has_secrets_file:
        try:
            # .get evita KeyError se a chave não existir
            return st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            # Qualquer problema aqui, cai para o .env
            pass

    # Se não tiver secrets.toml ou não tiver a chave lá, usa o .env / variáveis de ambiente
    load_dotenv()
    return os.getenv("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# Funções auxiliares: execução de código Python com acesso ao df
# ---------------------------------------------------------------------------

def python_repl_ast(code: str, df: pd.DataFrame = None):
    """
    Executa código Python usando o REPL com acesso ao df.
    Captura:
    - saída de texto (print, etc.)
    - gráfico (se existir figura aberta)
    """
    TEMP_PLOT_PATH = "temp_plot.png"

    # Limpa gráfico anterior
    if os.path.exists(TEMP_PLOT_PATH):
        os.remove(TEMP_PLOT_PATH)

    # Ambiente controlado
    local_env = {
        "df": df,
        "pd": pd,
        "sns": sns,
        "plt": plt,
    }

    text_output_buffer = StringIO()

    try:
        import sys
        original_stdout = sys.stdout
        sys.stdout = text_output_buffer

        # 1) Executa o código como veio
        exec(code, {}, local_env)

        # 2) Se o código parece ser APENAS uma expressão,
        # tenta imprimir o resultado explicitamente
        try:
            raw = code.strip()
            if (
                "\n" not in raw          # uma linha só
                and "=" not in raw       # sem atribuição
                and not raw.startswith("print(")  # já não é print
            ):
                exec(f"print({raw})", {}, local_env)
        except Exception:
            # se não der pra avaliar como expressão, ignora
            pass

        # 3) Só salva figura se existir gráfico aberto
        if plt.get_fignums():
            plt.savefig(TEMP_PLOT_PATH)
            plt.close()

        # Restaura stdout
        sys.stdout = original_stdout

        output_text = text_output_buffer.getvalue()
        has_image = os.path.exists(TEMP_PLOT_PATH)

        if output_text.strip() and has_image:
            return (
                "Código executado com sucesso.\n"
                "Saída:\n"
                f"{output_text}\n"
                f"Gráfico salvo em {TEMP_PLOT_PATH}"
            )
        elif output_text.strip():
            return (
                "Código executado com sucesso.\n"
                "Saída:\n"
                f"{output_text}"
            )
        elif has_image:
            return "Código executado com sucesso. Gráfico salvo em temp_plot.png"
        else:
            return "Código executado com sucesso (sem saída de texto e sem gráfico)."

    except Exception as e:
        sys.stdout = original_stdout
        return f"Erro ao executar o código Python: {e}"


def python_repl_ast_tool(code: str, df: pd.DataFrame = None) -> str:
    """
    Limpeza básica do código vindo do modelo antes de executar de fato.
    """
    cleaned = code

    # remove ```python ... ```
    if cleaned.startswith("```"):
        cleaned = cleaned.strip()
        cleaned = cleaned.lstrip("`")
        cleaned = cleaned.rstrip("`")

        # remove 'python' no início se existir
        if cleaned.startswith("python"):
            cleaned = cleaned[len("python"):].lstrip()

    # remove possíveis backticks restantes
    cleaned = cleaned.replace("```", "").strip()

    # remove aspas externas se o código vier entre "" ou ''
    if (
        (cleaned.startswith('"') and cleaned.endswith('"')) or
        (cleaned.startswith("'") and cleaned.endswith("'"))
    ):
        cleaned = cleaned[1:-1].strip()

    # REMOVE plt.show / plt.savefig
    cleaned = re.sub(r"plt\.show\s*\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"plt\.savefig\s*\([^)]*\)", "", cleaned)

    # opcional: evita "df.mean(), df.median()"
    if (
        "," in cleaned
        and "\n" not in cleaned
        and "(" not in cleaned
        and cleaned.startswith("df.")
    ):
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        cleaned = parts[-1]

    return python_repl_ast(cleaned, df=df)


# ---------------------------------------------------------------------------
# EDA inicial + memória
# ---------------------------------------------------------------------------

def initial_analysis_and_memory(df: pd.DataFrame, memory: ConversationBufferMemory, llm) -> None:
    """
    Realiza uma Análise Exploratória de Dados (EDA) genérica em uma AMOSTRA do dataset
    e grava um resumo na memória, cuidando para não ultrapassar limites de tokens.
    A função NÃO assume nenhum domínio específico.
    """
    # 1) Amostra
    if len(df) > 500:
        sample_df = df.sample(500, random_state=42)
    else:
        sample_df = df.copy()

    numeric_cols = sample_df.select_dtypes(include="number").columns.tolist()
    object_cols = sample_df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(numeric_cols) > 10:
        numeric_cols = numeric_cols[:10]

    # Info geral
    info_str = f"Shape do dataset original (linhas, colunas): {df.shape}\n"
    info_str += f"Shape da amostra usada na EDA: {sample_df.shape}\n\n"
    info_str += "Tipos de dados das colunas da amostra:\n"
    info_str += sample_df.dtypes.to_string()

    # Estatísticas numéricas
    if numeric_cols:
        describe_num_str = sample_df[numeric_cols].describe().to_string()
        corr = sample_df[numeric_cols].corr()
        corr_str = corr.to_string()
    else:
        describe_num_str = "Não foram encontradas colunas numéricas na amostra."
        corr_str = "Não foi possível calcular correlação (sem colunas numéricas)."

    # Estatísticas categóricas simples
    cat_summary_parts = []
    max_cat_cols = 5
    for col in object_cols[:max_cat_cols]:
        vc = sample_df[col].value_counts(dropna=False).head(10)
        cat_summary_parts.append(f"Frequências da coluna categórica '{col}':\n{vc.to_string()}\n")
    if cat_summary_parts:
        cat_str = "\n".join(cat_summary_parts)
    else:
        cat_str = "Não foram encontradas colunas categóricas relevantes na amostra."

    # Amostra de linhas
    head_str = sample_df.head(5).to_string()

    # Função de truncamento
    def trim(text: str, max_len: int = 2000) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + "\n...[texto truncado para caber no limite de tokens]..."

    info_str = trim(info_str, 2000)
    describe_num_str = trim(describe_num_str, 2000)
    head_str = trim(head_str, 2000)
    corr_str = trim(corr_str, 2000)
    cat_str = trim(cat_str, 2000)

    prompt_analise = f"""
Você é um especialista em Análise Exploratória de Dados (EDA) para QUALQUER dataset tabular em CSV.

Abaixo estão resultados de EDA executados em uma AMOSTRA do dataset carregado pelo usuário:

1) Informações gerais (shape e tipos de dados):
{info_str}

2) Estatísticas descritivas das principais variáveis numéricas:
{describe_num_str}

3) Amostra das primeiras linhas:
{head_str}

4) Matriz de correlação entre variáveis numéricas:
{corr_str}

5) Resumo de frequências de algumas colunas categóricas (se houver):
{cat_str}

TAREFA (sempre em Português do Brasil):
- Descreva, em detalhes, as principais características do dataset.
- Comente possíveis relações interessantes entre variáveis.
- Destaque desequilíbrios de categorias, possíveis outliers e pontos que merecem investigação.
- Sugira exemplos de perguntas úteis que o usuário poderia fazer ao agente.

Organize a resposta em tópicos e parágrafos curtos.
"""

    analise_inicial = llm.predict(prompt_analise)

    memory.chat_memory.add_message(
        HumanMessage(content="Resultado da análise exploratória inicial (EDA) sobre o dataset carregado.")
    )
    memory.chat_memory.add_message(
        AIMessage(content=analise_inicial)
    )

    memory.save_context(
        {"input": "Resumo sintético da EDA do dataset carregado."},
        {"output": analise_inicial[:1000]}
    )


# ---------------------------------------------------------------------------
# Consulta à memória (resumos / conclusões da EDA)
# ---------------------------------------------------------------------------

def smart_memory_lookup_tool(query: str, llm, memory) -> str:
    """
    Usa o texto salvo da EDA na memória para responder perguntas
    sem recalcular nada. Otimizado para consumir poucos tokens.
    """
    memory_content = memory.buffer_as_str

    if not memory_content or not memory_content.strip():
        return "Ainda não há análise exploratória (EDA) salva na memória."

    MAX_MEMORY_CHARS = 1000
    if len(memory_content) > MAX_MEMORY_CHARS:
        memory_content = (
            memory_content[:MAX_MEMORY_CHARS]
            + "\n...[trecho da EDA truncado para caber no limite de tokens]..."
        )

    q = query.strip().lower()

    if q in [
        "análise exploratória completa",
        "analise exploratoria completa",
        "resumo da eda",
        "resumo da análise exploratória",
        "resumo da analise exploratoria",
        "conclusões da eda",
        "conclusoes da eda",
        "quais conclusões você tirou dos dados",
        "quais as principais conclusões da eda",
    ]:
        sub_prompt = f"""
Você recebeu abaixo o texto de uma Análise Exploratória de Dados (EDA) sobre um dataset tabular em CSV.

[EDA]
{memory_content}
[/EDA]

Resuma, em Português do Brasil e em no máximo 5 parágrafos curtos:
- principais características do dataset;
- desequilíbrios relevantes;
- relações importantes entre variáveis;
- pontos de atenção para análises futuras.
"""
    else:
        sub_prompt = f"""
Abaixo está o texto de uma Análise Exploratória de Dados (EDA) sobre um dataset tabular em CSV:

[EDA]
{memory_content}
[/EDA]

Pergunta do usuário: "{query}"

Responda em Português do Brasil, em 1 ou 2 frases,
apenas com o que está explícito na EDA.
Se não houver informação suficiente, responda:
"Não encontrei essa informação na EDA.".
"""

    resposta = llm.predict(sub_prompt)
    return resposta


def is_summary_question(text: str) -> bool:
    """
    Heurística simples para identificar perguntas que pedem RESUMO ou CONCLUSÕES da EDA.
    """
    if not text:
        return False
    t = text.lower()
    keywords = [
        "resumo da eda",
        "resumo da análise exploratória",
        "resumo da analise exploratoria",
        "análise exploratória completa",
        "analise exploratoria completa",
        "quais as conclusões",
        "quais conclusoes",
        "quais as principais conclusões",
        "quais as principais conclusoes",
        "o que você observou",
        "o que voce observou",
        "conclusões da eda",
        "conclusoes da eda",
    ]
    return any(k in t for k in keywords)


# ---------------------------------------------------------------------------
# Instruções para o agente que usa apenas python_repl_ast
# ---------------------------------------------------------------------------

prefix_python_agent = """
Você é um especialista em Análise Exploratória de Dados (EDA) para QUALQUER dataset tabular em CSV.

Você tem acesso a UMA ferramenta:

1) python_repl_ast
   - Executa código Python usando o DataFrame 'df'.
   - Use para: cálculos numéricos, estatísticas, proporções, contagens, agrupamentos e geração de gráficos.

REGRAS:

- Sempre que a pergunta envolver números, estatísticas, colunas, tipos de dados OU gráficos,
  você DEVE chamar a ferramenta python_repl_ast ao menos uma vez.

- Antes de usar qualquer coluna, se não souber os nomes ainda, chame:
  Action: python_repl_ast
  Action Input: df.columns

- Não invente nomes de colunas. Use apenas nomes que existam em df.columns.

PARA HISTOGRAMAS, SEMPRE SIGA O PADRÃO:

  plt.figure(figsize=(10,6))
  serie = df['NOME_DA_COLUNA'].dropna()
  # opcional: filtragem entre percentis 1% e 99%
  # serie = serie[(serie >= serie.quantile(0.01)) & (serie <= serie.quantile(0.99))]
  plt.hist(serie, bins=50)
  plt.title('Histograma da coluna NOME_DA_COLUNA')
  plt.xlabel('NOME_DA_COLUNA')
  plt.ylabel('Frequência')

- Para escala logarítmica no eixo X:
  plt.xscale('log')

RESPOSTA FINAL:

- Depois de receber a Observation de python_repl_ast, responda em Português do Brasil, usando:

  Final Answer: <texto>

- Quando fizer sentido, cite 1 ou 2 valores numéricos concretos da Observation
  e faça 1 ou 2 frases de interpretação simples.
"""


# ---------------------------------------------------------------------------
# Carregamento dos dados e inicialização do LLM
# ---------------------------------------------------------------------------

def load_data(uploaded_file, openai_api_key, memory, llm_container):
    """
    Carrega o dataset, inicializa df, realiza a análise inicial (EDA) e armazena tudo no session_state.
    """
    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df

        # 1) Memória
        last_file = st.session_state.get("current_file_name")
        if last_file != uploaded_file.name:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            st.session_state['memory_instance'] = memory
            st.session_state['current_file_name'] = uploaded_file.name
        else:
            if memory is None:
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
                st.session_state['memory_instance'] = memory

        # 2) LLM
        if llm_container.get("llm") is None:
            if openai_api_key:
                llm = ChatGroq(
                    api_key=openai_api_key,
                    temperature=0.0,
                    model="llama-3.1-8b-instant",
                )
            else:
                llm = None
            llm_container["llm"] = llm
        else:
            llm = llm_container["llm"]

        # 3) EDA inicial
        if llm is not None:
            with st.spinner("Executando análise exploratória inicial com o LLM..."):
                initial_analysis_and_memory(df, memory, llm)
            st.success("Análise inicial registrada na memória com sucesso!")
        else:
            st.error("Não foi possível inicializar o LLM. Verifique sua chave Groq API.")

        return df, memory, llm_container

    return None, memory, llm_container


# ---------------------------------------------------------------------------
# Layout / UI do Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DataScout – Explorador Inteligente de CSV",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_dark_cyan_blade_theme()

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:0.4rem;margin-top:0.4rem;">
        <div style="font-size:34px;">🧭</div>
        <div><br>
            <div class="main-title">DataScout – Explorador Inteligente de Dados CSV</div>
            <div class="main-subtitle">
                Um painel interativo para explorar qualquer arquivo CSV com apoio de um agente de IA.
            </div>
        </div>
    </div>
    <div class="header-divider"></div>
    """,
    unsafe_allow_html=True,
)

# Estado global
if 'df' not in st.session_state:
    st.session_state['df'] = None
    st.session_state['memory_instance'] = None
    st.session_state['llm_container'] = {"llm": None}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("🎛️ Painel de Controle")

uploaded_file = st.sidebar.file_uploader("Selecione um arquivo CSV", type=["csv"])

# Tenta pegar automaticamente (secrets no Cloud ou .env/local)
default_key = get_groq_api_key()

if default_key:
    api_key = default_key
    st.sidebar.caption("Usando a chave Groq configurada no ambiente.")
else:
    api_key = st.sidebar.text_input(
        "Chave Groq API",
        type="password",
        help="Informe apenas se a chave não estiver configurada no ambiente."
    )

if st.sidebar.button("Carregar e analisar arquivo"):
    with st.spinner("Carregando dados e executando análise inicial..."):
        df, memory_instance, llm_container = load_data(
            uploaded_file,
            api_key,
            st.session_state.get('memory_instance'),
            st.session_state.get('llm_container'),
        )
        st.session_state['memory_instance'] = memory_instance
        st.session_state['llm_container'] = llm_container
else:
    df = st.session_state.get('df')
    memory_instance = st.session_state.get('memory_instance')
    llm_container = st.session_state.get('llm_container')

# ---------------------------------------------------------------------------
# Conteúdo principal com abas
# ---------------------------------------------------------------------------

if df is not None and memory_instance is not None and llm_container.get("llm") is not None:
    llm = llm_container.get("llm")

    tab_overview, tab_agent, tab_eda = st.tabs(
        ["📁 Visão geral do dataset", "🤖 Conversar com o agente", "🧾 Resumo automático da EDA"]
    )

    # -----------------------------------------------------------------------
    # Aba 1 – Visão geral
    # -----------------------------------------------------------------------
    with tab_overview:
        st.markdown('<div class="datacard">', unsafe_allow_html=True)

        st.subheader("Panorama rápido do arquivo carregado")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Número de linhas", f"{len(df):,}".replace(",", "."))
        with col2:
            st.metric("Número de colunas", df.shape[1])
        with col3:
            num_cols = len(df.select_dtypes(include='number').columns)
            st.metric("Colunas numéricas", num_cols)

        st.markdown("#### Amostra do dataset")
        st.dataframe(df.head(), use_container_width=True)

        with st.expander("Ver tipos de dados por coluna"):
            st.write(df.dtypes)

        st.markdown(
            f"<span class='helper-text'>Shape completo: {df.shape[0]} linhas × {df.shape[1]} colunas.</span>",
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)
    # -----------------------------------------------------------------------
    # Aba 2 – Conversa com o agente (EDA interativa + gráficos)
    # -----------------------------------------------------------------------
    with tab_agent:
        st.markdown('<div class="datacard">', unsafe_allow_html=True)

        st.subheader("Pergunte ao agente sobre o dataset")

        st.markdown(
            """
**Sugestões de perguntas:**
- Quais são os tipos de dados (numéricos, categóricos) presentes no dataset?  
- Quais colunas numéricas têm maior média e maior desvio padrão?  
- Existem variáveis com forte correlação entre si? Quais?  
- Gere um histograma da coluna `X`.  
- Gere um boxplot da coluna `Y` para identificar outliers.  
            """
        )

        user_question = st.text_input("Digite sua pergunta sobre o dataset:")

        if st.button("Consultar agente", key="ask_agent"):
            if user_question.strip() == "":
                st.warning("Por favor, digite uma pergunta.")
            else:
                # Se for pergunta de resumo/conclusão, usa diretamente a memória
                if is_summary_question(user_question):
                    with st.spinner("Consultando resumo salvo da EDA..."):
                        response = smart_memory_lookup_tool(
                            user_question,
                            llm=llm,
                            memory=memory_instance,
                        )
                    st.markdown("### Resposta do agente")
                    st.write(response)
                else:
                    # Ferramenta python_repl_ast para cálculos/gráficos
                    tools = [
                        Tool(
                            name="python_repl_ast",
                            func=lambda code: python_repl_ast_tool(code, df=df),
                            description=(
                                "Executa código Python diretamente no DataFrame 'df' "
                                "para cálculos estatísticos e geração de gráficos."
                            ),
                        )
                    ]

                    agent = initialize_agent(
                        tools,
                        llm,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=True,
                        memory=memory_instance,
                        handle_parsing_errors=True,
                        max_iterations=4,
                        early_stopping_method="generate",
                    )

                    full_prompt = f"{prefix_python_agent}\n\nPergunta do usuário: {user_question}"

                    with st.spinner("Consultando o agente..."):
                        response = agent.run(full_prompt)

                    st.markdown("### Resposta do agente")
                    st.write(response)

                    # Se o código gerou gráfico, exibe o temp_plot.png
                    TEMP_PLOT_PATH = "temp_plot.png"
                    if os.path.exists(TEMP_PLOT_PATH):
                        st.markdown("#### Visualização gerada")
                        st.image(TEMP_PLOT_PATH)
                        os.remove(TEMP_PLOT_PATH)

        st.markdown("</div>", unsafe_allow_html=True)
    # -----------------------------------------------------------------------
    # Aba 3 – Resumo automático da EDA
    # -----------------------------------------------------------------------
    with tab_eda:
        st.markdown('<div class="datacard">', unsafe_allow_html=True)

        st.subheader("Resumo textual da Análise Exploratória (EDA)")

        st.markdown(
            "Esta aba exibe um resumo gerado automaticamente a partir da EDA inicial "
            "executada quando o arquivo foi carregado."
        )

        if st.button("Atualizar resumo da EDA", key="refresh_eda"):
            with st.spinner("Consultando resumo da EDA na memória..."):
                resumo = smart_memory_lookup_tool(
                    "resumo da eda",
                    llm=llm,
                    memory=memory_instance,
                )
            st.markdown("### Resumo da EDA")
            st.write(resumo)
        else:
            st.info("Clique em **Atualizar resumo da EDA** para ver a interpretação atual dos dados.")

        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Carregue um arquivo CSV e informe sua chave da API para começar a exploração.")
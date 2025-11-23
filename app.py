import re
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_groq import ChatGroq  # usando Groq Llama 3
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage  # Novo import necessário
from io import StringIO

# Carrega variáveis de ambiente do arquivo .env (se houver)
load_dotenv()

# --- Funções Auxiliares ---

# Função do REPL Python que será utilizada pelo LangChain
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


# Função que será usada como ferramenta pelo agente
def python_repl_ast_tool(code: str, df: pd.DataFrame = None) -> str:
    # Normaliza entrada removendo marcas de markdown
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

    # 🔹 REMOVE plt.show(...) e plt.savefig(...) se o modelo insistir em usar
    cleaned = re.sub(r"plt\.show\s*\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"plt\.savefig\s*\([^)]*\)", "", cleaned)

    # opcional: remove vírgulas soltas tipo "df.mean(), df.median()"
    # que geram duas expressões na mesma linha
    if (
        "," in cleaned
        and "\n" not in cleaned
        and "(" not in cleaned
        and cleaned.startswith("df.")
    ):
        # força o modelo a executar uma coisa só por vez
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        cleaned = parts[-1]  # fica só com a última expressão

    return python_repl_ast(cleaned, df=df)

# --- Função de Análise Inicial (EDA) e Gravação na Memória ---

def initial_analysis_and_memory(df: pd.DataFrame, memory: ConversationBufferMemory, llm) -> None:
    """
    Realiza uma Análise Exploratória de Dados (EDA) genérica em uma AMOSTRA do dataset
    e grava um resumo na memória, cuidando para não ultrapassar limites de tokens.
    A função NÃO assume nenhum domínio específico (fraude, cartão, etc.).
    """

    # ---------- 1) Amostra para evitar prompts gigantes ----------
    if len(df) > 500:
        sample_df = df.sample(500, random_state=42)
    else:
        sample_df = df.copy()

    # Colunas numéricas e categóricas
    numeric_cols = sample_df.select_dtypes(include="number").columns.tolist()
    object_cols = sample_df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Limitar número de colunas numéricas no resumo
    if len(numeric_cols) > 10:
        numeric_cols = numeric_cols[:10]

    # ---------- 2) Textos base de EDA ----------
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

    # Estatísticas categóricas simples (frequências das primeiras colunas categóricas)
    cat_summary_parts = []
    max_cat_cols = 5
    for col in object_cols[:max_cat_cols]:
        vc = sample_df[col].value_counts(dropna=False).head(10)
        cat_summary_parts.append(f"Frequências da coluna categórica '{col}':\n{vc.to_string()}\n")
    if cat_summary_parts:
        cat_str = "\n".join(cat_summary_parts)
    else:
        cat_str = "Não foram encontradas colunas categóricas relevantes na amostra."

    # Algumas linhas de exemplo
    head_str = sample_df.head(5).to_string()

    # ---------- 3) Função auxiliar para truncar textos muito grandes ----------
    def trim(text: str, max_len: int = 2000) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + "\n...[texto truncado para caber no limite de tokens]..."

    info_str = trim(info_str, 2000)
    describe_num_str = trim(describe_num_str, 2000)
    head_str = trim(head_str, 2000)
    corr_str = trim(corr_str, 2000)
    cat_str = trim(cat_str, 2000)

    # ---------- 4) Prompt genérico para o LLM ----------
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
- Descreva, em detalhes, as principais características do dataset:
  - tipos de variáveis (numéricas, categóricas, datas, etc.);
  - distribuição geral (ordens de grandeza, variáveis com pouca variação, possíveis outliers);
  - presença ou ausência de valores ausentes.
- Comente possíveis relações interessantes entre variáveis com base na correlação numérica.
- Se existir alguma coluna que pareça ser "alvo" (por exemplo, colunas binárias ou com poucos valores distintos),
  mencione isso apenas como HIPÓTESE, sem assumir domínio específico (não assuma que é fraude, classe, rótulo, etc.).
- Destaque qualquer desequilíbrio forte de categorias (por exemplo, variável muito desbalanceada).
- Sugira exemplos de perguntas úteis que o usuário poderia fazer ao agente para aprofundar a análise
  (por exemplo: estatísticas de uma coluna específica, comparação entre grupos, gráficos, etc.).

Organize a resposta em tópicos e parágrafos curtos.
"""

    # ---------- 5) Chamar o LLM ----------
    analise_inicial = llm.predict(prompt_analise)

    # ---------- 6) Registrar na memória ----------
    memory.chat_memory.add_message(
        HumanMessage(content="Resultado da análise exploratória inicial (EDA) sobre o dataset carregado.")
    )
    memory.chat_memory.add_message(
        AIMessage(content=analise_inicial)
    )

    # Resumo sintético adicional (opcional)
    memory.save_context(
        {"input": "Resumo sintético da EDA do dataset carregado."},
        {"output": analise_inicial[:1000]}
    )

# --- Função da Ferramenta de Consulta à Memória (smart_memory_lookup_tool) ---

def smart_memory_lookup_tool(query: str, llm, memory) -> str:
    """
    Usa o texto salvo da EDA na memória para responder perguntas
    sem recalcular nada. Otimizado para consumir poucos tokens.
    Funciona para QUALQUER dataset tabular.
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

    # Perguntas do tipo "resuma a EDA", "quais as conclusões", etc.
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
- as principais características do dataset (tipos de variáveis, distribuição geral, eventuais outliers);
- quaisquer desequilíbrios importantes entre categorias ou grupos;
- relações relevantes entre variáveis que foram destacadas na EDA;
- pontos de atenção que merecem análises adicionais.

Não repita o texto original; produza apenas um resumo claro e direto.
"""
    else:
        # Pergunta específica sobre algum ponto da EDA
        sub_prompt = f"""
Abaixo está o texto de uma Análise Exploratória de Dados (EDA) sobre um dataset tabular em CSV:

[EDA]
{memory_content}
[/EDA]

Pergunta do usuário: "{query}"

Responda em Português do Brasil, em 1 ou 2 frases no máximo,
trazendo apenas a informação mais diretamente relacionada à pergunta,
com base NO TEXTO da EDA acima.

Se o texto da EDA não contiver informação suficiente, responda exatamente:
"Não encontrei essa informação na EDA.".
"""

    resposta = llm.predict(sub_prompt)
    return resposta


def make_memory_tool(memory, llm):
    """
    (Mantido para possível uso futuro, mas não é usado no fluxo principal.)
    Cria a ferramenta 'buscar_memoria_EDA' que o agente pode chamar.
    """
    def _inner(query: str) -> str:
        return smart_memory_lookup_tool(query, llm=llm, memory=memory)
    
    return Tool(
        name="buscar_memoria_EDA",
        func=_inner,
        description=(
            "Use esta ferramenta para consultar o texto da análise exploratória (EDA) que já foi salva na memória. "
            "Ideal para: resumos, conclusões gerais, interpretação textual da base e explicações qualitativas."
        )
    )

# --- Função para detectar perguntas de RESUMO / CONCLUSÕES da EDA ---

def is_summary_question(text: str) -> bool:
    """
    Heurística simples para identificar perguntas que pedem RESUMO ou CONCLUSÕES da EDA,
    em vez de novos cálculos numéricos ou gráficos.
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

# --- Prefixo para o Agente Principal focado em python_repl_ast ---

prefix_python_agent = """
Você é um especialista em Análise Exploratória de Dados (EDA) para QUALQUER dataset tabular em CSV.

Você tem acesso a UMA ferramenta:

1) python_repl_ast
   - Executa código Python usando o DataFrame 'df'.
   - Use para: cálculos numéricos, estatísticas, proporções, contagens, agrupamentos e geração de gráficos.

REGRAS GERAIS:

- Sempre que a pergunta envolver números, estatísticas, colunas, tipos de dados OU gráficos,
  você DEVE chamar a ferramenta python_repl_ast ao menos uma vez.

- Antes de usar qualquer coluna, se você não souber os nomes ainda, execute:
  Action: python_repl_ast
  Action Input: df.columns

- Não invente nomes de colunas. Use apenas nomes que existam em df.columns.

REGRAS PARA GRÁFICOS:

- Se a pergunta pedir um histograma de uma coluna 'X':
    - Use SEMPRE o padrão:

      plt.figure(figsize=(10,6))
      serie = df['X'].dropna()
      plt.hist(serie, bins=50)
      plt.title('Histograma da coluna X')
      plt.xlabel('X')
      plt.ylabel('Frequência')

- Se a pergunta pedir filtragem entre percentis 1% e 99%:
      serie = df['X'].dropna()
      serie = serie[(serie >= serie.quantile(0.01)) & (serie <= serie.quantile(0.99))]

- Se a pergunta pedir escala logarítmica no eixo X:
      plt.xscale('log')

- Nunca use parâmetros estranhos em plt.hist (como `kde=True`).

RESPOSTA FINAL:

- Depois de receber a Observation de python_repl_ast, responda diretamente ao usuário em Português do Brasil,
  usando a palavra-chave:

  Final Answer: <texto>

- Na resposta final, cite pelo menos 1 ou 2 valores numéricos concretos retornados pela Observation,
  quando fizer sentido, e faça 1 ou 2 frases de interpretação simples sobre esses valores.

- Não chame nenhuma outra ferramenta além de python_repl_ast.
"""

# --- Função para Inicialização e Carregamento dos Dados ---

def load_data(uploaded_file, openai_api_key, memory, llm_container):
    """
    Carrega o dataset, inicializa df, realiza a análise inicial (EDA) e armazena tudo no session_state.
    """
    
    if uploaded_file is not None:
        
        # Lê o CSV em um DataFrame
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df

        # -----------------------------------------
        # 1. Reset de memória SOMENTE se o arquivo mudou
        # -----------------------------------------
        last_file = st.session_state.get("current_file_name")

        if last_file != uploaded_file.name:
            # Novo arquivo → reset total da memória
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            st.session_state['memory_instance'] = memory
            st.session_state['current_file_name'] = uploaded_file.name

        else:
            # Mesmo arquivo → mantém memória existente
            if memory is None:
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
                st.session_state['memory_instance'] = memory

        # -----------------------------------------
        # 2. Inicializa o LLM se necessário
        # -----------------------------------------
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

        # -----------------------------------------
        # 3. Executa EDA inicial
        # -----------------------------------------
        if llm is not None:
            with st.spinner("Executando análise exploratória inicial com o LLM..."):
                initial_analysis_and_memory(df, memory, llm)

            st.success("Análise inicial registrada na memória com sucesso!")
        else:
            st.error("Não foi possível inicializar o LLM. Verifique sua chave Groq API.")

        return df, memory, llm_container

    return None, memory, llm_container

# --- Configurações Iniciais e Layout do Streamlit ---

st.set_page_config(
    page_title="Agente Inteligente de EDA Genérico",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Agente de Análise Exploratória de Dados (EDA)")
st.markdown(
    "Este agente realiza **Análise Exploratória de Dados (EDA)** em qualquer dataset tabular em CSV, "
    "permitindo cálculos estatísticos, visualizações e um resumo textual automático da base."
)

# Garante que o DataFrame persista durante as interações
if 'df' not in st.session_state:
    st.session_state['df'] = None
    st.session_state['memory_instance'] = None
    st.session_state['llm_container'] = {"llm": None}

# --- Sidebar: Upload de Dados e Configuração da API ---

st.sidebar.header("Configurações")

uploaded_file = st.sidebar.file_uploader("Faça upload do arquivo CSV", type=["csv"])

api_key = st.sidebar.text_input("Sua Chave Groq API", type="password", value=os.getenv("GROQ_API_KEY", ""))

# Botão para carregar dados e inicializar tudo
if st.sidebar.button("Carregar dados e executar análise inicial"):
    with st.spinner("Carregando dados e executando análise inicial..."):
        df, memory_instance, llm_container = load_data(
            uploaded_file,
            api_key,
            st.session_state.get('memory_instance'),
            st.session_state.get('llm_container')
        )
        st.session_state['memory_instance'] = memory_instance
        st.session_state['llm_container'] = llm_container
else:
    df = st.session_state.get('df')
    memory_instance = st.session_state.get('memory_instance')
    llm_container = st.session_state.get('llm_container')

# --- Seção Principal: Exibição do Dataset e Interação com o Agente ---

if df is not None and memory_instance is not None and llm_container.get("llm") is not None:
    st.subheader("Pré-visualização dos Dados")
    st.dataframe(df.head())

    st.markdown("### Perguntas sugeridas para o desafio")
    st.markdown(
    """
1. **Quais são os tipos de dados (numéricos, categóricos) presentes no dataset?**  
2. **Quais colunas numéricas têm maior média e maior desvio padrão?**  
3. **Existem variáveis com forte correlação entre si? Quais?**  
4. **Há colunas com distribuição muito desbalanceada (por exemplo, uma categoria quase sempre igual)?**  
5. **Gere um gráfico (histograma ou boxplot) para uma coluna numérica de interesse.**  
        """
)

    st.markdown("### Faça sua pergunta ao agente")

    user_question = st.text_input("Digite sua pergunta sobre o dataset:")

    if st.button("Perguntar ao agente"):
        if user_question.strip() == "":
            st.warning("Por favor, digite uma pergunta.")
        else:
            llm = llm_container.get("llm")
            if llm is None:
                st.error("LLM não está inicializado. Recarregue os dados e a análise inicial.")
            else:
                # Roteamento simples:
                # - se for pergunta de RESUMO / CONCLUSÕES da EDA → usa diretamente a memória
                # - caso contrário → usa agente com python_repl_ast
                if is_summary_question(user_question):
                    with st.spinner("Consultando o agente (resumo da EDA)..."):
                        response = smart_memory_lookup_tool(
                            user_question,
                            llm=llm,
                            memory=memory_instance,
                        )

                    st.markdown("### Resposta do agente:")
                    st.write(response)

                else:
                    # Cria ferramenta específica para este dataset (apenas python_repl_ast)
                    tools = [
                        Tool(
                            name="python_repl_ast",
                            func=lambda code: python_repl_ast_tool(code, df=df),
                            description=(
                                "Use esta ferramenta para executar código Python diretamente no DataFrame 'df'. "
                                "Ideal para: contagens, proporções, médias, correlações, gráficos, agrupamentos, etc. "
                                "Sempre que a pergunta envolver números, estatísticas ou visualizações, use esta ferramenta."
                            )
                        )
                    ]

                    # Inicializa o agente com a ferramenta python_repl_ast
                    agent = initialize_agent(
                        tools,
                        llm,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=True,
                        memory=memory_instance,
                        handle_parsing_errors=True,   # não tenta “consertar” para sempre
                        max_iterations=4,              # limite duro de passos
                        early_stopping_method="generate",
                    )

                    # Chama o agente com o prefixo de instruções + pergunta do usuário
                    full_prompt = f"{prefix_python_agent}\n\nPergunta do usuário: {user_question}"
                    
                    with st.spinner("Consultando o agente..."):
                        response = agent.run(full_prompt)
                    
                    st.markdown("### Resposta do agente:")
                    st.write(response)

                    # Após a chamada do agente, verifica se um gráfico foi salvo
                    TEMP_PLOT_PATH = "temp_plot.png"  # Recria a constante para este escopo

                    if os.path.exists(TEMP_PLOT_PATH):
                        st.subheader("Visualização Gerada:")
                        
                        # Exibe a imagem salva no disco
                        st.image(TEMP_PLOT_PATH)
                        
                        # Opcional: Remova o arquivo para que a próxima execução não pegue o gráfico antigo
                        os.remove(TEMP_PLOT_PATH)
else:
    st.info("Por favor, carregue um arquivo CSV e insira sua chave da API para começar.")
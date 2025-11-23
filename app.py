import streamlit as st
import pandas as pd
import os
import io
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

def executar_codigo_python(codigo: str, df: pd.DataFrame = None):
    """
    Executa código Python arbitrário dentro de um escopo controlado,
    incluindo 'df', 'pd', 'sns' e 'plt'.
    """
    # Criar um ambiente seguro com apenas as variáveis necessárias
    local_vars = {
        "df": df,
        "pd": pd,
        "sns": sns,
        "plt": plt
    }
    
    try:
        # Usa exec para executar o código Python com acesso ao df, pd, sns, plt
        exec(codigo, {}, local_vars)
        
        # Se o código gerar um gráfico, podemos salvar o resultado em buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        
        return buf  # Retorna o buffer da imagem
    except Exception as e:
        return str(e)

# Função do REPL Python que será utilizada pelo LangChain
def python_repl_ast(code: str, df: pd.DataFrame = None):
    """
    Executa código Python usando o REPL com acesso ao df.
    """
    # Cria diretório temporário, se necessário
    TEMP_PLOT_PATH = "temp_plot.png"

    # Limpa gráfico anterior
    if os.path.exists(TEMP_PLOT_PATH):
        os.remove(TEMP_PLOT_PATH)
    
    # Ambiente controlado
    local_env = {
        "df": df,
        "pd": pd,
        "sns": sns,
        "plt": plt
    }
    
    # Cria um buffer para capturar qualquer saída de texto
    text_output_buffer = StringIO()
    
    try:
        # Redireciona a saída padrão para o buffer
        import sys
        original_stdout = sys.stdout
        sys.stdout = text_output_buffer
        
        # Executa o código
        exec(code, {}, local_env)
        
        # Se um gráfico foi criado, salva em arquivo
        plt.savefig(TEMP_PLOT_PATH)
        plt.close()
        
        # Restaura stdout
        sys.stdout = original_stdout
        
        # Retorna a saída (se houver) e a indicação de que a imagem foi salva
        output_text = text_output_buffer.getvalue()
        if output_text.strip():
            return f"Código executado com sucesso.\nSaída:\n{output_text}\nGráfico salvo em {TEMP_PLOT_PATH}"
        else:
            return f"Código executado com sucesso. Gráfico salvo em {TEMP_PLOT_PATH}"
    
    except Exception as e:
        # Restaura stdout em caso de erro
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

    return python_repl_ast(cleaned, df=df)

# --- Função de Análise Inicial (EDA) e Gravação na Memória ---

def initial_analysis_and_memory(df: pd.DataFrame, memory: ConversationBufferMemory, llm) -> None:
    """
    Realiza análise exploratória inicial (EDA) em uma AMOSTRA do dataset
    e grava um resumo na memória, cuidando para não ultrapassar limites
    de tokens da API (Groq).
    """

    # ---------- 1) Reduzir o tamanho da base usada na EDA ----------
    # Amostra de no máximo 500 linhas para evitar prompts gigantes
    if len(df) > 500:
        sample_df = df.sample(500, random_state=42)
    else:
        sample_df = df.copy()

    # Considerar apenas colunas numéricas (para describe/correlação)
    numeric_cols = sample_df.select_dtypes(include="number").columns.tolist()

    # Limitar para, no máximo, 10 colunas numéricas
    if len(numeric_cols) > 10:
        numeric_cols = numeric_cols[:10]

    # ---------- 2) Gerar textos de EDA de forma controlada ----------
    # Vamos evitar df.info() completo, que é muito verboso.
    # Em vez disso, usamos dtypes + shape.
    info_str = f"Shape do dataset original: {df.shape}\n"
    info_str += f"Shape da amostra usada na EDA: {sample_df.shape}\n\n"
    info_str += "Tipos de dados das colunas numéricas consideradas:\n"
    info_str += sample_df[numeric_cols].dtypes.to_string()

    # Estatísticas descritivas apenas das colunas numéricas selecionadas
    if numeric_cols:
        describe_str = sample_df[numeric_cols].describe().to_string()
        corr = sample_df[numeric_cols].corr()
        corr_str = corr.to_string()
    else:
        describe_str = "Não foram encontradas colunas numéricas na amostra."
        corr_str = "Não foi possível calcular correlação (sem colunas numéricas)."

    # Algumas linhas de exemplo (poucas)
    head_str = sample_df.head(5).to_string()

    # ---------- 3) Função auxiliar para truncar textos muito grandes ----------
    def trim(text: str, max_len: int = 2000) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + "\n...[texto truncado para caber no limite de tokens]..."

    info_str = trim(info_str, 2000)
    describe_str = trim(describe_str, 2000)
    head_str = trim(head_str, 2000)
    corr_str = trim(corr_str, 2000)

    # ---------- 4) Construir o prompt final (bem mais enxuto) ----------
    prompt_analise = f"""
Você é um especialista em análise de dados, com foco em detecção de fraudes em transações de cartão de crédito.

Recebeu os seguintes resultados de uma Análise Exploratória de Dados (EDA) executada em uma **amostra** do dataset:

1) Informações gerais do dataset:
{info_str}

2) Estatísticas descritivas das principais variáveis numéricas:
{describe_str}

3) Primeiras linhas da amostra:
{head_str}

4) Matriz de correlação (apenas das principais variáveis numéricas):
{corr_str}

Tarefa:
- Explique, em detalhes, os principais pontos de atenção da base.
- Destaque o desbalanceamento da variável alvo (fraude vs não fraude), se existir.
- Explique o significado das variáveis de forma geral, considerando que elas foram geradas por PCA e podem não ter interpretação direta.
- Dê uma visão geral do que seria importante o usuário perguntar ao agente para entender melhor riscos e padrões de fraude nesse dataset.

Responda de forma organizada, com tópicos, sempre em Português do Brasil.
"""

    # ---------- 5) Chamar o LLM com o prompt reduzido ----------
    analise_inicial = llm.predict(prompt_analise)

    # ---------- 6) Registrar na memória ----------
    memory.chat_memory.add_message(
        HumanMessage(content="Resultado da análise exploratória inicial (EDA) sobre o dataset de fraudes.")
    )
    memory.chat_memory.add_message(
        AIMessage(content=analise_inicial)
    )

    # Também podemos gravar um resumo curto adicional (opcional)
    memory.save_context(
        {"input": "Resumo sintético da EDA de fraudes."},
        {"output": analise_inicial[:1000]}
    )
# --- Função da Ferramenta de Consulta à Memória (smart_memory_lookup_tool) ---

def smart_memory_lookup_tool(query: str, llm, memory) -> str:
    """
    Ferramenta que usa um sub-LLM para buscar a resposta na memória e retornar apenas o texto ou o dado.
    
    - Se a query for 'Análise Exploratória Completa', extrai o bloco de conclusões.
    - Caso contrário, extrai um dado específico (média, correlação) da tabela.
    """
    
    # 1. Obtém o conteúdo completo da memória (o texto longo da EDA)
    memory_content = memory.buffer_as_str 
    
    # 2. Lógica para definir o prompt de busca interno (o 'sub-LLM')
    
    # ⚠️ CASO 1: EXTRAÇÃO DE RESUMO/CONCLUSÕES (Query Genérica)
    if query.strip().lower() in [
        "análise exploratória completa",
        "analise exploratoria completa",
        "análise exploratória",
        "analise exploratoria",
        "resuma a eda",
        "resumo da eda",
        "conclusões da eda",
        "conclusoes da eda"
    ]:
        sub_prompt = f"""
Você recebeu o seguinte texto da memória, que contém a análise exploratória de dados (EDA) sobre um dataset de fraudes:

[INÍCIO DA MEMÓRIA]
{memory_content}
[FIM DA MEMÓRIA]

Sua tarefa:
- Extraia APENAS a parte de conclusões/resumo geral da EDA, explicando os principais insights e riscos.
- Responda de forma organizada, em tópicos, SEM mencionar 'memória', 'EDA original' ou 'texto acima'.
- Responda em Português do Brasil.
"""
        
        resposta = llm.predict(sub_prompt)
        return resposta
    
    # ⚠️ CASO 2: EXTRAÇÃO DE UM DADO ESPECÍFICO
    else:
        sub_prompt = f"""
Você recebeu o seguinte texto da memória (resultado de uma EDA de fraudes):

[INÍCIO DA MEMÓRIA]
{memory_content}
[FIM DA MEMÓRIA]

A pergunta do usuário é:
'{query}'

Sua tarefa:
- Localize, dentro dos números do texto da memória, APENAS o valor ou informação que responda diretamente a esta pergunta.
- Não explique contexto, não resuma nada além do necessário.
- Se a pergunta for sobre média, correlação, máximo, mínimo ou contagem, devolva apenas o número ou a frase curta.
- Se não encontrar exatamente o dado, responda 'Não encontrei esse valor na memória.'.

Responda em Português do Brasil.
"""
        resposta = llm.predict(sub_prompt)
        return resposta

# Cria o Tool do LangChain para usar no agente
def make_memory_tool(memory, llm):
    """
    Cria a ferramenta 'buscar_memoria_EDA' que o agente pode chamar.
    """
    def _inner(query: str) -> str:
        return smart_memory_lookup_tool(query, llm=llm, memory=memory)
    
    return Tool(
        name="buscar_memoria_EDA",
        func=_inner,
        description=(
            "Use esta ferramenta para buscar, na memória, informações da análise exploratória inicial (EDA) "
            "do dataset de fraudes. Ideal para: conclusões, resumos e dados específicos (ex: média, correlação)."
        )
    )

# --- Prefixo Completo para o Agente Principal ---
prefix_completo = (
    "Você é um especialista em análise de dados. Suas ferramentas são 'python_repl_ast' e 'buscar_memoria_EDA'. "
    "Sua missão é SEMPRE fornecer uma ANÁLISE DETALHADA e SEMPRE responder em Português do Brasil. "
    "A fonte primária e mais confiável de informação é a sua memória, acessada por 'buscar_memoria_EDA'."
    
    # ⚠️ REGRAS DE BUSCA E EXTRAÇÃO DE DADOS 
    
    # Regra 1: Ação para Resumo/Conclusões (Reforça a conversão da intenção)
    "1. SE A PERGUNTA DO USUÁRIO BUSCAR ANÁLISE INICIAL, CONCLUSÕES, RESUMO GERAL OU INTERPRETAÇÃO DO DATASET, "
    "VOCÊ DEVE CHAMAR 'buscar_memoria_EDA' com uma query genérica como 'Resumo da EDA', 'Conclusões da EDA', "
    "ou 'Análise exploratória completa'. Sempre retorne uma explicação detalhada em tópicos, traduzida para "
    "português, com foco em Fraudes em transações de Cartão de Crédito. Essa ação deve ser sua prioridade absoluta "
    "para essas perguntas."
    
    # Regra 2: Ação para Dados Específicos
    "2. Se a pergunta for sobre um dado específico (média, correlação, máximo, mínimo, contagem de fraudes etc.), "
    "você deve chamar 'buscar_memoria_EDA' com a pergunta completa (ex: 'Qual a correlação de V17 com Class?')."
    
    # ⚠️ REGRAS DE BUSCA E EXTRAÇÃO DE DADOS (Vamos focar na prioridade)
    # Regra 3 e 4: Como extrair
    "3. Após usar 'buscar_memoria_EDA', leia a 'Observation' e **responda ao usuário apenas com as informações "
    "pertinentes** à pergunta atual, sem repetir o texto inteiro da memória."
    "4. Para extrair correlação, procure o valor na linha da variável e coluna 'Class' dentro da tabela de "
    "estatísticas da Observation."
    
    # ⚠️ SOBRE O USO DA FERRAMENTA PYTHON_REPL_AST
    "5. Só chame 'python_repl_ast' quando a pergunta do usuário requerer um NOVO CÁLCULO ou NOVA VISUALIZAÇÃO, "
    "como 'faça um boxplot da variável X por Y' ou 'gere um novo gráfico de barras das fraudes por faixa de valor'. "
    "Nesses casos, gere código Python limpo, claro e comentado, usando sempre 'df' como DataFrame."
    "6. SEMPRE retorne o resultado em formato amigável, descrevendo brevemente o que o gráfico ou cálculo significa."
    
    # ⚠️ ESTILO DA RESPOSTA
    "7. Responda em Português do Brasil, com linguagem clara e organizada em tópicos ou parágrafos curtos."
    "8. Seja didático, focando em explicar o que os resultados significam em termos de risco de fraude em cartão."
)

# --- Função para Inicialização e Carregamento dos Dados ---

def load_data(uploaded_file, openai_api_key, memory, llm_container):
    """
    Carrega o dataset, inicializa df, realiza a análise inicial (EDA) e armazena tudo no session_state.
    """
    if uploaded_file is not None:
        # Lê o CSV em um DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Armazena o DataFrame no session_state para uso posterior
        st.session_state['df'] = df
        
        # Inicializa a memória, se ainda não estiver criada
        if memory is None:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            st.session_state['memory_instance'] = memory
        
        # Inicializa o LLM se ainda não existir no container
        if llm_container.get("llm") is None:
            # Aqui é onde adaptamos para Groq
            if openai_api_key:
                # Nota: Você pode usar a otimização com st.cache_resource aqui!
                #llm = ChatGroq(api_key=openai_api_key, temperature=0.0, model="llama3-70b-8192")
                llm = ChatGroq(
                    api_key=openai_api_key,
                    temperature=0.0,
                    #model="llama-3.1-8b-instant",
                    model="llama-3.3-70b-versatile",
                )
            else:
                llm = None
            
            # 2. Lógica de Carregamento de Dados (BLOCO 2)
            # Garante que o llm esteja no container (poderia ser no session_state)
            llm_container["llm"] = llm
        
        else:
            llm = llm_container["llm"]
        
        # Executa a análise exploratória inicial e grava tudo na memória
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
    page_title="Agente Inteligente de EDA para Fraudes",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💳 Agente de Análise de Transações de Cartão de Crédito")
st.markdown(
    "Este agente é especializado em **Análise Exploratória de Dados (EDA)** para detecção de fraudes em transações de cartão de crédito, "
    "especialmente em conjuntos de dados desbalanceados e transformados via PCA."
)

# Garante que o DataFrame persista durante as interações
if 'df' not in st.session_state:
    st.session_state['df'] = None
    st.session_state['memory_instance'] = None
    st.session_state['llm_container'] = {"llm": None}

# --- Sidebar: Upload de Dados e Configuração da API ---

st.sidebar.header("Configurações")

uploaded_file = st.sidebar.file_uploader("Faça upload do arquivo CSV de transações", type=["csv"])

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
1. **Qual a proporção de transações fraudulentas em relação ao total da base?**  
2. **Quais variáveis parecem ter maior correlação com a variável alvo (fraude)?**  
3. **Há diferenças relevantes no comportamento das variáveis entre transações fraudulentas e não fraudulentas?**  
4. **Quais riscos principais podem ser inferidos a partir desta base de dados?**  
5. **Gere um gráfico que ajude a visualizar a relação entre as transações fraudulentas e o valor das transações.**  
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
                # Cria ferramentas específicas para este dataset
                memory_tool = make_memory_tool(memory_instance, llm)

                tools = [
                    Tool(
                        name="python_repl_ast",
                        func=lambda code: python_repl_ast_tool(code, df=df),
                        description=(
                            "Executa código Python para criar novos gráficos e cálculos usando o DataFrame 'df'. "
                            "Use quando precisar de visualizações ou estatísticas adicionais."
                        )
                    ),
                    memory_tool
                ]

                # Inicializa o agente com as ferramentas
                agent = initialize_agent(
                    tools,
                    llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    memory=memory_instance,
                    handle_parsing_errors=True
                )

                # Chama o agente com o prefixo completo + pergunta do usuário
                full_prompt = f"{prefix_completo}\n\nPergunta do usuário: {user_question}"
                
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
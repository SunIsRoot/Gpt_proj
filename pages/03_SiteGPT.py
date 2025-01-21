from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
import streamlit as st

page_title = "SiteGPT"

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

# 수정된 답변 생성 프롬프트
answers_prompt =  ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give an answer within 5 sentences.

    If you don't know, say you don't know and recommend search terms with as many related words as possible.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: Who created Google first?
    Answer: Google was first created by Larry Page.
                                                  
    Question: Does God Exist?
    Answer: I don't know, but I recommend searching for the word Pascal's Wager.
                                                  
    Question: {question}
    """
)
def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            
            Answer the most relevant one and don't change the source.
            
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    result = choose_chain.invoke(
        {"question": question, "answers": condensed}
    )
    return result

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")

@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
        filter_urls=["https://developers.cloudflare.com/ai-gateway/", "https://developers.cloudflare.com/vectorize/","https://developers.cloudflare.com/workers-ai/"]
    )
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()

def load_memory(_):
    return memory.load_memory_variables({})["history"]

st.markdown(
    """
    # SiteGPT
    open sidebar and input your openai key. 
    Ask questions about the content of the Cloudflare website.
    """
)

with st.sidebar:
    url = "https://developers.cloudflare.com/sitemap.xml"
    openai_api_key = st.text_input("Input your OpenAI API Key", type="password")

    if openai_api_key:
        llm = ChatOpenAI(
            temperature=0.1,
            streaming=True,
            callbacks=[ChatCallbackHandler()],
            openai_api_key=openai_api_key,
        )

        memory = ConversationBufferMemory(
            llm=llm,
            return_messages=True,
        )
    

    retriever = load_website(url)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

send_message("I'm familiar with the document, feel free to ask questions", "ai", save=False)
paint_history()
message = st.chat_input("Ask anything about the Cloudflare developers website.")

if message:
    send_message(message, "human")
    chain = (
        {
            "docs": retriever,
            "question": RunnablePassthrough(),
            "history":load_memory,
        }
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_answer)
    )
    with st.chat_message("ai"):
        result = chain.invoke(message)
        memory.save_context({"input": message}, {"output": result.content})

    st.markdown(result.content.replace("$", "\$"))
else:
    st.session_state["messages"] = []

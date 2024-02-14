from langid.langid import LanguageIdentifier, model  # type: ignore
from langchain import PromptTemplate  # type: ignore
from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
from langchain.vectorstores import FAISS  # type: ignore
from langchain.llms import CTransformers  # type: ignore
from langchain.chains import RetrievalQA, ConversationChain # type: ignore
from langchain.chains.conversation.memory import ConversationBufferMemory  # type: ignore
from google.cloud import translate_v2 as translate
from tenacity import retry, wait_exponential, stop_after_attempt
from thefuzz import fuzz  # type: ignore
import chainlit as cl  # type: ignore
import os
import re


cl.config.debug = True
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/jose/Downloads/massive-catfish-411714-89ecb4eed938.json"
# |_ change it for your own Cloud Translate permissions
DB_FAISS_PATH = "vectorstores/db_faiss"



custom_prompt_template = """Use the following information to answer the user's question.
You're designed to be a helpful chatbot that infers or tries to determine what medical condition the user may have
based on the symptoms gived. 

Please try to avoid using the phrase "I don't know" or "I don't know what to do".
Please try to be as specific and accurate as possible.

No matter if you are not a doctor, you can make a diagnosis based on the symptoms the user gives you and recommend some 
specialists the user can consult.

If the user asks you a question in another language that is not English, please follow these steps:
1. Detect the language the user is using.
2. Translate the question to English.
3. Answer the question in English.
4. Translate your English response to the language the user is using.

If you don't know the answer, please type "Currently I don't have enough information to answer your question.", don't
try to make up an answer.

Context: {context}
Question: {question}

Only returns the helpful answer below and nothing else
Helpful Answer:
"""

# --- TRANSLATION FUNCTIONS ---
def detect_language(text):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    return identifier.classify(text)[0]


def translate_google(text, target_language):
    client = translate.Client()
    result = client.translate(text, target_language=target_language)
    return result['translatedText']


# --- PROMPTS AND BOT FUNCTIONS ---
def set_custom_prompt():
    """ 
    Prompt template for QA retrieval for each vectorstore
    """

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt


def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q4_K_S.bin",
        model_type="llama",
        config={
            'max_new_tokens': 2500,
            'temperature': 0.001,  # -> unpredicatbility | Higher = Less predictable "Nonsense"
            'context_length': 3500,
        }
    )
    return llm


def retrieval_qa_chain(llm, prompt, db):  # -> creates a question-answering chain (retrieval)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),  # -> retrieves the top 2 docs from db
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


def qa_bot():  # -> creates a question-answering bot
    embeddings = HuggingFaceEmbeddings(model_name='distilbert-base-nli-mean-tokens', 
                                       model_kwargs= {'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()

    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa


@retry(wait=wait_exponential(), stop=stop_after_attempt(50))
async def call_chain_with_retry(chain, query, cb):
    return await chain.acall(query, callbacks=[cb])


# --- CHAINLINT CODE ---
conversation_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


@cl.on_chat_start
async def start():
    app_user = cl.user_session.get("user")
    
    chain = qa_bot()  # -> start the bot...
    msg = cl.Message(content="Making qeue in Triage...")  # -> first try-message
    await msg.send()

    msg.content = f"Hi, Welcome to Qhali Medical Bot! What is your query?"  # -> real message
    await msg.update()

    cl.user_session.set("memory", conversation_memory)
    cl.user_session.set("chain", chain)  # -> sets the question-answering chain


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # -> gets the question-answering chain
    cl.user_session.set('answer_sent', False)  # -> set answer sent to false to prevent double answers

    cb = cl.AsyncLangchainCallbackHandler(  # -> for handling callbacks to get the answer
        stream_final_answer = True, 
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )

    # Language detection Algorithm:
    user_language = detect_language(message.content)  

    final_query = message.content
    if user_language != 'en':
        final_query = translate_google(message.content, 'en')
    
    #conversation_history = conversation_memory.get("conversation_history", [])
    conversation_history = conversation_memory.buffer
    conversation_history.append(final_query)
    context_aware_query = ' '.join(conversation_history)

    try:
        res = await chain.acall(final_query, callbacks=[cb]) #  Async. calls the question-answering chain with the final query 
        answer = res["result"] 

        final_answer = translate_google(answer, user_language) if user_language != 'en' else answer

        # Check if the answer has already been sent to avoid repetition
        if not cl.user_session.get("answer_sent", False) and answer != cl.user_session.get("last_answer"):
            await cl.Message(content=final_answer).send()
            cl.user_session.set("answer_sent", True)
            cl.user_session.set("last_answer", answer)

        #clear_memory(message.content, conversation_history)

    except Exception as e:
        print(f"An error occured: {e}")

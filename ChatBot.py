from llama_index import StorageContext,SimpleDirectoryReader, GPTVectorStoreIndex,SimpleDirectoryReader, load_index_from_storage, LLMPredictor, PromptHelper, ServiceContext
from langchain.chat_models import ChatOpenAI
import os
from colorama import Fore, Style


def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600 

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)

    index.set_index_id("vector_index")
    index.storage_context.persist('storage')
   
    return index

def ask_ai():
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir='storage')
    # load index
    index = load_index_from_storage(storage_context, index_id="vector_index")
    print(
            Fore.GREEN
            + Style.BRIGHT
            + "Đây là ChatBot là có vấn cho Viện quốc tế của đại học Hutech.\n"
            + Fore.GREEN + Style.BRIGHT + "ChatBot: Tôi có thể giúp gì được cho bạn?"
        )
    while True: 
        query_engine = index.as_query_engine()
        query =  input(Fore.BLUE + Style.BRIGHT +"User: ")
        response = query_engine.query(query)
        print(Fore.LIGHTMAGENTA_EX + Style.BRIGHT + "Chatbot: " + response.response)
        


os.environ["OPENAI_API_KEY"] = "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"

construct_index("data")
ask_ai()
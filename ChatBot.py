from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
import sys
import os
from colorama import Fore, Back, Style

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 4096
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600 

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk('index.json')

    return index

def ask_ai():
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    print(
            Fore.GREEN
            + Style.BRIGHT
            + "Đây là ChatBot là có vấn cho Viện quốc tế của đại học Hutech.\n"
            + Fore.GREEN + Style.BRIGHT + "ChatBot: Tôi có thể giúp gì được cho bạn?"
        )
    while True: 
        query =  input(Fore.WHITE + Style.BRIGHT +"User: ")
        response = index.query(query)
        print(Fore.YELLOW + Style.BRIGHT + Style.NORMAL +"Chatbot: " + response.response)


os.environ["OPENAI_API_KEY"] = "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"

construct_index("data")
ask_ai()
import logging
from io import BytesIO
from langchain.document_loaders import PyPDFLoader

from pathlib import Path

from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceInstructEmbeddings
from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY
from run_localGPT import load_mpt


class Controller:

    def __init__(self):
        device_type = 'cuda'
        logging.info(f"Running on: {device_type}")

        self.embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                                   model_kwargs={"device": device_type})

        # uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
        # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        # load the vectorstore
        self.db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
            client_settings=CHROMA_SETTINGS,
        )
        retriever = self.db.as_retriever()

        # load the LLM for generating Natural Language responses

        # for HF models
        model_id = "TheBloke/vicuna-7B-1.1-HF"
        model_basename = None
        # model_id = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"
        # model_id = "TheBloke/guanaco-7B-HF"
        # model_id = 'NousResearch/Nous-Hermes-13b' # Requires ~ 23GB VRAM. Using STransformers
        # alongside will 100% create OOM on 24GB cards.
        # llm = load_model(device_type, model_id=model_id)

        # for GPTQ (quantized) models
        # model_id = "TheBloke/Nous-Hermes-13B-GPTQ"
        # model_basename = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
        # model_id = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
        # model_basename = "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors" # Requires
        # ~21GB VRAM. Using STransformers alongside can potentially create OOM on 24GB cards.
        # model_id = "TheBloke/wizardLM-7B-GPTQ"
        # model_basename = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"
        # model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"
        # model_basename = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"

        llm = load_mpt()

        self.qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                              return_source_documents=True)

    def chat(self, content: str):
        res = self.qa(content)
        answer, docs = res["result"], res["source_documents"]

        return answer

    def upload_file(self, file):

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len
        )
        texts = self.parse_file2str(file=file)
        docs = text_splitter.create_documents([texts])
        if len(docs) == 0:
            return 'Empty document.'
        self.db.add_documents(
            documents=docs,
            persist_directory=PERSIST_DIRECTORY,
            client_settings=CHROMA_SETTINGS,
        )
        self.db.persist()
        return 'Upload succeed.'

    @staticmethod
    def parse_file2str(file) -> str:
        file_contents = file.read()
        file_stream = BytesIO(file_contents)
        if Path(file.name).name.endswith('.pdf'):
            loader = PyPDFLoader(file.name)
            pages = loader.load_and_split()

            text = ""
            for page in pages:
                text += page.page_content
        else:
            text = file_stream.read().decode('utf-8')
        return text

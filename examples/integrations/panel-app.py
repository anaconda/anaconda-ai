import panel as pn
from anaconda_ai.integrations.panel import AnacondaModelHandler

pn.extension("echarts", "tabulator", "terminal")

llm = AnacondaModelHandler(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0_Q4_K_M.gguf", display_throughput=True
)

chat = pn.chat.ChatInterface(callback=llm.callback, show_button_name=False)

chat.send(
    "I am your assistant. How can I help you?",
    user=llm.model_id,
    avatar=llm.avatar,
    respond=False,
)
chat.servable()

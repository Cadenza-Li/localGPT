import gradio as gr

from controller import Controller

controller = Controller()


def backend(text: str = None, file=None):
    ret = ''
    if text:
        ret = controller.chat(text)
    if file:
        ret = controller.upload_file(file)
    return ret


def web():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            <div style="text-align: center;">

            # Sparkle LLM
            </div>

            ### Chat with your private file.
            """)
        with gr.Tab('ChatBot'):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label='input')
            gr.ClearButton([msg, chatbot])

            def respond(message, chat_history):
                bot_message = controller.chat(message)
                chat_history.append((message, bot_message))
                return "", chat_history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])

        with gr.Tab("File upload"):
            file_input = gr.File()
            file_output = gr.Textbox(label='result')
            file_button = gr.Button("Upload")

            file_button.click(controller.upload_file, inputs=file_input, outputs=file_output)

    demo.launch(share=True, server_name='0.0.0.0', ssl_verify=False, server_port=80)


if __name__ == '__main__':
    web()

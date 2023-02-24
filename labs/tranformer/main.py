import os
os.environ["HF_ENDPOINT"] = "https://huggingface.co"



from transformers import pipeline, set_seed
import gradio as gr
import argparse

def generate(prompt):
    generator = pipeline('text-generation', model='bigscience/mt0-large', max_length=512, device='cuda')
    return generator(prompt)

def main(args):
    demo = gr.Interface(fn=generate, 
                        inputs=[gr.Textbox(placeholder='Enter prompt...', lines=20)], 
                        outputs='text')
    demo.launch()
    
    

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    main(arg_parser.parse_args())


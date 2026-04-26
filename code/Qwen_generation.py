from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os
import sys

################## functions ##################

################## loading models ##################

# using suggested parameters 
# Temperature=0.7, 
# TopP=0.8, 
# TopK=20,
# MinP=0
# turning off the thinking module for efficient generation

# Qwen model
model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model
tokenizer_qwen = AutoTokenizer.from_pretrained(model_name)
model_qwen = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# huggingface
def qwen_gen(messages, tokenizer, model):
    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
    Temperature=0.7, 
    TopP=0.8, 
    TopK=20,
    MinP=0)
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    
    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return content

################## loading models ##################



################## process bg ##################

def process_bg(data):
    data['processed_bg'] = data.apply(lambda row: row['bg'].replace(row['query'], '', 1)  if row['bg'].startswith(row['query']) else row['bg'], axis=1)
    data = data.reset_index(drop = True)
    return data

################## process bg ##################

################## method 1 ##################

# directly asking LLMs for generation
def qwen_conv(query, bg, tokenizer, model):


    messages = [
        {"role": "user", "content": f"Given the query {query} and background information {bg}, generate a natural conversation between a user and a system. In the conversation, the system does not have access to the background information and must ask clarifying questions to better understand and refine the query. The system should always ask clarifying questions based solely on the conversational history, without using any background information. The user knows the background information and can answer questions to clarify the query. Ensure that the user's answers are strictly based on the provided background information. The user should not introduce any new or additional information beyond what is included in the background. Use the following format:\n System: [clarifying question]\n User: [answer based on the background information]"}
    ]
    resutls = qwen_gen(messages, tokenizer, model)

    return resutls
    
################## method 1 ##################   

################## method 2 ##################
# stage 1: extract key points from bg
# stage 2: cq from sys <-> answers from extracted key points
# stage 3: make the whole conversation more natural

def qwen_stage1(query, bg, tokenizer, model, prompt1_path):
    
    prompt1 = Path(prompt1_path).read_text()

    messages = [
        {"role": "system", "content": prompt1},
        {"role": "user", "content": f"original query: {query}, background information: {bg}"}
    ]

    resutls = qwen_gen(messages, tokenizer, model)
    
    #print("thinking content:", thinking_content)
    #print("content:", content)


    return resutls

def qwen_stage2(query, kp, tokenizer, model, prompt2_path):

    prompt2 = Path(prompt2_path).read_text()
    

    messages = [
        {"role": "system", "content": prompt2},
        {"role": "user", "content": f"original query: {query}; extracted key points: {kp}"}
    ]

    resutls = qwen_gen(messages, tokenizer, model)
    
    #print("thinking content:", thinking_content)
    #print("content:", content)


    return resutls

def qwen_stage3(conv, tokenizer, model, prompt3_path):

    prompt3 = Path(prompt3_path).read_text()

    messages = [
        {"role": "system", "content": prompt3},
        {"role": "user", "content": f"conversation: {conv}"}
         ]

    resutls = qwen_gen(messages, tokenizer, model)
    
    #print("thinking content:", thinking_content)
    #print("content:", content)


    return resutls


'''
def qwen_all_the_stage(query, bg, tokenizer, model):

    key_points = qwen_stage1(query,bg, tokenizer, model)
    #print(key_points)
    gen_conversation = qwen_stage2(query, key_points, tokenizer, model)
    #print(gen_conversation)
    refined_conversation = qwen_stage3(gen_conversation, tokenizer, model)

    return key_points, gen_conversation, refined_conversation
'''

################## method 2 ##################

################## functions ##################


################## main ##################

def main(args):
    
    # Step 1: load data
    print(f"Running on {os.path.basename(args.input)} dataset.")
    
    data = pd.read_csv(args.input, lineterminator = '\n')
    processed_data = process_bg(data)

    # Step 2: generating conversations with different methods
    
    # method 1
    if args.method == 1:
        print("Running method 1 for generation.")
        
        for i in tqdm(range(len(processed_data))):
            processed_data.loc[i, 'gen_conv'] = 'User: ' + processed_data['query'][i] + '\n' + qwen_conv(processed_data['query'][i], processed_data['processed_bg'][i], tokenizer_qwen, model_qwen)

        print(f"Saving the dataset on {args.output} for method 1.")
        processed_data.to_csv(args.output)

    # method 2
    elif args.method == 2:
        print("Running method 2 for generation.")

        for i in tqdm(range(len(processed_data))): 
            processed_data.loc[i, 'extracted_points'] = qwen_stage1(processed_data['query'][i], processed_data['processed_bg'][i], tokenizer_qwen, model_qwen, '/workspace/prompt_stage1.txt')
            processed_data.loc[i, 'gen_conv'] = qwen_stage2(processed_data['query'][i], processed_data['extracted_points'][i], tokenizer_qwen, model_qwen, '/workspace/prompt_stage2.txt')
            processed_data.loc[i, 'refined_conv'] = qwen_stage3('User:' + processed_data['query'][i] + '\n' + processed_data['gen_conv'][i],  tokenizer_qwen, model_qwen, '/workspace/prompt_stage3.txt')
           
        print(f"Saving the dataset on {args.output} for method 2.")
        processed_data.to_csv(args.output)
        
################## main ##################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen generation"
    )

    parser.add_argument(
        "--input", "-i", required=True, type=str,
        help="Path to input CSV file"
    )

    parser.add_argument(
        "--output", "-o", required=True, type=str,
        help="Path to save the processed data"
    )

    parser.add_argument(
        "--method", "-m", type=int, choices=[1, 2], required=True,
        help="Choose a method: 1 or 2"
    )

    args = parser.parse_args()
    main(args)

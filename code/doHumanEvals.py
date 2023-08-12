# Author: Shawn Lee
# Date: Aug 2023
# Description: Script to run model simulations given human perspectives on ToMi + BigToM

import csv
from llm_utils import *
from tqdm import tqdm

# Both these functions have identical structures -- the difference is evaluate_question.

def evaluate_tomi(data_dir):
    
    print("#################") 
    print(" Evaluating ToMi ")
    print("#################") 
    
    gpt = ChatGPT("gpt-3.5-turbo")
    
    def evaluate_question(row):
        # Internal function to run simulation on each question.
        perspective = row[0]
        question = row[1]
        answer = row[2]
        question_type = row[3]
        prompt = perspective + "\n" + question
        predicted_answer = gpt.getOutput(prompt)
        if args.verbose:
            print(predicted_answer)
        if answer in predicted_answer:
            return True
        else:
            return False
        
    total = 0
    correct = 0
    
    with open(data_dir, mode='r') as file:
        reader = csv.reader(file)
        for row in tqdm(reader):
            # Evaluate each question
            result = evaluate_question(row)
            if result == True: # writing it like this for clarity's sake
                correct += 1
            total += 1
    
    accuracy = (correct / total) * 100
    
    print(f"Accuracy: {accuracy}") 
    return accuracy
    
def evaluate_bigtom(data_dir):
    
    print("###################") 
    print(" Evaluating BigToM ")
    print("###################") 
    
    gpt = ChatGPT("gpt-3.5-turbo")
    
    def evaluate_question(row):
        perspective = row[0]
        question = row[1]
        answer = row[2]
        answer_key = answer[:2]
        name = answer.split(" ")[1]
        prompt = f"""\
{perspective}
You are {name}.
Based on the above information, answer the following question:
{question}
Answer the questions based on the context. Keep your answer concise, few words are enough, maximum one sentence. Answer as 'Answer:<option>)<answer>'.
"""
        predicted_answer = gpt.getOutput(prompt)
        if args.verbose:
            print(predicted_answer)
        if answer_key in predicted_answer:
            return True
        else:
            return False
        
    total = 0
    correct = 0
    
    with open(data_dir, mode='r') as file:
        reader = csv.reader(file)
        for row in tqdm(reader):
            # Evaluate each question
            result = evaluate_question(row)
            if result == True: # writing it like this for clarity's sake
                correct += 1
            total += 1
    
    accuracy = (correct / total) * 100
    
    print("#################") 
    print(f"Accuracy: {accuracy}") 
    print("#################") 
    return accuracy



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='human_eval_data/tomi_human_evals_all.csv')
    parser.add_argument('--dataset', type=str, default='tomi')
    parser.add_argument('--verbose', action='store_true')
    
    global args
    args = parser.parse_args()
    
    if args.dataset == "tomi":
        evaluate_tomi(args.data_dir)
        
    elif args.dataset == "bigtom":
        evaluate_bigtom(args.data_dir)
    
    else:
        print(f"{args.dataset} is not a valid dataset to evaluate on!")


if __name__ == "__main__":
    main()
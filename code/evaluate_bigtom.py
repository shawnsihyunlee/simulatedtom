# Author: Shawn Lee
# Date: Aug 2023
# Description: Evaluation code to evaluate BigToM.
# Heavily based off of code from github.com/cicl-stanford/procedural-evals-tom/
# Copyright (c) 2023 Kanishk Gandhi, Jan-Philipp Fränken, Tobias Gerstenberg, Noah D. Goodman

import os
import random
import csv
import tqdm
import argparse
import itertools
import wandb
from sim_utils_bigtom import *


DATA_DIR = '../data'
CONDITION_DIR = os.path.join(DATA_DIR, 'conditions')
PROMPT_DIR = 'prompt_instructions'
random.seed(0)


def evaluate_condition(init_belief, variable, condition, gotRight):
    if args.wandb == 1:
        run = wandb.init(
            project=args.project,
            entity=args.entity, 
            config={
                "method": args.method,
                "eval_model": args.eval_model,
                "perspective_model": args.perspective_model,
                "sim_model": args.sim_model,
                "num_probs": args.num_probs,
                "init_belief": init_belief, 
                "variable": variable,
                "condition": condition,   
            },
            tags=args.tags.split(","),
        )
        # Log code as well
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
    
    # Create wandb table
    if args.wandb == 1:
        table = wandb.Table(columns=["question", "gold", "world_state", "agent_state", "prediction", "grade"])
    
    # Print stuff
    print("\n------------------------")
    print("    EVALUATING WITH      ")
    print("------------------------")
    if args.eval_model == None:
        print(f"PER MODEL: {args.perspective_model}")
        print(f"SIM MODEL: {args.sim_model}")
    else:
        print(f"EVAL MODEL: {args.eval_model}")
    print(f"NUM PROBS: {args.num_probs}")
    print(f"METHOD: {args.method}")
    print(f"CONDITION: {init_belief} {variable}, {condition}")
    print("------------------------\n")

    # load condition csv    
    csv_name = os.path.join(CONDITION_DIR, f'{init_belief}_{variable}_{condition}/stories.csv')
    with open(csv_name, "r") as f:
        reader = csv.reader(f, delimiter=";")
        condition_rows = list(reader)

    predicted_answers = []
    graded_answers = []
    

    if condition == "true_belief":
        gotRight[f'{init_belief}_{variable}'] = set()
    
    
    # If we want to use separate models for PT and simulation:
    if args.eval_model == None:
        if "gpt" in args.perspective_model:
            perspectiveModel = ChatGPT(args.perspective_model, temperature=args.temperature, verbose=args.verbose)
        else:
            perspectiveModel = LLM(args.perspective_model, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
        
        if "gpt" in args.sim_model:
            simModel = ChatGPT(args.sim_model, temperature=args.temperature, verbose=args.verbose)
        else:
            simModel = LLM(args.sim_model, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)

    else:
        simModel = None
        if "gpt" in args.eval_model:
            perspectiveModel = ChatGPT(args.eval_model, temperature=args.temperature, verbose=args.verbose)
        else:
            perspectiveModel = LLM(args.eval_model, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)


    for index, row in enumerate(tqdm.tqdm(condition_rows[:args.num_probs])):
        story = row[0]
        question_orig = row[1]
        question = row[1]
        true_answer, wrong_answer = row[2], row[3]
        answers = [true_answer, wrong_answer]
        random.shuffle(answers)
        # Give the answers letters (like choice a, choice b)
        question = f"{question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"
        
        
        # Load different prompts for different methods.
        if args.method == "cot":
            with(open(f'{PROMPT_DIR}/evaluate_cot.txt', 'r')) as f:
                instruction = f.read()
        elif args.method == "oneshot":
            with(open(f'{PROMPT_DIR}/evaluate_1shot.txt', 'r')) as f:
                instruction = f.read()
        elif args.method == "oneshotcot":
            with(open(f'{PROMPT_DIR}/evaluate_cot_1shot.txt', 'r')) as f:
                instruction = f.read()
        else:
            # Baseline prompt
            with(open(f'{PROMPT_DIR}/evaluate.txt', 'r')) as f:
                instruction = f.read()
                
        # Base prompt. The instruction corresponds to the above instruction prompts for diff. methods.
        prompt = f"""{instruction}

Story: {story}
Question: {question}"""

        if args.method == "simulation":
            # If we are doing simulated ToM
            predicted_answer, world = evalQuestion(perspectiveModel, story, question, knowsChange=('true' in condition), perspectiveGold=args.perspectiveGold, simModel=simModel)
            
            # Debugging purposes.
            world_state = world.getWorldState()
            agent_state = world.getAgentState()
            
        elif args.method == "onePromptSimulation":
            # If we are doing simulated ToM but with one prompt (ablation)
            predicted_answer, full_response = oneBigPrompt(perspectiveModel, story, question)
            world_state = full_response
            agent_state = "N/A"
            if len(predicted_answer.strip()) == 0:
                predicted_answer = "Refused to respond"
                
        else:
            # If we are using any other method
            predicted_answer = perspectiveModel.getOutput(prompt)
            world_state = "N/A"
            agent_state = "N/A"
            
            
        # Figure out correct answer key
        if answers[0] == true_answer:
            answer_key = 'a)'
            negative_answer_key = 'b)'
            true_answer = 'a) ' + true_answer
            wrong_answer = 'b) ' + wrong_answer
        else:
            answer_key = 'b)'
            negative_answer_key = 'a)'
            true_answer = 'b) ' + true_answer
            wrong_answer = 'a) ' + wrong_answer
        
        
        if not args.gradeGPT: 
            # We try seeing if the answer key is in the model's response.
            predicted_answer_parsed = predicted_answer
            if answer_key in predicted_answer_parsed.lower():
                graded_answer = 'True'
            elif negative_answer_key in predicted_answer_parsed.lower():
                graded_answer = 'False'
            else:
                graded_answer = 'False'
        else:
            # We ask GPT to grade the question if we cannot parse accurately.
            # This is only used for llama-2 Chain-of-thought right now, because CoT outputs both answers in its reasoning chain.
            grader = ChatGPT("gpt-3.5-turbo")
            
            prompt = f"""\
This is someone's response to a question:

{predicted_answer}

This is the correct answer:

{true_answer}

Is their final answer correct? Output 'True' or 'False' only. If they chose option c) or said something like "neither", output 'False'. If they consider both options but ultimately don't decide, also output 'False'.
"""

            graded_answer = grader.getOutput(prompt)
        
        # If true_belief and got right, track it
        if condition == "true_belief" and graded_answer == 'True':
            gotRight[f'{init_belief}_{variable}'].add(index)
        
        # If false_belief and it didn't get it right for true belief, assess it as wrong.
        if args.dotruefalse and condition == "false_belief" and index not in gotRight[f'{init_belief}_{variable}']:
            graded_answer = 'False'
        
        predicted_answers.append(predicted_answer)
        graded_answers.append(graded_answer)
        
        if args.verbose:
            try:
                print("### Perspective ###")
                print(world.perspective)
            except:
                pass
            print(f"graded answer: {graded_answer}")

        # running accuracy 
        accuracy = graded_answers.count('True') / len(graded_answers)
        if args.wandb == 1:
            wandb.log({"accuracy":accuracy})
        if args.wandb == 1:
            table.add_data(prompt, true_answer, world_state, agent_state, predicted_answer, graded_answer)
    
    if args.wandb == 1:
        wandb.log({"wrong_answers":table})
    if args.wandb == 1:
        wandb.finish()
    
    # Print results
    print("\n------------------------")
    print("         RESULTS        ")
    print("------------------------")
    if args.eval_model == None:
        print(f"PER MODEL: {args.perspective_model}")
        print(f"SIM MODEL: {args.sim_model}")
    else:
        print(f"EVAL MODEL: {args.eval_model}")
    print(f"CONDITION: {init_belief} {variable}, {condition}")
    print(f"ACCURACY: {accuracy:.2%}")
    print("------------------------\n")
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='simulation')
    parser.add_argument('--init_belief', type=str, default="0_forward")
    parser.add_argument('--variable', type=str, default='belief')
    parser.add_argument('--condition', type=str, default='true_belief')
    parser.add_argument('--eval_model', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--num_probs', '-n', type=int, default=200)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--gradeGPT', action='store_true')        # Ask GPT to grade the answers.
    parser.add_argument('--wandb', type=int, default=0)
    parser.add_argument('--tags', type=str, default='debug')
    parser.add_argument('--dotruefalse', action='store_true')
    parser.add_argument('--perspectiveGold', action='store_true') # whether to give gold labels for baseContext, knowsChange by parsing sentence and question type
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--perspective_model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--sim_model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--gpu', type=int, default=0) # which gpu to load on
    parser.add_argument('--eight_bit', action='store_true') # load model in 8-bit?
    
    global args
    args = parser.parse_args()

    
    INIT_BELIEF_LIST = ["0_forward"]
    VARIABLE_LIST = ["belief", "action"]
    CONDITION_LIST = ["true_belief", "false_belief"]
    
    categories = list(itertools.product(INIT_BELIEF_LIST, VARIABLE_LIST, CONDITION_LIST))
    
    
    # The flag "--all" will make us evaluate all categories.
    if args.all:
        gotRight = {}
        for (init_belief, variable, condition) in categories:
            if "backward" in init_belief and "action" in variable:
                continue # The form of "backward action" is not a valid parameter
            
            evaluate_condition(init_belief, variable, condition, gotRight)
    else:
        gotRight = {}
        evaluate_condition(args.init_belief, args.variable, args.condition, gotRight)

if __name__ == '__main__':
    main()
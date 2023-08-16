# Simulated Theory of Mind

Public repository for "Think Twice: Perspective-Taking Improves Large Language Models’ Theory-of-Mind Capabilities".

## Example Commands

*Evaluate BigToM across all categories with GPT-3.5 with simulation method:*
```
$ python evaluate_bigtom.py --eval_model=gpt-3.5-turbo --all --method=simulation
```

*Evaluate BigToM on false belief action with llama2-13b with chain-of-thought method:*
```
$ python evaluate_bigtom.py --eval_model=meta-llama/Llama-2-13b-chat-hf --condition=false_belief --variable=action --method=cot
```

*Evaluate ToMi across all categories with GPT-4 with baseline method, while seeing inputs and outputs of model:*
```
$ python evaluate_tomi.py --eval_model=gpt-4 --method=baseline --verbose
```

*To log to wandb:*

```
$ python evaluate_tomi.py --eval_model=gpt-4 --method=baseline --wandb=1 --tags=gpt4baseline
```

## Requirements

You need the `transformers` library and the `openai` library to run most of the code.
You also need a txt file called `api_key.txt` that contains the OpenAI API key in the same directory as the code to query ChatGPT.

To run local models, we use the HuggingFace pipeline.
Make sure to change the model cache directory -- currently, it is set to `CACHE_DIR = '/scratch/weights/llama2'` within `llm_utils.py`.

## Important key things:
If any accuracies seem abnormally high, please try running with the flag: `--gradeGPT`. Currently most grading is done by manual parsing, but sometimes this fails. For example, when evaluating chain-of-thought on BigToM with Llama2-7b, Llama tends to output both answer choices in its "chain of thought". Thus, PLEASE USE `--gradeGPT` for that specific scenario, e.g.
```
$ python evaluate_bigtom.py --eval_model=meta-llama/Llama-2-7b-chat-hf --all --gradeGPT
```

## Running Evaluations

There are two files used to run evaluations on BigToM and ToMi:

`evaluate_bigtom.py`

`evaluate_tomi.py`

---

### evaluate_bigtom.py



`--eval_model`: model to evaluate on. 
-  `gpt-4`
- `gpt-3.5-turbo`
- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Llama-2-13b-chat-hf`


`--method`: prompting method to evaluate with. 
- `baseline`
- `cot`
- `oneshot`
- `oneshotcot`
- `simulation`
- `onepromptsimulation` : This is the ablation (PT + sim in one prompt)


`--condition`: true or false belief.
- `true_belief`
- `false_belief`



`--variable`: action or belief.
- `belief`
- `action`


Other arguments:

- `--num_probs`: # of problems to evaluate. Default = 200.

- `--temperature`: Model temperature. Default = 0.0.

- `--wandb`: 1 for logging wandb, 0 for local run. Default = 0.

- `--tags`: Comma-separated tags for wandb.

- `--perspective_model`, `--sim_model` : If you want to use different models for perspective-taking and simulating. Default just uses `--eval_model` for both.

- `--gpu` : What GPU you want to run your model on. (Default = 0)


Boolean flags:

- `--all`: This will evaluate all four categories and ignore the `--condition` and `--variable` flags. 

- `--verbose`: Verbosity flag.

- `--gradeGPT`: This uses ChatGPT to grade the responses.

- `--eight_bit`: loads model in 8-bit.

---

### evaluate_tomi.py

The following flags are identical to above:

- `--eval_model`
- `--method`
    - `baseline`
    - `baselineRules`
    - `cot`
    - `cotRules`
    - `oneshot`
    - `oneshotcot`
    - `onePromptSimulation` (ablation)
    - `simulation`
- `--temperature`
- `--num_probs` (default=1000)
- `--wandb`, `--tags`
- `--perspective_model`, `--sim_model`
- `--gpu`
- `--verbose`

The difference for ToMi is the following: there are no `--condition`, `--variable`, `--all` flags, and instead:

- `--category` (Default = `"all"`): This is the question category you want to evaluate for ToMi. For example, `reality`, `first_order_0_no_tom`, etc...

## Repository Structure

`data/` : contains BigToM and ToMI datasets.
- BigToM has all generated data from the original BigToM paper, but we only use `0_forward` conditions.
- `data/fixedtomi/test_balanced.jsonl` contains 1000 ToMI questions, with 100 of each category, disambiguated and without the mislabelled second-order questions (courtesy of Sclar et al.)

`code/` : contains code for evaluations + simulation.
- `code/prompt_instructions` contains baseline prompts for BigToM, as in the original BigToM paper.
- All other prompts are in `prompts_bigtom.py` and `prompts_tomi.py`.
- `evaluate_bigtom.py` and `evaluate_tomi.py` are the evaluation scripts.
- `llm_utils.py` : Wrappers for ChatGPT and local models.
- `sim_utils_*.py` : Utilities for simulation for both datasets.
- `doHumanEvals.py` : Script for running ablation studies involving human perspectives or simulations. Data is in `human_eval_data`.

## Logging to wandb
Feel free to change the command line args of `--project` and `--entity` to log to your own wandb project.

Currently, the scripts log debugging content, including but not limited to the inputs and outputs of the model, in a wandb table, along with the total accuracy and accuracy across categories in wandb graphs.

The code is also logged to wandb for each run for posterity's sake.




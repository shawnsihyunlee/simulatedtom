# Author: Shawn Lee
# Date: Aug 2023
# Description: File with simulation-related prompts for ToMi.

PERSPECTIVE_PROMPT = """\
The following is a sequence of events about some characters, that takes place in multiple locations.
Your job is to output only the events that the specified character, {character}, knows about.
Here are a few rules:
1. A character knows about all events that they do.
2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place.
3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location.

Story:
{story}

What events does {character} know about? Only output the events according to the above rules, do not provide an explanation."""

LLAMA_SIM_PROMPT = """\
{perspective}
You are {name}.
Based on the above information, answer the following question:
{question}
You must choose one of the above choices, do not say there is not enough information. Answer with a single word, do not output anything else. 
"""

GPT_SIM_PROMPT = """\
{perspective}
You are {name}.
Based on the above information, answer the following question:
{question}
Keep your answer concise, one sentence is enough. You must choose one of the above choices.
"""

ABLATION_ONEPROMPT_SIM = """
I will give you a sequence of events about some characters, that takes place in multiple locations. and a question that asks about the sequence of events. Your task is in two steps.

Step 1. You will first output only the events that the specified character, {character}, knows about.
Here are a few rules:
    1. A character knows about all events that they do.
    2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place.
    3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location.

Step 2. You will then imagine you are the main character, {character}, then answer the question given to you based on the story you have rewritten. Ignore the previous sequence of events -- your rewritten sequence of events are now the new events. Do not output a blank answer or say you do not have enough information.

Do Step 1 and Step 2 combined.

Story: {story}
Output the sentences that only {character} knows about.
Question: {question}


Format your answer as follows:
Step 1: (list of events)
Step 2: Answer: (answer to question)
"""


######## Prompts for other methods ##########

baselinePrompt = """\
{story}
{question}
Choose from the following:
{containers_0}, {containers_1}
"""
                
questionPrompt = """\
{question}
Choose from the following:
{containers_0}, {containers_1}
""" 

rulesPrompt = """\
I will give you a story and ask you a question. Answer the question based on the rules.

Here are a few rules:
1. A character knows about all events that they do.
2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place.
3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location.

{story}
{question}
Choose from the following:
{containers_0}, {containers_1}
Keep your answer concise. Answer with a single word.
"""

rulesCoTPrompt = """\
I will give you a story and ask you a question. Answer the question based on the rules.

Here are a few rules:
1. A character knows about all events that they do.
2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place.
3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location.

{story}
{question}
Choose from the following:
{containers_0}, {containers_1}
Reason step by step before answering in 'Thought: Let's think step by step'. Write your final answer as 'Answer: <answer>'. Answer with a single word.
"""

cotPrompt = """
{story}
{question}
Choose from the following:
{containers_0}, {containers_1}
Reason step by step before answering in 'Thought: Let's think step by step'. Write your final answer as 'Answer: <answer>'. Answer with a single word.
"""

oneShotPrompt = """
I will give you a story, then ask you to answer a question. For example:
1 James dislikes the suit.
2 James entered the staircase.
3 Noah entered the staircase.
4 The slacks is in the basket.
5 The basket is in the staircase.
6 Noah exited the staircase.
7 James moved the slacks to the treasure_chest.
8 The treasure_chest is in the staircase.
Where will Noah look for the slacks?
Answer: basket

Now, it is your turn.
{story}
{question}
Choose from the following:
{containers}
"""

oneShotCotPrompt = """
I will give you a story, then ask you to answer a question. For example:
1 James dislikes the suit.
2 James entered the staircase.
3 Noah entered the staircase.
4 The slacks is in the basket.
5 The basket is in the staircase.
6 Noah exited the staircase.
7 James moved the slacks to the treasure_chest.
8 The treasure_chest is in the staircase.
Where will Noah look for the slacks?
Noah sees that the slacks are in the basket, which is located in the staircase. Noah then leaves the staircase. James moves the slacks to the treasure_chest, but Noah does not see this as he is not in the staircase. Thus, Noah is not aware that the location of the slacks has been changed, and he still believes that the slacks are in the basket.
Thus, Noah will look for the slacks in the basket.
Answer: basket

Now, it is your turn.
{story}
{question}
Choose from the following:
{containers}
Think step by step, then output your answer at the end.
"""

            
llamaprompt = """\
{story}
{question}
Choose from the following:
{containers_0}, {containers_1}
Keep your answer concise. Answer with a single word.

The answer is
"""

llamaCotPrompt = """
{story}
{question}
Choose from the following:
{containers_0}, {containers_1}
Reason step by step before answering in 'Thought: Let's think step by step'. Write your final answer as 'Answer: <answer>'. Answer with a single word.

Thought: Let's think step by step.
"""
# Author: Shawn Lee
# Date: Aug 2023
# Description: File with simulation-related prompts for BigToM.

GPT_PERSPECTIVE_PROMPT = """
Imagine you are {name}, and consider this story that has an unexpected event.
{story}
If the last sentence of the story says {name} notices, sees or realizes the unexpected event in this story, simply output the original story with nothing changed.
However, if the sentence says you are not aware of the changes in this story, output only the events you know, i.e., the sentences before the unexpected event happens.
Output either the original story or the edited story, nothing else.

Format your answer as follows:
Sees/Notices/Realizes: (Yes/No)
Story:
"""

LLAMA_PERSPECTIVE_PROMPT = """
Consider this story with an unexpected event.

{story}

Does the story say that {name} notices/sees/realizes the unexpected event?

If so, simply output the original story with nothing changed. 
However, if {name} is not aware of the changes in this story, output only the events that {name} knows, i.e., the events before the unexpected event happens.
"""

LLAMA_SIM_PROMPT = """\
Answer the questions based on the context. Keep your answer concise, few words are enough, maximum one sentence. Answer as 'Answer:<option>)<answer>'.

{perspective}

You are {name}.
{question}
Choose the most straightforward answer.
"""

GPT_SIM_PROMPT = """\
{perspective}
You are {name}.
Based on the above information, answer the following question:
{question}
Answer the questions based on the context. Keep your answer concise, few words are enough, maximum one sentence. Answer as 'Answer:<option>)<answer>'.
"""

ABLATION_ONEPROMPT_SIM = """
I will give you a short story, and a question that asks about the story. Your task is in two steps.

Step 1: Imagine you are {name}, and consider this story that has an unexpected event.
{story}
If the last sentence of the story says {name} notices, sees or realizes the unexpected event in this story, simply output the original story with nothing changed.
However, if the sentence says you are not aware of the changes in this story, output only the events you know, i.e., the sentences before the unexpected event happens.
Output either the original story or the edited story, nothing else.

Format your answer for step 1 as follows:
Sees/Notices/Realizes: (Yes/No)
Story:

2. You will then imagine you are the main character, {name}, then answer the question given to you based on the story you have rewritten. Ignore the previous story -- your rewritten story is now the new story. Do not output a blank answer or say you do not have enough information -- you must choose either choice a) or choice b). Answer as  '<option>) <answer>'.

Here is the story and question.

Story: {story}
Question: {question}

Format your answer as follows:
Step 1:
Step 2: <option>) <answer>
"""
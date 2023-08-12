# Author: Shawn Lee
# Date: Aug 2023
# Description: All utilities for running simulated ToM on BigToM.

from llm_utils import *
from prompts_bigtom import *

class Agent:
    """Agent class for simulation.
    """
    def __init__(self, llm:LLM, name=""):
        self.name = name
        self.perspective = ""
        self.llm = llm
        self.debug = False
        if "llama" in self.llm.model_name:
            # Llama evaluation prompt.
            self.evalPrompt = LLAMA_SIM_PROMPT
        else:
            # ChatGPT evaluation prompt.
            self.evalPrompt = GPT_SIM_PROMPT
    
        # Debugging
        self.wasAsked = None
        self.replied = None

    
    def evalQuestion(self, question:str) -> str:
        """Simulation function.

        Args:
            question (str): BigToM question + answer choices

        Returns:
            str: Answer choice
        """
        
        prompt = self.evalPrompt.format(perspective=self.perspective, name=self.name, question=question)
        choice = self.llm.getOutput(prompt)
        self.wasAsked, self.replied = prompt, choice
        return choice

class World:
    """World class that keeps track of true world state.
    """
    def __init__(self, llm, context:str, agent:Agent = None, perspective:str = None, knowsChange:bool = None, simModel=None):
        self.llm = llm
        self.agent = agent     
        self.context = context # Original BigToM Question
        self.perspective = perspective # Perspective
        self.knowsChange = knowsChange # Does the agent know the change?
        self.agentName = None 
        self.simModel = simModel
        if self.simModel == None: # If sim model not specified, we are just using the same model for PT and simulation.
            self.simModel = self.llm
        
    def takePerspective(self) -> None:
        """Function that takes perspective.
        """
        
        self.agentName = self.context.split(" ")[0]
        
        story = self.context
        name = self.agentName
        
        # Prompt for GPT
        prompt = GPT_PERSPECTIVE_PROMPT
        if "llama" in self.simModel.model_name:
            # Prompt for llama, as it outputs useless things
            prompt = LLAMA_PERSPECTIVE_PROMPT

        perspective = self.simModel.getOutput(prompt.format(story=story, name=name)).split(":")[-1].strip()

        if self.perspective is None: # using model, not perspective gold
            self.perspective = perspective


    def setupAgent(self) -> None:
        # Create agent if no agent was given (which is always lol. just futureproofing)
        if self.agent is None:
            self.agent = Agent(self.simModel, "")
        
        # Give it its name! (play god a little here)        
        self.agent.name = self.agentName
        
        # Give it the perspective
        self.agent.perspective = self.perspective

    # Debugging Function
    def getWorldState(self) -> str:
        return f"""\
Agent Name: {self.agentName}

Context: {self.context}

Perspective: {self.perspective}
""" 
    
    # Debugging Function
    def getAgentState(self) -> str:
        return(f"""
Agent was prompted:
{self.agent.wasAsked}

and outputted:
{self.agent.replied}
""")

# Evaluation function
def evalQuestion(llm:ChatGPT, context:str, question:str, knowsChange:bool, perspectiveGold:bool, simModel=None) -> Tuple[str, World]:
    if perspectiveGold: # give gold labels for perspective, knowsChange by parsing sentence and question type
        perspective = '.'.join(context.split('.')[:-3]) + '.'
        knowsChange = knowsChange
    else:
        perspective = None
        knowsChange = None
    world = World(llm, context, perspective=perspective, knowsChange=knowsChange, simModel=simModel)
    world.takePerspective()
    world.setupAgent()
    answer = world.agent.evalQuestion(question)
    return answer, world


# Function for ablation
def oneBigPrompt(llm, story, question):
    name = story.split()[0]
    prompt = ABLATION_ONEPROMPT_SIM
    output = llm.getOutput(prompt.format(name=name, story=story, question=question))
    answer = output[output.rfind("Answer:")+len("Answer:"):].strip()
    return answer, output

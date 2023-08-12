# Author: Shawn Lee
# Date: Aug 2023
# Description: All utilities for running simulated ToM on ToMi.

from llm_utils import *
from prompts_tomi import *


class Agent:
    """Agent class for simulation.
    """
    def __init__(self, llm, name=""):
        self.name = name
        self.perspective = ""
        self.llm = llm
        
        # Here, we differ the simulation prompts per model.
        if "llama" in llm.model_name:
            # Simulation prompt for Llama
            self.evalPrompt = LLAMA_SIM_PROMPT
        else: 
            # Simulation prompt for ChatGPT
            self.evalPrompt = GPT_SIM_PROMPT
        # DEBUGGING PURPOSES
        self.wasAsked = None
        self.replied = None
        self.debug = False
    
    def evalQuestion(self, question:str) -> str:
        """Answers BigToM question based on agent belief.

        Args:
            question (str): BigToM question + answer choices

        Returns:
            str: Answer choice
        """
        prompt = self.evalPrompt.format(perspective=self.perspective, name=self.name, question=question)
        choice = self.llm.getOutput(prompt)
        self.wasAsked, self.replied = prompt, choice
        return choice, self.perspective # Return perspective for debugging purposes

class World:
    """World class that keeps track of true world state.
    """
    def __init__(self, llm, context:str, debug=False, simModel=None):
        self.llm = llm
        self.agentNames = []
        self.agents:Dict[str, Agent] = {}                 # Note this is a dictioary for maybe future multi-agent simulations
        self.context = context                            # Original ToMi Question (JSON format)
        self.story:str = context["story"]                 # ToMI story
        self.perspectives : Dict[str, str] = {}           # Perspectives (Dict for same reason as above)
        self.debug = debug                                # This is verbose
        self.simModel = simModel
        if self.simModel == None:
            self.simModel = self.llm
        
    def parseCharacters(self) -> None:
        # We use ChatGPT here a little bit, just to parse the characters in this story.
        prompt = f"""\
{self.story}
What are the characters in this story?
Output only the character names, separated by commas."""
        gpt = ChatGPT("gpt-3.5-turbo")
        self.agentNames = gpt.getOutput(prompt).replace(" ", "").split(",")
        if self.debug:
            print("Agent names:", self.agentNames)
    
    def takePerspective(self, characterName=None) -> None:
        
        # Here is the perspective-taking.
        prompt = PERSPECTIVE_PROMPT
        
        if characterName is None:
        # Then parse perspective for each character. We don't use this though.
            for character in self.agentNames:
                self.perspectives[character] = self.llm.getOutput(prompt.format(story=self.story, character=character))
                if self.debug:
                    print(f"Perspective of {character}:", self.perspectives[character])
        else:
            # Take perspective for given character.
            self.perspectives[characterName] = self.llm.getOutput(prompt.format(story=self.story, character=characterName))
            if self.debug:
                print(f"Perspective of {characterName}:", self.perspectives[characterName])
        

    def setupAgent(self) -> None:
        """Create agents.
        """
        for agentName, perspective in self.perspectives.items():
            agent = Agent(self.simModel, name=agentName)
            agent.perspective = perspective
            # Add to agent dictionary
            self.agents[agentName] = agent
        
        
    def getWorldState(self) -> str:
        return f"""\
TBD! I don't have anything here yet.
""" 
    
    def getAgentState(self) -> str:
        return(f"""
Agent was prompted:

{self.agent.wasAsked}

and outputted:

{self.agent.replied}
""")

    def evalQuestion(self, question, agentName=None) -> str:
        if agentName is None:
            # This is a question about the truth.
            # For truth questions, there's really no point of simulation/perspective taking, so we just ask the LLM the question (same as baseline).
            return self.llm.getOutput(f"{self.story}\nBased on the above information, answer the following question:\n{question}"), "Truth Question"
        else:
            # Here we ask the agent to simulate.
            return self.agents[agentName].evalQuestion(question)
        

# Evaluation function
def evalQuestion(llm:ChatGPT, context:str, question:str, debug=False, simModel=None) -> Tuple[str, str]:
    """End to end function for Tomi evaluation.
    """
    # Create world for perspective taking
    world = World(llm, context, debug=debug, simModel=simModel)
    
    # What's the subject of the question (who's perspective do we have to take?)
    questionSubject = question.split(" ")[2]
    
    # What are the characters in this story?
    world.parseCharacters()
    
    # Is this a reality/memory question?
    if questionSubject not in world.agentNames:
        # Question about zeroth-order belief (Reality/Memory question)
        answer, perspective = world.evalQuestion(question)
        return answer, perspective
    
    # Else, run the pipeline.
    
    # 1. Perspective-taking
    world.takePerspective(characterName=questionSubject)
    world.setupAgent()
    
    # 2. Simulation
    answer, perspective = world.evalQuestion(question, agentName=questionSubject)
    
    # ...and we have our answer!
    return answer, perspective





def oneBigPrompt(llm, story, question):
    """
    Function for the one-prompt-simulation ablation.
    """
    
    # What's the subject of the question (who's perspective do we have to take?)
    character = question.split(" ")[2]
    prompt = f"""\
{story}
What are the characters in this story?
Output only the character names, separated by commas."""
    agentNames = llm.getOutput(prompt).replace(" ", "").split(",")
    # Is this a reality/memory question?
    if character not in agentNames:
        answer, perspective = llm.getOutput(f"{story}\nBased on the above information, answer the following question:\n{question}"), "Truth Question"
        return answer, perspective
    # Else, run the one-prompt simulation.
    prompt = ABLATION_ONEPROMPT_SIM
    output = llm.getOutput(prompt.format(character=character, story=story, question=question))
    # Parse output.
    # ChatGPT (unlike llama-2) is pretty good at following formatting instructions, so this pretty much never fails. 
    answer = output[output.rfind("Answer:")+len("Answer:"):].strip()
    return answer, output

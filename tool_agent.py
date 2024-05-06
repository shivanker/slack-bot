import argparse
import logging
import json
from typing import List, Optional

from haystack import Pipeline, component  # type: ignore
from haystack.components.builders import PromptBuilder  # type: ignore
from pydantic import BaseModel, ValidationError
from haystack.components.generators import OpenAIGenerator  # type: ignore

from chat_models import *

# Configure the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log-level", default="INFO", help="Set the log level")
args = parser.parse_args()

# Configure logger
logging.basicConfig(
    level=getattr(logging, args.log_level.upper()),
    format="%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s",
    datefmt="%m%d %H:%M:%S",
)


class City(BaseModel):
    name: str
    country: str
    population: int


class CitiesData(BaseModel):
    cities: List[City]


# Define the component input parameters
@component
class OutputValidator:
    def __init__(self, pydantic_model: BaseModel):
        self.pydantic_model = pydantic_model
        self.iteration_counter = 0

    # Define the component output
    @component.output_types(
        valid_replies=List[str],
        invalid_replies=Optional[List[str]],
        error_message=Optional[str],
    )
    def run(self, replies: List[str]):

        self.iteration_counter += 1
        logging.getLogger().info("Got message from LLM: %s", replies[0])

        ## Try to parse the LLM's reply ##
        # If the LLM's reply is a valid object, return `"valid_replies"`
        try:
            output_dict = json.loads(replies[0])
            self.pydantic_model.parse_obj(output_dict)
            print(
                f"OutputValidator at Iteration {self.iteration_counter}: Valid JSON from LLM - No need for looping: {replies[0]}"
            )
            return {"valid_replies": replies}

        # If the LLM's reply is corrupted or not valid, return "invalid_replies" and the "error_message" for LLM to try again
        except (ValueError, ValidationError) as e:
            print(
                f"OutputValidator at Iteration {self.iteration_counter}: Invalid JSON from LLM - Let's try again.\n"
                f"Output from LLM:\n {replies[0]} \n"
                f"Error from OutputValidator: {e}"
            )
            return {"invalid_replies": replies, "error_message": str(e)}


prompt_template = """
Create a JSON object from the information present in this passage: {{passage}}.
Only use information that is present in the passage. Follow this JSON schema, but only return the actual instances without any additional schema definition:
{{schema}}
Make sure your response is a dict and not a list.
{% if invalid_replies and error_message %}
  You already created the following output in a previous attempt: {{invalid_replies}}
  However, this doesn't comply with the format requirements from above and triggered this Python exception: {{error_message}}
  Correct the output and try again. Just return the corrected output without any extra explanations.
{% endif %}
"""
prompt_builder = PromptBuilder(
    template=prompt_template, required_variables=["passage", "schema"]
)
output_validator = OutputValidator(pydantic_model=CitiesData)  # type: ignore


llama8 = OpenAIGenerator(
    model="llama3-8b-8192",
    api_base_url="https://api.groq.com/openai/v1",
    api_key=Secret.from_env_var("GROQ_API_KEY"),
)


pipeline = Pipeline(max_loops_allowed=5)

# Add components to your pipeline
pipeline.add_component(instance=prompt_builder, name="prompt_builder")
pipeline.add_component(instance=llama8, name="llm")
pipeline.add_component(instance=output_validator, name="output_validator")

# Now, connect the components to each other
pipeline.connect("prompt_builder", "llm")
pipeline.connect("llm", "output_validator")
# If a component has more than one output or input, explicitly specify the connections:
pipeline.connect("output_validator.invalid_replies", "prompt_builder.invalid_replies")
pipeline.connect("output_validator.error_message", "prompt_builder.error_message")
pipeline.draw("./auto-correct-pipeline.png")  # type: ignore

json_schema = CitiesData.schema_json(indent=2)
passage = "Berlin is the capital of Germany. It has a population of 3.850.809. Paris, France's capital, has 2.161 million residents. Lisbon is the capital and the largest city of Portugal with the population of 504,718."

import argparse
import json
import logging
from typing import Any, List, Optional

from haystack import Pipeline, component  # type: ignore
from haystack.components.builders import DynamicChatPromptBuilder  # type: ignore
from haystack.components.generators.chat import OpenAIChatGenerator  # type: ignore
from haystack.dataclasses import ChatMessage  # type: ignore
from chat_models import *
from pydantic import BaseModel, ValidationError


# Configure the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log-level", default="INFO", help="Set the log level")
args = parser.parse_args()

# Configure logger
logging.basicConfig(
    format="%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s",
    datefmt="%m%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, args.log_level.upper()))


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
        valid_replies=List[ChatMessage],
        invalid_replies=Optional[List[str]],
        error_message=Optional[str],
    )
    def run(self, replies: List[ChatMessage]):

        self.iteration_counter += 1
        logger.debug(f"[{self.iteration_counter}] Got message from LLM: \n{replies[0]}")

        ## Try to parse the LLM's reply ##
        # If the LLM's reply is a valid object, return `"valid_replies"`
        try:
            content = replies[0].content
            if content.count("```") >= 2:
                output_dict = json.loads(content.split("```")[1])
            else:
                output_dict = json.loads(content)
            self.pydantic_model.parse_obj(output_dict)
            logger.info(
                f"OutputValidator at Iteration {self.iteration_counter}: Valid JSON from LLM - No need for looping: {replies[0]}"
            )
            return {"valid_replies": replies}

        # If the LLM's reply is corrupted or not valid, return "invalid_replies" and the "error_message" for LLM to try again
        except (ValueError, ValidationError) as e:
            logger.error(
                f"OutputValidator at Iteration {self.iteration_counter}: Invalid JSON from LLM - Let's try again.\n"
                f"Output from LLM:\n {replies[0].content} \n"
                f"Error from OutputValidator: {e}"
            )
            return {"invalid_replies": [replies[0].content], "error_message": str(e)}


messages = [
    ChatMessage.from_user(
        """
Create a JSON object from the information present in this passage: {{passage}}.
Only use information that is present in the passage.
Follow this JSON schema, but only return the actual instances without any additional schema definition:
{{schema}}
Make sure your response is a dict and not a list.
{% if invalid_replies and error_message %}
  You already created the following output in a previous attempt: {{invalid_replies}}
  However, this doesn't comply with the format requirements from above and triggered this Python exception: {{error_message}}
  Correct the output and try again. Just return the corrected output without any extra explanations.
{% endif %}
"""
    )
]
prompt_builder = DynamicChatPromptBuilder(
    runtime_variables=["invalid_replies", "error_message"]
)
for optional_var in ["invalid_replies", "error_message"]:
    component.set_input_type(prompt_builder, optional_var, Optional[Any], None)

output_validator = OutputValidator(pydantic_model=CitiesData)  # type: ignore


pipeline = Pipeline(max_loops_allowed=5)

# Add components to your pipeline
pipeline.add_component(instance=prompt_builder, name="prompt_builder")
pipeline.add_component(instance=LLAMA3_8B, name="llm")
pipeline.connect("prompt_builder.prompt", "llm.messages")


pipeline.add_component(instance=output_validator, name="output_validator")
pipeline.connect("llm", "output_validator.replies")
pipeline.connect("output_validator.invalid_replies", "prompt_builder.invalid_replies")
pipeline.connect("output_validator.error_message", "prompt_builder.error_message")

pipeline.draw("./auto-correct-pipeline.png")  # type: ignore

json_schema = CitiesData.schema_json(indent=2)
passage = "Berlin is the capital of Germany. It has a population of 3.850.809. Paris, France's capital, has 2.161 million residents. Lisbon is the capital and the largest city of Portugal with the population of 504,718."

res = pipeline.run(
    data={
        "prompt_builder": {
            "template_variables": {"passage": passage, "schema": json_schema},
            "prompt_source": messages,
        },
    }
)

print("Structured output from LLM: \n" + res["output_validator"]["valid_replies"][0].content)  # type: ignore

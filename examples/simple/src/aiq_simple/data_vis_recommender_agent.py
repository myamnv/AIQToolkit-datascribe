# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from pydantic import Field

from aiq.data_models.component_ref import LLMRef


# You are a classifier that determines if user input is a dataset or a chat message.
# Respond based on these rules:
# 1. If the input appears to be data (CSV-like, table-like, or structured data) -> 'dataset'
# 2. If the input is a natural language message or question -> 'chat'
# 3. Only respond with one word: either 'dataset' or 'chat'.

SYSTEM_MESSAGE = """You are a data visualization expert that recommends charts for a given dataset.
Respond based on these rules:
1. If there are multiple charts that could be appropriate, only recommend up to three of the most suitable charts.
2. If there are only one or two appropriate charts, that is fine. Don't try to force more recommendations than needed
3. If there are no appropriate charts, respond with an empty JSON array.
4. Include a short description of the chart type and why it was chosen.
5. Only return valid raw JSON without any markdown formatting or backticks. Never wrap the response in ```json or any other markdown markers. 
   Example JSON Structure: {[{"chartType": "string", "description": "string"}, ... ]}
6. Only recommend charts that are available in the Nivo library and use the same naming convention as the Nivo library.
"""

class DataVisRecommenderAgentConfig(FunctionBaseConfig, name="data_vis_recommender_agent"):
    description: str = Field(default=("You are a data visualization expert that recommends charts for a given dataset. Args: dataset_input: str"),
                             description="Description of the tool for the agent.")
    tool_names: list[str] = []
    llm_name: LLMRef

@register_function(config_type=DataVisRecommenderAgentConfig)
async def data_vis_recommender_agent(config: DataVisRecommenderAgentConfig, builder: Builder):

    async def _arun(dataset_input: str) -> str:
        """
        data visualization expert that recommends charts for a given dataset

        Args:
            dataset_input (str): dataset to analyze and recommend visualizations for. Can be any structured data format like CSV, JSON, etc.

        Returns:
            str: Analysis conclusion from the LLM agent
        """

        tools = builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        # Bind tools to LLM for parallel execution
        llm_n_tools = llm.bind_tools(tools, parallel_tool_calls=True)

        # Define agent function that processes messages with LLM
        def data_vis_recommender_llm_agent(state: MessagesState):
            sys_msg = SystemMessage(content=SYSTEM_MESSAGE)
            return {"messages": [llm_n_tools.invoke([sys_msg] + state["messages"])]}

        # Build the agent execution graph
        builder_graph = StateGraph(MessagesState)

        # Add nodes for agent and tools
        builder_graph.add_node("data_vis_recommender_llm_agent", data_vis_recommender_llm_agent)
        builder_graph.add_node("tools", ToolNode(tools))

        # Configure graph edges for execution flow
        builder_graph.add_edge(START, "data_vis_recommender_llm_agent")
        builder_graph.add_conditional_edges(
            "data_vis_recommender_llm_agent",
            tools_condition,
        )
        builder_graph.add_edge("tools", "data_vis_recommender_llm_agent")

        # Compile the execution graph
        agent_executor = builder_graph.compile()

        # Execute analysis and get response
        input_message = dataset_input
        response = await agent_executor.ainvoke({"messages": [HumanMessage(content=input_message)]})

        conclusion = response["messages"][-1].content

        return conclusion

    yield FunctionInfo.from_fn(
        _arun,
        description=config.description,
    )
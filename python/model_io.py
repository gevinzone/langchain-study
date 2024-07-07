#!/usr/bin/env python3

import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

deployment = "gpt-35-turbo-instruct"


def create_output_parser() -> StructuredOutputParser:
    response_schemas = [
        ResponseSchema(
            name="description",
            description="鲜花的描述文案",
        ),
        ResponseSchema(
            name="reason",
            description="为什么要写这个文案",
        ),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    return output_parser


def create_prompt(price: str, flower_name: str, response_format_instructions: str) -> str:
    template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
{response_format_instructions}
"""
    prompt = PromptTemplate.from_template(
        template,
        partial_variables={"response_format_instructions": response_format_instructions})
    return prompt.format(price=price, flower_name=flower_name)


def create_llm():
    llm = AzureOpenAI(
        deployment_name=deployment,
        model_name=deployment,
        temperature=0.8,
        max_tokens=512,
    )
    return llm


def create_result(flower, price, response_format_instructions, llm, output_parser):
    prompt = create_prompt(price, flower, response_format_instructions)
    output = llm.invoke(prompt)
    result = output_parser.parse(output)
    result["flower"] = flower
    result["price"] = price
    return result


def create(flowers: list, prices: list) -> pd.DataFrame:
    data_frame = pd.DataFrame(columns=["flower", "price", "description", "reason"])
    llm = create_llm()
    output_parser = create_output_parser()
    output_format_instruction = output_parser.get_format_instructions()
    for flower, price in zip(flowers, prices):
        result = create_result(flower, price, output_format_instruction, llm, output_parser)
        data_frame.loc[len(data_frame)] = result

    return data_frame


if __name__ == "__main__":
    flowers = ["玫瑰", "百合", "康乃馨"]
    prices = ["50", "30", "20"]

    df = create(flowers, prices)
    print(df.to_dict(orient="records"))
    df.to_csv("output.csv", index=False)

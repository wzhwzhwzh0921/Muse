# -*- coding: utf-8 -*-
import base64
import json
import os
import random

from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from openai import OpenAI
from pydantic import BaseModel, Field

from constants import (CHIT_CHAT, CLARIFIER, CONVERSATION_ANALYZER, MINI_MODEL,
                       MODEL, PRODUCT_SELECTOR, QUERY_GENERATOR, RECOMMENDER)
from create_item_db import ItemVector


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class Recsys:
    def __init__(self, db_path: str, data_path: str, model_path: str, base_url: str, api_key: str) -> None:
        self.client = self.get_client(base_url=base_url, api_key=api_key)
        self.db_path = db_path
        self.data_path = data_path
        self.model_path = model_path
        self.item_db = self.get_item_db()
        self.recommended_items = []
        self.last_query = ""

    def get_item_db(self):
        myDB = ItemVector(
            db_path=self.db_path,
            model_name=self.model_path,
            llm=None,
            data_path=self.data_path,
            force_create=False,
            use_multi_query=False,
        )
        return myDB
    def get_client(self, base_url, api_key):
        client = OpenAI(base_url=base_url,
                        api_key=api_key)
        return client

    def clear(self):
        self.recommended_items = []
        self.last_query = ""

    def find_target_item(self, background, requirement):
        class SelectorResponse(BaseModel):
            index: int = Field(..., description="The index of the item")
            reason: str = Field(..., description="The reason for the selection")

        self.item_db.retriever.search_kwargs = {"k": 10}
        results = self.item_db.search_retriever(requirement)
        content_system = PRODUCT_SELECTOR
        content_user = []
        item_texts = ""
        item_images = []
        for i, raw_item in enumerate(results, 1):
            item = raw_item.metadata
            item_id = item["item_id"]
            categories = item['categories']
            new_description = item['new_description']
            image_url = f"images_main/{item_id}.jpg"
            item_texts += f"{i}: New_descriptions: {new_description}, Categories:{categories}. \n"
            item_images.append(image_url)
        content_user.append(
            {"type": "text", "text": f"Scenario: {background}. \n Item List: {item_texts}"}, )

        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})

        completion = self.client.beta.chat.completions.parse(
            model=MINI_MODEL,
            messages=messages,
            temperature=0.2,
            response_format=SelectorResponse,
        )
        selection = completion.choices[0].message.parsed
        index = selection.index-1
        result = results[int(index)].metadata
        return result

    def get_requirements(self, conversations):
        content_system = CONVERSATION_ANALYZER
        content_user = f"Conversations: {conversations}"

        response = self.client.chat.completions.create(
            model=MINI_MODEL,
            messages=[
                {"role": "system", "content": content_system},
                {"role": "user", "content": content_user}
            ],
            temperature=0.5,
        )

        requirements = response.choices[0].message.content
        return requirements

    def once_query(self, last_query, conversations, mentioned_ids):
        requirements = self.get_requirements(conversations)
        new_query = self.clarifier(requirements)
        result_item, final_query = self.querier(last_query, new_query, mentioned_ids)

        return result_item, final_query

    def clarifier(self, new_query):
        content_system = CLARIFIER
        content_user = f"Initial query: {new_query} \n "
        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        response = self.client.chat.completions.create(
            model=MINI_MODEL,
            messages=messages,
            temperature=0.5,
        )
        final_query = response.choices[0].message.content

        return final_query

    def querier(self, last_query, new_query, mentioned_ids):
        content_system = QUERY_GENERATOR
        content_user = f"Old query: {last_query} \n " \
                       f"New query: {new_query} \n"
        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        response = self.client.chat.completions.create(
            model=MINI_MODEL,
            messages=messages,
            temperature=0.5,
        )
        new_query = response.choices[0].message.content
        final_query = self.clarifier(new_query)
        self.item_db.retriever.search_kwargs = {"k": 10}
        results = self.item_db.search_retriever(final_query)

        filtered_results = []
        for result in results:
            if result.metadata['item_id'] in mentioned_ids:
                continue
            filtered_results.append(result)
        # results = rerank(results)
        result = random.choice(filtered_results).metadata
        return result, final_query

    def chit_chat(self, conversations):
        content_system = CHIT_CHAT
        content_user = f"History conversations: {conversations} \n "
        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.5,
        )
        chit_chat = response.choices[0].message.content
        return chit_chat

    def recommender(self, conversations, recommended_item):
        content_system = RECOMMENDER
        content_user = []
        raw_item = recommended_item
        item = raw_item
        item_id = item["item_id"]
        description = item['description']
        features = item['features']
        title = item['title']
        categories = item['categories']
        image_url = f"images_main/{item_id}.jpg"
        item_texts = f" Title: {title}; Description: {description}; Some attributes: {features}; Categories: {categories}. \n"
        content_user.append(
            {"type": "text", "text": f"Conversation History: {conversations}. \n Recommended Item: {item_texts}"}, )

        base64_image = encode_image(image_url)
        content_user.append({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}})

        response = self.client.chat.completions.create(
            model=MINI_MODEL,
            messages=[
                {"role": "system", "content": content_system},
                {"role": "user", "content": content_user}
            ],
            temperature=0.5,
        )
        recommendation = response.choices[0].message.content
        return recommendation


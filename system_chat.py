# -*- coding: utf-8 -*-
import json
import random

from openai import OpenAI
from pydantic import BaseModel


from create_item_db import ItemVector
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.callbacks import get_openai_callback
import base64


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
        class MathResponse(BaseModel):
            index: int

        self.item_db.retriever.search_kwargs = {"k": 10}
        results = self.item_db.search_retriever(requirement)
        content_system = "You are a product selector. " \
                         "You will receive two pieces of information: " \
                         "1. The scenario for the user's purchase trip. " \
                         "2. The text descriptions and pictures of the three alternative products. " \
                         "Please select the most suitable product as the user's final choice and give your reasons." \
                         "Output the index of the item and reasons"
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
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            response_format=MathResponse,
        )
        selection = completion.choices[0].message.parsed
        index = selection.index-1
        result = results[int(index)].metadata
        return result

    def get_requirements(self, conversations):
        content_system = "You are a conversation analyzer. " \
                         "Please carefully analyze the following conversations between a user and a conversational recommendation system. \n" \
                         "Based on the conversations, identify and summarize the core needs and intentions of the user. \n" \
                         "Pay special attention to: " \
                         "1. The user's directly expressed needs or questions " \
                         "2. The user's implicit needs or areas of interest " \
                         "3. The user's emotional state and tone " \
                         "4. The user's feedback on the system's responses " \
                         "5. Any specific preferences, limitations, or criteria mentioned by the user " \
                         "Clearly summary the user's main needs for the products. " \
                         "Output only the summary."
        content_user = f"Conversations: {conversations}"

        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
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
        content_system = "You are a professional product query clarification assistant. " \
                         "You will be given a initial query from a user for shopping products." \
                         "Your task is to transform vague or general product requests in the query into more specific, precise queries. " \
                         "This will help to more accurately match the user's actual needs. \n" \
                         "Please follow these guidelines:" \
                         "1. Analyze the user's initial query, identifying any vague or unclear parts." \
                         "2. Convert ambiguous descriptions into specific, searchable product features or categories." \
                         "3. Retain any parts of the user's query that are already clear." \
                         "4. If the user's query contains multiple aspects, clarify each aspect separately. " \
                         "5. If there is no vague part, directly return the initial query.\n" \
                         "Here are some examples: " \
                         "User Query: Waterproof bag. " \
                         "Clarified: Waterproof backpack or handbag, made of nylon or PVC material. " \
                         "Explanation: Specified bag types and common waterproof materials. \n" \
                         "Now, please clarify the user's query according to the above guidelines and examples." \
                         "Output only the clarified query"
        content_user = f"Initial query: {new_query} \n "
        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0.5,
        )
        final_query = response.choices[0].message.content

        return final_query

    def querier(self, last_query, new_query, mentioned_ids):
        content_system = "You are a useful query generator. " \
                         "Specifically, you are tasked to help user find suitable products in the dataset by generate specific query. \n" \
                         "You will be given two information: " \
                         "1. Old query you generated before suggesting the user's preference extracted in the history. " \
                         "2. New query analyzed from current conversations. \n" \
                         "Use these two information to generate a new query. " \
                         "Please return only the query. "
        content_user = f"Old query: {last_query} \n " \
                       f"New query: {new_query} \n"
        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
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
        content_system = "You are an advanced conversational recommender system engaging in a friendly chat with a user. \n" \
                         "In the previous round of conversation, you recommended an item to the user, but instead of directly accepting or rejecting the recommendation, " \
                         "the user responded with a piece of casual chit-chat related to the recommended item. \n" \
                         "Your task is to: \n" \
                         "1. Respond to the user's chit-chat, maintaining a natural flow of conversation. " \
                         "2. Utilize all previous dialogue content with this user to demonstrate an understanding of their interests and experiences. " \
                         "3. Subtly showcase your personality and humor in the chit-chat, but maintain an appropriate balance without overshadowing the conversation. " \
                         "4. Reguide the conversation back to your last round of recommendations in an appropriate way. " \
                         "Remember, your response should: \n" \
                         "1. Be relevant to the user's chit-chat topic. " \
                         "2. Show empathy and understanding and be natural and engaging. " \
                         "3. Length should be limited to 1-2 sentences. " \
                         "Now, based on the user's previous chit-chat statement, generate an appropriate chit-chat response. \n" \
                         "Output only the chit-chat statement."
        content_user = f"History conversations: {conversations} \n "
        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        response = self.client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            temperature=0.5,
        )
        chit_chat = response.choices[0].message.content
        return chit_chat

    def recommender(self, conversations, recommended_item):
        content_system = "You are an advanced conversational recommender system engaged in a chat with a user." \
                         "Your task now is to recommend a product to the user. " \
                         "You have access to two key resources: " \
                         "The entire conversation history with this user up to this point. " \
                         "A detailed text description and image of the product you need to recommend. " \
                         "Your objective is to craft a product recommendation that: " \
                         "1. Avoid repeating the same start sentences in history conversations." \
                         "2. Seamlessly fits into the current conversation flow. " \
                         "3. Demonstrates understanding of the user's preferences, needs, and previous interactions. " \
                         "4. Highlights the most relevant features of the product based on what you know about the user. " \
                         "5. Incorporates relevant details from the product's text description and visual elements. " \
                         "6. Aiming for a length of 1-2 sentences. " \
                         "Output only the the introduction for the product!"
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
            model='gpt-4o-mini',
            messages=[
                {"role": "system", "content": content_system},
                {"role": "user", "content": content_user}
            ],
            temperature=0.5,
        )
        recommendation = response.choices[0].message.content
        return recommendation


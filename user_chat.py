# -*- coding: utf-8 -*-
import base64
import json
import random

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI




def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class User:
    def __init__(self, base_url: str, api_key: str):
        self.client = self.get_client(base_url, api_key)
        self.scenario = None
        self.target_item = None

    def get_client(self, base_url, api_key):
        client = OpenAI(base_url=base_url,
                        api_key=api_key)
        return client

    def load_user(self, scenario, target_item):
        self.scenario = scenario
        self.target_item = target_item

    def clear_user(self):
        self.scenario = None
        self.target_item = None

    def chit_chat(self, conversations, recommended_item):
        content_system = "You are roleplaying as a user engaging in a conversation with a recommender system. " \
                         "In the previous turn, the system recommended a product to you. " \
                         "Your task now is to respond with a chit-chat message rather than directly accepting or rejecting the recommendation. \n" \
                         "You are given three information. " \
                         "1. The entire conversation history between you and the recommender system. " \
                         "2. The product's text and image information that was just recommended to you. " \
                         "3. Background of the purchase trip \n" \
                         "Your chit-chat response should: " \
                         "1. Be related to the recommended product or the general topic it belongs to, but not directly address whether you want to purchase it or not. " \
                         "2. Draw from your 'personal experiences' or interests as established in the previous conversation. " \
                         "3. It can be reasonably connected with the history conversations" \
                         "4. Potentially include a brief anecdote, opinion, or question that's tangentially related to the product or its use. " \
                         "5. The length is limited in 1-2 sentences. " \
                         "6. Optionally, hint at your lifestyle, preferences, or future plans without explicitly connecting them to the product recommendation. \n" \
                         "Now, based on the context of the conversation and the product just recommended to you, generate a natural chit-chat response that avoids directly addressing the recommendation. \n" \
                         "Output only the chit-chat response. "
        text_information = f"Title: {recommended_item['title']}; Description: {recommended_item['description']}"
        content_user = f"Conversations: {conversations}\n; Scenario: {self.scenario}\n; Item information: {text_information}\n"
        messages =[]
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0.5,
        )
        chit_chat = response.choices[0].message.content

        return chit_chat

    def accept(self, conversations):
        content_system = "You are now playing the role of a user who has just had an in-depth conversation with a dialogue recommendation system. " \
                         "During this conversation, you successfully found a product that meets your needs. \n" \
                         "You will receive two information: " \
                         "1. The content of your conversations with the recommendation system " \
                         "2. The image and detailed information of the product you finally accepted" \
                         "Based on this information, please write a concluding statement expressing your feelings about the recommendation experience and your thoughts on the final product choice. \n " \
                         "Your statement should: " \
                         "1. Express gratitude for the recommendation system" \
                         "2. Explain why you think this product suits you" \
                         "3. Mention specific features of the product and relate them to your personal preferences" \
                         "Please ensure your response is natural and authentic, as if you were a real user expressing their thoughts. " \
                         "The length of your statement should be between 1-3 sentences." \
                         "Example statement:'Thank you, I think these pants are perfect for me. I love the fabric, and the pattern really fits my usual style of dressing. \n" \
                         "Please generate your personalized concluding statement based on the three pieces of information provided. " \
                         "Output only the personalized concluding statement. "
        content_user = []
        item_id = self.target_item["item_id"]
        description = self.target_item['description']
        features = self.target_item['features']
        title = self.target_item['title']
        image_url = f"images_main/{item_id}.jpg"
        item_texts = f"'Title: {title}; Description: {description}; some attributes: {features};'. \n"
        content_text = f"Item information: {item_texts}, Conversations: {conversations} \n"
        base64_image = encode_image(image_url)

        content_user.append({"type": "text", "text": content_text})
        content_user.append({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}})

        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0.5,
        )
        final_sentence = response.choices[0].message.content

        return final_sentence

    def find_reject_reasons(self, item):
        content_system = "You are roleplaying as a user engaged in a conversation with a recommender system. " \
                         "You already know exactly what item you want (your target product), but you cannot directly reveal this to the system. " \
                         "The recommender system has just suggested a product to you. \n" \
                         "Your task is to compare the features of the recommended product with your target product (including visual features) and formulate reasons to decline the recommendation. " \
                         "These reasons should be based on the differences between the two products. " \
                         "You have access to the following information: " \
                         "1. Detailed text and visual information about your target product. " \
                         "2. Detailed text and visual information about the recommended product. \n" \
                         "Your response should: " \
                         "1. Identify key differences between the recommended product and your target product. " \
                         "2. Focus on only two specific features or attributes that don't match the target item, using these as reasons for declining. " \
                         "3. Describe these differences in a way that hints at what you actually want, without being too obvious. \n" \
                         "Now, based on the information about your target product and the recommended product, craft responses that declines the current recommendation. \n"
        content_user = []
        content_text = ""

        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        target_item = f"TargetItem: Description: {self.target_item['description']}; Categories: {self.target_item['categories']}; Image: <Image_1>) \n"
        target_img = encode_image(f"images_main/{self.target_item['item_id']}.jpg")
        content_text += target_item
        recommended_item = f"RecommendedItem: Description: {item['description']}; Categories: {item['categories']}; Image: <Image_2>) \n"
        content_text += recommended_item
        recommended_img = encode_image(f"images_main/{item['item_id']}.jpg")
        content_user.append({"type": "text", "text": content_text})
        content_user.append({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{target_img}"}})
        content_user.append({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{recommended_img}"}})

        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})

        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0.2,
        )
        reject_reasons = response.choices[0].message.content

        return reject_reasons

    def reject(self, conversations, item):
        reject_reason = self.find_reject_reasons(item)
        content_system = "You are roleplaying as a user engaged in a conversation with a dialogue recommender system. \n" \
                         "In a previous turn of the conversation (which may not be the immediately preceding turn due to potential chit-chat), the recommender system suggested a product to you. " \
                         "You have already found a suitable reason to decline this recommendation, and this reason also includes some of your own needs or preferences. " \
                         "Your task now is to respond to the recommender system, declining the suggested product while expressing your needs in a natural and engaging manner. \n" \
                         "You have access to the following information: " \
                         "1. The conversation history up to this point. " \
                         "2. The specific reason for declining the recommendation, which includes some of your needs or preferences. \n"  \
                         "Your response should: " \
                         "1. Express yourself in different ways and avoid repeating the same sentences in history conversations. " \
                         "2. Politely decline the recommendation based the reason you've been provided. " \
                         "3. Clearly express your needs or preferences that are embedded in the reason for declining. " \
                         "4. Reference relevant parts of the conversation history to maintain context and continuity. " \
                         "5. The length is limited to 2-3 sentences.\n" \
                         "Now, based on the given reason for declining (which includes your needs) and the conversation history, " \
                         "craft a response that declines the current recommendation while expressing your needs and keeping the conversation flowing naturally. \n" \
                         "Output only the response."
        content_user = []
        content_text = f"History Conversations: {conversations}; Reason: {reject_reason}; "
        content_user.append({"type": "text", "text": content_text})
        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0.5,
        )
        reject_sentence = response.choices[0].message.content

        return reject_sentence






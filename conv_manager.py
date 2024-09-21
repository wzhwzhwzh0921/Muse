# -*- coding: utf-8 -*-
import json
import random

from openai import OpenAI
from pydantic import BaseModel

from user_chat import User
from system_chat import Recsys


class Cmanager:
    def __init__(self, user: User, recsys: Recsys, base_url: str, api_key: str):
        self.user = user
        self.recsys = recsys
        self.conversations = []
        self.last_query = ""
        self.actions = []
        self.client = self.get_client(base_url, api_key)
        self.mentioned_items = []
        self.mentioned_ids = []
        self.action_conv = []
        self.max_round = 6
        self.current_round = 1
        self.conv_num = 1

    def get_client(self, base_url, api_key):
        client = OpenAI(base_url=base_url,
                        api_key=api_key)
        return client

    def action_control(self, last_action=None):
        """
        """
        if self.current_round == self.max_round:
            return 'recommend'  # 对话结束

        # 根据轮次设置chit-chat的概率
        chit_chat_probabilities = {
            2: 0.5,
            3: 0.4,
            4: 0.3,
            5: 0.3,
            6: 0
        }

        # 如果连续两轮都是chit-chat，则这轮必须是推荐轮
        if last_action == 'chit-chat':
            return 'recommend'

        # 否则，根据概率决定
        if random.random() < chit_chat_probabilities[self.current_round]:
            return 'chit-chat'
        else:
            return 'recommend'

    def prepare_conv(self):
        self.conversations = []
        self.mentioned_items = []
        self.current_round = 1
        self.actions = []
        self.action_conv = []
        self.last_query = ""

    def conv_process(self, user):
        total_loader = {}
        profile = user['profile']
        scenario = user['scenario']
        requirement = user['requirements']
        target_item = user['target_item']
        total_loader['Persona'] = profile
        total_loader['Scenario'] = scenario
        total_loader['Target_item'] = target_item
        self.user.clear_user()
        self.user.load_user(scenario + requirement, target_item)
        self.prepare_conv()
        print('Prepared for conversation!')
        self.first_round(scenario, requirement)
        print('Start conversation!')
        finish_flag = self.one_round_conv()
        while not finish_flag:
            self.current_round += 1
            finish_flag = self.one_round_conv()
            print(self.current_round, self.actions[-1])
        print('Finish conversation!')

        total_loader['Conversations'] = self.action_conv

        with open(f'convs/conv_{self.conv_num}.json', 'w') as file:
            json.dump(self.conversations, file, indent=4)

        with open(f'detail_convs/conv_{self.conv_num}.json', 'w') as file:
            json.dump(total_loader, file, indent=4)
        self.conv_num+=1

    def first_round(self, background, requirement):
        class Round(BaseModel):
            System: str
            User: str

        class MathResponse(BaseModel):
            First_round: Round
            Second_round: Round

        content_system = "1. Generate a conversation between a user and a conversational recommendation system. " \
                         "The conversation should follow these guidelines: " \
                         "1. The dialogue begins with the recommendation system greeting the user. " \
                         "2. The conversation should last for 2 rounds between the system and the user. " \
                         "3. The dialogue should consist of chitchat related to a given shopping context and the user's surface-level needs. " \
                         "4. No product recommendations should be made in this conversation. " \
                         "Please generate the conversation, clearly indicating which part is spoken by the system and which by the user . " \
                         "Warning: The system won't know the context unless the user has mentioned!" \
                         "Ensure the dialogue feels natural and flows logically based on the given context and needs." \
                         "Remember to keep the conversation focused on chit-chat related to the shopping context and user's needs, without making specific product recommendations." \
                         "Output format: " \
                         "Round1: System: [Greating and statement]; User: [Reply, mention some information of the context]; " \
                         "Round2: System: [Reply, express empathic and ask for more]; User: [Reply, mention other information of the context]"
        content_user = f"Background: {background}; Requirement: {requirement}"
        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=MathResponse,
            temperature=0.7,
        )
        parse = completion.choices[0].message.parsed
        first = parse.First_round
        second = parse.Second_round
        self.conversations.append({'Assistant': first.System})
        self.conversations.append({'User': first.User})
        self.conversations.append({'Assistant': second.System})
        self.conversations.append({'User': second.User})

        self.action_conv.append({'Assistant': first.System, 'Action': 'chit-chat', 'Mentioned_item': [], 'Image': []})
        self.action_conv.append({'User': first.User, 'Action': 'chit-chat', 'Image': []})
        self.action_conv.append({'Assistant': second.System, 'Action': 'chit-chat', 'Mentioned_item': [], 'Image': []})
        self.action_conv.append({'User': second.User, 'Action': 'chit-chat', 'Image': []})

    def one_round_conv(self):
        if self.current_round == 1:
            # 推荐一个商品.
            result_item, last_query = self.recsys.once_query("", self.conversations, self.mentioned_ids)
            self.mentioned_ids.append(result_item['item_id'])
            self.mentioned_items.append(result_item)
            sys_conv = self.recsys.recommender(self.conversations, result_item)
            self.conversations.append({'Assistant': sys_conv})
            ##########logger
            self.action_conv.append({'Assistant': sys_conv, 'Action': 'recommend',
                                     'Mentioned_item': [result_item['item_id']],
                                     'Image': [f"images_main/{result_item['item_id']}"]})
            self.actions.append('recommend')
            ##########logger
            return False
        else:
            if self.actions[-1] == 'recommend' and self.mentioned_items[-1]['item_id'] == self.user.target_item['item_id']:
                acc_sentence = self.user.accept(self.conversations)

                self.conversations.append({'User': acc_sentence})
                ##########logger
                self.action_conv.append({'User': acc_sentence, 'Action': 'accept',
                                           'Mentioned_item': [], 'Image': []})
                ##########logger
                return True

            current_action = self.action_control(self.actions[-1])
            self.actions.append(current_action)

            if current_action == 'chit-chat':
                chitchat_user = self.user.chit_chat(self.conversations, self.mentioned_items[-1])
                self.conversations.append({'User': chitchat_user})
                chitchat_sys = self.recsys.chit_chat(self.conversations)
                self.conversations.append({'Assistant': chitchat_sys})

                self.action_conv.append({'User': chitchat_user, 'Action': 'chit-chat',
                                         'Mentioned_item': [], 'Image': []})
                self.action_conv.append({'Assistant': chitchat_sys, 'Action': 'chit-chat',
                                         'Mentioned_item': [], 'Image': []})

                return False
            else:
                #推荐
                reject_user = self.user.reject(self.conversations, self.mentioned_items[-1])
                self.conversations.append({'User': reject_user})

                self.action_conv.append({'User': reject_user, 'Action': 'reject',
                                         'Mentioned_item': [], 'Image': []})

                #判断是否到达了最后一轮
                if self.current_round == self.max_round:
                    result_item = self.user.target_item
                    recommend_sys = self.recsys.recommender(self.conversations, result_item)
                    self.conversations.append({'Assistant': recommend_sys})
                    self.mentioned_items.append(result_item)
                    self.mentioned_ids.append(result_item['item_id'])
                    acc_sentence = self.user.accept(self.conversations)
                    self.conversations.append({'User': acc_sentence})

                    self.action_conv.append({'Assistant': recommend_sys, 'Action': 'recommend',
                                               'Mentioned_item': [result_item['item_id']],
                                               'Image': [f"images_main/{result_item['item_id']}"]})
                    self.action_conv.append({'User': acc_sentence, 'Action': 'accept',
                                             'Mentioned_item': [], 'Image': []})

                    return True
                else:
                    result_item, final_query = self.recsys.once_query(self.last_query, self.conversations, self.mentioned_ids)
                    self.last_query = final_query
                    self.mentioned_items.append(result_item)
                    self.mentioned_ids.append(result_item['item_id'])
                    recommend_sys = self.recsys.recommender(self.conversations, result_item)
                    self.conversations.append({'Assistant': recommend_sys})

                    self.action_conv.append({'Assistant': recommend_sys, 'Action': 'recommend',
                                             'Mentioned_item': [result_item['item_id']],
                                             'Image': [f"images_main/{result_item['item_id']}"]})

                    return False


with open('user_scenarios.json', 'r') as file:
    users = json.load(file)

api_base = 'https://neudm.zeabur.app/v1'
api_key = 'sk-2sEiilPsN7H8nx3y6fBb89192370487b9eF3373c6586E8Dd'
db_path = "/datas/wangzihan/mmrec/preprocess/cloth/"
data_path = "/datas/wangzihan/mmrec/preprocess/cloth/item_profile.json"
model_name = "/datas/huggingface/bge-m3"
user = User(base_url=api_base, api_key=api_key)
recsys = Recsys(db_path=db_path, data_path=data_path, model_path=model_name, base_url=api_base, api_key=api_key)
cmanager = Cmanager(user=user, recsys=recsys, base_url=api_base, api_key=api_key)

for user in users:
    cmanager.conv_process(user)

import base64
import json
import random

from openai import OpenAI
from faker import Faker
from pydantic import BaseModel

from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

from create_item_db import ItemVector

fake = Faker()
reasons = [
    "Practical need: Replacing worn-out clothes/shoes/",
    "Work requires specific attire. ",
    "Seasonal shopping for clothes/shoes suitable for the new season",
    "Purchasing for a specific event",
    "Occasion need: Attending special events like weddings, graduation ceremonies, "
    "Outfits for dates or important social activities",
    "Following current trends, trying new fashion elements or styles",
    "Buying clothes for specific indoors or outdoors activities. "
    "Self-reward: Celebrating personal achievements or milestones",
    "Purchasing luxury items that reflect social status",
    "Purchasing clothes, shoes, or accessories as gifts for others",
    "Urchasing for families "
]

ages = list(range(3, 71))

genders = ['male', 'female', 'unknown']
gender_weights = [45, 45, 10]  # 总和为100，unknown的权重为10

with open('category2items.json', 'r') as file:
    cloth_types = json.load(file)

category_weights = {category: len(items) for category, items in cloth_types.items()}

with open('updated_item_profile.json', 'r') as file:
    item_profiles = json.load(file)

def generate_user_profile():
    selected_category = random.choices(list(category_weights.keys()),
                                       weights=list(category_weights.values()),
                                       k=1)[0]
    selected_item_id = random.choice(cloth_types[selected_category])
    selected_item = item_profiles[selected_item_id]
    profession = fake.job()
    reason = random.choice(reasons)
    age = random.choice(ages)
    gender = random.choices(genders, weights=gender_weights, k=1)[0]
    if gender == 'male':
        name = fake.name_male()
    elif gender == 'female':
        name = fake.name_female()
    else:
        name = fake.name()
    if age <= 15:
        profession = 'no-profession'
    selected_item['item_id'] = selected_item_id
    selected_item['images'] = f'images_main/{selected_item_id}.jpg'
    user = {'name': name, 'gender': gender, 'age': age,
            'profession': profession, 'reason': reason, 'cloth_type': selected_category,
            'target_item': selected_item}

    return user

def calculate_bleu_similarities(sentence, sentence_list):
    # 将输入句子分词
    reference = word_tokenize(sentence.lower())

    similarities = []
    for s in sentence_list:
        # 将列表中的每个句子分词
        candidate = word_tokenize(s.lower())

        # 计算BLEU得分
        score = sentence_bleu([reference], candidate)
        similarities.append(score)

    return similarities
    # 打印结果
db_path = "/datas/wangzihan/mmrec/preprocess/cloth/"
data_path = "/datas/wangzihan/mmrec/preprocess/cloth/updated_item_profile.json"
model_name = "/datas/huggingface/bge-m3"
client = OpenAI(base_url='https://neudm.zeabur.app/v1', api_key='sk-2sEiilPsN7H8nx3y6fBb89192370487b9eF3373c6586E8Dd')
user_scenarios = []
print('Start Loading Dataset...')
myDB = ItemVector(
            db_path=db_path,
            model_name=model_name,
            llm=None,
            data_path=data_path,
            force_create=False,
            use_multi_query=False,
        )
print('Dataset Loaded!')
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

while len(user_scenarios)< 20:
    user = generate_user_profile()
    content_system_pre = "Given a user's age, gender, shopping motivation and reason, and his/her shopping type, " \
                         "please judge whether it is reasonable. " \
                         "Output only 'Yes' or 'No'"
    content_system_user_pre = [{"type": "text", "text": f"1 gender: {user['gender']}, age: {user['age']} \n "
                                                    f"2. Basic reason for this purchase trip. {user['reason']}"
                                                    f"3. Target cloth type for the purchase trip. {user['cloth_type']}"}]
    response_pre = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": content_system_pre},
            {"role": "user", "content": content_system_user_pre}
        ],
        temperature=0.2,
    )
    print(response_pre.choices[0].message.content)
    if 'Yes' in response_pre.choices[0].message.content:
        class MathReasoning(BaseModel):
            scenario: str
            requirement: str
        user_profile = f"'Name: {user['name']}, Gender: {user['gender']}, Age: {user['age']},Profession: {user['profession']}"
        content_system = "You are scenarios generator for a consumer's purchase motivation. \n" \
                         "Your goal is to create a scenario that could naturally lead to a purchase, \n" \
                         "without explicitly mentioning the actual item bought. \n" \
                         "You will be given three information:" \
                         "1. User information. 2. Basic reason for the backstory. 3. Information of the target item." \
                         "Use the provided user information to craft a believable and engaging narrative. \n" \
                         "The descriptions should: " \
                         "1. An upcoming event (It can be a significant life event or a minor everyday occurrence.)" \
                         "2. Include relevant contextual details such as events, emotions. " \
                         "3. Reveal the user's requirements for the product, but only towards two attributes!, no more!" \
                         "4. Related to the revealed requirements. " \
                         "Warning: Do not mention or describe the actual purchased item. \n " \
                         "Please generate a backstory suits the case."
        content_user = f"1. User information: f{user_profile}\n " \
                       f"2. Basic reason for this purchase trip: '{user['reason']}' \n" \
                       f"3. Target item: Cloth_type: {user['cloth_type']}; Descriptions:{user['target_item']['new_description']}, "
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": content_system},
                {"role": "user", "content": content_user}
            ],
            response_format=MathReasoning,
            temperature=0.2
        )
        response = completion.choices[0].message.parsed
        sce = {'profile': user_profile, 'scenario': response.scenario,
               'requirements': response.requirement, 'target_item': user['target_item']}
        print(sce['target_item'])
        user_scenarios.append(sce)
    else:
        continue

with open('user_scenarios.json', 'w') as file:
    json.dump(user_scenarios, file, indent=4)


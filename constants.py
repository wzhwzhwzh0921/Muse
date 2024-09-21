# model setting
MINI_MODEL="gpt-4o-mini"
MODEL="gpt-4o-2024-08-06"

# system setting
PRODUCT_SELECTOR=open("prompt/product_selector.txt").read().strip()
CONVERSATION_ANALYZER=open("prompt/conversation_analyzer.txt").read().strip()
CLARIFIER=open("prompt/clarifier.txt").read().strip()
CHIT_CHAT=open("prompt/chit_chat.txt").read().strip()
RECOMMENDER=open("prompt/recommender.txt").read().strip()
QUERY_GENERATOR=open("prompt/query_generator.txt").read().strip()
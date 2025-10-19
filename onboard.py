from create_vectorstore import create_vectorstore

def train(user_id: str, content: str):
    result = create_vectorstore(user_id, content)
    print(result)

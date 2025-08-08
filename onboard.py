from create_vectorstore import create_vectorstore

def train(user_id: str):
    result = create_vectorstore(user_id)
    print(result)

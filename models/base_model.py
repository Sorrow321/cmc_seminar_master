class BaseModel():
    def __init__(self, *args):
        raise NotImplementedError("Please Implement this method")

    def get_embeddings(self, input):
        raise NotImplementedError("Please Implement this method") 
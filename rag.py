import requests

class ArtAssistant:

    def __init__(self, vector_store, embedding_model, llm_url, api_key, model, k_docs):
        self.vector_store = vector_store
        self.llm_url = llm_url
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.model = model
        self.k_docs = k_docs
    
    def handle_user_query(self, user_query: str) -> str:
        hypo = self.make_hypothesis(user_query)
        response_embedding = self.embedding_model.encode([hypo])[0]
        docs = self.get_similar_docs(response_embedding)
        
        return self.respond_with_docs(docs, user_query)

    def make_hypothesis(self, user_query: str) -> str:
        system_prompt = "you are an assistant who helps user to answer his questions. you should answer in english"
        user_prompt = f"Answer the users query: {user_query}. You should answer in english."
        
        return self.make_llm_request(system_prompt, user_prompt)

    def respond_with_docs(self, docs, user_query):
        context = ""

        for doc in docs:
            context += f"Информация об авторах: {doc.metadata["wiki"]}\n\nСписок картин автора: {doc.metadata["description"]}\n\n"
        
        user_prompt = f"Ответь на запрос пользователя: {user_query}\n\nТы можешь использовать эту информацию: {context}"
        system_prompt = "Ты - ассистент, который помогает пользователю отвечать на его вопросы об искусстве. Твой ответ должен быть кратким (несколько предложений), но точным. Ты должен отвечать на русском языке."
        
        return self.make_llm_request(system_prompt, user_prompt)

    def make_llm_request(self, system_prompt, user_prompt):
        llm_response = requests.post(
            self.llm_url,
            json={"model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]},
            headers={
                "Authorization": f"Bearer {self.api_key}",
            }
        ).json()["choices"][0]["message"]["content"]

        return llm_response

    def get_similar_docs(self, embedding):
        docs = self.vector_store.similarity_search_by_vector(embedding, self.k_docs)
        return docs

from fastapi import FastAPI
# from ml import nlp
# from pydantic import BaseModel
# import starlette
# app = FastAPI()


app = FastAPI(
    title='Henry - DS API',
    description='This is to visualize my metrics from Gapminder data',
    version='0.1',
    docs_url='/',
    )


@app.get("/")
def read_main():
    return {"message": "Hello World, this is DS17 team"}


# @app.get("/article/{article_id}")
# def analyze_article(article_id: int, q: str = None):
#     return {"article_id": article_id, "q": q}



@app.get("/article/{article_id}")
def analyze_article(article_id: int, q: str = None):
    return {"article_id": article_id, "q": q}
    # return {"article_id": article_id, "previous_id": article_id - 1}

    # count the numpber of entities
    # count = 0
    # if q:
    #     doc = nlp(q)
    #     count = len(doc.ents)

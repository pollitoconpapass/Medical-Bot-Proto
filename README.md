# Qhali: Medical Chatbot Prototype 💊
### Steps guide
1. Install all the necessary dependencies: ```pip install -r requirements.txt```
2. Create a `vectorstores` folder, inside create another folder named `db_faiss`
3. Download a LLAMA-2 model from HuggingFace: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML 
4. Run the ingest.py file: ```python3 ingest.py```
5. Go to the Cloud Translate API and create a Service Account Permission
6. Add the json path file to the ```model.py```
7. Configure the Flask application with:
      - ```export FLASK_APP=app.py```
      - ```export FLASK_DEBUG=1```
      - ```flask run```
   

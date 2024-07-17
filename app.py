from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from FlaskHelpers.forms import QueryForm
from FlaskHelpers.file_upload import allowed_file
from Helpers.prompts import PROMPT_TEMPLATE
import os
from VectorDB import VectorDB
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

SECRET_KEY = 'randomkey'
UPLOAD_FOLDER = 'knowledge_source'
CHROMA_PATH = 'vectordb'

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
vdb = VectorDB(UPLOAD_FOLDER)
model = Ollama(model="mistral")


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = QueryForm()
    query = ''
    response = None
    if request.method == 'POST':
        query = request.form.get('query')
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        if vdb.nb_of_existing_doc() > 0:
            similarity_results = vdb.db.similarity_search_with_score(query, k=3)
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in similarity_results])
            context_text = context_text.replace('\n', ' ')
            prompt = prompt_template.format(context=context_text, question=query)
            response_text = model.invoke(prompt)
            sources = [doc.metadata.get("id", None) for doc, _score in similarity_results]
            response = {"Response": response_text, "Sources": sources}
        else:
            prompt = prompt_template.format(context='', question=query)
            response_text = model.invoke(prompt)
            sources = ''
            response = {"Response": response_text, "Sources": sources}
    return render_template('index.html', answer=response, form=form, query=query)


@app.route('/vdb_operations', methods=['GET', 'POST'])
def vdb_operations():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            vdb.add_new_source()
            return redirect(url_for('vdb_operations',
                                    filename=filename))
    return render_template('vdb_operations.html', db_size=vdb.nb_of_existing_doc())


@app.route('/statistics')
def statistics():
    return render_template('statistics.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

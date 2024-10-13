from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import io
import base64
from lsa_model import get_top_documents, documents

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')  # Use get to avoid KeyError
    if not query:
        return render_template('index.html', error="Please enter a query.")
    
    try:
        # Retrieve the top 5 documents and their similarity scores
        top_doc_indices, similarities = get_top_documents(query)

        # Check if there are no results
        if len(top_doc_indices) == 0:
            return render_template('results.html', query=query, results=[], plot_url=None)

        # Extract the document contents for display (limit to 300 characters)
        result_docs = [(documents[i][:300] + "...", similarities[j]) for j, i in enumerate(top_doc_indices)]
        
        # Plot cosine similarities with document indices as labels
        fig, ax = plt.subplots()
        ax.bar(range(len(similarities)), similarities, color='skyblue')
        ax.set_xticks(range(len(similarities)))
        ax.set_xticklabels([f'Doc {i+1}' for i in top_doc_indices], rotation=45)
        plt.ylim(0, 1)  # Ensure the y-axis range is 0 to 1
        plt.xlabel('Document Index')
        plt.ylabel('Cosine Similarity')

        # Convert plot to PNG image and then to base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        # Render the results page
        return render_template('results.html', query=query, results=result_docs, plot_url=plot_url)

    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template('index.html', error="An error occurred while processing your request.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)

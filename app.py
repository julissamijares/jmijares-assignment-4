from flask import Flask, request, render_template
from lsa_model import get_top_documents, documents
import plotly.graph_objects as go

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    
    # Retrieve the top 5 documents and their similarity scores
    top_doc_indices, similarities = get_top_documents(query)
    
    # Extract the document contents for display
    result_docs = [(documents[i][:300] + "...", similarities[j]) for j, i in enumerate(top_doc_indices)]  # Display first 300 characters
    
    # Create a bar chart using Plotly
    fig = go.Figure(data=[go.Bar(
        x=[f'Index {i}' for i in top_doc_indices],
        y=similarities,
        hoverinfo='text',
        text=[f'Similarity: {score:.3f}' for score in similarities],
        marker=dict(color='skyblue')
    )])
    
    fig.update_layout(
        title='Cosine Similarity',
        #xaxis_title='Document Index',
        yaxis_title='Cosine Similarity',
        yaxis=dict(range=[0, 1]),
        width= 600,  # Increased width for better fit
        height=400,  # Height remains the same
        margin=dict(l=40, r=40, t=40, b=80),  # Adjust bottom margin for x-axis labels
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for cleaner look
        template='plotly_white',  # Use a white template for better aesthetics
        xaxis=dict(tickmode='array', tickvals=[f'Index {i}' for i in top_doc_indices], ticktext=[f'Index {i}' for i in top_doc_indices])  # Ensure all indices are shown
    )

    # Save the plot as an HTML string for rendering in template
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Render the results page
    return render_template('results.html', query=query, results=result_docs, plot_html=plot_html, top_doc_indices=top_doc_indices)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)

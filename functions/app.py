from flask import Flask, render_template, request, jsonify
from groq import Groq
from dataclasses import dataclass
from typing import List, Dict, Optional
import uuid
import plotly.graph_objects as go
import plotly.express as px
import json
import numpy as np

app = Flask(__name__)

# Initialize Groq client
client = Groq(api_key="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")

@dataclass
class StoryNode:
    id: str
    content: str
    weight: float
    parent_id: Optional[str]
    children: List[str]
    genre: str
    is_selected: bool = False

class StoryTree:
    def __init__(self):
        self.nodes: Dict[str, StoryNode] = {}
        self.current_path: List[str] = []
        self.genre_history: List[str] = []
        self.weights_history: List[float] = []

    def add_node(self, content: str, weight: float, parent_id: Optional[str], genre: str) -> str:
        node_id = str(uuid.uuid4())
        node = StoryNode(
            id=node_id,
            content=content,
            weight=weight,
            parent_id=parent_id,
            children=[],
            genre=genre
        )
        self.nodes[node_id] = node
        
        if parent_id:
            self.nodes[parent_id].children.append(node_id)
        
        self.genre_history.append(genre)
        self.weights_history.append(weight)
        return node_id

    def get_current_path(self) -> List[StoryNode]:
        return [self.nodes[node_id] for node_id in self.current_path]

    def get_genre_distribution(self) -> Dict[str, int]:
        genre_counts = {}
        for node in self.nodes.values():
            genre_counts[node.genre] = genre_counts.get(node.genre, 0) + 1
        return genre_counts

# Initialize story tree
story_tree = StoryTree()

def get_llm_response(prompt: str) -> str:
    """Get response from Groq LLM"""
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with LLM: {e}")
        return ""

def generate_next_situations(current_situation: str, genre: str, n: int = 3) -> List[Dict]:
    """Generate next possible situations using LLM"""
    prompt = f"""
    Given the current story situation in the {genre} genre:
    "{current_situation}"
    
    Generate {n} possible next future situations that could follow. Make them creative and distinct from each other.
    Rate each situation with a score from 0 to 1 based on how well it fits the genre and advances the story.the answers should be veryshort, i am craeting a story plot point.
    
    Format the response as:
    1. [Situation 1] | Score: [0-1]
    2. [Situation 2] | Score: [0-1]
    3. [Situation 3] | Score: [0-1]
    """
    
    response = get_llm_response(prompt)
    situations = []
    
    for line in response.split('\n'):
        if '|' in line:
            situation, score = line.split('|')
            situation = situation.strip().lstrip('123.').strip()
            try:
                score = float(score.strip().replace('Score:', '').strip())
            except:
                score = 0.5
            situations.append({
                "content": situation,
                "weight": score
            })
    
    return situations

def generate_analytics_data():
    """Generate plot data for analytics"""
    # Story Progress Plot Data
    progress_data = {
        'data': [{
            'x': list(range(1, len(story_tree.weights_history) + 1)),
            'y': story_tree.weights_history,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Story Progress',
            'line': {'color': '#3498db'}
        }],
        'layout': {
            'title': 'Story Progress',
            'xaxis': {'title': 'Situation Number'},
            'yaxis': {'title': 'Weight'},
            'template': 'plotly_dark',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)'
        }
    }

    # Genre Distribution Plot Data
    genre_dist = story_tree.get_genre_distribution()
    genre_data = {
        'data': [{
            'values': list(genre_dist.values()),
            'labels': list(genre_dist.keys()),
            'type': 'pie',
            'hole': 0.4,
            'marker': {'colors': px.colors.qualitative.Set3}
        }],
        'layout': {
            'title': 'Genre Distribution',
            'template': 'plotly_dark',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)'
        }
    }

    # Path Complexity Plot Data
    path_lengths = [len(node.children) for node in story_tree.nodes.values()]
    if path_lengths:
        path_data = {
            'data': [{
                'type': 'bar',
                'x': ['Min', 'Avg', 'Max'],
                'y': [min(path_lengths), np.mean(path_lengths), max(path_lengths)],
                'marker': {'color': '#2ecc71'}
            }],
            'layout': {
                'title': 'Path Complexity',
                'template': 'plotly_dark',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)'
            }
        }
    else:
        path_data = {'data': [], 'layout': {'title': 'Path Complexity'}}

    return {
        'progress_data': json.dumps(progress_data),
        'genre_data': json.dumps(genre_data),
        'path_data': json.dumps(path_data)
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_story', methods=['POST'])
def start_story():
    data = request.get_json()
    initial_situation = data['situation']
    genre = data['genre']
    
    weight_prompt = f"Rate how well this initial situation works for a {genre} story on a scale of 0 to 1: '{initial_situation}'"
    weight_response = get_llm_response(weight_prompt)
    try:
        weight = float(weight_response.strip())
    except:
        weight = 0.5
    
    initial_node_id = story_tree.add_node(initial_situation, weight, None, genre)
    story_tree.current_path = [initial_node_id]
    
    next_situations = generate_next_situations(initial_situation, genre)
    
    return jsonify({
        "current_node": initial_node_id,
        "next_situations": next_situations,
        "initial_weight": weight
    })

@app.route('/select_situation', methods=['POST'])
def select_situation():
    data = request.get_json()
    selected_situation = data['situation']
    parent_id = story_tree.current_path[-1]
    genre = data['genre']  # Use the current genre from frontend
    
    new_node_id = story_tree.add_node(selected_situation['content'], selected_situation['weight'], parent_id, genre)
    story_tree.current_path.append(new_node_id)
    
    next_situations = generate_next_situations(selected_situation['content'], genre)
    
    return jsonify({
        "current_node": new_node_id,
        "next_situations": next_situations,
        "current_path": [
            {"id": node.id, "content": node.content, "weight": node.weight}
            for node in story_tree.get_current_path()
        ]
    })

@app.route('/regenerate_situations', methods=['POST'])
def regenerate_situations():
    data = request.get_json()
    current_node_id = data['current_node']
    genre = data['genre']
    
    current_situation = story_tree.nodes[current_node_id].content
    next_situations = generate_next_situations(current_situation, genre)
    
    return jsonify({
        "next_situations": next_situations
    })

@app.route('/update_genre', methods=['POST'])
def update_genre():
    data = request.get_json()
    node_id = data['node_id']
    new_genre = data['genre']
    
    node = story_tree.nodes[node_id]
    node.genre = new_genre
    
    next_situations = generate_next_situations(node.content, new_genre)
    
    return jsonify({
        "next_situations": next_situations
    })

@app.route('/get_analytics', methods=['GET'])
def get_analytics():
    return jsonify(generate_analytics_data())

@app.route('/get_unexplored_paths', methods=['GET'])
def get_unexplored_paths():
    current_node_id = story_tree.current_path[-1]
    current_node = story_tree.nodes[current_node_id]
    
    unexplored_children = [
        story_tree.nodes[child_id]
        for child_id in current_node.children
        if child_id not in story_tree.current_path
    ]
    
    return jsonify({
        "unexplored_paths": [
            {"id": node.id, "content": node.content, "weight": node.weight}
            for node in unexplored_children
        ]
    })

if __name__ == '__main__':
    app.run(debug=True) #

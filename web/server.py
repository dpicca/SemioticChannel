"""
Flask Web Server for Taboo Semiotic Experiment.
Provides real-time streaming of experiment progress via SSE.
"""

import json
import time
import sys
import os
from flask import Flask, render_template, request, Response, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semiotic_channel.concepts_dataset import get_concepts, get_concept_category
from semiotic_channel.llm_interface import (
    SemioticEmitter,
    SemioticReceiver,
    OllamaEmbeddings,
    check_ollama_available,
    list_available_models,
    DEFAULT_MODEL
)
from semiotic_channel.metrics import SemioticMetrics, cosine_similarity

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Cache directory for descriptions
CACHE_DIR = Path(__file__).parent.parent / "data" / "description_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Store experiment results
current_experiment = {
    "running": False,
    "results": [],
    "logs": []
}


def get_cache_path(emitter_model: str) -> Path:
    """Get cache file path for a specific emitter model."""
    safe_model_name = emitter_model.replace(":", "_").replace("/", "_")
    return CACHE_DIR / f"{safe_model_name}.json"


def load_cache(emitter_model: str) -> dict:
    """Load cached descriptions for a model."""
    cache_path = get_cache_path(emitter_model)
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_cache(emitter_model: str, cache: dict):
    """Save descriptions cache for a model."""
    cache_path = get_cache_path(emitter_model)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)


def get_cached_description(cache: dict, concept: str, temperature: float, index: int) -> str | None:
    """Get a cached description if available."""
    key = f"{concept}|{temperature}"
    if key in cache and len(cache[key]) > index:
        return cache[key][index]
    return None


def add_to_cache(cache: dict, concept: str, temperature: float, description: str):
    """Add a description to the cache."""
    key = f"{concept}|{temperature}"
    if key not in cache:
        cache[key] = []
    cache[key].append(description)


def stream_experiment(n_concepts, n_descriptions, category, emitter_model, receiver_model, temperatures):
    """Generator that yields SSE events during experiment execution."""
    
    global current_experiment
    current_experiment = {"running": True, "results": [], "logs": []}
    
    # Send start event
    yield f"data: {json.dumps({'type': 'start', 'message': 'Experiment started'})}\n\n"
    
    # Check Ollama
    if not check_ollama_available():
        yield f"data: {json.dumps({'type': 'error', 'message': 'Ollama server not running!'})}\n\n"
        return
    
    yield f"data: {json.dumps({'type': 'log', 'message': 'Ollama server connected'})}\n\n"
    yield f"data: {json.dumps({'type': 'log', 'message': f'Emitter: {emitter_model} | Receiver: {receiver_model}'})}\n\n"
    
    # Get concepts
    concepts = get_concepts(category=category, n=n_concepts)
    yield f"data: {json.dumps({'type': 'concepts', 'concepts': concepts})}\n\n"
    
    # Initialize components
    emitter = SemioticEmitter(model_id=emitter_model)
    receiver = SemioticReceiver(model_id=receiver_model)
    # embeddings = OllamaEmbeddings()  # Not used for Ranked Decipherability
    
    # Load cache for this emitter model
    cache = load_cache(emitter_model)
    cache_hits = 0
    cache_misses = 0
    
    # Target embeddings not needed for Ranked Decipherability
    
    all_results = []
    
    for temp in temperatures:
        temp_results = {
            "temperature": temp,
            "concepts": [],
            "avg_S": 0,
            "avg_D": 0
        }
        
        yield f"data: {json.dumps({'type': 'temperature_start', 'temperature': temp})}\n\n"
        
        s_values = []
        d_values = []
        
        for concept in concepts:
            concept_data = {
                "name": concept,
                "category": get_concept_category(concept),
                "descriptions": [],
                "interpretations": [],
                "S": 0,
                "D": 0
            }
            
            yield f"data: {json.dumps({'type': 'concept_start', 'concept': concept, 'temperature': temp})}\n\n"
            
            # Phase A: Generate or retrieve descriptions
            descriptions = []
            for i in range(n_descriptions):
                try:
                    # Check cache first
                    cached = get_cached_description(cache, concept, temp, i)
                    if cached:
                        desc = cached
                        cache_hits += 1
                        yield f"data: {json.dumps({'type': 'description', 'concept': concept, 'index': i+1, 'text': desc, 'cached': True})}\n\n"
                    else:
                        desc = emitter.generate_description(concept, temperature=temp)
                        add_to_cache(cache, concept, temp, desc)
                        cache_misses += 1
                        yield f"data: {json.dumps({'type': 'description', 'concept': concept, 'index': i+1, 'text': desc, 'cached': False})}\n\n"
                    descriptions.append(desc)
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Generation error: {str(e)}'})}\n\n"
            
            concept_data["descriptions"] = descriptions
            
            # Phase B: Interpret descriptions
            interpretations = []
            for desc in descriptions:
                try:
                    # Get Top 5 guesses for Ranked Decipherability
                    full_guesses = receiver.interpret_top_k(desc)
                    interpretations.append(full_guesses)
                    # For UI log, just show the top guess
                    top_guess = full_guesses[0] if full_guesses else ""
                    yield f"data: {json.dumps({'type': 'interpretation', 'concept': concept, 'interpretation': top_guess})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Interpretation error: {str(e)}'})}\n\n"
            
            concept_data["interpretations"] = interpretations
            
            # Phase C: Calculate metrics
            if descriptions:
                # Semiotic Breadth (lexical diversity)
                S = SemioticMetrics.breadth(descriptions)
                concept_data["S"] = round(S, 4)
                s_values.append(S)
                
                # Decipherability (Rank-based)
                # interpretations is now a list of lists of strings
                D = SemioticMetrics.decipherability_rank(concept, interpretations)
                concept_data["D"] = round(D, 4)
                d_values.append(D)
                
                yield f"data: {json.dumps({'type': 'metrics', 'concept': concept, 'S': concept_data['S'], 'D': concept_data['D']})}\n\n"
            
            temp_results["concepts"].append(concept_data)
        
        # Calculate averages for this temperature
        temp_results["avg_S"] = round(sum(s_values) / len(s_values), 4) if s_values else 0
        temp_results["avg_D"] = round(sum(d_values) / len(d_values), 4) if d_values else 0
        
        yield f"data: {json.dumps({'type': 'temperature_complete', 'temperature': temp, 'avg_S': temp_results['avg_S'], 'avg_D': temp_results['avg_D']})}\n\n"
        
        all_results.append(temp_results)
    
    # Save cache after experiment
    save_cache(emitter_model, cache)
    yield f"data: {json.dumps({'type': 'log', 'message': f'Cache: {cache_hits} hits, {cache_misses} new generations saved'})}\n\n"
    
    # Calculate Semiotic Capacity
    capacities = [r["avg_D"] for r in all_results]
    max_capacity = max(capacities) if capacities else 0
    optimal_temp = all_results[capacities.index(max_capacity)]["temperature"] if capacities else 0
    
    final_data = {
        "type": "complete",
        "results": all_results,
        "semiotic_capacity": round(max_capacity, 4),
        "optimal_temperature": optimal_temp
    }
    
    yield f"data: {json.dumps(final_data)}\n\n"
    
    current_experiment["running"] = False
    current_experiment["results"] = all_results


@app.route('/')
def index():
    """Serve the main HTML interface."""
    return render_template('index.html')


@app.route('/api/models')
def get_models():
    """Get available Ollama models."""
    models = list_available_models()
    return jsonify({"models": models, "default": DEFAULT_MODEL})


@app.route('/api/concepts')
def get_concepts_list():
    """Get available concepts by category."""
    concrete = get_concepts(category="concrete", n=50)
    abstract = get_concepts(category="abstract", n=50)
    return jsonify({"concrete": concrete, "abstract": abstract})


@app.route('/api/run', methods=['POST'])
def run_experiment():
    """Start experiment with SSE streaming."""
    data = request.json
    
    n_concepts = int(data.get('n_concepts', 5))
    n_descriptions = int(data.get('n_descriptions', 3))
    category = data.get('category', 'abstract')
    emitter_model = data.get('emitter_model', DEFAULT_MODEL)
    receiver_model = data.get('receiver_model', DEFAULT_MODEL)
    temp_min = float(data.get('temp_min', 0.1))
    temp_max = float(data.get('temp_max', 1.5))
    temp_steps = int(data.get('temp_steps', 8))
    
    temperatures = [round(temp_min + i * (temp_max - temp_min) / (temp_steps - 1), 2) 
                    for i in range(temp_steps)]
    
    return Response(
        stream_experiment(n_concepts, n_descriptions, category, emitter_model, receiver_model, temperatures),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/status')
def get_status():
    """Check if Ollama is available."""
    available = check_ollama_available()
    models = list_available_models() if available else []
    return jsonify({
        "ollama_available": available,
        "models": models
    })


@app.route('/api/cache')
def get_cache_info():
    """Get cache statistics."""
    cache_files = list(CACHE_DIR.glob("*.json"))
    stats = []
    for f in cache_files:
        try:
            with open(f) as cf:
                data = json.load(cf)
                stats.append({
                    "model": f.stem.replace("_", ":"),
                    "entries": len(data),
                    "size_kb": round(f.stat().st_size / 1024, 1)
                })
        except:
            pass
    return jsonify({"cache_files": stats})


if __name__ == '__main__':
    print("Starting Semiotic Experiment Web Server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000, threaded=True)

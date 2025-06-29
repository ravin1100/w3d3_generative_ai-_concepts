from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please create a .env file with your API key.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

app = Flask(__name__)

# Load tool configurations
with open('tool_analysis.json', 'r') as f:
    TOOL_CONFIG = json.load(f)

def analyze_prompt_with_gemini(prompt):
    """Analyze the prompt using Gemini API to understand intent and requirements."""
    try:
        analysis_prompt = f"""
        Analyze this coding prompt and provide detailed insights:
        
        PROMPT: {prompt}
        
        Provide analysis in the following format:
        1. Intent: [Main purpose and goal of the prompt]
        2. Complexity: [simple/moderate/complex with brief explanation]
        3. Key Requirements:
           - [Requirement 1]
           - [Requirement 2]
           etc.
        4. Technical Considerations:
           - [Technical aspect 1]
           - [Technical aspect 2]
           etc.
        """
        
        response = model.generate_content(analysis_prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error in analyze_prompt_with_gemini: {str(e)}")
        raise

def optimize_for_tool(prompt, tool_config):
    """
    Use Gemini API to optimize the prompt for a specific tool based on its optimization strategies.
    This function directly uses the tool configuration from tool_analysis.json.
    """
    try:
        optimization_prompt = f"""
        Original prompt: {prompt}
        
        Tool: {tool_config['name']}
        Optimization strategies for this tool:
        {json.dumps(tool_config['optimization_strategies'], indent=2)}
        
        Please optimize the prompt for this specific tool by:
        1. Applying the tool's optimization strategies listed above
        2. Formatting according to tool's requirements
        3. Adding any tool-specific context or parameters
        
        Provide your response in this format:
        OPTIMIZED PROMPT:
        [Your optimized prompt here]

        EXPLANATIONS:
        - [First change made and why]
        - [Second change made and why]
        etc.
        """
        
        response = model.generate_content(optimization_prompt)
        optimization_result = response.text
        
        # Split the response into optimized prompt and explanations
        parts = optimization_result.split("\nEXPLANATIONS:")
        optimized_prompt = parts[0].replace("OPTIMIZED PROMPT:", "").strip()
        explanations = []
        if len(parts) > 1:
            explanations = [exp.strip().strip('- ') for exp in parts[1].strip().split('\n-')]
            explanations = [exp for exp in explanations if exp]
        
        return {
            'optimized_prompt': optimized_prompt,
            'explanations': explanations,
            'tool_name': tool_config['name']
        }
    except Exception as e:
        logger.error(f"Error in optimize_for_tool: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html', tools=TOOL_CONFIG['tools'])

@app.route('/optimize', methods=['POST'])
def optimize_prompt():
    try:
        data = request.json
        base_prompt = data.get('prompt')
        selected_tools = data.get('tools', [])
        
        if not base_prompt or not selected_tools:
            return jsonify({'error': 'Missing prompt or tool selection'}), 400
        
        # Step 1: Analyze prompt intent and requirements
        analysis = analyze_prompt_with_gemini(base_prompt)
        
        # Step 2: Generate optimized prompts for each selected tool
        results = []
        for tool_id in selected_tools:
            if tool_id in TOOL_CONFIG['tools']:
                tool_config = TOOL_CONFIG['tools'][tool_id]
                result = optimize_for_tool(base_prompt, tool_config)
                results.append(result)
        
        return jsonify({
            'original_prompt': base_prompt,
            'analysis': analysis,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html template
    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Prompt Optimizer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .tool-card { margin-bottom: 1rem; }
        .optimization-result { margin-top: 2rem; }
        pre { background: #f8f9fa; padding: 1rem; border-radius: 0.25rem; white-space: pre-wrap; }
        .analysis-section { margin-bottom: 2rem; }
    </style>
</head>
<body>
    <div class="container mt-5">
        <h1 class="mb-4">AI Prompt Optimizer</h1>
        
        <div class="row">
            <div class="col-md-5">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Input</h5>
                        <div class="mb-3">
                            <label class="form-label">Base Prompt:</label>
                            <textarea id="basePrompt" class="form-control" rows="5" placeholder="Enter your prompt here..."></textarea>
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label">Select Tool:</label>
                            <div id="toolSelection">
                                {% for tool_id, tool in tools.items() %}
                                <div class="form-check">
                                    <input class="form-check-input" type="radio" name="toolSelect" value="{{ tool_id }}" id="{{ tool_id }}">
                                    <label class="form-check-label" for="{{ tool_id }}">
                                        {{ tool.name }}
                                    </label>
                                </div>
                                {% endfor %}
                            </div>
                        </div>
                        
                        <button onclick="optimizePrompt()" class="btn btn-primary">Analyze & Optimize</button>
                    </div>
                </div>
            </div>
            
            <div class="col-md-7">
                <div id="results"></div>
            </div>
        </div>
    </div>

    <script>
        async function optimizePrompt() {
            const prompt = document.getElementById('basePrompt').value;
            const selectedTool = document.querySelector('input[name="toolSelect"]:checked');
            
            if (!prompt || !selectedTool) {
                alert('Please enter a prompt and select a tool');
                return;
            }
            
            try {
                const response = await fetch('/optimize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        prompt,
                        tools: [selectedTool.value]
                    })
                });
                
                const data = await response.json();
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                displayResults(data);
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred while optimizing the prompt');
            }
        }
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '';
            
            // Display original prompt
            resultsDiv.innerHTML += `
                <div class="card mb-4">
                    <div class="card-header bg-light">
                        <h5 class="mb-0">Original Prompt</h5>
                    </div>
                    <div class="card-body">
                        <pre>${data.original_prompt}</pre>
                    </div>
                </div>
            `;
            
            // Display analysis
            resultsDiv.innerHTML += `
                <div class="card mb-4">
                    <div class="card-header bg-info text-white">
                        <h5 class="mb-0">Prompt Analysis</h5>
                    </div>
                    <div class="card-body">
                        <pre>${data.analysis}</pre>
                    </div>
                </div>
            `;
            
            // Display tool-specific optimizations
            data.results.forEach(result => {
                resultsDiv.innerHTML += `
                    <div class="card mb-4">
                        <div class="card-header bg-success text-white">
                            <h5 class="mb-0">Optimized for ${result.tool_name}</h5>
                        </div>
                        <div class="card-body">
                            <h6>Optimized Prompt:</h6>
                            <pre>${result.optimized_prompt}</pre>
                            
                            <h6 class="mt-3">Optimizations Applied:</h6>
                            <ul>
                                ${result.explanations.map(exp => `<li>${exp}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                `;
            });
        }
    </script>
</body>
</html>
        """)
    
    app.run(debug=True) 
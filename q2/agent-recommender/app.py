from quart import Quart, render_template, request, jsonify
from recommendation_engine import RecommendationEngine
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Quart(__name__)
engine = RecommendationEngine()

@app.route('/')
async def index():
    """Render the main page with the task input form."""
    return await render_template('index.html')

@app.route('/recommend', methods=['POST'])
async def recommend():
    """Process the task description and return recommendations."""
    data = await request.get_json()
    task_description = data.get('task_description', '')
    
    if not task_description:
        return jsonify({
            'error': 'Task description is required'
        }), 400
    
    try:
        recommendations = await engine.get_recommendations(task_description)
        return jsonify({
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

# Create templates directory and template files
os.makedirs('templates', exist_ok=True)

# Create index.html template
with open('templates/index.html', 'w') as f:
    f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Coding Agent Recommender</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-4xl font-bold text-center mb-8">AI Coding Agent Recommender</h1>
        
        <div class="max-w-4xl mx-auto">
            <div class="bg-white rounded-lg shadow-md p-8 mb-8">
                <h2 class="text-2xl font-semibold mb-4">Describe Your Coding Task</h2>
                <textarea 
                    id="taskDescription" 
                    class="w-full h-40 p-4 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 text-lg"
                    placeholder="Enter your coding task description here..."></textarea>
                <button 
                    onclick="getRecommendations()"
                    class="mt-6 bg-blue-500 text-white px-8 py-3 rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-400 text-lg">
                    Get Recommendations
                </button>
            </div>

            <div id="results" class="hidden">
                <h2 class="text-2xl font-semibold mb-6">Recommended Coding Agents</h2>
                <div id="recommendationsList" class="space-y-6"></div>
            </div>

            <div id="loading" class="hidden text-center py-12">
                <div class="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mx-auto"></div>
                <p class="mt-6 text-gray-600 text-lg">Analyzing your task...</p>
            </div>
        </div>
    </div>

    <script>
        async function getRecommendations() {
            const taskDescription = document.getElementById('taskDescription').value;
            if (!taskDescription) {
                alert('Please enter a task description');
                return;
            }

            // Show loading
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('results').classList.add('hidden');

            try {
                const response = await fetch('/recommend', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ task_description: taskDescription }),
                });

                const data = await response.json();

                if (response.ok) {
                    displayRecommendations(data.recommendations);
                } else {
                    alert(data.error || 'An error occurred');
                }
            } catch (error) {
                alert('An error occurred while fetching recommendations');
            } finally {
                document.getElementById('loading').classList.add('hidden');
            }
        }

        function displayRecommendations(recommendations) {
            const resultsDiv = document.getElementById('results');
            const listDiv = document.getElementById('recommendationsList');
            listDiv.innerHTML = '';

            recommendations.forEach((rec, index) => {
                const card = document.createElement('div');
                card.className = 'bg-white rounded-lg shadow-md p-8 mb-6';
                card.innerHTML = `
                    <div class="flex items-start justify-between mb-4">
                        <div class="flex-1">
                            <h3 class="text-2xl font-semibold">${index + 1}. ${rec.name}</h3>
                            <p class="text-gray-600 mt-2 text-lg">${rec.description}</p>
                        </div>
                        <div class="bg-blue-100 text-blue-800 px-4 py-2 rounded-full text-lg ml-4">
                            Score: ${(rec.score * 100).toFixed(1)}%
                        </div>
                    </div>

                    <div class="mt-6">
                        <h4 class="font-semibold text-gray-700 text-xl mb-2">Why this agent?</h4>
                        <ul class="list-disc list-inside text-gray-600 text-lg space-y-2">
                            ${rec.justification.map(j => `<li>${j}</li>`).join('')}
                        </ul>
                    </div>

                    <div class="mt-6 grid grid-cols-2 gap-8">
                        <div>
                            <h4 class="font-semibold text-gray-700 text-xl mb-2">Strengths</h4>
                            <ul class="list-disc list-inside text-gray-600 text-lg space-y-2">
                                ${rec.strengths.map(s => `<li>${s}</li>`).join('')}
                            </ul>
                        </div>
                        <div>
                            <h4 class="font-semibold text-gray-700 text-xl mb-2">Limitations</h4>
                            <ul class="list-disc list-inside text-gray-600 text-lg space-y-2">
                                ${rec.limitations.map(l => `<li>${l}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                `;
                listDiv.appendChild(card);
            });

            resultsDiv.classList.remove('hidden');
        }
    </script>
</body>
</html>
    """)

if __name__ == '__main__':
    app.run(debug=True) 
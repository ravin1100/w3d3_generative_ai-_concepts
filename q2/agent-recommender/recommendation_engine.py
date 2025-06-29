import json
import os
from typing import List, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

class RecommendationEngine:
    def __init__(self, db_path: str = 'agents_db.json'):
        """Initialize the recommendation engine with the agents database."""
        self.db_path = db_path
        self.agents = self._load_agents()

    def _load_agents(self) -> Dict[str, Any]:
        """Load the agents database from JSON file."""
        with open(self.db_path, 'r') as f:
            return json.load(f)

    async def analyze_task(self, task_description: str) -> Dict[str, Any]:
        """Analyze the task using Gemini API to extract key features."""
        prompt = f"""
        Analyze the following coding task and extract key features:
        Task: {task_description}
        
        Please provide a structured analysis with:
        1. Task complexity (simple/medium/complex)
        2. Main programming concepts involved
        3. Special requirements or constraints
        4. Development environment needs
        5. Scale of the project
        
        Format your response like this:
        Complexity: [complexity level]
        Concepts: [concept1], [concept2], ...
        Requirements: [req1], [req2], ...
        Environment: [env1], [env2], ...
        Scale: [scale]
        """

        # Generate content using the async API
        response = await model.generate_content_async(prompt)
        return self._parse_gemini_response(response.text)

    def _parse_gemini_response(self, response: str) -> Dict[str, Any]:
        """Parse Gemini's response into structured data."""
        analysis = {
            'complexity': 'medium',  # default
            'concepts': [],
            'requirements': [],
            'environment': [],
            'scale': 'medium'  # default
        }
        
        # Parse line by line
        for line in response.split('\n'):
            line = line.strip().lower()
            if line.startswith('complexity:'):
                complexity = line.split(':')[1].strip()
                if 'simple' in complexity:
                    analysis['complexity'] = 'simple'
                elif 'complex' in complexity:
                    analysis['complexity'] = 'complex'
            elif line.startswith('concepts:'):
                concepts = line.split(':')[1].strip()
                analysis['concepts'] = [c.strip() for c in concepts.split(',') if c.strip()]
            elif line.startswith('requirements:'):
                reqs = line.split(':')[1].strip()
                analysis['requirements'] = [r.strip() for r in reqs.split(',') if r.strip()]
            elif line.startswith('environment:'):
                envs = line.split(':')[1].strip()
                analysis['environment'] = [e.strip() for e in envs.split(',') if e.strip()]
            elif line.startswith('scale:'):
                scale = line.split(':')[1].strip()
                if 'small' in scale:
                    analysis['scale'] = 'small'
                elif 'large' in scale:
                    analysis['scale'] = 'large'

        return analysis

    def calculate_agent_scores(self, task_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate scores for each agent based on task analysis."""
        scored_agents = []
        
        for agent in self.agents['agents']:
            score = 0.0
            justification = []
            
            # Score based on complexity handling
            complexity_score = agent['complexity_handling'][task_analysis['complexity']]
            score += complexity_score * 0.4  # 40% weight for complexity match
            justification.append(
                f"Complexity handling score: {complexity_score:.2f} for {task_analysis['complexity']} tasks"
            )
            
            # Score based on capabilities match
            capabilities_score = self._calculate_capabilities_score(agent, task_analysis)
            score += capabilities_score * 0.3  # 30% weight for capabilities
            justification.append(
                f"Capabilities match score: {capabilities_score:.2f}"
            )
            
            # Score based on use case match
            use_case_score = self._calculate_use_case_score(agent, task_analysis)
            score += use_case_score * 0.3  # 30% weight for use case match
            justification.append(
                f"Use case match score: {use_case_score:.2f}"
            )
            
            scored_agents.append({
                'id': agent['id'],
                'name': agent['name'],
                'score': score,
                'justification': justification,
                'description': agent['description'],
                'strengths': agent['strengths'],
                'limitations': agent['limitations']
            })
        
        # Sort by score in descending order
        return sorted(scored_agents, key=lambda x: x['score'], reverse=True)

    def _calculate_capabilities_score(self, agent: Dict[str, Any], task_analysis: Dict[str, Any]) -> float:
        """Calculate how well agent capabilities match task requirements."""
        score = 0.0
        required_capabilities = set(task_analysis['concepts'] + task_analysis['requirements'])
        agent_capabilities = set([cap.lower() for cap in agent['capabilities']])
        
        if required_capabilities:
            matches = sum(1 for req in required_capabilities 
                        if any(cap in req.lower() for cap in agent_capabilities))
            score = matches / len(required_capabilities)
        else:
            score = 0.5  # Default score if no specific requirements
            
        return score

    def _calculate_use_case_score(self, agent: Dict[str, Any], task_analysis: Dict[str, Any]) -> float:
        """Calculate how well the agent's use cases match the task."""
        score = 0.0
        scale_mapping = {
            'small': ['individual developers', 'rapid prototyping', 'quick prototypes'],
            'medium': ['small to medium projects', 'full project development'],
            'large': ['enterprise development', 'large projects']
        }
        
        relevant_use_cases = scale_mapping.get(task_analysis['scale'], [])
        agent_use_cases = [uc.lower() for uc in agent['best_use_cases']]
        
        if relevant_use_cases:
            matches = sum(1 for uc in relevant_use_cases 
                        if any(agent_uc in uc.lower() for agent_uc in agent_use_cases))
            score = matches / len(relevant_use_cases)
        else:
            score = 0.5  # Default score if no specific use cases
            
        return score

    async def get_recommendations(self, task_description: str) -> List[Dict[str, Any]]:
        """Get top 3 agent recommendations for the given task."""
        task_analysis = await self.analyze_task(task_description)
        scored_agents = self.calculate_agent_scores(task_analysis)
        return scored_agents[:3]  # Return top 3 recommendations 
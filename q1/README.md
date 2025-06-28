# LLM Inference Calculator

This tool calculates and benchmarks inference metrics (latency, memory usage, and throughput) for different language models.

## Supported Models

- TinyLlama (1.1B parameters) - Local inference
- OpenAI GPT API - Cloud API
- Google Gemini Pro API - Cloud API

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with the following:
```
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

Run the calculator with:
```bash
python inference_calculator.py
```

The script will:
1. Load the local TinyLlama model
2. Initialize API clients for OpenAI and Gemini
3. Run benchmarks for each model
4. Calculate and display metrics including:
   - Average latency (ms)
   - Memory usage (MB)
   - Tokens per second
   - Cost estimates (for API models)

## Notes

- The calculator will fall back to default metrics if a model fails to load or API keys are not configured
- Memory usage is only tracked for local models
- Token counts for Gemini API are estimated based on word count
- All metrics are averaged over multiple runs for accuracy

## Project Structure

```
.
├── inference_calculator.py    # Main calculator implementation
├── research_notes.md         # Research findings and model comparisons
├── scenario_analysis.md      # Use case analysis and recommendations
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Features

- Latency estimation
- Memory usage calculation
- Cost prediction
- Hardware compatibility checking
- Support for both local and API-based models
- Batch processing considerations

## Documentation

- See `research_notes.md` for detailed model comparisons and inference basics
- See `scenario_analysis.md` for real-world use cases and recommendations

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

MIT License 
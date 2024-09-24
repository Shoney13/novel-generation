# Setup Instructions
1. Clone the Repository:

```bash
git clone https://github.com/Shoney13/novel-generation.git
```

2. Create a Virtual Environment:

Using venv:

```bash
python3 -m venv .venv
```

Using conda:

```bash
conda create -n .conda python=3.8
```

3. Activate the Virtual Environment:

For venv on Unix/macOS:

```bash
source .venv/bin/activate
```

For venv on Windows:

```bash
.venv\Scripts\activate
```

For conda:

```bash
conda activate .conda
```

Install Dependencies:

```bash
pip install -r requirements.txt
```

This installs all required packages listed in requirements.txt.

4. Configure Environment Variables:

Create a copy of the .env.example file and rename it to .env.
In the .env file, add your OpenAI API key:
```makefile
OPENAI_API_KEY="your-api-key-here"
```
Replace your-api-key-here with your actual API key.

5. Run the Application:

```bash
python3 app.py
```

This command starts the application.
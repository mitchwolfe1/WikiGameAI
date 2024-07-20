# WikiGame Solver

This project is a solver for the WikiGame, which uses web scraping and text embeddings to find a path between two Wikipedia pages. The solver fetches links on each page and selects the most relevant one based on cosine similarity of the embeddings.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/mitchwolfe1/WikiGameAI.git
    cd WikiGameAI 
    ```

2. Create a virtual environment and install dependencies:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage

Run the solver with:
```sh
python main.py "start_url" "end_url"


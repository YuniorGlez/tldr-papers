import logging
from serpapi import GoogleSearch
from llama_cpp import Llama
from rich.console import Console
from rich.logging import RichHandler
from config import SERP_API_KEY

# Definir palabras clave para la búsqueda
keywords = ["inteligencia artificial", "aprendizaje automático", "aprendizaje profundo", "visión artificial", "procesamiento del lenguaje natural", "robótica"]

# Configurar el logging con RichHandler
console = Console()
logging.basicConfig(
    level="WARNING",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)

# Cargar el LLM local (llama-cpp-python)
model_path = "models/mistral-7b-instruct-v0.2.Q3_K_M.gguf"  # Ajusta la ruta si es necesario
llm = Llama(model_path=model_path, n_ctx=32768, n_threads=8)  # Ajusta los parámetros según sea necesario

# Función para buscar papers con SerpApi
def search_papers(keywords, num_papers=100):
    params = {
        "engine": "google_scholar",
        "q": " OR ".join(keywords),
        "hl": "es",  # Buscar en español
        "num": num_papers,
        "api_key": SERP_API_KEY,
    }
    console.log(f"[bold blue]Buscando papers en google schoolar sobre [{', '.join(keywords)}].[/]")
    search = GoogleSearch(params)
    results = search.get_dict()

    papers = []
    for index, result in enumerate(results["organic_results"]):
        paper = {
            "index": index,
            "title": result.get("title"),
            "authors": result.get("authors"),
            "summary": result.get("publication_info").get("summary"),
            "publication_date": result.get("publication_info", {}).get("date"),
            "publication_name": result.get("publication_info", {}).get("journal"),
            "link": result.get("link"),
        }
        papers.append(paper)

    console.log(f"[bold green]Se encontraron {len(papers)} papers.[/]")
    return papers


def process_papers_with_llm(papers, prompt_path):
    with open(prompt_path, "r") as f:
        prompt = f.read()

    # Preparar la entrada para el LLM
    papers_text = "\n\n".join([
        f"## Paper {paper['index'] + 1}:\n"
        f"**Título:** {paper['title']}\n"
        # f"**Autores:** {', '.join(paper['authors'])}\n"
        f"**Resumen:** {paper['summary']}\n"
        f"**Fecha de publicación:** {paper.get('publication_date', 'N/A')}\n"
        f"**Publicación:** {paper.get('publication_name', 'N/A')}\n"
        for paper in papers
    ])
    llm_input = prompt.replace("{{PAPERS}}", papers_text)

    console.log("[bold magenta]Enviando metadatos de los papers al LLM Local (Mixtral)...[/]")

    # Obtener la salida del LLM
    output = llm(llm_input)

    # Procesar la salida (extraer los índices relevantes)
    predicted_text = output["choices"][0]["text"].strip()
    relevant_indices = [int(index) - 1 for index in predicted_text.split(",") if index.isdigit()]

    console.log(f"[bold green]El LLM identificó {len(relevant_indices)} papers relevantes.[/]")
    return relevant_indices


# Ejemplo de uso
papers = search_papers(keywords)
relevant_indices = process_papers_with_llm(papers, "prompts/pre_filtering_papers.txt")

# Mostrar los papers relevantes
relevant_papers = [papers[index] for index in relevant_indices]
for paper in relevant_papers:
    console.log(f"[bold blue]Título: {paper['title']}[/]")
    console.log(f"[bold green]Resumen: {paper['summary']}[/]")
    # ... mostrar otra información relevante ...
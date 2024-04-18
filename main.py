import logging
from serpapi import GoogleSearch
from llama_cpp import Llama
from rich.console import Console
from rich.logging import RichHandler
from config import SERP_API_KEY


# Definir palabras clave para la búsqueda
keywords = ["inteligencia artificial", "aprendizaje automático", "aprendizaje profundo", "visión artificial", "procesamiento del lenguaje natural", "robótica"]

# Configurar el logging con RichHandler
console = Console()  # Crear un objeto Console global

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
def search_papers(keywords, num_papers=10, console=console):  # Pasar console como argumento
    params = {
        "engine": "google_scholar",
        "q": " OR ".join(keywords),
        "hl": "es",  # Buscar en español
        "num": num_papers,
        "api_key": SERP_API_KEY, 
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    papers = []
    for result in results["organic_results"]:
        paper = {
            "title": result.get("title"),
            "authors": result.get("authors"),
            "summary": result.get("publication_info").get("summary"),
            "link": result.get("link"),
        }
        papers.append(paper)

    console.log(f"[bold green]Se encontraron {len(papers)} papers.[/]")
    return papers

# Ejemplo de uso
papers = search_papers(keywords)
import logging
import json
from serpapi import GoogleSearch
import ollama
from ollama import Client
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

# Cargar el modelo LLaMA3 con OLLAMA
ollama.pull('llama3:8b')
client = Client(host='http://localhost:11434')
model = 'llama3:8b'

# Función para buscar papers con SerpApi
def search_papers(keywords, num_papers=10):
    params = {
        "engine": "google_scholar",
        "q": " OR ".join(keywords),
        "hl": "es",  # Buscar en español
        "num": num_papers,
        "api_key": SERP_API_KEY,
    }
    console.log(f"[bold blue]Buscando papers en google schoolar sobre {','.join(keywords)}.[/]")
    search = GoogleSearch(params)
    results = search.get_dict()

    papers = []
    for index, result in enumerate(results["organic_results"]):
        paper = {
            "index": index -1,
            "title": result.get("title"),
            "authors": result.get("authors"),
            "summary": result.get("publication_info", {}).get("summary", ""),
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

    console.log("[bold magenta]Enviando metadatos de los papers al LLM Local (LLaMA3)...[/]")

    # Obtener la salida del LLM
    response = client.chat(model=model, messages=[{"role": "user", "content": llm_input}])
    # console.log(f"[bold magenta]Salida del LLM: {response["message"]}[/]")


    if "message" in response and "content" in response["message"]:
        predicted_text = response["message"]["content"].strip()
        console.log(f"[bold magenta]Salida del LLM (texto): {predicted_text}[/]")
    else:
        console.log(f"[bold red]La respuesta del LLM no es válida[/]")
        raise ValueError("La respuesta del LLM no es válida")

    try:
        response_json = json.loads(predicted_text)
    except json.JSONDecodeError:
        console.log(f"[bold red]La respuesta del LLM no es un JSON válido: {predicted_text}[/]")
        raise ValueError("La respuesta del LLM no es un JSON válido")

    # Extraer los índices relevantes y no relevantes
    relevant_papers = response_json["relevant_papers"]
    non_relevant_papers = response_json["non_relevant_papers"]

    # Mostrar los resultados
    console.log(f"[bold green]El LLM identificó {len(relevant_papers)} papers relevantes.[/]")
    console.log(f"[bold red]El LLM identificó {len(non_relevant_papers)} papers no relevantes.[/]")
    for paper in relevant_papers:
        index = paper["index"]
        title = papers[index]["title"]
        summary = papers[index]["summary"]
        reason = paper["reason"]
        console.log(f"[bold blue]Título: {title}[/]")
        console.log(f"[bold green]Resumen: {summary}[/]")
        console.log(f"[bold yellow]Razón: {reason}[/]")
    for paper in non_relevant_papers:
        index = paper["index"]
        title = papers[index]["title"]
        summary = papers[index]["summary"]
        reason = paper["reason"]
        console.log(f"[bold red]Título: {title}[/]")
        console.log(f"[bold red]Resumen: {summary}[/]")
        console.log(f"[bold red]Razón: {reason}[/]")

    return relevant_papers, non_relevant_papers

# Ejemplo de uso
papers = search_papers(keywords)
relevant_papers, non_relevant_papers = process_papers_with_llm(papers, "prompts/pre_filtering_papers.txt")

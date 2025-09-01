import time
from agents import Runner, trace
from ddgs import DDGS
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from models import SearchResult

from research_agents.follow_up_agent import FollowUpDecisionResponse
from research_agents.follow_up_agent import follow_up_decision_agent
from research_agents.search_agent import search_agent

from research_agents.synthesis_agent import synthesis_agent
from research_agents.query_agent import QueryResponse, query_agent


console = Console()

# The number of sub queries in the original query
MAX_SEARCH_RESULTS = 2


class ResearchCoordinator:
    def __init__(self, query: str):
        self.query = query
        self.search_results = []
        self.iteration = 1

    async def research(self) -> str:
        with trace("Deep Research Workflow"):

            query_response = await self.generate_queries()

            await self.perform_research_for_queries(queries=query_response.queries)

            final_report = await self.synthesis_report()

            console.print("\n[bold green]✓ Research complete![/bold green]\n")
            console.print(Markdown(final_report))
            with open("./src/case_study2/deep_research/output/report.md", "w") as f:
                f.write(final_report)

            return final_report

    async def generate_queries(self) -> QueryResponse:
        """This generates a number od query terms to be used for the research as well as its thoughts and queries to run for each."""
        with console.status("[bold cyan]Analyzing query...[/bold cyan]") as status:

            # Run the query agent
            result = await Runner.run(query_agent, input=self.query)

            # Display the results
            console.print(Panel("[bold cyan]Query Analysis[/bold cyan]"))
            console.print(f"[yellow]Thoughts:[/yellow] {result.final_output.thoughts}")
            console.print("\n[yellow]Generated Search Queries:[/yellow]")
            for i, query in enumerate(result.final_output.queries, 1):
                console.print(f"  {i}. {query}")
            # This will log THOUGHTSD and QUERIES it will use (3)
            with open("./src/case_study2/deep_research/output/log.md", "w") as f:
                f.write(f"# THOUGHTS - {query}\n")
                f.write(f"\n{result.final_output.thoughts}\n")
                f.write("\n# GENERATED SEARCH QUERIES\n")
                for i, query in enumerate(result.final_output.queries, 1):
                    f.write(f"\n  {i}. {query}\n")

            return result.final_output

    def duckduckgo_search(self, query: str):
        """
        Executes a search query using the DuckDuckGo search engine.

        Args:
            query (str): The search query string.

        Returns:
            list: A list of search results, each containing details such as
                title and URL. Returns an empty list if an error occurs
                during the search process.
        """

        try:
            results = DDGS().text(
                query,
                region="us-en",
                safesearch="on",
                timelimit="y",
                max_results=MAX_SEARCH_RESULTS,
            )
            return results
        except Exception as ex:
            console.print(f"[bold red]Search error:[/bold red] {str(ex)}")
            return []

    async def perform_research_for_queries(self, queries: list[str]) -> None:

        # get all of the search results for each query
        all_search_results = {}
        for query in queries:
            search_results = self.duckduckgo_search(query)
            all_search_results[query] = search_results
        # Query items for each of the three main queries
        for query in queries:
            console.print(f"\n[bold cyan]Searching for:[/bold cyan] {query}")
            with open("./src/case_study2/deep_research/output/log.md", "a") as f:
                f.write(f"\n\n## SUB QUERY: {query}\n")
            for result in all_search_results[query]:
                console.print(f"  [green]Result:[/green] {result['title']}")
                console.print(f"  [dim]URL:[/dim] {result['href']}")
                console.print("  [cyan]Analyzing content...[/cyan]")

                start_analysis_time = time.time()
                search_input = f"Title: {result['title']}\nURL: {result['href']}"
                agent_result = await Runner.run(search_agent, input=search_input)
                analysis_time = time.time() - start_analysis_time
                with open(
                    "./src/case_study2/deep_research/output/log.md",
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        f"\nWeb Query: {result['title']} - URL: {result['href']} - TIME: {analysis_time} now analyzing content...\n\n"
                    )

                search_result = SearchResult(
                    title=result["title"],
                    url=result["href"],
                    summary=agent_result.final_output,
                )

                self.search_results.append(search_result)

                summary_preview = agent_result.final_output[:100] + (
                    "..." if len(agent_result.final_output) > 100 else ""
                )

                console.print(f"  [green]Summary:[/green] {summary_preview}")

        for query in queries:
            console.print(f"\n[bold cyan]Searching for:[/bold cyan] {query}")

            for result in all_search_results[query]:
                console.print(f"  [green]Result:[/green] {result['title']}")
                console.print(f"  [dim]URL:[/dim] {result['href']}")
                console.print("  [cyan]Analyzing content...[/cyan]")

                start_analysis_time = time.time()
                search_input = f"Title: {result['title']}\nURL: {result['href']}"
                agent_result = await Runner.run(search_agent, input=search_input)
                analysis_time = time.time() - start_analysis_time

                search_result = SearchResult(
                    title=result["title"],
                    url=result["href"],
                    summary=agent_result.final_output,
                )

                self.search_results.append(search_result)

                summary_preview = agent_result.final_output[:100] + (
                    "..." if len(agent_result.final_output) > 100 else ""
                )

                console.print(f"  [green]Summary:[/green] {summary_preview}")
                console.print(
                    f"  [dim]Analysis completed in {analysis_time:.2f}s[/dim]\n"
                )

        console.print(
            f"\n[bold green]✓ Research round complete![/bold green] Found {len(all_search_results)} sources across {len(queries)} queries."
        )

    async def generate_followup(self) -> FollowUpDecisionResponse:
        with console.status(
            "[bold cyan]Evaluating if more research is needed...[/bold cyan]"
        ) as status:
            findings_text = f"Original Query: {self.query}\n\nCurrent Findings:\n"
            for i, result in enumerate(self.search_results, 1):
                findings_text += f"\n{i}. Title: {result.title}\n   URL: {result.url}\n   Summary: {result.summary}\n"

            result = await Runner.run(follow_up_decision_agent, input=findings_text)

            console.print(Panel("[bold cyan]Follow-up Decision[/bold cyan]"))
            console.print(
                f"[yellow]Decision:[/yellow] {'More research needed' if result.final_output.should_follow_up else 'Research complete'}"
            )
            console.print(
                f"[yellow]Reasoning:[/yellow] {result.final_output.reasoning}"
            )

            if result.final_output.should_follow_up:
                console.print("\n[yellow]Follow-up Queries:[/yellow]")
                for i, query in enumerate(result.final_output.queries, 1):
                    console.print(f"  {i}. {query}")

            return result.final_output

    async def synthesis_report(self) -> str:
        with console.status(
            "[bold cyan]Synthesizing research findings...[/bold cyan]"
        ) as status:
            findings_text = f"Query: {self.query}\n\nSearch Results:\n"
            for i, result in enumerate(self.search_results, 1):
                findings_text += f"\n{i}. Title: {result.title}\n   URL: {result.url}\n   Summary: {result.summary}\n"
            with open(
                "./src/case_study2/deep_research/output/log.md", "a", encoding="utf-8"
            ) as f:
                f.write("\n### Synthesising Report...\n")
            result = await Runner.run(synthesis_agent, input=findings_text)

            console.print(Panel("[bold cyan]Synthesis Report[/bold cyan]"))
            console.print(f"[yellow]Report:[/yellow] {result.final_output}")

            with open(
                "./src/case_study2/deep_research/output/log.md", "a", encoding="utf-8"
            ) as f:
                f.write("\n### Final Report:\n")
                f.write(f"\n{result.final_output}\n")
            return result.final_output

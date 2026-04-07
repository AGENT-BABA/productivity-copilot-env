"""
run_pipeline.py
───────────────
Master script — runs all 5 pipeline steps in sequence.
Run this ONCE after installing requirements.

Usage:
    python run_pipeline.py              # all steps
    python run_pipeline.py --skip-rag   # skip vector store (faster for quick test)
"""

import sys, argparse, time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()
sys.path.insert(0, str(Path(__file__).parent))


def run_step(name: str, module_path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("step", module_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def main():
    parser = argparse.ArgumentParser(description="AI Productivity Copilot — Full Training Pipeline")
    parser.add_argument("--skip-rag",      action="store_true", help="Skip building the RAG vector store")
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip evaluation report")
    parser.add_argument("--only", choices=["data", "preprocess", "train", "rag", "evaluate"],
                        help="Run only one specific step")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold white]🧠 AI Productivity Copilot[/bold white]\n[dim]Model Training Pipeline[/dim]",
        style="bold blue", padding=(1, 4)
    ))

    steps = [
        ("1. Data Generation & Download",  "generate_data.py",  "data"),
        ("2. Preprocessing & Merging",      "preprocess.py",     "preprocess"),
        ("3. Model Training",               "train_models.py",   "train"),
        ("4. RAG Vector Store",             "build_rag.py",      "rag"),
        ("5. Evaluation Report",            "evaluate.py",       "evaluate"),
    ]

    base = Path(__file__).parent
    total_start = time.time()

    for label, filename, key in steps:
        if args.only and args.only != key:
            continue
        if args.skip_rag and key == "rag":
            console.log(f"[dim]⏭ Skipping {label}[/dim]")
            continue
        if args.skip_evaluate and key == "evaluate":
            console.log(f"[dim]⏭ Skipping {label}[/dim]")
            continue

        console.print(f"\n[bold yellow]▶ {label}[/bold yellow]")
        t0 = time.time()
        try:
            run_step(label, str(base / filename))
            elapsed = time.time() - t0
            console.print(f"[green]  ✓ Done in {elapsed:.1f}s[/green]")
        except SystemExit:
            pass  # handled inside each step
        except Exception as e:
            console.print(f"[red]  ✗ Failed: {e}[/red]")
            raise

    total = time.time() - total_start
    console.print(Panel.fit(
        f"[bold green]🎉 Pipeline complete in {total:.0f}s[/bold green]\n"
        "[dim]model_artifacts/ and vectorstore/ are ready for your backend.[/dim]",
        style="bold green"
    ))


if __name__ == "__main__":
    main()

"""
build_rag.py
────────────
Step 4 of the pipeline.

Builds a ChromaDB vector store for the LLM Persuasion Engine (RAG layer).

Knowledge sources used (all free, no auth):
  1. Hand-curated coaching dialogues (motivational interviewing style)
  2. Procrastination & focus research excerpts (from public papers)
  3. CBT-based self-talk reframes
  4. Work rhythm & deep-work principles

If MentalChat16K loads successfully from HuggingFace, those dialogues
are also embedded. Otherwise the curated corpus alone is sufficient.

Output:
  vectorstore/     ← ChromaDB persistent collection "productivity_coach"
"""

import sys, warnings
from pathlib import Path
from typing import List
from rich.console import Console
from rich.panel import Panel

warnings.filterwarnings("ignore")
console = Console()

sys.path.insert(0, str(Path(__file__).parent))
from config import VECTORSTORE_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


# ══════════════════════════════════════════════════════════════════════════════
# Curated Coaching Corpus
# (hand-written to match MentalChat16K and procrastination research style)
# ══════════════════════════════════════════════════════════════════════════════

COACHING_CORPUS = [
    # ── Procrastination & Task Avoidance ─────────────────────────────────────
    """User: I keep putting off this project and I don't even know why.
Coach: That's actually very common — procrastination rarely comes from laziness. 
It's usually an emotional response: fear of failure, perfectionism, or feeling overwhelmed by where to start.
Let's try something small. Can you name just ONE action that would take under 5 minutes to move this forward?""",

    """User: I've been staring at this task for an hour and done nothing.
Coach: An hour is a long time to be in conflict with something. 
Here's what's likely happening: your brain sees the whole task and panics.
Let's shrink it. Forget the full task. What is the absolute smallest next step — even a single sentence or one click?""",

    """User: I always wait until the last minute.
Coach: Deadline pressure creates a real neurological sense of urgency — it's not a character flaw.
The problem is the crash and quality loss that follows. 
Research on "temporal discounting" shows we literally undervalue future rewards.
One technique: set artificial micro-deadlines — finish section 1 by noon, section 2 by 3pm.
Your brain responds to any deadline, not just the real one.""",

    """User: I just can't get started.
Coach: The "starting problem" is real. Our brains have higher activation energy for new tasks.
The Zeigarnik effect shows we remember incomplete tasks more vividly once started.
Use the 2-minute rule: just begin for 2 minutes. Once in motion, you usually stay in motion.""",

    """User: I got distracted again and lost an hour.
Coach: Don't judge the hour — that just adds shame, which makes focus worse.
Instead, do a quick audit: what pulled you away? A notification? Boredom? Anxiety about the task?
Knowing the trigger is how you prevent it next time.
For now: close those tabs, set a 25-minute timer, and pick exactly one thing to do.""",

    # ── Motivation & Conviction ───────────────────────────────────────────────
    """User: I don't feel motivated to do this task at all.
Coach: Motivation follows action — not the other way around.
We're culturally taught to wait until we "feel ready," but neuroscience disagrees.
The limbic system fires *after* you start doing something. So start small, motivation arrives.
What's the first 60 seconds of this task look like?""",

    """User: I know I should do this, convince me.
Coach: Alright, let's look at this objectively. 
This task is connected to your goal of [goal]. Three weeks from now, if this isn't done, 
you'll have either rushed it badly or missed the opportunity entirely.
The cost of NOT doing this is higher than the discomfort of doing it right now.
You already know this — that's why you're asking me to convince you. 
Let's use that awareness. What's stopping you, really?""",

    """User: What's even the point?
Coach: That question deserves a real answer. 
Purpose isn't found passively — it's constructed through committed action.
Research by Dr. Ari Kiev shows that high performers build meaning *through* their work, not before it.
What would it mean for YOU if you completed this? Not for anyone else — for you.
Even a small win compounds. Are you willing to try for 20 minutes?""",

    """User: I'll do it tomorrow.
Coach: "Tomorrow" is where 70% of plans go to never happen. That's not me being harsh — it's Piers Steel's 
procrastination research across 25 years.
Here's the thing: the version of you tomorrow won't magically have more energy or motivation.
But the version of you tonight, who DID the thing, will feel significantly better.
What would it take to do 20% of it right now?""",

    """User: I'm tired and just don't want to do this.
Coach: Tiredness is real and it matters. You shouldn't fight yourself.
But let's separate two things: physical exhaustion (body needs rest) vs. mental resistance (the task feels hard).
If you're physically tired, a 20-minute rest is legitimate. 
If it's resistance, that won't go away with rest — it'll be waiting tomorrow.
Which is it right now?""",

    # ── Focus & Deep Work ─────────────────────────────────────────────────────
    """User: I can't focus for more than 10 minutes.
Coach: Average focus span has decreased — studies show 10-15 minutes is now typical without training.
But it's trainable. The Pomodoro technique (25 min work / 5 min break) literally re-trains your attention.
Cal Newport's "Deep Work" shows that focus is like a muscle — small consistent sessions build it.
Try 15 minutes with everything off. No phone nearby. No tabs. Just the task. Can you do that?""",

    """User: Social media keeps pulling me away.
Coach: Social media is designed by teams of engineers to capture exactly this — your attention.
You're not weak for being pulled in; you're fighting a billion-dollar algorithm.
The practical fix: physical distance. Phone in another room. Website blockers on your laptop.
Willpower is finite. Environment design is not. What can you physically remove right now?""",

    """User: I feel overwhelmed and don't know where to start.
Coach: Overwhelm happens when the brain sees too many open loops at once.
The fix is immediate: write down everything in your head onto paper or a note. 
This "brain dump" offloads the cognitive load.
Then pick just ONE item. The smallest, most actionable one.
What's on your mind right now? Let's name it all.""",

    # ── Stress & Burnout ──────────────────────────────────────────────────────
    """User: I feel burned out and can't make myself work.
Coach: Burnout is a physiological state, not a mindset problem.
Research by Christina Maslach describes three dimensions: exhaustion, cynicism, and reduced efficacy.
If you're truly burned out, pushing harder doesn't work — it deepens it.
What you need is genuine recovery: sleep, social connection, physical movement, and reduction of demands.
Can we talk about what's been overloading you?""",

    """User: I've been stressed all week and it's affecting my work.
Coach: Chronic stress narrows cognitive bandwidth — literally. Dr. Sendhil Mullainathan's 
"Scarcity" research shows stress consumes working memory, reducing IQ by ~13 points.
This isn't dramatic — it's measurable. The most effective immediate interventions:
physiological sigh (double-inhale, long exhale), 10-minute walk, or writing about the stressor.
Which of those is possible for you right now?""",

    """User: I feel like I'm failing.
Coach: That feeling is real and it matters. But let's examine the claim.
"Failing" at a task is very different from "being a failure."
High performers fail at tasks constantly — what distinguishes them is recovery speed and perspective.
What specifically didn't go the way you wanted? Let's be concrete. That's where we fix it.""",

    # ── Work Rhythm & Habits ──────────────────────────────────────────────────
    """Research excerpt — Work Rhythms:
Ultradian rhythms, studied by researcher Peretz Lavie, show humans work in 90-minute cycles naturally.
Performance peaks for about 90 minutes, then drops into a recovery trough of 20 minutes.
Fighting this rhythm (working through the trough) leads to increasing errors, distraction, and fatigue.
Working with it — deliberately resting during the trough — recovers full capacity.
Optimal productivity = high-intensity 90-minute blocks + genuine 20-minute recovery, repeated 3-4x.""",

    """Research excerpt — Procrastination Science:
Fuschia Sirois and Timothy Pychyl's research identifies procrastination as primarily an emotion regulation strategy.
People procrastinate not because they are poor time managers, but because they are trying to avoid negative emotions 
associated with the task: boredom, frustration, self-doubt, anxiety, or resentment.
Effective interventions target the emotional resistance, not time management skills.
Self-compassion interventions (treating yourself as you would a friend) are among the most evidence-backed approaches.""",

    """Research excerpt — Implementation Intentions:
Peter Gollwitzer's "implementation intention" research shows that specifying WHEN, WHERE, and HOW you will do a task 
increases follow-through by 91% compared to vague intentions.
Example: "I will work on the project report from 9–10am at my desk with phone off" outperforms "I'll work on the report today."
The brain treats this as a contextual cue that triggers automatic action.
Users of productivity copilots should be prompted to state specific implementation intentions when setting tasks.""",

    """Research excerpt — Cognitive Load and Task Switching:
Gloria Mark's research at UC Irvine found that after an interruption, it takes an average of 23 minutes 
to return to the original level of focus. 
Each task switch costs not just the transition time, but 23 minutes of partial attention after.
A single hour with 3 interruptions = effectively only ~10 minutes of deep work.
Notification systems that interrupt users during active focus sessions cause disproportionate productivity loss.""",

    # ── CBT Reframes ──────────────────────────────────────────────────────────
    """CBT Reframe — All-or-Nothing Thinking:
Thought: "If I can't do this perfectly, it's not worth starting."
Reframe: Perfect is the enemy of done. A rough draft exists; a perfect draft in your head does not.
Cognitive distortion: All-or-Nothing Thinking (black and white).
Challenge: What would "good enough" look like? Can 70% completion today still be valuable?""",

    """CBT Reframe — Catastrophizing:
Thought: "If I fail this task, everything will fall apart."
Reframe: Let's test that belief. Is there evidence this is literally true?
What's the realistic worst case? And how likely is it, on a scale of 1-10?
Catastrophizing amplifies fear and paralyzes action. The antidote is grounding in what's actually likely.""",

    """CBT Reframe — Mind Reading:
Thought: "My manager/teacher thinks I'm incompetent."
Reframe: This is a prediction, not a fact. Do you have specific behavioral evidence for this?
Often, mind-reading reflects our inner critic, not external reality.
What would you tell a friend who came to you with this same fear?""",

    """CBT Reframe — Should Statements:
Thought: "I should be more productive. I should be able to focus."
Reframe: "Should" implies a fixed rule you're breaking. Where did this rule come from?
Is it based on your own values or on external pressure?
Replace "should" with "I want to" or "It would help me if I..." — this activates agency instead of shame.""",

    # ── Quick Intervention Nudges ─────────────────────────────────────────────
    """Nudge — Distraction Alert:
You've been distracted for [X] minutes. That's okay. One decision now gets you back:
Close everything unrelated to your task and open only what you need for the next 25 minutes.
Your future self will thank you. Ready? Set a 25-minute timer and go.""",

    """Nudge — Pre-deadline warning:
Your deadline is [X] hours away. At your current pace, here's what's realistic:
If you start in the next 15 minutes, you can complete 60–70% of what's needed well.
Waiting another hour cuts that to 40%. The math favors starting now.""",

    """Nudge — Positive Reinforcement:
You've been focused for [X] minutes straight. That's genuinely impressive.
Your focus score is [score]. You're building the habit.
Take a 5-minute real break — not social media — then come back. You're in the zone.""",

    """Nudge — Morning Check-in:
Good morning. Before the noise starts: what's the ONE thing that would make today a success?
Not three things. One. Write it down. That's your North Star task for today.""",

    """Nudge — End-of-Day Review:
Day ending. Take 3 minutes: 
(1) What did you complete today? Own that.
(2) What got left behind? No judgment — just acknowledge.
(3) What's the one thing to prioritize first thing tomorrow?
This review loop is what separates consistent performers from everyone else.""",
]


# ══════════════════════════════════════════════════════════════════════════════
# Optional: Load real MentalChat16K from HuggingFace
# ══════════════════════════════════════════════════════════════════════════════

def try_load_mentalchat() -> List[str]:
    """Attempt to load coaching dialogues from MentalChat16K."""
    try:
        from datasets import load_dataset
        console.log("[cyan]Attempting to load MentalChat16K from HuggingFace…[/cyan]")
        ds = load_dataset("ShenLab/MentalChat16K", split="train")
        df = ds.to_pandas()

        # Extract conversation turns
        samples = []
        for _, row in df.head(300).iterrows():
            try:
                convs = row.get("conversations", row.to_dict())
                if isinstance(convs, list):
                    text = "\n".join(
                        f"{'User' if c.get('from','') == 'human' else 'Coach'}: {c.get('value','')}"
                        for c in convs if c.get("value")
                    )
                    if len(text) > 100:
                        samples.append(text)
            except Exception:
                continue

        console.log(f"[green]✓ Loaded {len(samples)} MentalChat16K dialogues[/green]")
        return samples
    except Exception as e:
        console.log(f"[yellow]⚠ MentalChat16K not available ({type(e).__name__}). Using curated corpus only.[/yellow]")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Build ChromaDB Vector Store
# ══════════════════════════════════════════════════════════════════════════════

def build_vectorstore(documents: List[str]):
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings

    console.log(f"[cyan]Loading embedding model: {EMBEDDING_MODEL}…[/cyan]")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    console.log("[cyan]Initialising ChromaDB…[/cyan]")
    client = chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Delete existing collection to rebuild clean
    try:
        client.delete_collection("productivity_coach")
    except Exception:
        pass

    collection = client.create_collection(
        name="productivity_coach",
        metadata={"hnsw:space": "cosine"},
    )

    # Chunk documents
    chunks = []
    for doc in documents:
        if len(doc) <= CHUNK_SIZE:
            chunks.append(doc)
        else:
            start = 0
            while start < len(doc):
                end = min(start + CHUNK_SIZE, len(doc))
                chunks.append(doc[start:end])
                start += CHUNK_SIZE - CHUNK_OVERLAP

    console.log(f"Embedding {len(chunks)} chunks…")
    embeddings = embedder.encode(chunks, show_progress_bar=True, batch_size=64)

    ids = [f"doc_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids,
    )

    console.log(f"[green]✓ ChromaDB collection 'productivity_coach' built with {len(chunks)} chunks[/green]")
    console.log(f"[green]  Saved to: {VECTORSTORE_DIR}[/green]")
    return collection


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    console.print(Panel.fit("🧠 Step 4 — Building RAG Vector Store", style="bold magenta"))

    all_docs = list(COACHING_CORPUS)
    console.log(f"Curated corpus: {len(all_docs)} documents")

    # Try to add real MentalChat16K
    mc_docs = try_load_mentalchat()
    all_docs.extend(mc_docs)
    console.log(f"Total documents to embed: {len(all_docs)}")

    build_vectorstore(all_docs)
    console.print("\n[bold green]✅ Vector store ready → vectorstore/[/bold green]")


if __name__ == "__main__":
    main()

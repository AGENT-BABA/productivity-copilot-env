"""
inference.py
────────────
Drop-in inference helper for your backend.

Import this module from your FastAPI routers.
All models are loaded once at startup (singleton pattern).

Usage example (FastAPI):
    from data_pipeline.inference import copilot

    @router.post("/predict/failure")
    def predict_failure(data: TaskInput):
        result = copilot.predict_failure(data.dict())
        return result

    @router.post("/predict/work-style")
    def predict_style(data: UserBehavior):
        return copilot.predict_work_style(data.dict())

    @router.post("/chat/persuade")
    async def persuade(data: ChatInput):
        return await copilot.persuade(data.user_message, data.task_context)
"""

from __future__ import annotations
import os, json, asyncio
from pathlib import Path
from typing import Any

import numpy as np
import joblib

# ── Resolve model artifact paths ──────────────────────────────────────────────
_HERE        = Path(__file__).parent
_ROOT        = _HERE.parent
ARTIFACTS    = _ROOT / "model_artifacts"
VECTORSTORE  = _ROOT / "vectorstore"
FEATURE_JSON = ARTIFACTS / "feature_columns.json"


# ══════════════════════════════════════════════════════════════════════════════
# Model Singleton
# ══════════════════════════════════════════════════════════════════════════════

class CopilotModels:
    _instance: "CopilotModels | None" = None

    def __new__(cls):
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._loaded = False
            cls._instance = instance
        return cls._instance

    def load(self):
        if self._loaded:
            return
        print("Loading Copilot models…")

        self.failure_model  = joblib.load(ARTIFACTS / "failure_predictor.pkl")
        self.scaler         = joblib.load(ARTIFACTS / "feature_scaler.pkl")
        self.style_model    = joblib.load(ARTIFACTS / "work_style_classifier.pkl")
        self.style_encoder  = joblib.load(ARTIFACTS / "work_style_label_encoder.pkl")
        self.distraction_model = joblib.load(ARTIFACTS / "distraction_scorer.pkl")

        with open(FEATURE_JSON) as f:
            self._features = json.load(f)

        self._load_rag()
        self._loaded = True
        print("✓ All models loaded.")

    def _load_rag(self):
        try:
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            client = chromadb.PersistentClient(
                path=str(VECTORSTORE),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = client.get_collection("productivity_coach")
            print(f"✓ RAG vector store loaded ({self._collection.count()} chunks)")
        except Exception as e:
            print(f"⚠ RAG not available: {e}. Persuasion will work without retrieval context.")
            self._collection = None
            self._embedder = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def failure_features(self) -> list[str]:
        return self._features["failure_predictor"]["features"]

    @property
    def style_features(self) -> list[str]:
        return self._features["work_style_classifier"]["features"]

    @property
    def distraction_features(self) -> list[str]:
        return self._features["distraction_scorer"]["features"]

    # ── Inference methods ─────────────────────────────────────────────────────

    def predict_failure(self, user_data: dict[str, Any]) -> dict:
        """
        Returns task failure risk for the given behavioral data.

        user_data keys (all optional — missing keys are filled with defaults):
            session_duration_minutes, break_count, social_media_minutes_before,
            task_complexity, work_style_score, time_of_day_hour, day_of_week,
            stress_level, sleep_hours, distraction_events, deadline_days_remaining,
            previous_completion_rate, focus_score, motivation_level,
            study_hours_weekly
        """
        DEFAULTS = {
            "session_duration_minutes":    120,
            "break_count":                 2,
            "social_media_minutes_before": 15,
            "task_complexity":             3,
            "work_style_score":            0.5,
            "time_of_day_hour":            10,
            "day_of_week":                 1,
            "stress_level":                5,
            "sleep_hours":                 7,
            "distraction_events":          5,
            "deadline_days_remaining":     3,
            "previous_completion_rate":    0.7,
            "focus_score":                 0.6,
            "motivation_level":            6,
            "study_hours_weekly":          20,
        }
        row = {**DEFAULTS, **user_data}
        X = np.array([[row.get(f, DEFAULTS.get(f, 0)) for f in self.failure_features]])
        X_scaled = self.scaler.transform(X)

        failure_proba = float(self.failure_model.predict_proba(X_scaled)[0][0])
        # Note: class 0 = failed → higher proba = higher failure risk
        risk_score = round(failure_proba, 4)

        return {
            "failure_probability": risk_score,
            "risk_level": (
                "high"   if risk_score >= 0.65 else
                "medium" if risk_score >= 0.40 else
                "low"
            ),
            "should_intervene": risk_score >= 0.65,
        }

    def predict_work_style(self, user_data: dict[str, Any]) -> dict:
        """Returns predicted work style: turtle | hare | hybrid."""
        DEFAULTS = {
            "session_duration_minutes": 240,
            "break_count": 3,
            "distraction_events": 8,
            "stress_level": 5,
            "motivation_level": 6,
            "previous_completion_rate": 0.7,
            "deadline_days_remaining": 3,
        }
        row = {**DEFAULTS, **user_data}
        X = np.array([[row.get(f, DEFAULTS.get(f, 0)) for f in self.style_features]])
        pred = self.style_model.predict(X)[0]
        proba = self.style_model.predict_proba(X)[0]
        label = self.style_encoder.inverse_transform([pred])[0]

        return {
            "work_style": label,
            "confidence": round(float(proba.max()), 3),
            "scores": {
                cls: round(float(p), 3)
                for cls, p in zip(self.style_encoder.classes_, proba)
            },
        }

    def score_distraction(self, user_data: dict[str, Any]) -> dict:
        """Returns a 0–1 distraction score for the current session."""
        DEFAULTS = {
            "distraction_events": 5,
            "social_media_minutes_before": 10,
            "break_count": 2,
            "session_duration_minutes": 120,
            "focus_score": 0.65,
        }
        row = {**DEFAULTS, **user_data}
        X = np.array([[row.get(f, DEFAULTS.get(f, 0)) for f in self.distraction_features]])
        score = float(np.clip(self.distraction_model.predict(X)[0], 0, 1))

        return {
            "distraction_score": round(score, 4),
            "level": (
                "high"   if score >= 0.65 else
                "medium" if score >= 0.35 else
                "low"
            ),
        }

    def retrieve_context(self, query: str, k: int = 5) -> list[str]:
        """Retrieve top-k relevant coaching snippets from ChromaDB."""
        if self._collection is None or self._embedder is None:
            return []
        embedding = self._embedder.encode([query])[0].tolist()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
        )
        return results["documents"][0] if results["documents"] else []

    async def persuade(
        self,
        user_message: str,
        task_context: dict | None = None,
        chat_history: list[dict] | None = None,
    ) -> dict:
        """
        Calls Groq (Llama 3.3-70B) with RAG context to generate a coaching response.
        Requires GROQ_API_KEY environment variable.

        Returns:
            {"response": str, "sources_used": int}
        """
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            return {
                "response": (
                    "I'm here to help you stay on track! It seems my coaching engine "
                    "isn't configured yet (missing GROQ_API_KEY). Once set up, I can "
                    "provide personalized motivation, CBT-based reframes, and help you "
                    "push through resistance. You've got this!"
                ),
                "sources_used": 0,
            }

        # Retrieve coaching context
        context_docs = self.retrieve_context(user_message)
        context_str  = "\n\n---\n\n".join(context_docs) if context_docs else ""

        # Build system prompt
        task_info = ""
        if task_context:
            task_info = (
                f"\nCurrent task: {task_context.get('title', 'Unknown')}"
                f"\nRisk level: {task_context.get('risk_level', 'unknown')}"
                f"\nDeadline: {task_context.get('deadline', 'unset')}"
                f"\nWork style: {task_context.get('work_style', 'unknown')}"
            )

        system_prompt = f"""You are an expert productivity coach and behavioral psychologist.
Your role is to help users overcome procrastination, stay focused, and complete their tasks.
You use evidence-based techniques: Motivational Interviewing, CBT reframes, implementation intentions,
and behavioral science. Be direct, empathetic, and genuinely persuasive — not preachy or robotic.
Keep responses under 150 words unless the user needs a deeper conversation.
{task_info}

Relevant coaching knowledge:
{context_str}"""

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history[-6:])   # last 3 turns
        messages.append({"role": "user", "content": user_message})

        try:
            from groq import AsyncGroq
            client = AsyncGroq(api_key=groq_key)
            response = await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=300,
                temperature=0.7,
            )
            return {
                "response": response.choices[0].message.content,
                "sources_used": len(context_docs),
            }
        except Exception as e:
            return {
                "response": f"Coaching engine error: {e}",
                "sources_used": 0,
            }


# ── Singleton instance — import directly in your backend ─────────────────────
copilot = CopilotModels()


# ══════════════════════════════════════════════════════════════════════════════
# Quick smoke-test
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    copilot.load()

    test_user = {
        "session_duration_minutes": 90,
        "break_count": 5,
        "social_media_minutes_before": 40,
        "task_complexity": 4,
        "work_style_score": 0.85,
        "time_of_day_hour": 22,
        "day_of_week": 6,
        "stress_level": 8,
        "sleep_hours": 5,
        "distraction_events": 12,
        "deadline_days_remaining": 0.5,
        "previous_completion_rate": 0.45,
        "focus_score": 0.3,
        "motivation_level": 3,
        "study_hours_weekly": 10,
    }

    print("\n── Failure Prediction ──")
    print(copilot.predict_failure(test_user))

    print("\n── Work Style ──")
    print(copilot.predict_work_style(test_user))

    print("\n── Distraction Score ──")
    print(copilot.score_distraction(test_user))

    print("\n── RAG Context Retrieved ──")
    docs = copilot.retrieve_context("I can't focus and keep procrastinating")
    print(f"Retrieved {len(docs)} coaching snippets")
    if docs:
        print("First snippet preview:", docs[0][:150], "…")

    print("\n── Persuasion (async) ──")
    async def test_persuade():
        r = await copilot.persuade(
            "I keep delaying my project. Convince me to start right now.",
            task_context={"title": "Machine Learning Project", "risk_level": "high",
                         "deadline": "tomorrow", "work_style": "hare"},
        )
        print(r["response"])
    asyncio.run(test_persuade())

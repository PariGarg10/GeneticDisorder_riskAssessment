import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

def _local_precautions(level, top_features, genetic_flag):
    """Fallback guidance when remote AI is unavailable."""
    base = f"""
### Your Risk Summary
Your current profile is classified as **{level}**.
Key contributing factors detected: **{top_features}**.
Family history present: **{genetic_flag}**.

### Precautionary Measures
1. **Monitor core vitals regularly**
   - Track blood pressure, fasting glucose, and cholesterol at routine intervals.
2. **Heart-protective lifestyle**
   - Follow a high-fiber, low-sodium diet and reduce saturated/trans fats.
   - Aim for at least 150 minutes/week of moderate physical activity.
3. **Clinical follow-up**
   - Discuss this risk profile with a clinician to decide on preventive labs or screening.
4. **Avoid high-risk habits**
   - Avoid smoking, limit alcohol, improve sleep quality, and manage stress.
5. **Family-history awareness**
   - If family history is positive, start preventive checkups earlier and maintain annual reviews.

### Important Note
This output is supportive and educational, not a clinical diagnosis.
"""
    return base.strip()


def generate_precautions(level, top_features, genetic_flag):
    try:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return _local_precautions(level, top_features, genetic_flag)

        client = genai.Client(api_key=api_key)

        prompt = f"""
        A patient has {level}.
        Top contributing risk factors: {top_features}.
        Family history present: {genetic_flag}.

        Provide:
        - Simple explanation of their cardiovascular risk
        - Preventive lifestyle recommendations
        - Medical precautions
        - Keep tone professional and medically responsible.
        """

        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )

        if hasattr(response, "text") and response.text:
            return response.text
        return _local_precautions(level, top_features, genetic_flag)

    except Exception as e:
        print("Gemini API Error:", e)
        return _local_precautions(level, top_features, genetic_flag)

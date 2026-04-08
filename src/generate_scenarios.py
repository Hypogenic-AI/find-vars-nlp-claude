"""
Generate message-writing scenarios and axis-varied response pairs using GPT-4.1.

For each scenario, we generate response pairs that differ along specific pragmatic axes:
- Formality (casual ↔ formal)
- Directness (hedged ↔ direct)
- Politeness (blunt ↔ polite)
- Emotional tone (neutral ↔ warm/empathetic)
- Specificity (vague ↔ specific)
- Control: synonym substitution (surface change only)
"""

import json
import os
import time
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PRAGMATIC_AXES = [
    "formality",
    "directness",
    "politeness",
    "emotional_tone",
    "specificity",
]

CONTROL_AXES = [
    "synonym",  # surface-level word swaps, no pragmatic change
]

ALL_AXES = PRAGMATIC_AXES + CONTROL_AXES

SCENARIO_GENERATION_PROMPT = """Generate 30 diverse scenarios where someone asks an LLM to help write a message.
Each scenario should specify:
- who the sender is
- who the recipient is
- what the message is about
- the relationship/context

Cover these domains evenly (10 each):
1. Workplace (emails to boss, colleagues, clients, HR)
2. Personal (messages to friends, family, acquaintances, neighbors)
3. Service (complaints, requests, feedback, applications)

Return a JSON array of objects with keys:
- "id": integer 1-30
- "domain": "workplace" | "personal" | "service"
- "prompt": the actual user prompt asking for help writing the message (e.g., "Help me write an email to my boss requesting a day off next Friday.")
- "sender": who is sending
- "recipient": who receives
- "context": brief context description

Make the scenarios realistic and diverse. Vary the stakes, topics, and relationships.
Return ONLY the JSON array, no other text."""


VARIANT_GENERATION_PROMPT = """Given this message-writing scenario:
{scenario_prompt}

Write TWO response messages that the scenario asks for. Both should be plausible, helpful messages that address the request.

The two messages should differ PRIMARILY along the axis of **{axis_name}**:
{axis_description}

CRITICAL CONSTRAINTS:
- Both messages should be roughly the same length (within 20% word count)
- Both should address the same core content/points
- The ONLY major difference should be the target axis
- Each message should be 50-150 words
- Write the actual message content only (no "Subject:" lines, no meta-commentary)

Return a JSON object with keys:
- "variant_a": the first variant (the "{pole_a}" end of the axis)
- "variant_b": the second variant (the "{pole_b}" end of the axis)
- "axis": "{axis_name}"

Return ONLY the JSON object."""

AXIS_DESCRIPTIONS = {
    "formality": {
        "description": "FORMALITY — how formal vs. casual the language is. This includes vocabulary choice (colloquial vs. professional), sentence structure (fragments vs. complete sentences), greetings/closings, contractions, etc.",
        "pole_a": "casual/informal",
        "pole_b": "formal/professional",
    },
    "directness": {
        "description": "DIRECTNESS — how direct vs. hedged the request/statement is. Direct = states things plainly, gets to the point. Indirect = uses hedging language, softeners, roundabout phrasing.",
        "pole_a": "indirect/hedged",
        "pole_b": "direct/assertive",
    },
    "politeness": {
        "description": "POLITENESS — how polite vs. blunt the message is. Polite = uses please/thank you, acknowledges the other person, shows deference. Blunt = states things without social niceties.",
        "pole_a": "blunt/terse",
        "pole_b": "polite/courteous",
    },
    "emotional_tone": {
        "description": "EMOTIONAL TONE — how emotionally warm vs. neutral the message is. Warm = expresses empathy, enthusiasm, personal connection. Neutral = sticks to facts, impersonal, businesslike.",
        "pole_a": "neutral/factual",
        "pole_b": "warm/empathetic",
    },
    "specificity": {
        "description": "SPECIFICITY — how specific vs. vague the message is. Specific = includes concrete details, examples, numbers, dates. Vague = uses general language, abstractions, approximations.",
        "pole_a": "vague/general",
        "pole_b": "detailed/specific",
    },
    "synonym": {
        "description": "SYNONYM SUBSTITUTION (CONTROL) — the two messages should convey the EXACT same pragmatic meaning with the same tone, formality, and directness. The only difference should be word-level synonym swaps (e.g., 'happy'→'glad', 'help'→'assist', 'big'→'large'). The pragmatic feel should be identical.",
        "pole_a": "version_a",
        "pole_b": "version_b",
    },
}


def generate_scenarios():
    """Generate 30 diverse message-writing scenarios."""
    print("Generating scenarios...")
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": SCENARIO_GENERATION_PROMPT}],
        temperature=0.8,
        max_tokens=4000,
    )
    content = response.choices[0].message.content.strip()
    # Parse JSON, handling possible markdown code fences
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    scenarios = json.loads(content)
    print(f"Generated {len(scenarios)} scenarios")
    return scenarios


def generate_variants(scenario, axis_name):
    """Generate a pair of response variants along a specific axis for a scenario."""
    axis_info = AXIS_DESCRIPTIONS[axis_name]
    prompt = VARIANT_GENERATION_PROMPT.format(
        scenario_prompt=scenario["prompt"],
        axis_name=axis_name,
        axis_description=axis_info["description"],
        pole_a=axis_info["pole_a"],
        pole_b=axis_info["pole_b"],
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)
            result["scenario_id"] = scenario["id"]
            result["axis"] = axis_name
            result["pole_a_label"] = axis_info["pole_a"]
            result["pole_b_label"] = axis_info["pole_b"]
            return result
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Retry {attempt+1} for scenario {scenario['id']}, axis {axis_name}: {e}")
            time.sleep(1)
    return None


def generate_introspection(scenario):
    """Ask GPT-4.1 what variables it considered when writing a response."""
    prompt = f"""You are helping someone write a message. Here is their request:

{scenario["prompt"]}

First, write a short response message (50-100 words).

Then, list ALL the pragmatic/communicative variables you considered when crafting this message. For each variable, explain briefly why it mattered for this scenario.

Format your answer as JSON:
{{
  "message": "your response message here",
  "variables_considered": [
    {{"variable": "variable name", "reason": "why it mattered"}},
    ...
  ]
}}

Return ONLY the JSON."""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)
            result["scenario_id"] = scenario["id"]
            return result
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Retry introspection {attempt+1} for scenario {scenario['id']}: {e}")
            time.sleep(1)
    return None


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/model_outputs", exist_ok=True)

    # Step 1: Generate scenarios
    scenarios = generate_scenarios()
    with open("results/scenarios.json", "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"Saved {len(scenarios)} scenarios to results/scenarios.json")

    # Step 2: Generate axis variants for each scenario
    all_variants = []
    for scenario in scenarios:
        print(f"Generating variants for scenario {scenario['id']}: {scenario['prompt'][:60]}...")
        for axis in ALL_AXES:
            variant = generate_variants(scenario, axis)
            if variant:
                all_variants.append(variant)
                print(f"  ✓ {axis}")
            else:
                print(f"  ✗ {axis} (failed)")
            time.sleep(0.3)  # rate limiting

    with open("results/variants.json", "w") as f:
        json.dump(all_variants, f, indent=2)
    print(f"\nSaved {len(all_variants)} variants to results/variants.json")

    # Step 3: Generate introspection data
    introspections = []
    print("\nGenerating introspection data...")
    for scenario in scenarios:
        intro = generate_introspection(scenario)
        if intro:
            introspections.append(intro)
            print(f"  ✓ Scenario {scenario['id']}")
        else:
            print(f"  ✗ Scenario {scenario['id']} (failed)")
        time.sleep(0.3)

    with open("results/introspections.json", "w") as f:
        json.dump(introspections, f, indent=2)
    print(f"Saved {len(introspections)} introspections to results/introspections.json")


if __name__ == "__main__":
    main()

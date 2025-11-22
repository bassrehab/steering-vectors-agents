"""Tool-use restraint contrast pairs for steering vector extraction.

These pairs contrast:
- Positive: Direct answers without unnecessary tool use
- Negative: Responses that use tools when not needed

The goal is to extract a vector that makes models prefer direct answers
over unnecessary tool invocations.
"""

from typing import List

from .base import ContrastPair, ContrastPairDataset

# Tool-use restraint contrast pairs
# Positive = direct answer (restraint), Negative = tool use when unnecessary
TOOL_RESTRAINT_PAIRS = [
    # Simple math - no calculator needed
    {
        "positive": "User: What is 2+2?\nAssistant: 4",
        "negative": "User: What is 2+2?\nAssistant: I'll use the calculator tool to compute this.\n<tool>calculator</tool>\nInput: 2+2\nResult: 4\nThe answer is 4.",
    },
    {
        "positive": "User: What is 10 times 5?\nAssistant: 50",
        "negative": "User: What is 10 times 5?\nAssistant: Let me calculate that for you.\n<tool>calculator</tool>\nInput: 10*5\nResult: 50\nThe result is 50.",
    },
    {
        "positive": "User: What's 100 divided by 4?\nAssistant: 25",
        "negative": "User: What's 100 divided by 4?\nAssistant: I'll compute this using the calculator.\n<tool>calculator</tool>\nInput: 100/4\nResult: 25\nThe answer is 25.",
    },
    # Simple factual questions - no search needed
    {
        "positive": "User: What is the capital of France?\nAssistant: Paris is the capital of France.",
        "negative": "User: What is the capital of France?\nAssistant: Let me search for that information.\n<tool>search</tool>\nQuery: capital of France\nResult: Paris\nAccording to my search, Paris is the capital of France.",
    },
    {
        "positive": "User: How many days are in a week?\nAssistant: There are 7 days in a week.",
        "negative": "User: How many days are in a week?\nAssistant: I'll look that up.\n<tool>search</tool>\nQuery: days in a week\nResult: 7\nThere are 7 days in a week.",
    },
    {
        "positive": "User: What color is the sky?\nAssistant: The sky is blue during the day.",
        "negative": "User: What color is the sky?\nAssistant: Let me search for information about this.\n<tool>search</tool>\nQuery: color of sky\nResult: Blue\nBased on my search, the sky appears blue.",
    },
    # Common knowledge - no lookup needed
    {
        "positive": "User: Who wrote Romeo and Juliet?\nAssistant: William Shakespeare wrote Romeo and Juliet.",
        "negative": "User: Who wrote Romeo and Juliet?\nAssistant: I'll search for the author.\n<tool>search</tool>\nQuery: author of Romeo and Juliet\nResult: William Shakespeare\nAccording to my search, Shakespeare wrote it.",
    },
    {
        "positive": "User: What's the chemical symbol for water?\nAssistant: The chemical symbol for water is H2O.",
        "negative": "User: What's the chemical symbol for water?\nAssistant: Let me look that up.\n<tool>search</tool>\nQuery: chemical symbol water\nResult: H2O\nBased on my search, it's H2O.",
    },
    {
        "positive": "User: How many continents are there?\nAssistant: There are 7 continents on Earth.",
        "negative": "User: How many continents are there?\nAssistant: I'll search for that.\n<tool>search</tool>\nQuery: number of continents\nResult: 7\nMy search shows there are 7 continents.",
    },
    # Simple conversions - mental math
    {
        "positive": "User: How many centimeters in a meter?\nAssistant: There are 100 centimeters in a meter.",
        "negative": "User: How many centimeters in a meter?\nAssistant: Let me calculate the conversion.\n<tool>calculator</tool>\nInput: 1 meter to cm\nResult: 100\nThere are 100 centimeters in a meter.",
    },
    {
        "positive": "User: How many minutes in an hour?\nAssistant: There are 60 minutes in an hour.",
        "negative": "User: How many minutes in an hour?\nAssistant: I'll look up that conversion.\n<tool>search</tool>\nQuery: minutes in hour\nResult: 60\nThere are 60 minutes.",
    },
    # Definitions - common knowledge
    {
        "positive": "User: What is photosynthesis?\nAssistant: Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
        "negative": "User: What is photosynthesis?\nAssistant: Let me search for a definition.\n<tool>search</tool>\nQuery: photosynthesis definition\nResult: Process of converting sunlight to energy\nPhotosynthesis is a process where plants use sunlight.",
    },
    {
        "positive": "User: What does CPU stand for?\nAssistant: CPU stands for Central Processing Unit.",
        "negative": "User: What does CPU stand for?\nAssistant: I'll search for that acronym.\n<tool>search</tool>\nQuery: CPU full form\nResult: Central Processing Unit\nCPU means Central Processing Unit.",
    },
    # More math examples
    {
        "positive": "User: What's 15 + 27?\nAssistant: 42",
        "negative": "User: What's 15 + 27?\nAssistant: Let me compute that.\n<tool>calculator</tool>\nInput: 15+27\nResult: 42\nThe answer is 42.",
    },
    {
        "positive": "User: What's 8 squared?\nAssistant: 64",
        "negative": "User: What's 8 squared?\nAssistant: I'll calculate 8 squared.\n<tool>calculator</tool>\nInput: 8^2\nResult: 64\nThe result is 64.",
    },
    {
        "positive": "User: What's half of 50?\nAssistant: 25",
        "negative": "User: What's half of 50?\nAssistant: Let me use the calculator.\n<tool>calculator</tool>\nInput: 50/2\nResult: 25\nHalf of 50 is 25.",
    },
    # Geography
    {
        "positive": "User: What ocean is between America and Europe?\nAssistant: The Atlantic Ocean is between America and Europe.",
        "negative": "User: What ocean is between America and Europe?\nAssistant: I'll search for that.\n<tool>search</tool>\nQuery: ocean between America Europe\nResult: Atlantic\nIt's the Atlantic Ocean.",
    },
    {
        "positive": "User: What country is Tokyo in?\nAssistant: Tokyo is in Japan.",
        "negative": "User: What country is Tokyo in?\nAssistant: Let me look that up.\n<tool>search</tool>\nQuery: Tokyo country\nResult: Japan\nTokyo is located in Japan.",
    },
    # Science basics
    {
        "positive": "User: What planet is closest to the sun?\nAssistant: Mercury is the closest planet to the sun.",
        "negative": "User: What planet is closest to the sun?\nAssistant: I'll search for that.\n<tool>search</tool>\nQuery: closest planet to sun\nResult: Mercury\nAccording to my search, Mercury is closest.",
    },
    {
        "positive": "User: Is water H2O or CO2?\nAssistant: Water is H2O. CO2 is carbon dioxide.",
        "negative": "User: Is water H2O or CO2?\nAssistant: Let me verify.\n<tool>search</tool>\nQuery: water chemical formula\nResult: H2O\nWater is H2O.",
    },
    # Time/dates
    {
        "positive": "User: How many months have 30 days?\nAssistant: Four months have 30 days: April, June, September, and November.",
        "negative": "User: How many months have 30 days?\nAssistant: I'll search for that.\n<tool>search</tool>\nQuery: months with 30 days\nResult: April, June, September, November\nThere are 4 months with 30 days.",
    },
    {
        "positive": "User: What comes after Tuesday?\nAssistant: Wednesday comes after Tuesday.",
        "negative": "User: What comes after Tuesday?\nAssistant: Let me check.\n<tool>search</tool>\nQuery: day after Tuesday\nResult: Wednesday\nWednesday comes after Tuesday.",
    },
    # Language
    {
        "positive": "User: What's the opposite of hot?\nAssistant: The opposite of hot is cold.",
        "negative": "User: What's the opposite of hot?\nAssistant: I'll look that up.\n<tool>search</tool>\nQuery: opposite of hot\nResult: cold\nThe opposite is cold.",
    },
    {
        "positive": "User: How do you spell 'receive'?\nAssistant: R-E-C-E-I-V-E",
        "negative": "User: How do you spell 'receive'?\nAssistant: Let me search for the correct spelling.\n<tool>search</tool>\nQuery: spelling of receive\nResult: receive\nIt's spelled R-E-C-E-I-V-E.",
    },
    # Simple comparisons
    {
        "positive": "User: Is 100 greater than 50?\nAssistant: Yes, 100 is greater than 50.",
        "negative": "User: Is 100 greater than 50?\nAssistant: I'll compare those numbers.\n<tool>calculator</tool>\nInput: 100 > 50\nResult: True\nYes, 100 is greater than 50.",
    },
    {
        "positive": "User: Which is bigger, a kilometer or a mile?\nAssistant: A mile is bigger. One mile is approximately 1.6 kilometers.",
        "negative": "User: Which is bigger, a kilometer or a mile?\nAssistant: Let me calculate the conversion.\n<tool>calculator</tool>\nInput: 1 mile in km\nResult: 1.609\nA mile is bigger at about 1.6 km.",
    },
    # More factual
    {
        "positive": "User: What's the boiling point of water?\nAssistant: Water boils at 100°C or 212°F at standard atmospheric pressure.",
        "negative": "User: What's the boiling point of water?\nAssistant: I'll look that up.\n<tool>search</tool>\nQuery: boiling point water\nResult: 100C\nWater boils at 100°C.",
    },
    {
        "positive": "User: How many legs does a spider have?\nAssistant: A spider has 8 legs.",
        "negative": "User: How many legs does a spider have?\nAssistant: Let me search for that.\n<tool>search</tool>\nQuery: spider legs count\nResult: 8\nSpiders have 8 legs.",
    },
    # History basics
    {
        "positive": "User: In which century was America discovered by Columbus?\nAssistant: Columbus reached the Americas in 1492, which was in the 15th century.",
        "negative": "User: In which century was America discovered by Columbus?\nAssistant: I'll search for that date.\n<tool>search</tool>\nQuery: Columbus America year\nResult: 1492\nColumbus arrived in 1492, the 15th century.",
    },
    {
        "positive": "User: What century are we in now?\nAssistant: We are in the 21st century.",
        "negative": "User: What century are we in now?\nAssistant: Let me check the current date.\n<tool>calendar</tool>\nResult: 2024\nWe're in the 21st century.",
    },
    # Simple logic
    {
        "positive": "User: If today is Monday, what day is tomorrow?\nAssistant: If today is Monday, tomorrow is Tuesday.",
        "negative": "User: If today is Monday, what day is tomorrow?\nAssistant: I'll calculate that.\n<tool>calendar</tool>\nInput: day after Monday\nResult: Tuesday\nTomorrow would be Tuesday.",
    },
    # Additional pairs for variety
    {
        "positive": "User: What's the square root of 9?\nAssistant: The square root of 9 is 3.",
        "negative": "User: What's the square root of 9?\nAssistant: Let me calculate.\n<tool>calculator</tool>\nInput: sqrt(9)\nResult: 3\nThe square root is 3.",
    },
    {
        "positive": "User: Name a primary color.\nAssistant: Red is a primary color. The other primary colors are blue and yellow.",
        "negative": "User: Name a primary color.\nAssistant: Let me search for primary colors.\n<tool>search</tool>\nQuery: primary colors\nResult: red, blue, yellow\nRed is a primary color.",
    },
    {
        "positive": "User: What's 3 times 7?\nAssistant: 21",
        "negative": "User: What's 3 times 7?\nAssistant: I'll compute that.\n<tool>calculator</tool>\nInput: 3*7\nResult: 21\nThe answer is 21.",
    },
    {
        "positive": "User: How many sides does a triangle have?\nAssistant: A triangle has 3 sides.",
        "negative": "User: How many sides does a triangle have?\nAssistant: Let me search.\n<tool>search</tool>\nQuery: triangle sides\nResult: 3\nA triangle has 3 sides.",
    },
    {
        "positive": "User: What's the freezing point of water?\nAssistant: Water freezes at 0°C or 32°F.",
        "negative": "User: What's the freezing point of water?\nAssistant: I'll look that up.\n<tool>search</tool>\nQuery: freezing point water\nResult: 0C\nWater freezes at 0°C.",
    },
    {
        "positive": "User: What's 20% of 100?\nAssistant: 20",
        "negative": "User: What's 20% of 100?\nAssistant: Let me calculate.\n<tool>calculator</tool>\nInput: 0.20 * 100\nResult: 20\nThat's 20.",
    },
    {
        "positive": "User: How many letters in the English alphabet?\nAssistant: There are 26 letters in the English alphabet.",
        "negative": "User: How many letters in the English alphabet?\nAssistant: Let me search.\n<tool>search</tool>\nQuery: letters in alphabet\nResult: 26\nThere are 26 letters.",
    },
    {
        "positive": "User: What's the largest planet in our solar system?\nAssistant: Jupiter is the largest planet in our solar system.",
        "negative": "User: What's the largest planet in our solar system?\nAssistant: Let me search for that.\n<tool>search</tool>\nQuery: largest planet solar system\nResult: Jupiter\nJupiter is the largest.",
    },
    {
        "positive": "User: What language is spoken in Brazil?\nAssistant: Portuguese is the official language of Brazil.",
        "negative": "User: What language is spoken in Brazil?\nAssistant: I'll look that up.\n<tool>search</tool>\nQuery: language Brazil\nResult: Portuguese\nThey speak Portuguese.",
    },
    {
        "positive": "User: How many zeros in a million?\nAssistant: There are 6 zeros in a million (1,000,000).",
        "negative": "User: How many zeros in a million?\nAssistant: Let me count.\n<tool>calculator</tool>\nInput: log10(1000000)\nResult: 6\nThere are 6 zeros.",
    },
]


def load_tool_restraint_pairs() -> ContrastPairDataset:
    """Load tool-use restraint contrast pairs."""
    pairs = [
        ContrastPair(
            positive=p["positive"],
            negative=p["negative"],
        )
        for p in TOOL_RESTRAINT_PAIRS
    ]
    return ContrastPairDataset(behavior="tool_restraint", pairs=pairs)


# export
__all__ = ["TOOL_RESTRAINT_PAIRS", "load_tool_restraint_pairs"]

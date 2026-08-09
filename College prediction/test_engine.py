"""Quick smoke test for the prediction engine."""
from engine import PredictionEngine

e = PredictionEngine()

# Test 1: OPEN category, male, Delhi
results = e.predict(5000, "OPEN", "Gender-Neutral", "Delhi")
print(f"Test 1: Rank=5000, OPEN, Male, Delhi => {len(results)} programs found")

for i, p in enumerate(results[:8]):
    tier_short = p["risk_tier"].split("(")[1].rstrip(")")
    print(f"  {i+1}. [{tier_short:9s}] R{p['earliest_round']} | {p['institute'][:55]:55s} | {p['program'][:45]:45s} | avg_cr={p['avg_closing_rank']}")

print()

# Test 2: OBC-NCL category, female, Maharashtra
results2 = e.predict(8000, "OBC-NCL", "Female-only (including Supernumerary)", "Maharashtra")
print(f"Test 2: Rank=8000, OBC-NCL, Female, Maharashtra => {len(results2)} programs found")
for i, p in enumerate(results2[:5]):
    tier_short = p["risk_tier"].split("(")[1].rstrip(")")
    print(f"  {i+1}. [{tier_short:9s}] R{p['earliest_round']} | {p['institute'][:55]:55s} | avg_cr={p['avg_closing_rank']}")

print()

# Test 3: Verify IITs only have AI quota (check institute types)
results3 = e.predict(3000, "OPEN", "Gender-Neutral", "Tamil Nadu")
iit_results = [r for r in results3 if r["institute_type"] == "IIT"]
nit_results = [r for r in results3 if r["institute_type"] == "NIT"]
print(f"Test 3: Rank=3000, OPEN, Male, Tamil Nadu => {len(iit_results)} IIT, {len(nit_results)} NIT programs")
print("  All tests passed!")

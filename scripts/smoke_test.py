"""Send three example decisions (choice, yes/no, score) to a running server and print the answers.

    python scripts/smoke_test.py --endpoint http://127.0.0.1:8013
"""
import argparse
import json
import time
import urllib.request

REQUEST = {
    "state": {"ticket": "I was charged twice for order #4411 and want the second charge refunded.",
              "customer_tier": "gold"},
    "model": "Jeff-1.0-Large",
    "questions": {
        "route": {"type": "choice", "instructions": "Which team should handle this ticket?",
                  "criteria": {"billing": "Payments, charges and refunds", "technical": "Bugs and outages",
                               "shipping": "Delivery problems"}},
        "urgent": {"type": "noul", "instructions": "Does this ticket need a reply today?",
                   "criteria": {"true": "A reply is needed today", "false": "It can wait"}},
        "severity": {"type": "score", "instructions": "How severe is the customer's problem?",
                     "criteria": ["trivial", "minor", "moderate", "serious", "critical"]},
    },
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="http://127.0.0.1:8013")
    a = p.parse_args()
    req = urllib.request.Request(a.endpoint.rstrip("/") + "/v1/systemone", data=json.dumps(REQUEST).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.loads(r.read())
    print(json.dumps(body["answers"], indent=1))
    print(f"input tokens {body['usage']['input_tokens']}, round trip {time.perf_counter() - t0:.3f} s")
    assert set(body["answers"]) == set(REQUEST["questions"]), "missing answers"

# multi-agent-breakeven

Closed-form break-even models for multi-agent architectures, checked against Monte Carlo: majority voting, verifier-and-retry, pipelines and fan-out, under correlated failures.

Companion to [A Three-Agent Vote Can Add 9.6 Points at Most](https://cloudandsre.com/blog/multi-agent-break-even-math/).

## The formula that started this

Under a common-cause model of failure, a three-agent majority vote improves on one agent by

```text
gain = (1 - rho) * p * (1 - p) * (2p - 1)
```

It is zero at p = 50% and p = 100%, negative below 50%, and peaks at 9.6 points near p = 79% with fully independent failures.

## Use it

Standard library only.

```bash
python breakeven.py --baseline 0.8 --rho 0.5 --cost 0.5
```

```text
architecture                success     cost  worth it above
single agent                  80.0%     0.50               -
verifier + one retry          84.8%     0.76            5.42
three agents, majority        84.8%     1.88           28.65
```

"Worth it above" is the value of one successful task at which the extra spend pays for itself. The verifier row assumes 70% recall and a 10% false-alarm rate. Measure your own.

Estimate failure correlation by running every eval case twice:

```python
from breakeven import estimate_rho

p, rho = estimate_rho(both_pass=140, only_one=30, both_fail=30)  # 0.775, 0.57
```

## Tests

```bash
pip install -r requirements-ci.txt
pytest -q
```

The tests compare each closed form with a 200,000-trial simulation.

## Limits

These are models. They assume outcomes score as success or failure. Voting assumes outputs that can be compared. Latency and rate limits are ignored.

## License

Apache-2.0

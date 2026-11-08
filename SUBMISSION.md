# BANG / LarpGentV4 submission

This repository uses the official Heads-Up starter bundle. The tournament
entry point is the root agent.py. The strategy uses strategy.py, hand_eval.py,
preflop_data.bin and brain.json. Keep these together at the repository root.
The included brain.json contains the trained generation 9 parameters.

## Local checks

    python -m pip install -r requirements-dev.txt
    python -m pytest -q

With Docker and Bash available, run the official container check.

    bash tools/check_build.sh

## Submit

Extract this ZIP into an empty folder. The files at the top of the ZIP belong
at the root of your public GitHub repository. Commit and push them together.
Use BANG as your team name if it matches your registration.

Enter your registered team name, public GitHub repository URL, and the full
40-character commit SHA. Run git rev-parse HEAD from the repository to print it.

Submission form: https://forms.gle/Jx9RWE5VgR8dbbcj9


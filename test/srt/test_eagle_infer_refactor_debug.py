import argparse
from types import SimpleNamespace

from sglang.test.few_shot_gsm8k import run_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Eagle inference test")
    parser.add_argument("--num_questions", "-n", type=int, default=1, help="Number of questions to evaluate")
    cli_args = parser.parse_args()
    
    args = SimpleNamespace(
        num_shots=5,
        data_path=None,
        num_questions=cli_args.num_questions,
        max_new_tokens=512,
        parallel=128,
        host="http://127.0.0.1",
        port=30000,
    )
    metrics = run_eval(args)
    print(f"TestEagleLargeBS -- {metrics=}")

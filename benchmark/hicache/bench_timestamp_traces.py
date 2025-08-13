import json
import queue
import time

import requests
from bench_multiturn import (
    ReadyQueue,
    WorkloadGenerator,
    gen_payload,
    log_to_jsonl_file,
    parse_args,
)
from tqdm.asyncio import tqdm

from sglang.bench_serving import get_tokenizer


def parse_jsonl(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data


class ContextWorkloadGenerator(WorkloadGenerator):
    def __init__(self, args):
        # Construct the base URL for requests
        self.baseurl = f"http://{args.host}:{args.port}/"
        self.url = self.baseurl + "generate"

        self.tokenizer = get_tokenizer(args.model_path)
        self.distribution = args.distribution
        self.request_rate = args.request_rate
        self.start_time = None
        self.finished_time = None

        self.sent_requests = 0
        self.completed_requests = 0

        self.dataset = json.load(open(args.dataset_path))
        traces = parse_jsonl("toolagent_trace.jsonl")

        init_requests = []
        for i in range(args.num_clients):
            prompt = "".join([self.dataset[str(j)] for j in traces[i]["hash_ids"]])
            init_requests.append(
                (
                    i,
                    gen_payload(
                        prompt,
                        traces[i]["output_length"],
                    ),
                )
            )
        self.ready_queue = ReadyQueue(init_requests=init_requests, policy="fifo")

        self.response_queue = queue.Queue()
        self.pbar = tqdm(total=args.num_clients * args.num_rounds)
        self.performance_metrics = {
            "ttft": [],
            "latency": [],
            "itl": [],
            "prompt_len": [],
            "cached_tokens": [],
        }

        self.max_parallel = args.max_parallel
        self.logfile = args.log_file

    def response_handler(self):
        while True:
            try:
                _, response = self.response_queue.get(
                    timeout=10
                )  # Block until response is available
                if not response.success:
                    raise ValueError(f"Request failed with error: {response.error}")
                self.performance_metrics["ttft"].append(response.ttft)
                self.performance_metrics["itl"].extend(response.itl)
                self.performance_metrics["latency"].append(response.latency)
                self.performance_metrics["prompt_len"].append(response.prompt_len)
                self.performance_metrics["cached_tokens"].append(response.cached_tokens)
                self.completed_requests += 1

            except queue.Empty:
                if self.pbar.n == self.pbar.total:
                    break


if __name__ == "__main__":
    args = parse_args()
    args.num_rounds = 1
    args.max_parallel = 128
    flush_cache_url = f"http://{args.host}:{args.port}/flush_cache"

    for request_rate in [24, 16, 12, 8, 4, 2, 1]:
        args.request_rate = request_rate
        requests.post(flush_cache_url)
        time.sleep(1)
        performance_data = ContextWorkloadGenerator(args).run()
        log_to_jsonl_file(performance_data, args.log_file, args.tag)

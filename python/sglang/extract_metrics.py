import json
import sys

# 定义要提取的字段及其在JSON中的对应键
field_mapping = {
    "每秒请求数": "request_rate",
    "最大并发数": "max_concurrency",
    "总请求数": "completed",
    "input长度": None,
    "output长度": None,
    "总耗时(s)": "duration",
    "Request throughput (req/s)": "request_throughput",
    "Output token throughput (tok/s)": "output_throughput",
    "Input token throughput (tok/s)": "input_throughput",
    "Total token throughput (tok/s)": None,  # 需要计算
    "Mean E2E Latency (ms)": "mean_e2e_latency_ms",
    "实时并发量": "concurrency",
    "Mean TTFT (ms)": "mean_ttft_ms",
    "P90 TTFT (ms)": None,  # JSON中没有直接对应
    "P95 TTFT (ms)": None,  # JSON中没有直接对应
    "P99 TTFT (ms)": "p99_ttft_ms",
    "Mean TPOT (ms)": "mean_tpot_ms",
    "P90 TPOT (ms)": None,  # JSON中没有直接对应
    "P95 TPOT (ms)": None,  # JSON中没有直接对应
    "P99 TPOT (ms)": "p99_tpot_ms",
    "Mean ITL (ms)": "mean_itl_ms",
    "P90 ITL (ms)": None,  # JSON中没有直接对应
    "P95 ITL (ms)": "p95_itl_ms",
    "P99 ITL (ms)": "p99_itl_ms",
    "P99.8 ITL (ms)": None  # JSON中没有直接对应
}

# 表头
header = "\t".join(field_mapping.keys())
'''
{"backend": "sglang", "dataset_name": "sharegpt", "request_rate": 5.0, "max_concurrency": 256,
"sharegpt_output_len": null, "random_input_len": 1024, "random_output_len": 1024, "random_range_ratio": 0.0,
"duration": 204.53694480800186, "completed": 1000, "total_input_tokens": 302118, "total_output_tokens": 195775,
"total_output_tokens_retokenized": 195324, "request_throughput": 4.88909229058201,
"input_throughput": 1477.0827846460556, "output_throughput": 957.162043188693,
"mean_e2e_latency_ms": 2409.499004891637, "median_e2e_latency_ms": 1537.8948565048631,
"std_e2e_latency_ms": 2620.270331568627, "p99_e2e_latency_ms": 9813.89673267491, "mean_ttft_ms": 91.69221715730964,
"median_ttft_ms": 76.41966750088613, "std_ttft_ms": 65.22224954470558, "p99_ttft_ms": 327.9139723521075,
"mean_tpot_ms": 11.808901276564441, "median_tpot_ms": 11.829237894316, "std_tpot_ms": 1.9529553892334461,
"p99_tpot_ms": 19.04175553412642, "mean_itl_ms": 11.917698144217361, "median_itl_ms": 10.71716399746947,
"std_itl_ms": 2.8155413018211033, "p95_itl_ms": 18.839622958330438, "p99_itl_ms": 21.477889133384465,
"concurrency": 11.780263008980727, "accept_length": null}
'''

def extract_data(line):
    """从一行JSON数据中提取所需字段"""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    result = []

    for field, key in field_mapping.items():
        if key is None:
            # 需要计算的字段
            if field == "Total token throughput (tok/s)":
                # 计算总token吞吐量 = 输入吞吐量 + 输出吞吐量
                input_tp = data.get("input_throughput", 0)
                output_tp = data.get("output_throughput", 0)
                value = input_tp + output_tp
            elif field == "input长度":
                # 计算平均输入长度
                total_input = data.get("total_input_tokens", 0)
                completed = data.get("completed", 1)
                value = total_input / completed if completed > 0 else 0
            elif field == "output长度":
                # 计算平均输出长度
                total_output = data.get("total_output_tokens", 0)
                completed = data.get("completed", 1)
                value = total_output / completed if completed > 0 else 0
            else:
                value = "N/A"  # 其他未实现的字段
        else:
            # 直接获取的字段
            value = data.get(key, "N/A")

        # 处理可能的None值
        if value is None:
            value = "N/A"

        if isinstance(value, (int, float)):
            # 使用格式化字符串将其转换为保留两位小数的字符串
            value = f"{value:.2f}"
        else:
            # 如果不是数字，确保它是字符串
            value = str(value)

        result.append(str(value))

    return " ".join(result)


def process_file(input_file):
    """处理输入文件"""
    print(header)  # 打印表头

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            extracted = extract_data(line)
            if extracted:
                print(extracted)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_metrics.py <input_json_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    process_file(input_file)

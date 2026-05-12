import requests
import json
import time
from datetime import datetime


def test_api_connectivity():
    """
    测试API服务器的连通性
    """
    base_url = "http://localhost:8000"

    print("API连通性测试")
    print("=" * 50)

    # 1. 测试健康检查端点
    print("1. 测试健康检查端点...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_result = response.json()
            print(f"   ✓ 健康检查: {health_result}")
        else:
            print(f"   ✗ 健康检查失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"   ✗ 健康检查异常: {str(e)}")

    # 2. 测试聊天端点
    print("\n2. 测试聊天端点...")
    try:
        chat_payload = {
            "user_input": "测试消息",
            "thread_id": f"test_{int(time.time())}",
        }
        response = requests.post(
            f"{base_url}/api/v1/chat", json=chat_payload, timeout=30
        )
        if response.status_code == 200:
            chat_result = response.json()
            print(
                f"   ✓ 聊天端点: 响应正常，thread_id={chat_result.get('thread_id', 'N/A')}"
            )
        else:
            print(f"   ✗ 聊天端点失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"   ✗ 聊天端点异常: {str(e)}")

    # 3. 测试回测端点（快速测试，不等待完成）
    print("\n3. 测试回测端点连接...")
    try:
        backtest_payload = {
            "symbol": "AAPL",
            "start": "2024-01-01",
            "end": "2024-01-05",  # 很短的时间段，快速测试
            "initial_cash": 10000.0,
            "commission": 0.001,
            "slippage": 0.0005,
            "min_confidence": 0.0,
            "rebalance_threshold": 0.02,
            "quiet": True,  # 静默模式
            "temperature": 0.0,
            "audit_path": "Trade/backtest_audit.jsonl",
        }

        print("   正在启动回测任务...")
        response = requests.post(
            f"{base_url}/api/v1/backtest",
            json=backtest_payload,
            timeout=10,  # 较短超时时间，只测试连接
        )

        if response.status_code == 200:
            task_result = response.json()
            print(
                f"   ✓ 回测端点: 任务启动成功，task_id={task_result.get('task_id', 'N/A')}"
            )

            # 如果获得了任务ID，尝试查询状态
            if "task_id" in task_result:
                task_id = task_result["task_id"]
                print(f"   查询任务状态...")
                status_response = requests.get(
                    f"{base_url}/api/v1/backtest/status/{task_id}"
                )
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(
                        f"   ✓ 任务状态: {status_data.get('status', 'N/A')}, 进度: {status_data.get('progress', 0)}%"
                    )
                else:
                    print(
                        f"   ? 任务状态查询失败，状态码: {status_response.status_code}"
                    )
        else:
            print(f"   ✗ 回测端点失败，状态码: {response.status_code}")
            print(f"   响应内容: {response.text}")
    except Exception as e:
        print(f"   ✗ 回测端点异常: {str(e)}")

    print("\n" + "=" * 50)
    print("API连通性测试完成")


def test_sse_connection():
    """
    测试SSE连接
    """
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    print("\nSSE连接测试")
    print("=" * 50)

    try:
        import requests
        from threading import Thread
        import time

        # 启动一个线程来处理SSE流
        received_events = []
        stream_finished = False

        def handle_stream():
            nonlocal stream_finished
            url = "http://localhost:8000/api/v1/backtest-stream"

            payload = {
                "symbol": "AAPL",
                "start": "2024-01-01",
                "end": "2024-01-03",  # 只测试2天，快速验证
                "initial_cash": 10000.0,
                "commission": 0.001,
                "slippage": 0.0005,
                "min_confidence": 0.0,
                "rebalance_threshold": 0.02,
                "quiet": True,
                "temperature": 0.0,
                "audit_path": "Trade/backtest_audit.jsonl",
            }

            try:
                response = requests.post(url, json=payload, stream=True)
                print(f"SSE响应状态: {response.status_code}")

                buffer = ""
                for chunk in response.iter_content(
                    chunk_size=None, decode_unicode=True
                ):
                    if chunk:
                        buffer += chunk

                        # 按行分割
                        lines = buffer.split("\n")
                        buffer = lines[-1]  # 保留最后一行

                        for line in lines[:-1]:
                            line = line.strip()
                            if line.startswith("data: "):
                                try:
                                    data_str = line[6:]  # 去掉 "data: "
                                    data = json.loads(data_str)

                                    if (
                                        isinstance(data, dict)
                                        and "event" in data
                                        and "data" in data
                                    ):
                                        event_type = data["event"]
                                        event_data = json.loads(data["data"])
                                        received_events.append((event_type, event_data))

                                        print(
                                            f"收到事件: {event_type}, 数据片段: {str(event_data)[:100]}..."
                                        )

                                        # 如果收到最终结果或错误，结束
                                        if event_type in ["final_result", "error"]:
                                            stream_finished = True
                                            return

                                except json.JSONDecodeError:
                                    print(f"JSON解析错误: {line}")

                stream_finished = True
            except Exception as e:
                print(f"SSE连接异常: {str(e)}")
                stream_finished = True

        # 启动SSE处理线程
        stream_thread = Thread(target=handle_stream)
        stream_thread.daemon = True
        stream_thread.start()

        # 等待最多30秒
        timeout = time.time() + 30
        while not stream_finished and time.time() < timeout:
            time.sleep(1)

        if not stream_finished:
            print("   ✗ SSE连接超时")
        else:
            print(f"   ✓ SSE连接成功，收到 {len(received_events)} 个事件")
            for i, (event_type, event_data) in enumerate(received_events):
                print(f"     [{i+1}] {event_type}: {str(event_data)[:100]}...")

    except ImportError:
        print("   - 需要安装requests库: pip install requests")
    except Exception as e:
        print(f"   ✗ SSE测试异常: {str(e)}")


if __name__ == "__main__":
    test_api_connectivity()
    test_sse_connection()

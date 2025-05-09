from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
import time

# 加载模型和分词器
model_id = "modernbert-llm-router"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# 性能测试函数
def run_performance_test(classifier, num_iterations=10, batch_size=10):
    """运行性能测试，测量推理时间"""
    test_queries = [
        "我的包裹什么时候到达？",
        "如何跟踪我的物流包裹？",
        "我的订单发货了吗？",
        "你们提供国际运输服务吗？",
        "我的包裹在运输过程中损坏了，如何申请赔偿？"
    ] * 2  # 复制以达到batch_size
    
    total_times = []
    
    for i in range(num_iterations):
        print(f"Running iteration {i+1}/{num_iterations}")
        
        # 测量批量推理时间
        start_time = time.time()
        results = classifier(test_queries[:batch_size])
        end_time = time.time()
        
        inference_time = end_time - start_time
        avg_time = inference_time / batch_size
        total_times.append(avg_time)
        
        print(f"Iteration {i+1} average inference time: {avg_time:.4f} seconds")
        print("-" * 50)
    
    # 计算平均推理时间
    overall_avg = sum(total_times) / len(total_times)
    print(f"\nOverall average inference time: {overall_avg:.4f} seconds per query")
    print(f"Throughput: {1/overall_avg:.2f} queries per second")

# 示例查询函数
def run_example_queries(classifier):
    """运行一些示例查询并显示结果"""
    example_queries = [
        "我的包裹什么时候到达？",
        "如何重置我的电子邮件密码？",
        "你们的国际运输费用是多少？",
        "能推荐一家好的意大利餐厅吗？",
        "我的包裹在运输过程中损坏了，如何申请赔偿？"
    ]
    
    print("\nExample Query Results:")
    print("-" * 50)
    
    for query in example_queries:
        result = classifier(query)[0]
        print(f"Query: {query}")
        print(f"Prediction: {result['label']} (Confidence: {result['score']:.4f})")
        print("-" * 50)

# 运行性能测试
print("Running performance test...")
run_performance_test(classifier)

# 运行示例查询
print("\nRunning example queries...")
run_example_queries(classifier)
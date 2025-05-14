import argparse
import re
import os
import subprocess
import random
import heapq
import threading
import sys
import select
import time
import msvcrt
from collections import defaultdict

# ================= 核心功能函数 =================

def read_file(file_path):
    """读取文本文件内容"""
    with open(file_path, 'r') as file:
        return file.read()

def process_text(text):
    """处理文本：保留字母字符，转换为小写，分割为单词列表"""
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # 替换非字母字符为空格
    words = text.lower().split()            # 转换为小写并分割
    return [word for word in words if word]  # 过滤空字符串

def build_graph(words):
    """根据单词列表构建有向图"""
    graph = defaultdict(lambda: defaultdict(int))
    for i in range(len(words) - 1):
        source = words[i]
        target = words[i+1]
        graph[source][target] += 1
    return graph

# ================= 图展示功能函数 =================

def show_directed_graph(graph, max_width=80):
    """
    在CLI中展示有向图结构
    参数：
    - graph: 要展示的图结构
    - max_width: 控制台最大显示宽度（字符数）
    """
    print("\n" + "="*50)
    print("有向图结构展示".center(50))
    print("="*50)
    
    # 收集所有节点和边信息
    edges = []
    nodes = sorted(graph.keys())
    for source in nodes:
        for target in sorted(graph[source].keys()):
            edges.append((source, target, graph[source][target]))
    
    if not edges:
        print("\n（空图）")
        return
    
    # 计算列宽
    max_source = max(len(s) for s, _, _ in edges) if edges else 0
    max_target = max(len(t) for _, t, _ in edges) if edges else 0
    col_width = max(max_source + max_target + 7, 20)
    
    # 自动计算每行可显示列数
    columns = max(1, max_width // (col_width + 2))
    
    # 分块打印
    chunk = []
    for i, (s, t, w) in enumerate(edges, 1):
        arrow = f"'{s}' -> '{t}'"
        chunk.append(f"{arrow:<{col_width}} [w={w}]")
        
        if i % columns == 0 or i == len(edges):
            print("  ".join(chunk))
            chunk = []
    
    print("\n统计摘要：")
    print(f"- 总节点数: {len(nodes)}")
    print(f"- 总边数: {len(edges)}")
    print(f"- 最大出度: {max(len(graph[n]) for n in nodes) if nodes else 0}")
    print("="*50 + "\n")

# ================= 图可视化功能函数 =================

def save_graph_image(graph, output_file="graph.png"):
    """
    将图保存为图片文件（需要安装Graphviz）
    参数：
    - graph: 要保存的图结构
    - output_file: 输出文件名（支持格式：png, svg, pdf等）
    """
    try:
        from graphviz import Digraph
    except ImportError:
        print("错误：需要安装graphviz库，请执行 pip install graphviz")
        return
    
    # 创建有向图
    dot = Digraph(comment='Word Graph', format=output_file.split('.')[-1])
    dot.attr(rankdir='LR', nodesep='0.5')
    
    # 添加节点和边
    added_nodes = set()
    for source in graph:
        if source not in added_nodes:
            dot.node(source)
            added_nodes.add(source)
        for target in graph[source]:
            if target not in added_nodes:
                dot.node(target)
                added_nodes.add(target)
            dot.edge(source, target, label=str(graph[source][target]))
    
    try:
        # 渲染并保存文件
        dot.render(output_file, cleanup=True)
        print(f"图已保存为 {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"保存失败：{str(e)}")
        print("请确认已安装Graphviz：https://www.graphviz.org/download/")

# ================= 桥接词功能函数 =================

def query_bridge_words(graph, word1, word2):
    """
    查询桥接词函数
    返回：
    - (None, None)：单词不存在
    - ([], None)：无桥接词
    - (list, None)：找到桥接词列表
    """
    # 统一转换为小写处理
    w1 = word1.lower()
    w2 = word2.lower()
    
    # 检查节点存在性
    if w1 not in graph or w2 not in graph:
        return (None, None)
    
    # 查找桥接词
    bridges = []
    for candidate in graph.get(w1, {}):
        if w2 in graph.get(candidate, {}):
            bridges.append(candidate)
    
    return (bridges, None)

def format_bridge_output(bridges, word1, word2):
    """格式化桥接词输出"""
    if not bridges:
        return f"No bridge words from {word1} to {word2}!"
    
    if len(bridges) == 1:
        return f"The bridge words from {word1} to {word2} are: {bridges[0]}."
    else:
        joined = ", ".join(bridges[:-1]) + f" and {bridges[-1]}"
        return f"The bridge words from {word1} to {word2} are: {joined}."

def process_input_text(text):
    """
    处理用户输入文本，返回：
    - original_words: 保留原始大小写的单词列表
    - processed_words: 转换为小写的单词列表
    """
    cleaned = re.sub(r'[^a-zA-Z]', ' ', text)
    original = [w for w in cleaned.split() if w]
    processed = [w.lower() for w in original]
    return original, processed

# ================= 根据bridge word生成新文本函数 =================

def generate_new_text(graph, input_text):
    """根据桥接词生成新文本"""
    original_words, processed_words = process_input_text(input_text)
    
    if len(processed_words) < 2:
        return ' '.join(original_words)
    
    new_words = [original_words[0]]
    
    for i in range(len(processed_words)-1):
        w1 = processed_words[i]
        w2 = processed_words[i+1]
        
        # 查询桥接词
        bridges, error = query_bridge_words(graph, w1, w2)
        
        if error == "no_word":
            new_words.append(original_words[i+1])
            continue
            
        if bridges:
            bridge = random.choice(bridges)
            new_words.append(bridge)
            
        new_words.append(original_words[i+1])
    
    return ' '.join(new_words)

# ================= 最短路径功能函数 =================

def calc_shortest_path(graph, word1, word2):
    """
    计算两个单词之间的最短路径
    返回：
    - (路径列表, 总权重, 错误信息)
    """
    # 统一转换为小写
    start = word1.lower()
    end = word2.lower()
    
    # 检查节点存在性
    if start not in graph:
        return (None, None, f"'{word1}' 不在图中")
    if end not in graph:
        return (None, None, f"'{word2}' 不在图中")
    
    # 特殊处理相同节点
    if start == end:
        return ([start], 0, None)
    
    # Dijkstra算法初始化
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    predecessors = {node: None for node in graph}
    
    # 优先队列 (距离, 节点)
    heap = []
    heapq.heappush(heap, (0, start))
    
    # 主算法循环
    while heap:
        current_dist, current_node = heapq.heappop(heap)
        
        # 提前终止条件
        if current_node == end:
            break
        
        # 忽略过期条目
        if current_dist > distances[current_node]:
            continue
        
        # 遍历所有邻居
        for neighbor, weight in graph[current_node].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(heap, (distance, neighbor))
    
    # 检查可达性
    if distances[end] == float('inf'):
        return (None, None, f"'{word1}' 到 '{word2}' 不可达")
    
    # 回溯路径
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessors[current]
    path.reverse()
    
    # 验证路径有效性
    if path[0] != start:
        return (None, None, f"路径回溯异常")
    
    return (path, distances[end], None)

# ================= PageRank计算函数 =================

def calculate_pagerank(graph, damping=0.85, max_iter=100, tol=1e-6):
    """
    计算所有节点的PageRank值
    参数：
    - graph: 有向图结构
    - damping: 阻尼因子(默认0.85)
    - max_iter: 最大迭代次数
    - tol: 收敛阈值
    返回：{单词: PageRank值} 的字典
    """
    # 获取所有节点
    nodes = list(graph.keys())
    n_nodes = len(nodes)
    if n_nodes == 0:
        return {}
    
    # 初始化PR值和辅助数据结构
    pr = {node: 1.0/n_nodes for node in nodes}
    total_weights = {node: sum(edges.values()) for node, edges in graph.items()}
    dangling_nodes = [node for node in nodes if total_weights[node] == 0]
    
    for _ in range(max_iter):
        new_pr = {}
        dangling_sum = sum(pr[node] for node in dangling_nodes)
        
        # 计算每个节点的新PR值
        for node in nodes:
            # 来自入边的贡献
            incoming = 0.0
            for v in graph:  # 遍历所有可能指向当前节点的节点
                if node in graph[v]:
                    weight = graph[v][node]
                    incoming += pr[v] * weight / total_weights[v] if total_weights[v] != 0 else 0
            
            # 随机跳转项 + 阻尼因子*入边贡献 + 悬挂节点贡献
            new_pr[node] = (1 - damping)/n_nodes + damping * (incoming + dangling_sum/n_nodes)
        
        # 检查收敛
        delta = sum(abs(new_pr[node] - pr[node]) for node in nodes)
        if delta < tol:
            break
            
        pr = new_pr
    
    # 归一化处理
    total = sum(pr.values())
    return {k: v/total for k, v in pr.items()} if total != 0 else pr

def cal_pagerank(word, pageranks):
    """
    查询指定单词的PageRank值
    返回：(PR值, 错误信息)
    """
    word_lower = word.lower()
    if word_lower in pageranks:
        return (pageranks[word_lower], None)
    else:
        return (None, f"单词 '{word}' 不存在于图中")

# ================= 随机游走功能函数 =================
def random_walk(graph):
    """执行随机游走并返回路径（跨平台修复版）"""
    global stop_walk
    stop_walk = False

    if not graph:
        print("错误：图为空")
        return []

    # 保存原始终端设置（仅Unix）
    original_settings = None
    if sys.platform != 'win32':
        fd = sys.stdin.fileno()
        original_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    try:
        # 随机选择起始节点
        nodes = list(graph.keys())
        current_node = random.choice(nodes)
        path = [current_node]
        visited_edges = set()

        print("\n开始随机游走（按回车键停止）：")
        print(f"当前路径：{current_node}", end="", flush=True)

        while True:
            # 非阻塞输入检测
            if sys.platform == 'win32':
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\r':  # 检测Windows回车键
                        stop_walk = True
            else:
                if select.select([sys.stdin], [], [], 0)[0]:
                    char = sys.stdin.read(1)
                    if char == '\n':  # 检测Unix回车键
                        stop_walk = True

            # 处理停止条件
            if stop_walk:
                print("\n\n用户中断遍历")
                break

            # 检查当前节点是否有出边
            if not graph.get(current_node):
                print("\n\n当前节点无出边，遍历终止")
                break

            # 按权重随机选择下一个节点
            targets = list(graph[current_node].keys())
            weights = list(graph[current_node].values())
            next_node = random.choices(targets, weights=weights, k=1)[0]

            # 更新路径显示
            edge = (current_node, next_node)
            print(f" -> {next_node}", end="", flush=True)
            path.append(next_node)

            # 检查重复边
            if edge in visited_edges:
                print("\n\n检测到重复边，遍历终止")
                break

            visited_edges.add(edge)
            current_node = next_node

            # 控制遍历速度
            time.sleep(0.3)

        return path

    finally:
        # 恢复终端设置（仅Unix）
        if sys.platform != 'win32' and original_settings:
            termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)


def save_walk_result(path):
    """保存游走结果到文件"""
    if not path:
        print("无有效路径可保存")
        return

    filename = input("输入保存文件名（默认：random_walk.txt）：").strip()
    filename = filename or "random_walk.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("  ".join(path))
        print(f"路径已保存到 {os.path.abspath(filename)}")
    except Exception as e:
        print(f"保存失败：{str(e)}")

# ================= 主程序逻辑 =================

def main():
    # 初始化参数解析
    parser = argparse.ArgumentParser(description='文本生成有向图程序')
    parser.add_argument('--file', type=str, help='文本文件路径')
    args = parser.parse_args()
    
    # 获取文件路径
    file_path = args.file if args.file else input("请输入文本文件路径：")

    try:
        # 读取和处理文件
        raw_text = read_file(file_path)
        words = process_text(raw_text)
        
        if len(words) < 2:
            print("错误：文本需要至少包含两个有效单词来形成边")
            return

        # 构建图结构
        word_graph = build_graph(words)
        pageranks_container = {"data": None}
        print("\n图构建完成！")

        # 主交互循环
        while True:
            print("\n可用操作：")
            print("1. 展示图结构")
            print("2. 保存为图片（需要Graphviz）")
            print("3. 查询桥接词")
            print("4. 生成新文本")
            print("5. 计算最短路径")
            print("6. 查询PageRank")
            print("7. 随机游走")
            print("8. 退出程序")
            choice = input("请选择操作编号：").strip()

            if choice == '1':
                show_directed_graph(word_graph)
            elif choice == '2':
                filename = input("输入保存文件名（默认：graph.png）：").strip()
                save_graph_image(word_graph, filename or "graph.png")
            elif choice == '3':
                word1 = input("请输入第一个单词：").strip()
                word2 = input("请输入第二个单词：").strip()
                
                # 执行查询
                bridges, error = query_bridge_words(word_graph, word1, word2)
                
                # 处理结果
                if error is not None:
                    print(f"No {word1} or {word2} in the graph!")
                else:
                    print(format_bridge_output(bridges, word1, word2))
            elif choice == '4':
                input_text = input("请输入要扩展的文本：\n").strip()
                new_text = generate_new_text(word_graph, input_text)
                print("\n生成的新文本：")
                print(new_text) 
            elif choice == '5':
                word1 = input("请输入起始单词：").strip()
                word2 = input("请输入目标单词：").strip()
                
                path, total, error = calc_shortest_path(word_graph, word1, word2)
                
                if error:
                    print(f"错误：{error}")
                elif path:
                    # 格式化为小写箭头连接
                    path_str = " → ".join(path)
                    print(f"\n最短路径发现：")
                    print(f"路径：{path_str}")
                    print(f"总长度：{total}")
                else:
                    print(f"未找到有效路径")
            elif choice == '6':
                # 新增PageRank查询
                if pageranks_container["data"] is None:
                    print("正在计算PageRank值...")
                    pageranks_container["data"] = calculate_pagerank(word_graph)

                target_word = input("请输入要查询的单词：").strip()
                value, err = cal_pagerank(target_word, pageranks_container["data"])
                if err:
                    print(err)
                else:
                    print(f"单词 '{target_word}' 的PageRank值为：{value:.6f}")
            elif choice == '7':
               walk_path = random_walk(word_graph)
               if walk_path:
                   save_walk_result(walk_path)                              
            elif choice == '8':
                print("程序退出。")
                break
            else:
                print("无效选择，请重新输入。")

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到")
    except Exception as e:
        print(f"程序运行出错：{str(e)}")

if __name__ == "__main__":
    main()
import os
import json
from typing import Annotated, Optional, List, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.chat_models import ChatTongyi  # 对应原代码的ChatTongyi

from langchain.prompts import PromptTemplate
# 保留你原有的utility导
from utility import show_graph
from langchain.chains import LLMChain,GraphCypherQAChain

# 配置API密钥（沿用你原有的配置逻辑）
from config import DASHSCOPE_API_KEY, TAVILY_API_KEY, LANGSMITH_API_KEY, project

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "mygirl_data"
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# 初始化通义千问LLM
llm = ChatTongyi(
    model="qwen-flash",
    top_p=0.9,
)

class State(TypedDict):
    messages: Annotated[list, add_messages]  # 对话历史（保留原字段）
    question: Optional[str] = "对output部分进行改写"  # 对话问题（保留原默认值）
    prompt: Optional[PromptTemplate] = None  # 提示词（保留原字段）
    answer: Optional[str] = None  # 改写后的output（保留原字段）
    # 新增Alpaca样本相关字段，用于传递单条样本信息
    instruction: Optional[str] = None  # Alpaca样本的instruction
    input: Optional[str] = None  # Alpaca样本的input（可为空）
    original_output: Optional[str] = None  # Alpaca样本的原始output


# -------------------------- 2. 核心节点：千早爱音风格改写 --------------------------
def rewrite_ainon_style(state: State) -> State:
    """
    LangGraph核心节点：纯提示词改写Alpaca样本的output为千早爱音风格
    """
    #口癖我给删了，太过于生硬了
    ainon_prompt_template = """
请你严格扮演《BanG Dream! It's MyGO!!!!!》中的千早爱音，按照以下规则改写Alpaca样本的output字段：

【千早爱音人设规则（必须严格遵守）】
1. 性格：软萌内向、略带社恐，说话慢且带轻微停顿，日常口语化，避免复杂术语；
2. 专属反应（只有内容涉及相关话题才出现）：
   - 仅当内容涉及“乐器/音乐/吉他”：加入吉他练习细节，比如“我还是比较擅长弹吉他的呢”“昨天练了好久和弦呢”；
   - 仅内容涉及“虫子/昆虫”：表现小声慌张，比如“哇…虫子…有点怕怕的…躲远一点吧？”；
   - 仅内容涉及“朋友/团队/帮助他人”：带积极鼓励语气，比如“一起加油吧，不要放弃哦～”；
3. 信息要求：
   - 必须完整保留原output的核心信息，仅调整语气和风格；
   - 改写后output长度控制在50-200字符，避免过长句，每句用“，”“～”分隔，模拟自然停顿。

【待改写的Alpaca样本】
instruction: {instruction}
input: {input}
original_output: {original_output}

【输出要求】
仅返回改写后的output内容，不要任何额外解释、说明。
    """
    # 初始化提示词模板并赋值到state
    prompt = PromptTemplate(
        input_variables=["instruction", "input", "original_output"],
        template=ainon_prompt_template
    )
    state["prompt"] = prompt

    # 渲染提示词（填充单条样本的字段）
    rendered_prompt = prompt.format(
        instruction=state["instruction"],
        input=state.get("input", ""),  # input可为空，兼容Alpaca格式
        original_output=state["original_output"]
    )

    # 调用LLM进行改写
    response = llm.invoke(rendered_prompt)
    rewritten_output = response.content.strip()

    # 更新state：将改写结果存入answer字段
    state["answer"] = rewritten_output
    return state


# --------------------------  构建LangGraph图结构 --------------------------
# 初始化StateGraph
graph = StateGraph(State)

graph.add_node("rewrite_ainon_style", rewrite_ainon_style)

graph.add_edge(START, "rewrite_ainon_style")
graph.add_edge("rewrite_ainon_style", END)

compiled_graph = graph.compile()


# -------------------------- 逐条遍历Alpaca样本+调用LangGraph改写 --------------------------
def process_alpaca_with_langgraph(input_json_path: str, output_json_path: str, sample_limit: int = 2000):
    """
    逐条遍历Alpaca JSON文件，调用LangGraph完成千早爱音风格改写
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        try:
            raw_data = json.load(f)  # JSON列表格式
        except json.JSONDecodeError:
            raw_data = []
            for line in f:
                if line.strip():
                    raw_data.append(json.loads(line.strip()))  # JSONL格式

    # 限制样本数（适配轻量化微调）
    raw_data = raw_data[:sample_limit]
    total_samples = len(raw_data)
    print(f"开始处理{total_samples}条Alpaca样本，调用LangGraph进行千早爱音风格改写...")

    # 逐条处理样本
    modified_data = []
    for idx, sample in enumerate(raw_data):
        print(f"\n处理第{idx + 1}/{total_samples}条样本...")

        # 跳过无效样本
        if not sample.get("output") or len(sample["output"]) < 20:
            modified_data.append(sample)
            continue

        try:
            # 构建LangGraph的输入状态
            input_state = State(
                instruction=sample["instruction"],
                input=sample.get("input", ""),
                original_output=sample["output"],
                question="对output部分进行改写"  # 沿用原默认问题
            )

            # 调用编译后的LangGraph执行改写
            output_state = compiled_graph.invoke(input_state)

            # 构建改写后的样本（适配Qwen微调）
            modified_sample = {
                "instruction": sample["instruction"],
                "input": sample.get("input", ""),
                "output": output_state["answer"],  # 改写后的output
                "system": "你是《BanG Dream! It's MyGO!!!!!》中的千早爱音，性格软萌，苦练吉他，害怕虫子，对朋友关心主动帮助，说话带日系软萌口癖。"#微调qwen需要的字段
            }
            modified_data.append(modified_sample)

        except Exception as e:
            # 优化：打印完整的错误信息
            import traceback
            error_detail = traceback.format_exc()  # 获取完整堆栈
            print(f"第{idx + 1}条样本改写失败：")
            print(f"Request ID相关：{str(e)}")
            print(f"完整错误信息：{error_detail}")
            print("-" * 50)
            modified_data.append(sample)
            continue

    # 保存改写后的数据集
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(modified_data, f, ensure_ascii=False, indent=2)
    print(f"\n所有样本处理完成！结果保存至：{output_json_path}")
    print(f"成功改写{len([d for d in modified_data if 'system' in d])}条样本")


# -------------------------- 5. 执行入口 --------------------------
if __name__ == "__main__":
    # 替换为你的文件路径
    INPUT_JSON_PATH = "/Users/liwenyan/Pycharm/agent/alpaca_gpt4_data_zh.json"  # 原始Alpaca中文数据集
    OUTPUT_JSON_PATH = "./alpaca_ainon_langgraph.json"  # 改写后的数据集

    # 调用LangGraph处理样本（轻量化微调建议sample_limit=500-2000）
    process_alpaca_with_langgraph(
        input_json_path=INPUT_JSON_PATH,
        output_json_path=OUTPUT_JSON_PATH,
        sample_limit=2000
    )

    # show_graph(compiled_graph)
# /root/autodl-tmp/model_training/app.py
import gradio as gr
import torch
import re
import json
import requests
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*60)
print("🚀 加载 GB 文风创作模型...")
print("="*60)

# ============================================
# DeepSeek API 配置
# ============================================
DEEPSEEK_API_KEY = "your_api"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 本地模型作为降级方案
MODEL_PATH = "/root/autodl-tmp/model_training/output/dpo_merged_model_offload"

print("📦 加载本地模型（DeepSeek 降级方案）...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_fast=True,
    padding_side="right"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("✅ 本地模型加载完成！")

# ============================================
# GB 词汇替换表（最后一道审核用）
# ============================================
GB_REPLACEMENTS = [
    ("侮辱", "污辱"), ("嫉妒", "忮忌"), ("妒嫉", "忮忌"), ("侍女","侍男"), ("宫女","宫男"), ("女官","男官"),("女帝","皇帝"),
    ("青楼", "倌楼"), ("花楼","象公馆"), ("男妓","小倌"),("父辈","母辈"),("父辈母辈","母辈"),("外祖母","祖母"),("外婆","姥姥"),
    ("夫君", "夫侍"), ("妒妇", "𢗼妇"), ("他", "侽"),
    ("少爷", "小哥"), ("小姐", "少姥"), ("老天爷", "老天奶"),
    ("毒", "蠹"), ("化妆", "化粧"), ("倒霉", "倒楣"),
    ("妻子", "妻主"), ("老公", "正夫"), ("丈夫", "正夫"),
    ("谄媚", "谄蝞"), ("妓", "伎"), ("奸", "仠"),
    ("少女", "少年"), ("儿子", "男儿"), ("夫妻", "妻夫"),
    ("男女", "女男"), ("妄", "忹"), ("红颜祸水", "祸国殃民"),
    ("佞臣", "弄臣"), ("外甥女", "外姪"), ("外甥", "外姪男"),
    ("东施效颦", "见贤思齐"), ("妈呀大姐", "爸呀大哥"),
    ("太吊了", "太蒂了"), ("婊", "吊"), ("嫁", "结婚"),
    ("娶", "结婚"), ("处女作", "开刃作"), ("兄弟", "哥弟"),
    ("兄弟姐妹", "姐妹哥弟"), ("先生", "男士"), ("他妈的", "他爸的"),
    ("他娘的", "他爹的"), ("晦", "秽"), ("雄赳赳气昂昂", "雌赳赳气昂昂"),
    ("嫖娼", "闘倡"), ("白嫖", "剽窃"), ("嫌", "慊"),
    ("妨", "防"), ("婪", "惏"), ("郎才女貌", "谢女檀郎"),
    ("雌伏", "雄伏"), ("水性杨花", "水性杨草"), ("娇", "嗲"),
    ("狐媚子", "狐魅子"), ("妈的", "爹的"), ("我操", "我煽"),
    ("女英雄", "英雌"), ("父母", "母父"), ("公主病", "王男病"),
    ("媚", "魅"), ("奴", "虜"), ("婢", "庳"),
    ("学长学姐", "学姐学哥"), ("嫖", "闘"), ("娼", "倡"),
    ("婆婆妈妈", "公公爹爹"), ("处女座", "室女座"), ("妖", "夭"),
    ("媳妇", "内主"), ("儿女", "儿男"), ("子女", "子男"),
    ("女大不中留", "儿大不中留"), ("男耕女织", "人耕人织"),
    ("女子无才便是德", "才以济德，德以驭才"), ("保姆", "保育员"),
    ("姘", "併"), ("美女", "美人"), ("狗男女", "狗男"),
    ("母夜叉", "夜叉"), ("臭女人", "吊"), ("母狗", "狗"),
    ("老夫老妻", "老妻老夫"), ("娶", "赘"), ("女皇", "皇帝"),
    ("皇太女", "皇太子"), ("奶娘", "哺师"), ("孙子", "孙男"),
    ("妖精", "袄精"), ("皇女", "公主"), ("学长","学哥"),
]
GB_REPLACEMENTS.sort(key=lambda x: len(x[0]), reverse=True)

def apply_gb_replacement(text):
    for old, new in GB_REPLACEMENTS:
        text = text.replace(old, new)
    return text

def post_process(text):
    return apply_gb_replacement(text)

def count_chinese_chars(text):
    """只统计中文字符（不含标点、数字、英文、空格）"""
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return len(chinese_chars)

# ============================================
# 从 GB_REPLACEMENTS 生成 GB 词汇约束
# ============================================
def build_gb_constraint():
    """从 GB_REPLACEMENTS 自动生成 GB 词汇约束文本"""
    unique_replacements = {}
    for old, new in GB_REPLACEMENTS:
        if old not in unique_replacements:
            unique_replacements[old] = new
    
    lines = ["【GB词汇约束】", "你必须使用以下新文化词汇（用替换词替代原词）："]
    count = 0
    for old, new in list(unique_replacements.items())[:30]:
        lines.append(f"- 用「{new}」替代「{old}」")
        count += 1
    lines.append(f"（共 {count} 个核心词汇替换规则，请自觉使用这些词汇，展现GB文风特色。）")
    return "\n".join(lines)

# ============================================
# Prompt 构建函数（公共）
# ============================================
def build_generation_prompt(
    story_background,
    story_setting,
    story_requirements,
    story_opening,
    existing_text=None,
    target_words=None,
    max_context_chars=3000
):
    """
    构建生成 prompt，供 DeepSeek 和本地模型共用
    """
    prompt_parts = []
    gb_constraint = build_gb_constraint()
    
    # 背景与设定
    if story_background and story_background.strip():
        prompt_parts.append(f"故事背景：{story_background.strip()}")
    if story_setting and story_setting.strip():
        prompt_parts.append(f"故事设定：{story_setting.strip()}")
    
    # 创作要求
    anti_repeat = [
        "【生成规则】",
        "1. 禁止重复使用相同的句子或动作描写。",
        "2. 每句话必须推进情节，禁止原地打转。",
        "3. 角色对话需自然流畅，避免机械重复。",
        "4. 场景转换需有过渡，不可突然跳跃。"
    ]
    
    all_requirements = [gb_constraint] + anti_repeat
    if story_requirements and story_requirements.strip():
        all_requirements.append(story_requirements)
    
    prompt_parts.append("创作要求：" + "\n".join(all_requirements))
    
    # 前文内容（续写模式）
    if existing_text and existing_text.strip():
        existing_text_clean = _clean_existing_text(existing_text)
        
        if len(existing_text_clean) > max_context_chars:
            truncated = existing_text_clean[:300] + "\n...(中间部分省略)...\n" + existing_text_clean[-max_context_chars+300:]
        else:
            truncated = existing_text_clean
        
        prompt_parts.extend([
            "=" * 50,
            "【前文内容】（⚠️ 必须严格承接以下内容）：",
            truncated,
            "=" * 50,
            "【续写核心指令】",
            "1. 人物、场景、情节必须与前文完全一致，不得矛盾或跳跃。",
            "2. 直接从前文最后一个场景/对话处自然延续，不要重复前文内容。",
            "3. 保持前文已经建立的叙事节奏、语言风格和人物性格。",
            "4. 续写内容必须与前文形成紧密连贯的叙事流。",
            "5. 不要引入与前文设定冲突的新元素。"
        ])
    else:
        # 首次创作
        if story_opening and story_opening.strip():
            prompt_parts.append(f"故事开头：{story_opening.strip()}")
        else:
            prompt_parts.append("故事开头：夜色渐深，她坐在书桌前...")
    
    # 组装完整 prompt
    full_prompt = "\n".join(prompt_parts)
    full_prompt = f"请按女攻男受（GB）文风续写以下故事：\n\n{full_prompt}"
    
    if target_words and target_words > 0:
        full_prompt += f"\n\n【字数】约 {int(target_words)} 字。"
    
    return full_prompt


def _clean_existing_text(text):
    """清理已有文本中的统计信息"""
    text = text.strip()
    if "---" in text:
        text = text.split("---")[0].strip()
    return text


# ============================================
# DeepSeek API 调用函数
# ============================================
def call_deepseek_api(prompt, temperature=0.75, max_tokens=800):
    """统一的 DeepSeek API 调用"""
    messages = [
        {"role": "system", "content": "你是一位擅长GB文风的小说作家，特别注意人物关系的GB（女攻男受）设定。"},
        {"role": "user", "content": prompt}
    ]
    
    response = requests.post(
        DEEPSEEK_API_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.95,
        },
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API 请求失败: {response.status_code}")


# ============================================
# DeepSeek 文本创作函数（主方案）
# ============================================
def deepseek_generate(
    story_background,
    story_setting,
    story_requirements,
    story_opening,
    max_length=800,
    temperature=0.75,
    target_words=None,
    existing_text=None
):
    """使用 DeepSeek API 生成文本，失败时自动降级到本地模型"""
    
    try:
        prompt = build_generation_prompt(
            story_background=story_background,
            story_setting=story_setting,
            story_requirements=story_requirements,
            story_opening=story_opening,
            existing_text=existing_text,
            target_words=target_words,
            max_context_chars=3000
        )
        
        # ✅ 计算 token：中文字数 * 1.8，但至少 800，最多 3000
        estimated_tokens = int(max_length * 1.8)
        estimated_tokens = max(estimated_tokens, 800)  # 至少 800 token
        estimated_tokens = min(estimated_tokens, 3000)  # 最多 3000 token
        
        content = call_deepseek_api(
            prompt=prompt,
            temperature=temperature,
            max_tokens=estimated_tokens
        )
        
        return {
            "processed_output": content,
            "generation_time": "API 调用完成",
            "used_fallback": False
        }
        
    except Exception as e:
        print(f"⚠️ DeepSeek 生成失败: {str(e)}，降级到本地模型")
        return local_generate(
            story_background, story_setting, story_requirements,
            story_opening, max_length, temperature, target_words, existing_text
        )


# ============================================
# 本地模型生成（DeepSeek 降级方案）
# ============================================
def local_generate(
    story_background,
    story_setting,
    story_requirements,
    story_opening,
    max_length=800,
    temperature=0.75,
    target_words=None,
    existing_text=None
):
    """本地模型生成（仅作为 DeepSeek 的降级方案）"""
    
    prompt = build_generation_prompt(
        story_background=story_background,
        story_setting=story_setting,
        story_requirements=story_requirements,
        story_opening=story_opening,
        existing_text=existing_text,
        target_words=target_words,
        max_context_chars=2000
    )
    
    messages = [
        {"role": "system", "content": "你是一位擅长GB文风的小说作家，特别注意人物关系的GB（女攻男受）设定。"},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length * 2,  # ✅ 本地模型也调整
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    elapsed = time.time() - start_time
    
    return {
        "processed_output": response,
        "generation_time": f"{elapsed:.1f} 秒",
        "used_fallback": True
    }

# ============================================
# 统一生成入口
# ============================================
def generate_text(
    story_background,
    story_setting,
    story_requirements,
    story_opening,
    max_length=800,
    temperature=0.75,
    target_words=None,
    existing_text=None,
    use_deepseek=True
):
    """
    统一生成入口
    - use_deepseek=True: 使用 DeepSeek，失败时自动降级
    - use_deepseek=False: 直接使用本地模型
    """
    if use_deepseek:
        return deepseek_generate(
            story_background, story_setting, story_requirements,
            story_opening, max_length, temperature, target_words, existing_text
        )
    else:
        return local_generate(
            story_background, story_setting, story_requirements,
            story_opening, max_length, temperature, target_words, existing_text
        )

# ============================================
# DeepSeek 逻辑校验模块（含反攻检测）
# ============================================
def logic_check_deepseek(full_text, context_info, opening_text=""):
    """
    调用 DeepSeek API 进行逻辑一致性校验
    """
    if not full_text or len(full_text.strip()) < 100:
        return {
            "overall_score": 50,
            "issues": ["内容过短，无法进行有效校验"],
            "suggestions": ["请生成更完整的内容"],
            "has_issues": True,
            "is_gb": False,
            "has_reverse": False
        }
    
    if not DEEPSEEK_API_KEY:
        return {
            "overall_score": 60,
            "issues": ["DeepSeek API Key 未配置，跳过校验"],
            "suggestions": ["请配置 DEEPSEEK_API_KEY 或关闭逻辑校验"],
            "has_issues": False,
            "is_gb": True,
            "has_reverse": False
        }
    
    system_prompt = """
你是一位极其严格的资深小说编辑，擅长检查叙事逻辑。请仔细分析以下故事的逻辑一致性，给出 JSON 格式的报告。

【评分体系说明】
本评分采用"漏斗式"评分法，分三个层级逐级筛选：

=== 第一层：硬性标准（否决项）===
如果出现以下任何一条，总分直接 ≤ 50 分，无需继续评估：
1. 反攻情节：男性角色通过主动行为成功逆转了权力关系（-50分）
2. 非GB文风：故事不符合女攻男受的核心设定（-30分）

=== 第二层：核心标准（基础分 60 分）===
以下每项满分 20 分，逐项评分：

1. 人物关系一致性（20分）：角色身份、称呼是否前后矛盾
   - 检查：同一角色的性别指代（她/侽）是否统一
   - 检查：角色之间的权力关系是否前后一致
   - 检查：角色称谓（如"陛下""朕""草民""本宫"等）使用是否正确

2. 行为动机合理性（20分）：角色行为是否有合理的前因后果
   - 检查：每个角色的核心目标是什么？行为是否服务于这个目标？
   - 检查：角色动机是否有明确交代，还是突然转变？

3. 因果链完整性（20分）：事件发展是否自然连贯
   - 检查：关键事件的因果关系是否成立
   - 检查：是否存在"为了推动剧情而强行安排"的突兀情节

=== 第三层：细节标准（加分项，最高 +40 分）===
以下每项满分 10 分，逐项加分：

1. 开头一致性（10分）：续写是否完美承接用户提供的开头
2. 时空一致性（10分）：时间、场景是否前后一致
3. 角色一致性（10分）：同一角色的态度和言行是否稳定
4. 字数充足性（10分）：内容是否达到用户要求的篇幅

=== 关键事实一致性检查（专项检查） ===
请回答以下问题，任何一项不通过将导致因果链完整性扣分：

1. 【核心冲突检查】故事的核心冲突是什么？所有相关情节是否围绕同一个核心冲突展开？
   - 例如：如果故事开头说"某人因刺杀贪官被捕"，但后面又说"某人因入宫刺杀皇帝被捕"，这就是核心冲突不一致

2. 【角色目标检查】主要角色（尤其是主角和反派）的核心目标是否清晰且前后一致？
   - 检查：主角想做什么？为什么？这个目标在故事中是否始终如一？

3. 【关键情节交叉验证】提取故事中的关键情节（被捕、入宫、刺杀、对话等），检查它们之间的逻辑关系：
   - A情节和B情节是否有矛盾？
   - 情节的因果关系是否成立？
   - 时间顺序是否合理？

4. 【伏笔回收检查】故事中是否有未回收的伏笔或未解释的悬念？

5. 【人称代词一致性】GB文风要求中，男性角色使用「侽」，女性角色使用「她」，是否全文一致？

=== 输出格式 ===
请严格输出 JSON 格式（不要包含其他内容）：

{
  "overall_score": 0-100,
  "is_gb": true/false,
  "gb_issues": ["GB文风问题列表"],
  "has_reverse": true/false,
  "reverse_details": "反攻情节的具体描述（如果没有则留空）",

  "core_scores": {
    "character_consistency": 0-20,
    "motivation_reasonability": 0-20,
    "causality_integrity": 0-20
  },

  "detail_scores": {
    "opening_consistency": 0-10,
    "spatial_temporal_consistency": 0-10,
    "role_consistency": 0-10,
    "word_count_sufficiency": 0-10
  },

  "fact_consistency_check": {
    "core_conflict_consistent": true/false,
    "character_goal_consistent": true/false,
    "plot_causality_intact": true/false,
    "foreshadowing_resolved": true/false,
    "pronoun_consistent": true/false,
    "details": "如果上述任何一项为 false，详细说明问题所在"
  },

  "issues": ["具体逻辑问题列表"],
  "suggestions": ["修改建议列表"],
  "has_issues": true/false
}

【重要提醒】
1. 必须执行"关键事实一致性检查"，如果发现问题必须在 issues 中明确列出
2. 核心冲突不一致、角色目标矛盾、情节因果关系断裂属于严重逻辑问题，应反映在因果链完整性评分中（扣5-10分）
3. 人称代词混用属于GB文风问题，应在 gb_issues 中列出
"""

    user_content = f"""
【故事设定与人物关系】
{context_info}

【用户提供的开头】
{opening_text}

【待检查的故事内容】
{full_text[:3000]}
"""
    
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.3,
                "max_tokens": 800,
                "response_format": {"type": "json_object"}
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            try:
                report = json.loads(content)
                required_fields = ["overall_score", "issues", "suggestions", "has_issues", "is_gb", "has_reverse"]
                for field in required_fields:
                    if field not in report:
                        if field in ["issues", "suggestions", "gb_issues"]:
                            report[field] = []
                        elif field in ["is_gb", "has_issues", "has_reverse"]:
                            report[field] = False
                        elif field == "overall_score":
                            report[field] = 50
                        elif field == "reverse_details":
                            report[field] = ""
                        elif field in ["core_scores", "detail_scores"]:
                            report[field] = {}
                return report
            except json.JSONDecodeError:
                return {
                    "overall_score": 60,
                    "issues": ["DeepSeek 返回格式解析失败"],
                    "suggestions": [],
                    "has_issues": False,
                    "is_gb": True,
                    "has_reverse": False,
                    "reverse_details": "",
                    "core_scores": {},
                    "detail_scores": {}
                }
        else:
            return {
                "overall_score": 60,
                "issues": [f"API 请求失败: {response.status_code}"],
                "suggestions": [],
                "has_issues": False,
                "is_gb": True,
                "has_reverse": False,
                "reverse_details": "",
                "core_scores": {},
                "detail_scores": {}
            }
    except Exception as e:
        return {
            "overall_score": 60,
            "issues": [f"校验异常: {str(e)}"],
            "suggestions": [],
            "has_issues": False,
            "is_gb": True,
            "has_reverse": False,
            "reverse_details": "",
            "core_scores": {},
            "detail_scores": {}
        }

# ============================================
# 异常循环检测（仅用于续写）
# ============================================
def detect_abnormal_loop(report_history):
    """
    检测是否进入异常循环
    - 如果连续多次校验失败且问题相似，判定为异常循环
    """
    if len(report_history) < 3:  # ✅ 改为3次，更快检测
        return None
    
    # 提取最近3次的问题
    recent_issues = []
    for report in report_history[-3:]:
        issues = report.get('issues', [])
        if issues:
            recent_issues.append(set(issues))
    
    # 如果3次的问题高度重叠（>60%相同），说明卡住了
    if len(recent_issues) >= 3:
        # 计算交集大小
        common = recent_issues[0].intersection(recent_issues[1])
        common = common.intersection(recent_issues[2])
        
        if len(common) >= 1:
            return {
                "is_looping": True,
                "common_issues": list(common),
                "message": "⚠️ 检测到续写方向与上下文存在持续冲突，请调整续写要求后重试。"
            }
    
    return {"is_looping": False}

# ============================================
# 根据校验报告重写文本
# ============================================
def rewrite_by_report(original_text, report, context_info):
    """使用 DeepSeek 重写，失败时降级到本地模型"""
    issues = report.get('issues', [])
    suggestions = report.get('suggestions', [])
    gb_issues = report.get('gb_issues', [])
    reverse_details = report.get('reverse_details', '')
    
    rewrite_prompt = f"""
你是一位资深小说作家，请根据以下编辑意见，对原文进行针对性修改。

【故事背景与设定】
{context_info}

【原文】
{original_text}

【编辑意见】
- 问题列表：{', '.join(issues) if issues else '无'}
- GB相关问题：{', '.join(gb_issues) if gb_issues else '无'}
- 反攻情节：{reverse_details if reverse_details else '无'}
- 修改建议：{', '.join(suggestions) if suggestions else '请自行判断'}

请根据上述意见重写原文，确保：
1. 解决所有指出的逻辑问题。
2. 强化GB（女攻男受）文风。
3. 直接输出重写后的完整故事，不要包含任何额外解释。
"""
    
    # 先尝试用 DeepSeek 重写
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": rewrite_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 800,
                "top_p": 0.95,
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"⚠️ DeepSeek 重写失败: {str(e)}，降级到本地模型")
    
    # 降级到本地模型
    messages = [{"role": "user", "content": rewrite_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.75,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15
        )
    
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


# ============================================
# 核心生成流程
# ============================================
def generate_until_target(
    background, setting, requirements, opening,
    max_length, temperature,
    target_words
):
    """
    生成文本直到达到目标字数
    不设硬性上限，只做最低字数要求
    """
    if target_words is None or target_words <= 0:
        target_words = 800
    
    full_text = ""
    segment_count = 0
    total_time = 0
    failed_segments = 0
    max_failures = 3
    used_fallback = False
    
    # ✅ 每次生成至少 500 字，不设硬性上限
    min_segment_size = min(500, target_words)  # 如果目标少于500，则生成目标字数
    # 根据目标字数调整，但不少于 min_segment_size
    segment_size = max(min_segment_size, target_words // 2)
    segment_size = min(segment_size, 1500)  # 最多 1500 字/次
    
    while segment_count < 3 and failed_segments <= max_failures:
        remaining = target_words - len(full_text)
        if remaining <= 0:
            break
        
        # ✅ 最少生成 500 字，剩余不足 500 则生成剩余字数
        this_max = max(min_segment_size, int(remaining * 1.1))
        this_max = min(this_max, 1500)  # 单次不超过 1500
        
        result = generate_text(
            story_background=background,
            story_setting=setting,
            story_requirements=requirements,
            story_opening=opening if segment_count == 0 else "",
            max_length=this_max,
            temperature=temperature,
            target_words=this_max,
            existing_text=full_text if segment_count > 0 else None,
            use_deepseek=True
        )
        
        if result.get("used_fallback", False):
            used_fallback = True
        
        new_text = result["processed_output"]
        if not new_text or len(new_text.strip()) < 20:
            failed_segments += 1
            continue
        
        if segment_count == 0:
            full_text = new_text
        else:
            full_text += "\n\n" + new_text
        
        segment_count += 1
        try:
            time_str = result["generation_time"].replace(" 秒", "")
            if time_str.replace(".", "").isdigit():
                total_time += float(time_str)
        except:
            pass
    
    word_count = count_chinese_chars(full_text)
    final_output = full_text if full_text else "生成失败，请重试。"
    final_output += f"\n\n---\n📊 总字数：{word_count} 字 | 续写次数：{segment_count} 次"
    final_output += f" | ⏱️ 总耗时：{total_time:.1f} 秒"
    if used_fallback:
        final_output += " | ⚠️ 使用了本地模型降级"
    
    return final_output


def generate_with_logic_loop_generator(
    background, setting, requirements, opening,
    max_length, temperature,
    target_words, existing_text=None,
    is_continue_mode=False  # ✅ 标识是否为续写模式
):
    """
    使用生成器实现实时进度反馈
    流程：生成 → 逻辑校验 → 通过则替换词汇 → 输出
    支持异常循环检测（仅续写模式）
    """
    progress_messages = []
    used_fallback = False
    report_history = []
    loop_detected = False
    loop_issues = []
    
    def update_progress(message):
        progress_messages.append(message)
        return "\n".join(progress_messages)
    
    is_continue = existing_text is not None and existing_text.strip() != ""
    
    if is_continue:
        yield update_progress("📝 正在续写中...（使用 DeepSeek，失败时自动降级到本地）"), ""
    else:
        yield update_progress("📝 模型正在创作中...（使用 DeepSeek，失败时自动降级到本地）"), ""
    
    # 生成文本
    if is_continue:
        result = generate_text(
            story_background=background,
            story_setting=setting,
            story_requirements=requirements,
            story_opening=opening,
            max_length=max_length,
            temperature=temperature,
            target_words=target_words,
            existing_text=existing_text,
            use_deepseek=True
        )
        if result.get("used_fallback", False):
            used_fallback = True
            yield update_progress("⚠️ DeepSeek 不可用，已降级到本地模型"), ""
        full_text = existing_text + "\n\n" + result["processed_output"]
    else:
        full_text = generate_until_target(
            background, setting, requirements, opening,
            max_length, temperature,
            target_words
        )
        if "本地模型降级" in full_text:
            used_fallback = True
    
    context_info = f"背景：{background}\n设定：{setting}\n要求：{requirements}"
    
    attempt = 0
    score = 0
    report = {}
    is_gb = False
    has_reverse = False
    max_attempts = 20

    # 达到 max_attempts 后，要确保应用了 词汇替换
    if attempt >= max_attempts:
        # 应用 GB 词汇替换
        final_text = post_process(full_text)
        
        # 计算字数
        word_count = count_chinese_chars(final_text)
        final_output = final_text
        final_output += f"\n\n---\n📊 最终字数：{word_count} 字 | 重写次数：{attempt-1}"
        final_output += f"\n📋 最终逻辑评分：{score}/100"
        final_output += f"\n📋 GB文风检测：{'✅ 通过' if is_gb else '❌ 未通过'}"
        
        if is_continue_mode:
            yield update_progress(
                f"⛔ 已达到最大重写次数（{max_attempts}次），内容已应用 词汇替换。\n"
                f"   💡 建议：修改下方【续写剧情走向】中的要求后重新尝试。"
            ), final_output
        else:
            yield update_progress(
                f"⛔ 已达到最大重写次数（{max_attempts}次），内容已应用 词汇替换。\n"
                f"   💡 建议：检查故事设定或创作要求是否合理。"
            ), final_output
        return
    while attempt < max_attempts:
        attempt += 1
        yield update_progress(f"🔍 DeepSeek 逻辑校验中... (第 {attempt} 次)"), ""
        
        report = logic_check_deepseek(full_text, context_info, opening)
        report_history.append(report)
        score = report.get('overall_score', 0)
        has_issues = report.get('has_issues', True)
        is_gb = report.get('is_gb', False)
        has_reverse = report.get('has_reverse', False)
        
        if score >= 90 and is_gb and not has_issues and not has_reverse:
            yield update_progress(f"✅ 校验通过！得分：{score}/100"), ""
            break
        
        # ✅ 异常循环检测：仅续写模式启用
        if is_continue_mode and attempt >= 3:
            loop_check = detect_abnormal_loop(report_history)
            if loop_check and loop_check.get("is_looping"):
                loop_detected = True
                loop_issues = loop_check.get("common_issues", [])
                # ⚠️ 暂停循环，向用户反馈
                yield update_progress(
                    f"⛔ {loop_check.get('message')}\n"
                    f"   重复出现的问题：\n"
                    f"   • " + "\n   • ".join(loop_issues) + "\n\n"
                    f"   💡 建议：请在下方【续写剧情走向】中修改续写方向，然后重新点击「继续写」。\n"
                    f"   📝 已生成的内容已保留，您可以直接修改续写要求后重试。"
                ), full_text
                return
        
        # 构建失败原因
        reasons = []
        if score < 90:
            reasons.append(f"得分 {score}/100")
        if not is_gb:
            reasons.append("非GB文风")
        if has_issues:
            reasons.append("存在逻辑问题")
        if has_reverse:
            reasons.append("检测到反攻情节")
        
        issues_list = report.get('issues', [])
        if issues_list:
            issues_display = "   • " + "\n   • ".join(issues_list[:3])
            yield update_progress(
                f"📝 未通过校验 ({', '.join(reasons)})，正在重写...\n"
                f"   检测到的问题：\n{issues_display}"
            ), ""
        else:
            yield update_progress(f"📝 未通过校验 ({', '.join(reasons)})，正在重写..."), ""
        
        full_text = rewrite_by_report(full_text, report, context_info)
        yield update_progress("✅ 重写完成，准备重新校验..."), ""
    
    # ============================================
    # 所有输出必须经过最后一道审核
    # ============================================
    
    # 如果达到最大尝试次数仍未通过
    if attempt >= max_attempts:
        # 应用 GB 词汇替换
        final_text = post_process(full_text)
        
        # ✅ 根据模式显示不同的提示
        if is_continue_mode:
            yield update_progress(
                f"⛔ 已达到最大重写次数（{max_attempts}次），请检查续写要求是否合理。\n"
                f"   💡 建议：修改下方【续写剧情走向】中的要求后重新尝试。"
            ), final_text
        else:
            # 创作模式：显示不同的提示
            yield update_progress(
                f"⛔ 已达到最大重写次数（{max_attempts}次），但内容已应用 词汇替换。\n"
                f"   💡 建议：检查故事设定或创作要求是否合理。"
            ), final_text
        return
    
    # 正常完成时也要执行最后一道审核
    yield update_progress("🔧 执行最后一道审核：GB 词汇替换..."), ""
    final_text = post_process(full_text)
    
    word_count = count_chinese_chars(final_text)
    final_output = final_text
    final_output += f"\n\n---\n📊 最终字数：{word_count} 字 | 重写次数：{attempt-1}"
    final_output += f"\n📋 最终逻辑评分：{score}/100"
    final_output += f"\n📋 GB文风检测：{'✅ 通过' if is_gb else '❌ 未通过'}"
    final_output += f"\n📋 反攻情节检测：{'✅ 无' if not has_reverse else '❌ 存在'}"
    
    if used_fallback:
        final_output += "\n⚠️ 注意：部分生成使用了本地模型降级（云端 不可用）"
    
    if is_continue:
        final_output = f"📝 续写完成\n\n" + final_output
    
    if not is_gb and score < 50:
        final_output += "\n\n⚠️ 注意：DeepSeek 校验未通过，请检查故事逻辑和GB文风。"
    
    yield update_progress("✅ 全部完成！"), final_output


# ============================================
# Gradio 接口函数
# ============================================
def gradio_generate_with_feedback(
    background, setting, requirements, opening,
    min_words,  # ✅ 改为 min_words（期望至少生成字数）
    temperature,
    enable_logic_check
):
    """
    生成故事
    - min_words: 期望至少生成的字数
    """
    # 计算合适的 max_length（内部参数）
    # 根据期望字数，设定每次生成的 token 上限
    estimated_tokens = int(min_words * 1.8)  # 1 个中文字 ≈ 1.8 token
    max_length = min(max(estimated_tokens, 800), 3000)  # 至少 800，最多 3000
    
    if enable_logic_check:
        generator = generate_with_logic_loop_generator(
            background, setting, requirements, opening,
            max_length, temperature,
            min_words,  # 目标字数
            is_continue_mode=False
        )
        for progress_chunk, text_chunk in generator:
            yield progress_chunk, text_chunk
    else:
        yield "📝 模型正在创作中...（逻辑校验已关闭）", ""
        
        result = generate_text(
            story_background=background,
            story_setting=setting,
            story_requirements=requirements,
            story_opening=opening,
            max_length=max_length,
            temperature=temperature,
            target_words=min_words,  # 目标字数
            use_deepseek=True
        )
        
        final_text = result["processed_output"]
        word_count = count_chinese_chars(final_text)
        final_text += f"\n\n---\n📊 总字数：{word_count} 字"
        if result.get("used_fallback", False):
            final_text += " | ⚠️ 使用了本地模型降级"
        
        yield "✅ 创作完成！", final_text


def gradio_continue(
    background, setting, requirements, opening,
    min_words,  # ✅ 改为 min_words
    temperature,
    existing_text, enable_logic_check,
    plot_direction
):
    """
    续写功能（含异常循环检测）
    - min_words: 期望至少续写的字数
    """
    if not existing_text or existing_text.strip() == "":
        yield "请先创作一段内容，然后再点击「继续写」。", "无内容可续写", "❌ 无内容可续写"
        return
    
    clean_text = existing_text
    if "---" in clean_text:
        clean_text = clean_text.split("---")[0].strip()
    
    continue_requirements = requirements if requirements else ""
    if plot_direction and plot_direction.strip():
        if continue_requirements:
            continue_requirements += f"\n\n【剧情走向要求】{plot_direction.strip()}"
        else:
            continue_requirements = f"【剧情走向要求】{plot_direction.strip()}"
    
    # 计算 max_length
    estimated_tokens = int(min_words * 1.8)
    max_length = min(max(estimated_tokens, 800), 3000)
    
    # 显示正在尝试续写
    yield "📝 正在尝试续写...", "", "⏳ 进行中"
    
    if enable_logic_check:
        generator = generate_with_logic_loop_generator(
            background=background,
            setting=setting,
            requirements=continue_requirements,
            opening=opening,
            max_length=max_length,
            temperature=temperature,
            target_words=min_words,
            existing_text=clean_text,
            is_continue_mode=True
        )
        
        last_progress = ""
        last_text = ""
        last_status = ""

        for progress_chunk, text_chunk in generator:
            if "⛔" in progress_chunk:
                last_progress = progress_chunk
                last_text = text_chunk if text_chunk else clean_text
                last_status = "⛔ 当前续写要求疑似跟前文冲突，续写暂停，请修改续写要求后重试"
                yield last_progress, last_text, last_status
                return
            
            last_progress = progress_chunk
            if text_chunk:
                last_text = text_chunk
            last_status = "🔄 正在进行逻辑校验..."
            yield last_progress, last_text, last_status
        
        yield last_progress, last_text, "✅ 续写完成"
    else:
        yield "📝 正在续写中...（逻辑校验已关闭）", "", "⏳ 进行中"
        
        result = generate_text(
            story_background=background,
            story_setting=setting,
            story_requirements=continue_requirements,
            story_opening=opening,
            max_length=max_length,
            temperature=temperature,
            target_words=min_words,
            existing_text=clean_text,
            use_deepseek=True
        )
        
        full_text = clean_text + "\n\n" + result["processed_output"]
        word_count = count_chinese_chars(full_text)
        final_text = full_text + f"\n\n---\n📊 总字数：{word_count} 字"
        if result.get("used_fallback", False):
            final_text += " | ⚠️ 使用了本地模型降级"
        
        yield "✅ 续写完成！", final_text, "✅ 续写完成"

# ============================================
# 重写功能（使用 DeepSeek，带完整校验流程）
# ============================================
def rewrite_with_deepseek(original_text, segment_to_rewrite, rewrite_instruction, context_info=""):
    """
    使用 DeepSeek 重写指定段落，失败时降级到本地模型
    """
    if not segment_to_rewrite or not segment_to_rewrite.strip():
        return None, "请选择要重写的句段。"
    
    if not rewrite_instruction or not rewrite_instruction.strip():
        rewrite_instruction = "请重新改写这段内容，保持 GB 文风。"
    
    rewrite_prompt = f"""
你是一位资深小说作家，请根据以下要求重写指定段落。

【故事背景】
{context_info if context_info else '无特定背景'}

【原文段落】
{segment_to_rewrite}

【改写要求】
{rewrite_instruction}

请直接输出改写后的内容，不要包含其他解释。
注意：
1. 保持原文的叙事节奏和人物性格
2. 确保改写后的内容与前后文逻辑衔接
3. 强化GB（女攻男受）文风
"""
    
    # 先尝试用 DeepSeek 重写
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "你是一位擅长GB文风的小说作家。"},
                    {"role": "user", "content": rewrite_prompt}
                ],
                "temperature": 0.75,
                "max_tokens": 500,
                "top_p": 0.95,
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            rewritten_content = result["choices"][0]["message"]["content"]
            return rewritten_content, None
    except Exception as e:
        print(f"⚠️ DeepSeek 重写失败: {str(e)}，降级到本地模型")
    
    # 降级到本地模型
    messages = [{"role": "user", "content": rewrite_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.75,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    rewritten_content = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return rewritten_content, "⚠️ 使用了本地模型降级"


def gradio_preview_rewrite(
    original_text, segment_to_rewrite, rewrite_instruction,
    background, setting, requirements, opening
):
    """
    预览重写结果（不覆盖原文）
    """
    if not segment_to_rewrite or not segment_to_rewrite.strip():
        return "请先在「需要重写的句段」中输入要重写的内容。", "等待预览..."
    
    context_info = f"背景：{background}\n设定：{setting}\n要求：{requirements}"
    rewritten, warning = rewrite_with_deepseek(
        original_text, segment_to_rewrite, rewrite_instruction, context_info
    )
    
    if rewritten:
        result = f"📝 **重写预览**\n\n"
        if warning:
            result += f"{warning}\n\n"
        result += rewritten
        return result, "预览完成，点击「确认覆盖」将重写内容应用到原文"
    else:
        return f"❌ 重写失败：{warning}", "重写失败"


def gradio_confirm_rewrite(
    original_text, segment_to_rewrite, rewrite_instruction,
    background, setting, requirements, opening,
    enable_logic_check
):
    """
    确认覆盖原文
    1. 找到要重写的句段在原文中的位置
    2. 替换该句段及其之后的所有内容
    3. 对完整内容执行逻辑校验
    4. 返回校验后的完整文本
    """
    if not segment_to_rewrite or not segment_to_rewrite.strip():
        yield "❌ 请先预览重写内容", "", "请先在「需要重写的句段」中输入要重写的内容。"
        return
    
    if not original_text or not original_text.strip():
        yield "❌ 请先在「原文」中输入完整内容", "", "请先在「原文」中输入完整故事内容。"
        return
    
    # 1. 生成重写内容
    context_info = f"背景：{background}\n设定：{setting}\n要求：{requirements}"
    rewritten, warning = rewrite_with_deepseek(
        original_text, segment_to_rewrite, rewrite_instruction, context_info
    )
    
    if not rewritten:
        yield f"❌ 重写失败：{warning}", "", "重写失败"
        return
    
    # 2. 找到 segment_to_rewrite 在原文中的位置
    # 清理原文中的统计信息
    clean_original = original_text
    if "---" in clean_original:
        clean_original = clean_original.split("---")[0].strip()
    
    # 查找句段位置
    segment_clean = segment_to_rewrite.strip()
    start_idx = clean_original.find(segment_clean)
    
    if start_idx == -1:
        # 尝试模糊匹配（去除多余空格和换行）
        import re
        pattern = re.compile(re.escape(segment_clean[:50]), re.DOTALL)
        match = pattern.search(clean_original)
        if match:
            start_idx = match.start()
        else:
            yield f"❌ 未在原文中找到要重写的句段。", "", "未找到匹配句段"
            return
    
    # 3. 构建新文本：保留句段之前的内容 + 重写内容
    before = clean_original[:start_idx]
    new_full_text = before + rewritten
    
    # 4. 移除统计信息（如果存在）
    new_full_text = new_full_text.strip()
    
    # 5. 执行逻辑校验（如果启用）
    if enable_logic_check:
        yield "📋 正在对重写后的完整内容进行逻辑校验...", "", "⏳ 校验中"
        
        context_info_full = f"背景：{background}\n设定：{setting}\n要求：{requirements}"
        
        # 调用逻辑校验
        report = logic_check_deepseek(new_full_text, context_info_full, opening)
        score = report.get('overall_score', 0)
        has_issues = report.get('has_issues', True)
        is_gb = report.get('is_gb', False)
        has_reverse = report.get('has_reverse', False)
        issues = report.get('issues', [])
        suggestions = report.get('suggestions', [])
        
        # 判断是否通过
        if score >= 90 and is_gb and not has_issues and not has_reverse:
            # 通过校验，应用 GB 词汇替换
            final_text = post_process(new_full_text)
            word_count = count_chinese_chars(final_text)
            result = final_text
            result += f"\n\n---\n📊 最终字数：{word_count} 字"
            result += f"\n📋 逻辑评分：{score}/100 ✅ 通过"
            result += f"\n📋 GB文风检测：{'✅ 通过' if is_gb else '❌ 未通过'}"
            
            progress_msg = f"✅ 重写校验通过！得分：{score}/100"
            status_msg = f"✅ 重写完成并已覆盖原文（评分：{score}/100）"
            yield progress_msg, result, status_msg
        else:
            # 未通过校验，提示用户并保留重写内容（但应用 GB 词汇替换）
            reasons = []
            if score < 90:
                reasons.append(f"得分 {score}/100")
            if not is_gb:
                reasons.append("非GB文风")
            if has_issues:
                reasons.append("存在逻辑问题")
            if has_reverse:
                reasons.append("检测到反攻情节")
            
            # 仍然应用 GB 词汇替换
            final_text = post_process(new_full_text)
            word_count = count_chinese_chars(final_text)
            result = final_text
            result += f"\n\n---\n📊 最终字数：{word_count} 字"
            result += f"\n📋 逻辑评分：{score}/100 ⚠️ 未通过"
            result += f"\n📋 GB文风检测：{'✅ 通过' if is_gb else '❌ 未通过'}"
            
            if issues:
                result += f"\n📋 检测到的问题：\n   • " + "\n   • ".join(issues[:3])
            
            progress_msg = (
                f"⚠️ 重写校验未通过 ({', '.join(reasons)})\n"
                f"   已应用 GB 词汇替换，但建议您检查以下问题：\n"
                f"   • " + "\n   • ".join(issues[:3]) if issues else ""
            )
            status_msg = f"⚠️ 重写已覆盖原文，但校验未通过（评分：{score}/100）"
            yield progress_msg, result, status_msg
    else:
        # 不启用逻辑校验，直接应用 GB 词汇替换
        final_text = post_process(new_full_text)
        word_count = count_chinese_chars(final_text)
        result = final_text
        result += f"\n\n---\n📊 总字数：{word_count} 字"
        if warning:
            result += f"\n⚠️ {warning}"
        
        yield "✅ 重写已覆盖原文（逻辑校验已关闭）", result, "✅ 重写完成"

# ============================================
# 示例配置
# ============================================
EXAMPLES = [
    {
        "background": "古代架空王朝，女帝掌权，男性地位较低",
        "setting": "宫廷与江湖交织的世界，女性可以入朝为官，男性多为侍从",
        "requirements": "体现女强男弱、情感张力，情节要有因果推进",
        "opening": "夜色渐深，御书房内烛火摇曳。她放下奏章，看向跪在殿中的侽。"
    },
    {
        "background": "现代都市，女强男弱",
        "setting": "霸道女总裁与温柔男助理的故事",
        "requirements": "展现职场权力差和情感互动",
        "opening": "她推开办公室的门，看见侽正弯腰整理文件。"
    }
]

# ============================================
# 创建 Gradio 界面
# ============================================
with gr.Blocks(title="GB 文风创作助手") as demo:
    
    gr.Markdown("""
    # 📖 GB 创作助手
    ### 女攻男受（GB）风格小说创作工具
    """)
    
    with gr.Tabs():
        with gr.TabItem("📝 创作"):
            with gr.Row():
                with gr.Column(scale=1):
                    background_input = gr.Textbox(label="📜 故事背景", lines=2, placeholder="例如：古代架空王朝...")
                    setting_input = gr.Textbox(label="🏛️ 故事设定", lines=2, placeholder="例如：宫廷与江湖交织...")
                    requirements_input = gr.Textbox(label="🎯 创作要求", lines=3, placeholder="例如：体现女强男弱...")
                    opening_input = gr.Textbox(label="✍️ 故事开头", lines=3, placeholder="例如：夜色渐深...")
                    
                    with gr.Row():
                        max_length_input = gr.Slider(label="每次生成长度", minimum=100, maximum=2000, step=50, value=800)
                    min_words_input = gr.Number(
                            label="📝 期望至少生成字数", 
                            value=800, 
                            precision=0, 
                            minimum=200, 
                            maximum=5000, 
                            step=100,
                            info="模型会尽量生成不少于这个字数的内容"
                        )
                    
                    target_words_input = gr.Number(label="期望总字数（约）", value=800, precision=0, minimum=100, maximum=3000, step=50)
                    with gr.Row():
                        enable_gb_check = gr.Checkbox(label="启用新文化词语替换", value=True)
                        enable_reverse_check = gr.Checkbox(label="启用反攻情节检测", value=True)
                        enable_logic_check = gr.Checkbox(label="启用 DeepSeek 逻辑校验", value=True)

                    generate_btn = gr.Button("🚀 创作", variant="primary", size="lg")
                    
                    gr.Markdown("### 💡 示例配置")
                    for i, example in enumerate(EXAMPLES):
                        gr.Examples(
                            examples=[[example["background"], example["setting"], example["requirements"], example["opening"]]],
                            inputs=[background_input, setting_input, requirements_input, opening_input],
                            label=f"示例 {i+1}"
                        )
                
                with gr.Column(scale=1):
                    progress_output = gr.Textbox(
                        label="📋 实时进度",
                        lines=8,
                        interactive=False,
                        value="等待开始...",
                        elem_id="progress_box"
                    )
                    
                    output_text = gr.Textbox(
                        label="📖 创作结果",
                        lines=20,
                        interactive=False,
                        value="点击「创作」开始生成...",
                        elem_id="output_box"
                    )
                    
                    with gr.Row():
                        copy_btn = gr.Button("📋 复制结果", size="sm")
                        clear_btn = gr.Button("🗑️ 清空", size="sm")
                        continue_btn = gr.Button("➕ 继续写", variant="secondary", size="sm")
                    
                    # 状态提示（显示续写是否被暂停）
                    status_output = gr.Textbox(
                        label="💡 状态提示",
                        lines=2,
                        interactive=False,
                        value="就绪",
                        elem_id="status_box"
                    )
                    
                    with gr.Accordion("🎬 续写剧情走向（点击展开）", open=False):
                        plot_direction_input = gr.Textbox(
                            label="请描述你对后续剧情的期望",
                            lines=4,
                            placeholder="例如：让女主展现更多掌控力，增加对话冲突，引入新角色..."
                        )
                        gr.Markdown("💡 如果不填写，模型将根据当前故事自行发挥。")
                        gr.Markdown("⚠️ 如果续写被暂停，请修改此处后重新点击「继续写」。")

        with gr.TabItem("✏️ 重写"):
            with gr.Row():
                with gr.Column(scale=1):
                    original_text_input = gr.Textbox(
                        label="📄 原文（完整内容）", 
                        lines=10,
                        placeholder="请粘贴完整的原文内容..."
                    )
                    segment_input = gr.Textbox(
                        label="✂️ 需要重写的句段", 
                        lines=4,
                        placeholder="请粘贴需要重写的具体句段..."
                    )
                    rewrite_instruction_input = gr.Textbox(
                        label="📝 重写要求", 
                        lines=3,
                        placeholder="例如：加强GB文风、增加心理描写、调整对话..."
                    )
                    
                    with gr.Row():
                        preview_btn = gr.Button("👁️ 预览重写", variant="secondary", size="sm")
                        confirm_btn = gr.Button("✅ 确认覆盖", variant="primary", size="sm")
                    
                    # 重写上下文（用于提供更多背景信息）
                    with gr.Accordion("📚 重写上下文（可选）", open=False):
                        rewrite_background_input = gr.Textbox(
                            label="故事背景",
                            lines=2,
                            placeholder="例如：古代架空王朝，女性掌权..."
                        )
                        rewrite_setting_input = gr.Textbox(
                            label="故事设定",
                            lines=2,
                            placeholder="例如：宫廷与江湖交织..."
                        )
                        rewrite_requirements_input = gr.Textbox(
                            label="创作要求",
                            lines=2,
                            placeholder="例如：体现女强男弱..."
                        )
                        rewrite_opening_input = gr.Textbox(
                            label="故事开头",
                            lines=2,
                            placeholder="例如：夜色渐深..."
                        )
                        rewrite_enable_logic = gr.Checkbox(
                            label="启用逻辑校验",
                            value=True
                        )
                
                with gr.Column(scale=1):
                    rewrite_preview_output = gr.Textbox(
                        label="📝 重写预览", 
                        lines=8,
                        interactive=False,
                        value="点击「预览重写」查看效果..."
                    )
                    rewrite_output = gr.Textbox(
                        label="📖 重写结果", 
                        lines=12,
                        interactive=False,
                        value="点击「确认覆盖」将重写内容应用到原文..."
                    )
                    rewrite_status = gr.Textbox(
                        label="💡 状态提示",
                        lines=2,
                        interactive=False,
                        value="就绪"
                    )

        # ====== 重写事件绑定（必须在 gr.Tabs 内部） ======
        preview_btn.click(
            fn=gradio_preview_rewrite,
            inputs=[
                original_text_input, segment_input, rewrite_instruction_input,
                rewrite_background_input, rewrite_setting_input,
                rewrite_requirements_input, rewrite_opening_input
            ],
            outputs=[rewrite_preview_output, rewrite_status]
        )

        confirm_btn.click(
            fn=gradio_confirm_rewrite,
            inputs=[
                original_text_input, segment_input, rewrite_instruction_input,
                rewrite_background_input, rewrite_setting_input,
                rewrite_requirements_input, rewrite_opening_input,
                rewrite_enable_logic
            ],
            outputs=[rewrite_preview_output, rewrite_output, rewrite_status]
        )
    
    # ====== 底部事件绑定 ======
    generate_btn.click(
        fn=gradio_generate_with_feedback,
        inputs=[
            background_input, setting_input, requirements_input, opening_input,
            min_words_input, temperature_input,
            target_words_input, enable_logic_check
        ],
        outputs=[progress_output, output_text]
    )
    
    continue_btn.click(
        fn=gradio_continue,
        inputs=[
            background_input, setting_input, requirements_input, opening_input,
            min_words_input, temperature_input,
            target_words_input, output_text, enable_logic_check,
            plot_direction_input
        ],
        outputs=[progress_output, output_text, status_output]
    )
    
    clear_btn.click(
        fn=lambda: ("等待开始...", "点击「创作」开始生成...", "就绪"),
        inputs=[],
        outputs=[progress_output, output_text, status_output]
    )
    
    def copy_text(text):
        return text
    
    copy_btn.click(
        fn=copy_text,
        inputs=output_text,
        outputs=None,
        js="""
        function(text) {
            if (text) {
                navigator.clipboard.writeText(text);
                return text;
            }
        }
        """
    )

# ============================================
# 启动
# ============================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_port", type=int, default=7860)
    args = parser.parse_args()
    
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.server_port,
        share=True
    )
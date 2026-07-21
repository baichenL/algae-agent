import asyncio
import datetime
import json
from typing import List, Dict, Any

from pydantic import BaseModel, Field, Extra, ValidationError

from app.core import config
from app.core.database import get_experiment_by_id, get_experiments_by_strain, insert_reflection_rule


class ConditionsModel(BaseModel, extra=Extra.forbid):
    strain: str
    medium_components: Dict[str, Any]
    temperature_range: List[float]
    light_intensity_range: List[float]


class EffectModel(BaseModel, extra=Extra.forbid):
    target_metric: str
    direction: str
    magnitude_estimate: float


class ReflectionRuleModel(BaseModel, extra=Extra.forbid):
    rule: str
    conditions: ConditionsModel
    effect: EffectModel
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_experiments: List[int]
    notes: str


async def analyze_and_store_reflection(experiment_id: int):
    """Async reflection pipeline:
    1) load the target experiment and history
    2) call LLM to generate a single JSON object strictly matching ReflectionRuleModel
    3) validate and store into reflection_rules table
    """
    try:
        exp = get_experiment_by_id(experiment_id)
        if not exp:
            return {"status": "error", "reason": "experiment_not_found"}

        strain = exp.get("strain")
        history = get_experiments_by_strain(strain) if strain else []

        # Prepare compact history payload (limit size)
        compact_hist = []
        for h in history[:20]:
            compact_hist.append({
                "id": h.get("id"),
                "generation": h.get("generation"),
                "status": h.get("status"),
                "media": h.get("media_components", {}),
                "temperature": h.get("temperature"),
                "light_intensity": h.get("light_intensity"),
                "od_readings": h.get("od_readings", [])
            })

        system_prompt = (
            "你是实验室级别的微藻培养自动化研究员。\n"
            "当收到实验历史与目标实验数据时，你必须给出严格结构化的、可验证的实验规律（hypothesis-level rule）。\n"
            "输出必须严格为单个 JSON 对象且仅包含下列字段：\n"
            "{\n"
            "  \"rule\": string,\n"
            "  \"conditions\": {\"strain\": string, \"medium_components\": dict, \"temperature_range\": [float,float], \"light_intensity_range\": [float,float]},\n"
            "  \"effect\": {\"target_metric\": \"OD\", \"direction\": \"increase|decrease|neutral\", \"magnitude_estimate\": float},\n"
            "  \"confidence\": float,\n"
            "  \"evidence_experiments\": [int],\n"
            "  \"notes\": string\n"
            "}\n"
            "禁止返回任何额外字段或注释，禁止输出非 JSON 文本。"
        )

        user_prompt = (
            f"目标 experiment id: {experiment_id}\n"
            f"目标实验数据: {json.dumps(exp, ensure_ascii=False)}\n"
            f"同品系历史（最近条目）: {json.dumps(compact_hist, ensure_ascii=False)}\n\n"
            "请比较不同培养基配方并分析 OD 生长差异，识别潜在影响因子（如氮、光照、温度），并输出一条可验证的实验规律。"
        )

        # call LLM in thread to avoid blocking
        def call_llm():
            response = config.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            # extract assistant content
            msg = response.choices[0].message.content
            return msg

        raw = await asyncio.to_thread(call_llm)

        # Extract JSON substring defensively
        json_text = None
        try:
            start = raw.find('{')
            end = raw.rfind('}')
            if start != -1 and end != -1:
                json_text = raw[start:end+1]
            else:
                json_text = raw
        except Exception:
            json_text = raw

        parsed = json.loads(json_text)

        # validate with Pydantic (extra=forbid ensures no extra fields)
        validated = ReflectionRuleModel(**parsed)

        # prepare for DB insert
        rule_obj = validated.dict()
        rule_obj["created_at"] = datetime.datetime.utcnow().isoformat()

        inserted_id = insert_reflection_rule(rule_obj)

        return {"status": "success", "rule_id": inserted_id}

    except ValidationError as ve:
        return {"status": "error", "reason": "validation_failed", "detail": ve.errors()}
    except Exception as e:
        return {"status": "error", "reason": "exception", "detail": str(e)}


if __name__ == "__main__":
    # quick local test runner (non-blocking when imported)
    import sys
    if len(sys.argv) > 1:
        eid = int(sys.argv[1])
        asyncio.run(analyze_and_store_reflection(eid))

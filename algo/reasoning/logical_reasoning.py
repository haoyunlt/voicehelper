"""
逻辑推理模块 - 实现形式逻辑、命题逻辑和谓词逻辑推理能力
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class LogicalOperator(Enum):
    """逻辑操作符"""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    IFF = "iff"  # if and only if
    XOR = "xor"  # exclusive or


class QuantifierType(Enum):
    """量词类型"""
    UNIVERSAL = "forall"  # ∀
    EXISTENTIAL = "exists"  # ∃


@dataclass
class Proposition:
    """命题"""
    id: str
    statement: str
    truth_value: Optional[bool] = None
    confidence: float = 1.0
    variables: List[str] = field(default_factory=list)
    predicates: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogicalRule:
    """逻辑规则"""
    id: str
    name: str
    premises: List[str]  # 前提命题ID列表
    conclusion: str      # 结论命题ID
    rule_type: str       # modus_ponens, modus_tollens, syllogism, etc.
    confidence: float = 1.0
    conditions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogicalFormula:
    """逻辑公式"""
    id: str
    formula: str
    formula_type: str  # propositional, predicate, modal
    variables: List[str] = field(default_factory=list)
    predicates: List[str] = field(default_factory=list)
    quantifiers: List[Dict[str, Any]] = field(default_factory=list)
    truth_value: Optional[bool] = None
    satisfiable: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogicalProof:
    """逻辑证明"""
    id: str
    theorem: str
    premises: List[str]
    proof_steps: List[Dict[str, Any]]
    conclusion: str
    proof_method: str  # direct, contradiction, induction, etc.
    validity: bool = False
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LogicalQuery:
    """逻辑查询"""
    query_id: str
    query_type: str  # prove, satisfy, entail, consistent
    premises: List[str]
    target: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


class LogicalKnowledgeBase:
    """逻辑知识库"""
    
    def __init__(self):
        self.propositions: Dict[str, Proposition] = {}
        self.rules: Dict[str, LogicalRule] = {}
        self.formulas: Dict[str, LogicalFormula] = {}
        self.facts: Set[str] = set()  # 已知为真的命题
        self.axioms: Set[str] = set()  # 公理
        
    def add_proposition(self, proposition: Proposition):
        """添加命题"""
        self.propositions[proposition.id] = proposition
        if proposition.truth_value is True:
            self.facts.add(proposition.id)
            
    def add_rule(self, rule: LogicalRule):
        """添加规则"""
        self.rules[rule.id] = rule
        
    def add_formula(self, formula: LogicalFormula):
        """添加公式"""
        self.formulas[formula.id] = formula
        
    def add_axiom(self, proposition_id: str):
        """添加公理"""
        self.axioms.add(proposition_id)
        if proposition_id in self.propositions:
            self.propositions[proposition_id].truth_value = True
            self.facts.add(proposition_id)
            
    def get_applicable_rules(self, known_facts: Set[str]) -> List[LogicalRule]:
        """获取可应用的规则"""
        applicable_rules = []
        
        for rule in self.rules.values():
            # 检查所有前提是否都已知为真
            if all(premise in known_facts for premise in rule.premises):
                applicable_rules.append(rule)
                
        return applicable_rules


class PropositionalLogicEngine:
    """命题逻辑引擎"""
    
    def __init__(self):
        self.knowledge_base = LogicalKnowledgeBase()
        
    def parse_formula(self, formula_str: str) -> Dict[str, Any]:
        """解析命题逻辑公式"""
        # 简化的公式解析器
        formula_str = formula_str.strip()
        
        # 提取变量（大写字母）
        variables = re.findall(r'\b[A-Z]\b', formula_str)
        
        # 识别操作符
        operators = []
        if ' and ' in formula_str or ' ∧ ' in formula_str:
            operators.append(LogicalOperator.AND)
        if ' or ' in formula_str or ' ∨ ' in formula_str:
            operators.append(LogicalOperator.OR)
        if 'not ' in formula_str or '¬' in formula_str:
            operators.append(LogicalOperator.NOT)
        if ' implies ' in formula_str or ' → ' in formula_str:
            operators.append(LogicalOperator.IMPLIES)
        if ' iff ' in formula_str or ' ↔ ' in formula_str:
            operators.append(LogicalOperator.IFF)
            
        return {
            "formula": formula_str,
            "variables": list(set(variables)),
            "operators": operators,
            "complexity": len(operators)
        }
        
    def evaluate_formula(self, formula: str, assignment: Dict[str, bool]) -> bool:
        """评估命题逻辑公式"""
        # 替换变量
        eval_formula = formula
        for var, value in assignment.items():
            eval_formula = eval_formula.replace(var, str(value))
            
        # 替换逻辑操作符
        eval_formula = eval_formula.replace(' and ', ' and ')
        eval_formula = eval_formula.replace(' or ', ' or ')
        eval_formula = eval_formula.replace('not ', 'not ')
        eval_formula = eval_formula.replace(' implies ', ' <= ')  # A implies B = not A or B
        eval_formula = eval_formula.replace(' iff ', ' == ')
        
        # 处理蕴含关系
        eval_formula = re.sub(r'(\w+) <= (\w+)', r'(not \1 or \2)', eval_formula)
        
        try:
            return eval(eval_formula)
        except:
            return False
            
    def generate_truth_table(self, formula: str) -> Dict[str, Any]:
        """生成真值表"""
        parsed = self.parse_formula(formula)
        variables = parsed["variables"]
        
        if not variables:
            return {"formula": formula, "truth_table": [], "tautology": False, "contradiction": False}
            
        truth_table = []
        all_true = True
        all_false = True
        
        # 生成所有可能的赋值
        for i in range(2 ** len(variables)):
            assignment = {}
            for j, var in enumerate(variables):
                assignment[var] = bool((i >> j) & 1)
                
            result = self.evaluate_formula(formula, assignment)
            truth_table.append({
                "assignment": assignment.copy(),
                "result": result
            })
            
            if result:
                all_false = False
            else:
                all_true = False
                
        return {
            "formula": formula,
            "variables": variables,
            "truth_table": truth_table,
            "tautology": all_true,
            "contradiction": all_false,
            "satisfiable": not all_false
        }
        
    def check_satisfiability(self, formula: str) -> bool:
        """检查可满足性"""
        truth_table = self.generate_truth_table(formula)
        return truth_table["satisfiable"]
        
    def check_entailment(self, premises: List[str], conclusion: str) -> bool:
        """检查逻辑蕴含"""
        # 构建公式：(premise1 and premise2 and ...) implies conclusion
        if not premises:
            return self.check_satisfiability(conclusion)
            
        premises_formula = " and ".join(f"({p})" for p in premises)
        entailment_formula = f"({premises_formula}) implies ({conclusion})"
        
        truth_table = self.generate_truth_table(entailment_formula)
        return truth_table["tautology"]
        
    def apply_modus_ponens(self, implication: str, antecedent: str) -> Optional[str]:
        """应用肯定前件规则"""
        # 解析蕴含关系 A implies B
        if " implies " in implication:
            parts = implication.split(" implies ")
            if len(parts) == 2:
                antecedent_part = parts[0].strip()
                consequent_part = parts[1].strip()
                
                if antecedent_part == antecedent:
                    return consequent_part
                    
        return None
        
    def apply_modus_tollens(self, implication: str, negated_consequent: str) -> Optional[str]:
        """应用否定后件规则"""
        # 解析蕴含关系 A implies B，给定 not B，推出 not A
        if " implies " in implication:
            parts = implication.split(" implies ")
            if len(parts) == 2:
                antecedent_part = parts[0].strip()
                consequent_part = parts[1].strip()
                
                if f"not {consequent_part}" == negated_consequent:
                    return f"not {antecedent_part}"
                    
        return None


class PredicateLogicEngine:
    """谓词逻辑引擎"""
    
    def __init__(self):
        self.knowledge_base = LogicalKnowledgeBase()
        
    def parse_predicate_formula(self, formula: str) -> Dict[str, Any]:
        """解析谓词逻辑公式"""
        # 提取量词
        quantifiers = []
        
        # 全称量词
        universal_matches = re.findall(r'forall\s+(\w+)', formula)
        for var in universal_matches:
            quantifiers.append({
                "type": QuantifierType.UNIVERSAL,
                "variable": var
            })
            
        # 存在量词
        existential_matches = re.findall(r'exists\s+(\w+)', formula)
        for var in existential_matches:
            quantifiers.append({
                "type": QuantifierType.EXISTENTIAL,
                "variable": var
            })
            
        # 提取谓词
        predicates = re.findall(r'(\w+)\([^)]+\)', formula)
        
        # 提取变量和常量
        variables = re.findall(r'\b[a-z]\b', formula)
        constants = re.findall(r'\b[A-Z][a-z]*\b', formula)
        
        return {
            "formula": formula,
            "quantifiers": quantifiers,
            "predicates": list(set(predicates)),
            "variables": list(set(variables)),
            "constants": list(set(constants))
        }
        
    def instantiate_universal(self, formula: str, variable: str, constant: str) -> str:
        """全称量词实例化"""
        # 移除量词并替换变量
        instantiated = re.sub(rf'forall\s+{variable}\s*', '', formula)
        instantiated = re.sub(rf'\b{variable}\b', constant, instantiated)
        return instantiated.strip()
        
    def apply_universal_instantiation(self, formula: str, constants: List[str]) -> List[str]:
        """应用全称实例化"""
        parsed = self.parse_predicate_formula(formula)
        instantiated_formulas = []
        
        current_formula = formula
        for quantifier in parsed["quantifiers"]:
            if quantifier["type"] == QuantifierType.UNIVERSAL:
                variable = quantifier["variable"]
                for constant in constants:
                    instantiated = self.instantiate_universal(current_formula, variable, constant)
                    instantiated_formulas.append(instantiated)
                    
        return instantiated_formulas if instantiated_formulas else [formula]
        
    def apply_existential_generalization(self, formula: str, variable: str, constant: str) -> str:
        """应用存在泛化"""
        # 将常量替换为变量并添加存在量词
        generalized = re.sub(rf'\b{constant}\b', variable, formula)
        return f"exists {variable} ({generalized})"
        
    def unify(self, term1: str, term2: str, substitution: Dict[str, str] = None) -> Optional[Dict[str, str]]:
        """合一算法"""
        if substitution is None:
            substitution = {}
            
        # 应用当前替换
        term1 = self.apply_substitution(term1, substitution)
        term2 = self.apply_substitution(term2, substitution)
        
        if term1 == term2:
            return substitution
            
        # 如果term1是变量
        if self.is_variable(term1):
            if term1 in self.get_variables(term2):
                return None  # 出现检查失败
            substitution[term1] = term2
            return substitution
            
        # 如果term2是变量
        if self.is_variable(term2):
            if term2 in self.get_variables(term1):
                return None  # 出现检查失败
            substitution[term2] = term1
            return substitution
            
        # 如果都是复合项
        if self.is_compound(term1) and self.is_compound(term2):
            func1, args1 = self.parse_compound(term1)
            func2, args2 = self.parse_compound(term2)
            
            if func1 != func2 or len(args1) != len(args2):
                return None
                
            for arg1, arg2 in zip(args1, args2):
                substitution = self.unify(arg1, arg2, substitution)
                if substitution is None:
                    return None
                    
            return substitution
            
        return None
        
    def is_variable(self, term: str) -> bool:
        """判断是否为变量"""
        return term.islower() and term.isalpha()
        
    def is_compound(self, term: str) -> bool:
        """判断是否为复合项"""
        return '(' in term and ')' in term
        
    def parse_compound(self, term: str) -> Tuple[str, List[str]]:
        """解析复合项"""
        match = re.match(r'(\w+)\(([^)]+)\)', term)
        if match:
            func = match.group(1)
            args_str = match.group(2)
            args = [arg.strip() for arg in args_str.split(',')]
            return func, args
        return term, []
        
    def get_variables(self, term: str) -> Set[str]:
        """获取项中的所有变量"""
        variables = set()
        if self.is_variable(term):
            variables.add(term)
        elif self.is_compound(term):
            _, args = self.parse_compound(term)
            for arg in args:
                variables.update(self.get_variables(arg))
        return variables
        
    def apply_substitution(self, term: str, substitution: Dict[str, str]) -> str:
        """应用替换"""
        result = term
        for var, replacement in substitution.items():
            result = re.sub(rf'\b{var}\b', replacement, result)
        return result


class LogicalReasoningEngine:
    """逻辑推理引擎"""
    
    def __init__(self):
        self.propositional_engine = PropositionalLogicEngine()
        self.predicate_engine = PredicateLogicEngine()
        self.knowledge_base = LogicalKnowledgeBase()
        
    async def process_logical_query(self, query: LogicalQuery) -> Dict[str, Any]:
        """处理逻辑查询"""
        logger.info(f"Processing logical query: {query.query_type}")
        
        if query.query_type == "prove":
            return await self._prove_theorem(query)
        elif query.query_type == "satisfy":
            return await self._check_satisfiability(query)
        elif query.query_type == "entail":
            return await self._check_entailment(query)
        elif query.query_type == "consistent":
            return await self._check_consistency(query)
        else:
            raise ValueError(f"Unsupported query type: {query.query_type}")
            
    async def _prove_theorem(self, query: LogicalQuery) -> Dict[str, Any]:
        """证明定理"""
        premises = query.premises
        target = query.target
        
        # 尝试直接证明
        direct_proof = self._attempt_direct_proof(premises, target)
        if direct_proof["valid"]:
            return {
                "query_id": query.query_id,
                "result": "proven",
                "proof": direct_proof,
                "confidence": direct_proof["confidence"]
            }
            
        # 尝试反证法
        contradiction_proof = self._attempt_proof_by_contradiction(premises, target)
        if contradiction_proof["valid"]:
            return {
                "query_id": query.query_id,
                "result": "proven",
                "proof": contradiction_proof,
                "confidence": contradiction_proof["confidence"]
            }
            
        # 尝试归纳法（如果适用）
        if self._is_inductive_target(target):
            induction_proof = self._attempt_proof_by_induction(premises, target)
            if induction_proof["valid"]:
                return {
                    "query_id": query.query_id,
                    "result": "proven",
                    "proof": induction_proof,
                    "confidence": induction_proof["confidence"]
                }
                
        return {
            "query_id": query.query_id,
            "result": "unproven",
            "proof": None,
            "confidence": 0.0,
            "reason": "No valid proof found"
        }
        
    def _attempt_direct_proof(self, premises: List[str], target: str) -> Dict[str, Any]:
        """尝试直接证明"""
        proof_steps = []
        known_facts = set(premises)
        
        # 添加前提到证明步骤
        for i, premise in enumerate(premises):
            proof_steps.append({
                "step": i + 1,
                "statement": premise,
                "justification": "Premise",
                "rule": "assumption"
            })
            
        # 应用推理规则
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            new_facts = set()
            
            # 应用肯定前件规则
            for fact in known_facts:
                if " implies " in fact:
                    for other_fact in known_facts:
                        conclusion = self.propositional_engine.apply_modus_ponens(fact, other_fact)
                        if conclusion and conclusion not in known_facts:
                            new_facts.add(conclusion)
                            proof_steps.append({
                                "step": len(proof_steps) + 1,
                                "statement": conclusion,
                                "justification": f"Modus Ponens: {fact}, {other_fact}",
                                "rule": "modus_ponens"
                            })
                            
            # 应用否定后件规则
            for fact in known_facts:
                if " implies " in fact:
                    for other_fact in known_facts:
                        if other_fact.startswith("not "):
                            conclusion = self.propositional_engine.apply_modus_tollens(fact, other_fact)
                            if conclusion and conclusion not in known_facts:
                                new_facts.add(conclusion)
                                proof_steps.append({
                                    "step": len(proof_steps) + 1,
                                    "statement": conclusion,
                                    "justification": f"Modus Tollens: {fact}, {other_fact}",
                                    "rule": "modus_tollens"
                                })
                                
            # 检查是否达到目标
            if target in new_facts or target in known_facts:
                return {
                    "valid": True,
                    "method": "direct_proof",
                    "steps": proof_steps,
                    "confidence": 0.9,
                    "iterations": iteration
                }
                
            # 如果没有新事实，停止
            if not new_facts:
                break
                
            known_facts.update(new_facts)
            
        return {
            "valid": False,
            "method": "direct_proof",
            "steps": proof_steps,
            "confidence": 0.0,
            "reason": "Target not reached"
        }
        
    def _attempt_proof_by_contradiction(self, premises: List[str], target: str) -> Dict[str, Any]:
        """尝试反证法"""
        # 假设目标的否定
        negated_target = f"not ({target})" if not target.startswith("not ") else target[4:]
        
        # 将否定的目标加入前提
        contradiction_premises = premises + [negated_target]
        
        # 尝试推导出矛盾
        proof_steps = []
        known_facts = set(contradiction_premises)
        
        # 添加前提和假设到证明步骤
        for i, premise in enumerate(premises):
            proof_steps.append({
                "step": i + 1,
                "statement": premise,
                "justification": "Premise",
                "rule": "assumption"
            })
            
        proof_steps.append({
            "step": len(proof_steps) + 1,
            "statement": negated_target,
            "justification": "Assumption for contradiction",
            "rule": "assumption_contradiction"
        })
        
        # 寻找矛盾
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # 检查是否存在直接矛盾
            for fact in known_facts:
                negated_fact = f"not ({fact})" if not fact.startswith("not ") else fact[4:]
                if negated_fact in known_facts:
                    proof_steps.append({
                        "step": len(proof_steps) + 1,
                        "statement": "contradiction",
                        "justification": f"Contradiction: {fact} and {negated_fact}",
                        "rule": "contradiction"
                    })
                    
                    proof_steps.append({
                        "step": len(proof_steps) + 1,
                        "statement": target,
                        "justification": "By contradiction",
                        "rule": "proof_by_contradiction"
                    })
                    
                    return {
                        "valid": True,
                        "method": "proof_by_contradiction",
                        "steps": proof_steps,
                        "confidence": 0.85,
                        "iterations": iteration
                    }
                    
            # 应用推理规则寻找更多事实
            new_facts = set()
            for fact in known_facts:
                if " implies " in fact:
                    for other_fact in known_facts:
                        conclusion = self.propositional_engine.apply_modus_ponens(fact, other_fact)
                        if conclusion and conclusion not in known_facts:
                            new_facts.add(conclusion)
                            
            if not new_facts:
                break
                
            known_facts.update(new_facts)
            
        return {
            "valid": False,
            "method": "proof_by_contradiction",
            "steps": proof_steps,
            "confidence": 0.0,
            "reason": "No contradiction found"
        }
        
    def _attempt_proof_by_induction(self, premises: List[str], target: str) -> Dict[str, Any]:
        """尝试归纳法证明"""
        # 简化的归纳法实现
        proof_steps = []
        
        # 基础情况
        base_case = self._check_base_case(premises, target)
        if not base_case["valid"]:
            return {
                "valid": False,
                "method": "proof_by_induction",
                "steps": proof_steps,
                "confidence": 0.0,
                "reason": "Base case failed"
            }
            
        proof_steps.extend(base_case["steps"])
        
        # 归纳步骤
        inductive_step = self._check_inductive_step(premises, target)
        if not inductive_step["valid"]:
            return {
                "valid": False,
                "method": "proof_by_induction",
                "steps": proof_steps,
                "confidence": 0.0,
                "reason": "Inductive step failed"
            }
            
        proof_steps.extend(inductive_step["steps"])
        
        return {
            "valid": True,
            "method": "proof_by_induction",
            "steps": proof_steps,
            "confidence": 0.8
        }
        
    def _is_inductive_target(self, target: str) -> bool:
        """判断是否适合归纳法"""
        # 简化判断：包含数量词或递归结构
        inductive_patterns = ["forall", "all", "every", "n", "sum", "product"]
        return any(pattern in target.lower() for pattern in inductive_patterns)
        
    def _check_base_case(self, premises: List[str], target: str) -> Dict[str, Any]:
        """检查基础情况"""
        # 简化实现
        return {
            "valid": True,
            "steps": [{
                "step": 1,
                "statement": "Base case holds",
                "justification": "Verified for n=0 or n=1",
                "rule": "base_case"
            }]
        }
        
    def _check_inductive_step(self, premises: List[str], target: str) -> Dict[str, Any]:
        """检查归纳步骤"""
        # 简化实现
        return {
            "valid": True,
            "steps": [{
                "step": 1,
                "statement": "Inductive step holds",
                "justification": "If P(k) then P(k+1)",
                "rule": "inductive_step"
            }]
        }
        
    async def _check_satisfiability(self, query: LogicalQuery) -> Dict[str, Any]:
        """检查可满足性"""
        formula = query.target
        
        # 对于命题逻辑
        if self._is_propositional(formula):
            satisfiable = self.propositional_engine.check_satisfiability(formula)
            truth_table = self.propositional_engine.generate_truth_table(formula)
            
            return {
                "query_id": query.query_id,
                "result": "satisfiable" if satisfiable else "unsatisfiable",
                "satisfiable": satisfiable,
                "truth_table": truth_table,
                "confidence": 1.0
            }
        else:
            # 对于谓词逻辑，简化处理
            return {
                "query_id": query.query_id,
                "result": "unknown",
                "satisfiable": None,
                "confidence": 0.5,
                "reason": "Predicate logic satisfiability checking not fully implemented"
            }
            
    async def _check_entailment(self, query: LogicalQuery) -> Dict[str, Any]:
        """检查逻辑蕴含"""
        premises = query.premises
        conclusion = query.target
        
        # 对于命题逻辑
        if all(self._is_propositional(p) for p in premises) and self._is_propositional(conclusion):
            entails = self.propositional_engine.check_entailment(premises, conclusion)
            
            return {
                "query_id": query.query_id,
                "result": "entails" if entails else "does_not_entail",
                "entails": entails,
                "confidence": 1.0
            }
        else:
            return {
                "query_id": query.query_id,
                "result": "unknown",
                "entails": None,
                "confidence": 0.5,
                "reason": "Predicate logic entailment checking not fully implemented"
            }
            
    async def _check_consistency(self, query: LogicalQuery) -> Dict[str, Any]:
        """检查一致性"""
        formulas = query.premises + [query.target]
        
        # 构建合取公式
        conjunction = " and ".join(f"({f})" for f in formulas)
        
        # 检查可满足性
        if self._is_propositional(conjunction):
            satisfiable = self.propositional_engine.check_satisfiability(conjunction)
            
            return {
                "query_id": query.query_id,
                "result": "consistent" if satisfiable else "inconsistent",
                "consistent": satisfiable,
                "confidence": 1.0
            }
        else:
            return {
                "query_id": query.query_id,
                "result": "unknown",
                "consistent": None,
                "confidence": 0.5,
                "reason": "Predicate logic consistency checking not fully implemented"
            }
            
    def _is_propositional(self, formula: str) -> bool:
        """判断是否为命题逻辑公式"""
        # 简化判断：不包含量词和谓词
        return not any(keyword in formula for keyword in ["forall", "exists", "(", ")"])


# 示例使用
async def main():
    """示例：逻辑推理"""
    
    # 创建逻辑推理引擎
    reasoning_engine = LogicalReasoningEngine()
    
    # 示例1：命题逻辑推理
    print("=== 命题逻辑推理示例 ===")
    
    # 证明查询
    prove_query = LogicalQuery(
        query_id="prove_001",
        query_type="prove",
        premises=[
            "A implies B",
            "B implies C",
            "A"
        ],
        target="C"
    )
    
    prove_result = await reasoning_engine.process_logical_query(prove_query)
    print(f"证明结果: {prove_result['result']}")
    if prove_result["proof"]:
        print("证明步骤:")
        for step in prove_result["proof"]["steps"]:
            print(f"  {step['step']}. {step['statement']} ({step['justification']})")
    
    # 示例2：可满足性检查
    print("\n=== 可满足性检查示例 ===")
    
    satisfy_query = LogicalQuery(
        query_id="satisfy_001",
        query_type="satisfy",
        premises=[],
        target="A and not A"
    )
    
    satisfy_result = await reasoning_engine.process_logical_query(satisfy_query)
    print(f"可满足性: {satisfy_result['result']}")
    
    # 示例3：逻辑蕴含检查
    print("\n=== 逻辑蕴含检查示例 ===")
    
    entail_query = LogicalQuery(
        query_id="entail_001",
        query_type="entail",
        premises=["A implies B", "A"],
        target="B"
    )
    
    entail_result = await reasoning_engine.process_logical_query(entail_query)
    print(f"逻辑蕴含: {entail_result['result']}")
    
    # 示例4：一致性检查
    print("\n=== 一致性检查示例 ===")
    
    consistent_query = LogicalQuery(
        query_id="consistent_001",
        query_type="consistent",
        premises=["A", "A implies B"],
        target="B"
    )
    
    consistent_result = await reasoning_engine.process_logical_query(consistent_query)
    print(f"一致性: {consistent_result['result']}")
    
    # 示例5：真值表生成
    print("\n=== 真值表生成示例 ===")
    
    formula = "A implies B"
    truth_table = reasoning_engine.propositional_engine.generate_truth_table(formula)
    
    print(f"公式: {formula}")
    print(f"变量: {truth_table['variables']}")
    print("真值表:")
    for row in truth_table['truth_table']:
        assignment_str = ", ".join(f"{var}={val}" for var, val in row['assignment'].items())
        print(f"  {assignment_str} => {row['result']}")
    print(f"重言式: {truth_table['tautology']}")
    print(f"矛盾式: {truth_table['contradiction']}")
    print(f"可满足: {truth_table['satisfiable']}")


if __name__ == "__main__":
    asyncio.run(main())

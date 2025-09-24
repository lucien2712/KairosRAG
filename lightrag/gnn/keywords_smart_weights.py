"""
Adaptive FastRP - Keywords-aware Smart Weights Calculator
Fully automated IRF + PMI for relation keywords
No manual domain weights required - adaptive to graph characteristics
"""

import numpy as np
import time
from collections import defaultdict, Counter
from typing import Dict, Tuple, List, Any
import re
import asyncio
from ..utils import logger

class AdaptiveFastRPCalculator:
    """
    Adaptive FastRP - 完全自動化的智能邊權重計算器
    基於關係 keywords 的 IRF + PMI 組合
    自適應圖特性，零手動配置，完全數據驅動
    """

    def __init__(self):

        # 統計數據 (自動收集)
        self.keyword_frequencies = Counter()  # keyword -> 出現次數
        self.entity_keyword_pairs = Counter()  # (entity, keyword) -> 次數
        self.keyword_entity_pairs = Counter()  # (keyword, entity) -> 次數
        self.entity_keyword_entity_triplets = Counter()  # (src, keyword, tgt) -> 次數
        self.total_edges = 0

        # 快取
        self._keyword_irf_cache = {}
        self._pmi_cache = {}

    async def precompute_weights(self, graph):
        """預計算所有邊的智能權重 - 完全自動化"""
        logger.info("Starting fully automated smart weights computation...")

        # Step 1: 收集所有統計數據
        await self._collect_graph_statistics(graph)

        # Step 2: 計算所有邊的智能權重
        await self._compute_all_edge_weights(graph)

        logger.info(f"Smart weights computed for {self.total_edges} edges")
        logger.info(f"Discovered {len(self.keyword_frequencies)} unique keywords")
        # logger.info(f"Top 5 keywords: {dict(self.keyword_frequencies.most_common(5))}")

    async def _collect_graph_statistics(self, graph):
        """收集圖的統計數據用於 IRF 和 PMI 計算"""
        logger.info("Collecting graph statistics for automated weight calculation...")

        edges = await graph.get_all_edges()
        self.total_edges = len(edges)

        for e in edges:
            src = e.get('source') if isinstance(e, dict) else (e[0] if len(e) > 0 else None)
            tgt = e.get('target') if isinstance(e, dict) else (e[1] if len(e) > 1 else None)
            if not src or not tgt:
                continue
            edge_data = await graph.get_edge(src, tgt)
            keywords_str = edge_data.get('keywords', '')

            if not keywords_str:
                continue

            # 解析 keywords
            keywords = self._parse_keywords(keywords_str)

            # 統計 keyword 頻率 (for IRF)
            for keyword in keywords:
                self.keyword_frequencies[keyword] += 1

            # 統計實體-keyword 共現 (for PMI)
            for keyword in keywords:
                self.entity_keyword_pairs[(src, keyword)] += 1
                self.keyword_entity_pairs[(keyword, tgt)] += 1
                self.entity_keyword_entity_triplets[(src, keyword, tgt)] += 1

        logger.info(f"Statistics collected: {len(self.keyword_frequencies)} keywords, {self.total_edges} edges")

    async def _compute_all_edge_weights(self, graph):
        """計算所有邊的智能權重"""
        edges = await graph.get_all_edges()

        for e in edges:
            src = e.get('source') if isinstance(e, dict) else (e[0] if len(e) > 0 else None)
            tgt = e.get('target') if isinstance(e, dict) else (e[1] if len(e) > 1 else None)
            if not src or not tgt:
                continue
            edge_data = await graph.get_edge(src, tgt)

            # 計算智能權重
            irf_score = self._compute_automated_irf(edge_data.get('keywords', ''))
            pmi_score = self._compute_automated_pmi(src, tgt, edge_data.get('keywords', ''))

            # 組合權重 (移除時間衰減)
            smart_weight = irf_score * pmi_score
            smart_weight = max(0.1, min(5.0, smart_weight))  # 自動歸一化

            # 更新邊權重
            edge_data['smart_weight'] = smart_weight
            await graph.upsert_edge(src, tgt, edge_data)

    def _parse_keywords(self, keywords_str: str) -> List[str]:
        """解析逗號分隔的 keywords"""
        if not keywords_str:
            return []
        return [k.strip() for k in keywords_str.split(',') if k.strip()]

    def _compute_automated_irf(self, keywords_str: str) -> float:
        """自動化 IRF 計算 - 無需手動權重"""
        if not keywords_str:
            return 0.1

        # 快取檢查
        if keywords_str in self._keyword_irf_cache:
            return self._keyword_irf_cache[keywords_str]

        keywords = self._parse_keywords(keywords_str)
        if not keywords:
            return 0.1

        irf_scores = []
        for keyword in keywords:
            frequency = self.keyword_frequencies.get(keyword, 1)
            # IRF = log(1 + 總邊數 / (1 + keyword頻率)) - 避免負值
            irf_raw = np.log(1 + self.total_edges / (1 + frequency))
            irf_scores.append(irf_raw)

        # 對 keyword 層進行 z-score 標準化然後映射到 [0.5, 2.0]
        if len(irf_scores) > 1:
            irf_mean = np.mean(irf_scores)
            irf_std = np.std(irf_scores) + 1e-9  # 避免除以零
            irf_normalized = [(score - irf_mean) / irf_std for score in irf_scores]
            # 使用 sigmoid 映射到 [0.5, 2.0]
            result_scores = [1.25 + 0.75 * (2 / (1 + np.exp(-score)) - 1) for score in irf_normalized]
            result = np.mean(result_scores)
        else:
            # 單個 keyword 時直接線性映射
            irf_raw = irf_scores[0]
            # 線性映射到 [0.5, 2.0]，假設合理範圍是 [0, log(total_edges)]
            max_possible = np.log(1 + self.total_edges)
            result = 0.5 + 1.5 * min(1.0, irf_raw / max_possible)
        self._keyword_irf_cache[keywords_str] = result
        return result

    def _compute_automated_pmi(self, src_entity: str, tgt_entity: str, keywords_str: str) -> float:
        """自動化 PMI 計算 - 基於實際共現統計"""
        if not keywords_str:
            return 0.5

        # 快取檢查
        cache_key = (src_entity, tgt_entity, keywords_str)
        if cache_key in self._pmi_cache:
            return self._pmi_cache[cache_key]

        keywords = self._parse_keywords(keywords_str)
        if not keywords:
            return 0.5

        pmi_scores = []

        for keyword in keywords:
            # 以 keyword 為條件的機率計算 + Dirichlet 平滑
            joint_count = self.entity_keyword_entity_triplets.get((src_entity, keyword, tgt_entity), 0)
            src_keyword_count = self.entity_keyword_pairs.get((src_entity, keyword), 0)
            keyword_tgt_count = self.keyword_entity_pairs.get((keyword, tgt_entity), 0)

            # 計算以 keyword 為條件的總計數（用於歸一化）
            keyword_total_pairs = sum([count for (kw_entity, kw), count in self.entity_keyword_pairs.items() if kw == keyword])
            keyword_total_triplets = sum([count for (src, kw, tgt), count in self.entity_keyword_entity_triplets.items() if kw == keyword])

            if keyword_total_triplets > 0:
                # 加一平滑 (Dirichlet smoothing)
                smooth_factor = 1.0

                # P(src, tgt | keyword) = #(src, keyword, tgt) / Σ(#(x, keyword, y))
                joint_prob = (joint_count + smooth_factor) / (keyword_total_triplets + smooth_factor * 100)

                # P(src | keyword) = #(src, keyword) / Σ(#(x, keyword))
                src_prob = (src_keyword_count + smooth_factor) / (keyword_total_pairs + smooth_factor * 50)

                # P(tgt | keyword) = #(keyword, tgt) / Σ(#(keyword, y))
                tgt_prob = (keyword_tgt_count + smooth_factor) / (keyword_total_pairs + smooth_factor * 50)

                # PMI = log(P(src,tgt|kw) / (P(src|kw) * P(tgt|kw)))
                if src_prob > 1e-9 and tgt_prob > 1e-9:
                    pmi = np.log(joint_prob / (src_prob * tgt_prob))
                    # 使用 tanh 映射到穩定區間 [0.5, 2.0]
                    pmi_normalized = 1.25 + 0.75 * np.tanh(pmi)  # 映射到 [0.5, 2.0]
                    pmi_scores.append(pmi_normalized)
                else:
                    pmi_scores.append(0.5)
            else:
                # keyword 無共現數據時的預設值
                pmi_scores.append(0.5)

        result = np.mean(pmi_scores) if pmi_scores else 0.5
        self._pmi_cache[cache_key] = result
        return result



async def compute_adaptive_fastrp_similarity(target_entity: str, seed_entities: List[str], graph) -> float:
    """
    Adaptive FastRP 相似度計算
    基於邊的 smart_weight 屬性，自適應關係語義
    """
    try:
        similarities = []

        for seed_entity in seed_entities:
            # 直接連接的邊權重
            direct_weight = 0.0

            # 檢查雙向連接
            for src, tgt in [(target_entity, seed_entity), (seed_entity, target_entity)]:
                try:
                    edge_data = await graph.get_edge(src, tgt)
                    if edge_data and 'smart_weight' in edge_data:
                        direct_weight = max(direct_weight, edge_data['smart_weight'])
                except:
                    continue

            if direct_weight > 0:
                similarities.append(direct_weight)
            else:
                # 透過共同鄰居的間接相似度
                try:
                    target_edges = await graph.get_node_edges(target_entity) or []
                    seed_edges = await graph.get_node_edges(seed_entity) or []
                    target_neighbors = {t if s == target_entity else s for s, t in target_edges}
                    seed_neighbors = {t if s == seed_entity else s for s, t in seed_edges}
                    common_neighbors = target_neighbors & seed_neighbors

                    if common_neighbors:
                        indirect_weights = []
                        for neighbor in common_neighbors:
                            # 獲取兩條邊的權重
                            e1 = await graph.get_edge(target_entity, neighbor) or {}
                            e2 = await graph.get_edge(seed_entity, neighbor) or {}
                            w1 = e1.get('smart_weight', 0.1)
                            w2 = e2.get('smart_weight', 0.1)
                            # 使用幾何平均作為間接權重
                            indirect_weights.append(np.sqrt(w1 * w2))

                        if indirect_weights:
                            similarities.append(max(indirect_weights) * 0.5)  # 間接權重打折
                except:
                    pass

        return float(np.mean(similarities)) if similarities else 0.0

    except Exception as e:
        logger.error(f"Error computing keywords smart similarity: {e}")
        return 0.0
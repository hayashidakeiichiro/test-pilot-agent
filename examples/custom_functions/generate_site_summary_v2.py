from browser_use import Controller, ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.dom.views import DOMBaseNode, DOMElementNode, DOMTextNode
from browser_use.dom.service import DomService
from Levenshtein import distance
from typing import List, Tuple, Optional
import asyncio
import datetime
import json
import os
import base64
import io
from dotenv import load_dotenv
from PIL import Image
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from browser_use import ActionResult, Agent, Browser, BrowserConfig, Controller
from langchain_core.language_models.chat_models import BaseChatModel
import re
import functools
from langchain_core.prompts import PromptTemplate
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage, BaseMessageChunk
from itertools import islice
from PIL import Image
from examples.custom_functions.generate_icon_list_image import generate_selector_thumbnail_grid_base64, extract_none_text_selector_map

logger = logging.getLogger(__name__)
import cv2
import numpy as np
from collections import deque
from statistics import stdev, quantiles
from zss import simple_distance

# Serialize DOM structure, filtering out text nodes
class ZssAdapter:
    def __init__(self, dom_node: DOMElementNode):
        self.dom_node = dom_node

    @property
    def children(self):
        # Only element children
        return [ZssAdapter(c) for c in self.dom_node.children if isinstance(c, DOMElementNode)]

    @property
    def label(self):
        return self.dom_node.tag_name

def serialize_structure(node):
    if not isinstance(node, DOMElementNode):
        return None
    children = [c for c in node.children if isinstance(c, DOMElementNode)]
    return (node.tag_name, tuple(serialize_structure(c) for c in children))


def generate_blocks(root):
    """
    Identify block nodes: those whose immediate children share identical subtree structures.
    """
    def init_block(node):
        if isinstance(node, DOMElementNode):
            setattr(node, 'is_block_node', False)
            for c in node.children:
                init_block(c)
    init_block(root)

    queue = deque([root])
    while queue:
        node = queue.popleft()
        if not isinstance(node, DOMElementNode):
            continue
        children = [c for c in node.children if isinstance(c, DOMElementNode)]
        if len(children) > 1:
            print(f"node: {node.get_all_text_till_next_clickable_element(max_depth=1)}, children: {[c.tag_name for c in children]}")
            sigs = [serialize_structure(c) for c in children]
            if sigs and all(s == sigs[0] for s in sigs[1:]):
                setattr(node, 'is_block_node', True)
                continue
        queue.extend(children)


def compute_heterogeneity(root:DOMBaseNode):
    blocks: list[DOMElementNode] = []
    def dfs(n):
        if isinstance(n, DOMElementNode) and getattr(n, 'is_block_node', False):
            blocks.append(n)
        for c in getattr(n, 'children', []): dfs(c)
    dfs(root)

    scores = []
    for node in blocks:
        children = [c for c in node.children if isinstance(c, DOMElementNode)]
        costs = []
        for i in range(len(children)):
            for j in range(i+1, len(children)):
                costs.append(simple_distance(ZssAdapter(children[i]), ZssAdapter(children[j])))
        H = stdev(costs) if len(costs)>=2 else 0.0
        scores.append((node, H))
    return scores


def mark_outliers(root):
    def init_out(node):
        if isinstance(node, DOMElementNode):
            setattr(node, 'is_outlier', False)
            for c in node.children: init_out(c)
    init_out(root)

    scores = compute_heterogeneity(root)
    if not scores: return
    H_vals = [h for (_,h) in scores]
    q1,q3 = quantiles(H_vals,n=4)[0], quantiles(H_vals,n=4)[2]
    thresh = q3 + 1.5*(q3-q1)
    out_ids = {id(n) for n,h in scores if h>thresh}

    def dfs_mark(n):
        if isinstance(n, DOMElementNode):
            setattr(n,'is_outlier', id(n) in out_ids)
            for c in n.children: dfs_mark(c)
    dfs_mark(root)


def eliminate_outliers(node):
    """
    Extract segments: either block nodes that are not outliers or descend.
    """
    if not isinstance(node, DOMElementNode):
        return []
    # If this node is a valid block (not an outlier), it's a segment
    if getattr(node, 'is_block_node', False) and not getattr(node, 'is_outlier', False):
        return [node]
    # If outlier, replace by children
    if getattr(node, 'is_outlier', False):
        segments = []
        for c in node.children:
            segments.extend(eliminate_outliers(c))
        return segments
    # Else, continue descending
    segments = []
    for c in node.children:
        segments.extend(eliminate_outliers(c))
    return segments
def get_top_level_segments(root):
    """
    root 以下のノードを走査し、
    - is_block_node == True かつ is_outlier == False のノード
    - かつ祖先に is_block_node == True のノードがいない（トップレベル）
    だけをセグメントとして返す
    """
    segments = []

    def dfs(node, has_block_ancestor):
        if not isinstance(node, DOMElementNode):
            return
        # もしこのノードが「まとまり候補」で、かつ外れ判定されていなければ
        if getattr(node, "is_block_node", False) and not getattr(node, "is_outlier", False):
            # トップレベルかどうかチェック
            if not has_block_ancestor:
                segments.append(node)
            # ここで descend せずに、このノード配下はスキップ
            return
        # それ以外のノードは子を探索。祖先に既にブロックがあればフラグを渡す
        for c in node.children:
            dfs(c, has_block_ancestor or getattr(node, "is_block_node", False))

    dfs(root, False)
    return segments

def segment_page(root, screenshot=None):
    generate_blocks(root)
    mark_outliers(root)
    segments = get_top_level_segments(root)
    # if screenshot is not None:
    #     for seg in segments:
    #         if getattr(seg,'viewport_info',None):
    #             optimize_region(seg, screenshot)
    return segments

# Utility to print concise summary

def summarize_segments(segments: list[DOMElementNode]):
    print('Segments summary:')
    for idx, s in enumerate(segments,1):
        print(f"{idx}: tag={s.tag_name}, xpath={getattr(s,'xpath',None)}", s.get_all_text_till_next_clickable_element(max_depth=1))



def attach_generate_site_summary_v2(controller: Controller):
    @controller.action("Find target element by context_block and target")
    async def generate_site_summary_v2(
        browser: BrowserContext,
    ):
        page = await browser.get_agent_current_page()
        dom_service = DomService(page)

        # ページからクリック可能要素を取得
        content = await dom_service.get_clickable_elements(
            focus_element=-1,
            viewport_expansion=-1,
            highlight_elements=True,
        )
        html_tree = content.element_tree
        segments = segment_page(browser, html_tree)
        summarize_segments(browser, segments)

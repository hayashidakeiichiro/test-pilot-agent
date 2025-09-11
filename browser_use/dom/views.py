from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Optional, List, Dict

from browser_use.dom.history_tree_processor.view import CoordinateSet, HashedDomElement, ViewportInfo
from browser_use.utils import time_execution_sync
from urllib.parse import urlparse, unquote
import re
from collections import defaultdict, Counter

# Avoid circular import issues
if TYPE_CHECKING:
	from .views import DOMElementNode


@dataclass(frozen=False)
class DOMBaseNode:
	is_visible: bool
	# Use None as default and set parent later to avoid circular reference issues
	parent: Optional['DOMElementNode']

	def __json__(self) -> dict:
		raise NotImplementedError('DOMBaseNode is an abstract class')


@dataclass(frozen=False)
class DOMTextNode(DOMBaseNode):
	text: str
	type: str = 'TEXT_NODE'

	def has_parent_with_highlight_index(self) -> bool:
		current = self.parent
		while current is not None:
			# stop if the element has a highlight index (will be handled separately)
			if current.highlight_index is not None:
				return True

			current = current.parent
		return False

	def is_parent_in_viewport(self) -> bool:
		if self.parent is None:
			return False
		return self.parent.is_in_viewport

	def is_parent_top_element(self) -> bool:
		if self.parent is None:
			return False
		return self.parent.is_top_element

	def __json__(self) -> dict:
		return {
			'text': self.text,
			'type': self.type,
		}


@dataclass(frozen=False)
class DOMElementNode(DOMBaseNode):
	"""
	xpath: the xpath of the element from the last root node (shadow root or iframe OR document if no shadow root or iframe).
	To properly reference the element we need to recursively switch the root node until we find the element (work you way up the tree with `.parent`)
	"""

	tag_name: str
	xpath: str
	attributes: dict[str, str]
	children: list[DOMBaseNode]
	is_interactive: bool = False
	is_top_element: bool = False
	is_in_viewport: bool = False
	shadow_root: bool = False
	highlight_index: int | None = None
	viewport_coordinates: CoordinateSet | None = None
	page_coordinates: CoordinateSet | None = None
	viewport_info: ViewportInfo | None = None

	"""
	### State injected by the browser context.

	The idea is that the clickable elements are sometimes persistent from the previous page -> tells the model which objects are new/_how_ the state has changed
	"""
	is_new: bool | None = None

	def __json__(self) -> dict:
		return {
			'tag_name': self.tag_name,
			'xpath': self.xpath,
			'attributes': self.attributes,
			'is_visible': self.is_visible,
			'is_interactive': self.is_interactive,
			'is_top_element': self.is_top_element,
			'is_in_viewport': self.is_in_viewport,
			'shadow_root': self.shadow_root,
			'highlight_index': self.highlight_index,
			'viewport_coordinates': self.viewport_coordinates,
			'page_coordinates': self.page_coordinates,
			'children': [child.__json__() for child in self.children],
		}

	def __repr__(self) -> str:
		tag_str = f'<{self.tag_name}'

		# Add attributes
		for key, value in self.attributes.items():
			tag_str += f' {key}="{value}"'
		tag_str += '>'

		# Add extra info
		extras = []
		if self.is_interactive:
			extras.append('interactive')
		if self.is_top_element:
			extras.append('top')
		if self.shadow_root:
			extras.append('shadow-root')
		if self.highlight_index is not None:
			extras.append(f'highlight:{self.highlight_index}')
		if self.is_in_viewport:
			extras.append('in-viewport')

		if extras:
			tag_str += f' [{", ".join(extras)}]'

		return tag_str

	@cached_property
	def hash(self) -> HashedDomElement:
		from browser_use.dom.history_tree_processor.service import (
			HistoryTreeProcessor,
		)

		return HistoryTreeProcessor._hash_dom_element(self)

	def get_all_text_till_next_clickable_element(self, max_depth: int = -1, include_attr_for_img: bool = True) -> str:
		"""
		1) DOM を「インデックス要素（= highlight_index:int がある要素）」基準でまず dict ツリーに変換
		2) その dict からテキストにレンダリング

		グルーピング規則（cap 方式）:
		- 各ノードの「配下に含まれるインデックス要素の個数 = indexed_count」を数える
		- 出力時に親から cap（親の indexed_count）を受け取り、現在ノードの cnt と比べて
			* cnt >= 2 かつ (cap is None or cnt < cap) のとき:
				=> <要素名> を出して中身をインデントして再帰（cap を cnt に更新）
			* cnt == cap のとき:
				=> その要素名は出さず、同じインデントで子だけを出力（= 無視）
			* それ以外:
				=> その要素がインデックス要素なら 1 行のアイテムとして出力
		- DOMTextNode も dict に含める（カウントには含めない）
		- **追加**: 要素内に `<img>` が存在する場合は、その画像情報（alt/src など）もアイテム情報に含める
				（別の index 要素配下は潜らない、浅い深さで集計）
		"""
		from urllib.parse import urlparse, unquote  # ローカルインポートで安全に

		# ----------------- 小道具 -----------------
		def trim_ws(s: str) -> str:
			return " ".join((s or "").split())

		def truncate(s: str, n: int = 160) -> str:
			s = s or ""
			return s if len(s) <= n else s[: n - 1] + "…"

		def summarize_image_url_segments(src_text: str) -> str:
			if not src_text:
				return ""
			parsed = urlparse(unquote(src_text))
			segs = [seg for seg in parsed.path.strip("/").split("/") if seg]
			short = [seg for seg in segs if len(seg) <= 12]
			picked = short[:3] if short else segs[:2]
			return "/".join(picked)

		def element_signature(node: "DOMElementNode") -> str:
			"""div#id.class1.class2（クラスは先頭2つ + … を付与）"""
			tag = (getattr(node, "tag_name", "") or "element").lower()
			attrs = getattr(node, "attributes", {}) or {}
			_id = trim_ws(attrs.get("id", ""))
			cls = trim_ws(attrs.get("class", ""))
			cls_list = [c for c in cls.split() if c]
			cls_head = "." + ".".join(cls_list[:2]) if cls_list else ""
			cls_tail = "…" if len(cls_list) > 2 else ""
			id_part = f"#{_id}" if _id else ""
			return f"{tag}{id_part}{cls_head}{cls_tail}"

		def element_open_meta(node: "DOMElementNode") -> dict:
			"""開始側に付ける簡易メタ（role/name/aria-label/title）"""
			attrs = getattr(node, "attributes", {}) or {}
			return {
				"role": trim_ws(attrs.get("role", "")) or "",
				"name": trim_ws(attrs.get("name", "")) or "",
				"label": trim_ws(attrs.get("aria-label", "") or attrs.get("title", "")) or "",
			}

		def is_indexed_element(node) -> bool:
			return isinstance(node, DOMElementNode) and isinstance(getattr(node, "highlight_index", None), int)

		# ----------------- 1st pass: indexed_count を数える -----------------
		indexed_count: dict[int, int] = {}

		def count_indexed(node) -> int:
			total = 0
			if is_indexed_element(node):
				total += 1
			for ch in getattr(node, "children", []):
				total += count_indexed(ch)
			indexed_count[id(node)] = total  # DOMTextNode は 0 のまま
			return total

		count_indexed(self)

		# ----------------- 画像の抽出（浅く、index 配下は潜らない） -----------------
		def collect_image_descendants(node: "DOMElementNode", depth_limit: int = 2) -> list[dict]:
			imgs: list[dict] = []
			if not include_attr_for_img:
				return imgs

			def walk(n, d):
				if max_depth != -1 and d > depth_limit:
					return
				if isinstance(n, DOMElementNode):
					tag = (getattr(n, "tag_name", "") or "").lower()
					if is_indexed_element(n) and n is not node:
						return  # 別の index 要素の内側は潜らない
					if tag == "img":
						attrs = getattr(n, "attributes", {}) or {}
						alt = trim_ws(attrs.get("alt", ""))
						src = trim_ws(attrs.get("src", ""))
						cls = trim_ws(attrs.get("class", ""))
						title = trim_ws(attrs.get("title", ""))
						imgs.append({
							"alt": truncate(alt, 80) if alt else "",
							"src": truncate(summarize_image_url_segments(src), 60) if src else "",
							"class": truncate(cls, 80) if cls else "",
							"title": truncate(title, 80) if title else "",
						})
						return  # img 自体の子は通常なし
					for c in getattr(n, "children", []):
						walk(c, d + 1)
				# DOMTextNode は無視（ここでは画像だけ）
			walk(node, 0)
			return imgs

		# ----------------- テキストの抽出（浅く、index 配下は潜らない） -----------------
		def collect_text_descendants(node: "DOMElementNode", depth_limit: int = 2) -> list[str]:
			out: list[str] = []
			def walk(n, d):
				if max_depth != -1 and d > depth_limit:
					return
				if isinstance(n, DOMTextNode):
					t = trim_ws(getattr(n, "text", ""))
					if t:
						out.append(t)
					return
				if isinstance(n, DOMElementNode):
					if is_indexed_element(n) and n is not node:
						return
					for c in getattr(n, "children", []):
						walk(c, d + 1)
			walk(node, 0)
			return out

		# ----------------- 2nd pass: dict ツリーを構築 -----------------
		# dict ノード:
		#   Group: {"type":"group","signature":str,"meta":{...},"items":int,"children":[...]}
		#   Item : {"type":"item","index":int,"signature":str,"attrs":{...},"kind":"TEXT|IMAGE",
		#           "label":str,"text_nodes":[str,...],"images":[{alt,src,class,title},...]}
		#   Text : {"type":"text","text":str}

		def build_item(node: "DOMElementNode") -> dict:
			idx = getattr(node, "highlight_index")
			sig = element_signature(node)
			attrs = getattr(node, "attributes", {}) or {}
			type_attr = trim_ws(attrs.get("type", ""))
			role_attr = trim_ws(attrs.get("role", ""))
			class_attr = trim_ws(attrs.get("class", ""))

			tag = (getattr(node, "tag_name", "") or "").lower()
			if tag == "img" and include_attr_for_img:
				alt = trim_ws(attrs.get("alt", ""))
				src = trim_ws(attrs.get("src", ""))
				label = " ".join(
					p for p in [
						f"alt='{truncate(alt,80)}'" if alt else "",
						f"src='{truncate(summarize_image_url_segments(src),60)}'" if src else "",
					] if p
				) or "(no-attr)"
				kind = "IMAGE"
				text_nodes: list[str] = []
				images: list[dict] = []  # 自身が img の場合は別リストは不要
			else:
				kind = "TEXT"
				text_nodes = [truncate(t, 160) for t in collect_text_descendants(node, depth_limit=2)]
				images = collect_image_descendants(node, depth_limit=2)
				# ラベルはテキストの先頭 or 画像の alt/src の先頭をフォールバック
				label = (text_nodes[0] if text_nodes else "") or (
					f"img:{images[0]['alt'] or images[0]['src']}" if images else ""
				)

			return {
				"type": "item",
				"index": idx,
				"signature": sig,
				"attrs": {
					"type": type_attr,
					"role": role_attr,
					"class": truncate(class_attr, 120) if class_attr else "",
				},
				"kind": kind,
				"label": truncate(label, 160) if label else "",
				"text_nodes": text_nodes,
				"images": images,  # ★ ここに画像情報を保持
			}

		def build_dict(node: "DOMBaseNode", cap: int | None, depth_from_root: int) -> list[dict]:
			if max_depth != -1 and depth_from_root > max_depth:
				return []

			# テキスト
			if isinstance(node, DOMTextNode):
				t = trim_ws(getattr(node, "text", ""))
				return [{"type": "text", "text": truncate(t, 160)}] if t else []

			if not isinstance(node, DOMElementNode):
				return []

			cnt = indexed_count.get(id(node), 0)

			# group 開始
			if cnt >= 2 and (cap is None or cnt < cap):
				g_children: list[dict] = []
				for ch in getattr(node, "children", []):
					if isinstance(ch, DOMTextNode) or indexed_count.get(id(ch), 0) > 0 or is_indexed_element(ch):
						g_children.extend(build_dict(ch, cnt, depth_from_root + 1))
				return [{
					"type": "group",
					"signature": element_signature(node),
					"meta": element_open_meta(node),
					"items": cnt,
					"children": g_children,
				}]

			# 自分がインデックス要素なら item
			if is_indexed_element(node):
				return [build_item(node)]

			# 素通り
			out: list[dict] = []
			for ch in getattr(node, "children", []):
				if isinstance(ch, DOMTextNode) or indexed_count.get(id(ch), 0) > 0 or is_indexed_element(ch):
					out.extend(build_dict(ch, cap, depth_from_root + 1))
			return out

		dict_root = {
			"type": "root",
			"children": build_dict(self, None, 0),
		}

		# ----------------- dict → テキスト レンダリング -----------------
		INDENT = "  "

		def render_node(n: dict, indent: int, out_lines: list[str]):
			t = n.get("type")
			if t == "group":
				sig = n.get("signature", "")
				meta = n.get("meta", {}) or {}
				items = n.get("items", 0)
				meta_str_parts = []
				if meta.get("role"):
					meta_str_parts.append(f"role='{meta['role']}'")
				if meta.get("name"):
					meta_str_parts.append(f"name='{truncate(meta['name'], 40)}'")
				meta_str = (" " + " ".join(meta_str_parts)) if meta_str_parts else ""
				label = meta.get("label", "")
				label_str = f' "{truncate(label, 60)}"' if label else ""
				out_lines.append(f"{INDENT*indent}<{sig}{meta_str}> {label_str}")
				for c in n.get("children", []):
					render_node(c, indent + 1, out_lines)
				out_lines.append(f"{INDENT*indent}</{sig}>")

			elif t == "item":
				sig = n.get("signature", "")
				idx = n.get("index")
				kind = n.get("kind", "TEXT")
				attrs = n.get("attrs", {}) or {}
				extra_parts = []
				if attrs.get("type"):
					extra_parts.append(f"type='{attrs['type']}'")
				if attrs.get("role"):
					extra_parts.append(f"role='{attrs['role']}'")
				if attrs.get("class"):
					extra_parts.append(f"class='{attrs['class']}'")
				extra = (" " + " ".join(extra_parts)) if extra_parts else ""
				label = n.get("label", "")
				label_str = f" {truncate(label, 160)}" if label else ""
				out_lines.append(f"{INDENT*indent}[{idx}]<{sig}{extra}> {label_str} />")

				# テキスト行（任意）
				for tx in n.get("text_nodes", []):
					if tx:
						out_lines.append(f"{INDENT*(indent+1)} {tx}")

				# ★ 画像行（任意）
				for img in n.get("images", []) or []:
					parts = []
					if img.get("alt"):   parts.append(f"alt='{img['alt']}'")
					if img.get("src"):   parts.append(f"src='{img['src']}'")
					if img.get("class"): parts.append(f"class='{img['class']}'")
					if img.get("title"): parts.append(f"title='{img['title']}'")
					payload = " ".join(parts) if parts else "(no-attr)"
					out_lines.append(f"{INDENT*(indent+1)}IMG {payload}")

			elif t == "text":
				text = n.get("text", "")
				if text:
					out_lines.append(f"{INDENT*indent} {text}")

			elif t == "root":
				for c in n.get("children", []):
					render_node(c, indent, out_lines)

		lines: list[str] = []
		render_node(dict_root, 0, lines)
		return "\n".join(lines).strip()

	@time_execution_sync('--clickable_elements_to_string')
	def clickable_elements_to_string(self, include_attributes: list[str] | None = None) -> str:
		"""Convert the processed DOM content to HTML."""
		formatted_text = []

		def process_node(node: DOMBaseNode, depth: int) -> None:
			next_depth = int(depth)
			depth_str = depth * '\t'

			if isinstance(node, DOMElementNode):
				# Add element with highlight_index
				if node.highlight_index is not None:
					next_depth += 1

					text = node.get_all_text_till_next_clickable_element()
					attributes_html_str = ''
					if include_attributes:
						attributes_to_include = {
							key: str(value) for key, value in node.attributes.items() if key in include_attributes
						}

						# Easy LLM optimizations
						# if tag == role attribute, don't include it
						if node.tag_name == attributes_to_include.get('role'):
							del attributes_to_include['role']

						# if aria-label == text of the node, don't include it
						if (
							attributes_to_include.get('aria-label')
							and attributes_to_include.get('aria-label', '').strip() == text.strip()
						):
							del attributes_to_include['aria-label']

						# if placeholder == text of the node, don't include it
						if (
							attributes_to_include.get('placeholder')
							and attributes_to_include.get('placeholder', '').strip() == text.strip()
						):
							del attributes_to_include['placeholder']

						if attributes_to_include:
							# Format as key1='value1' key2='value2'
							attributes_html_str = ' '.join(f"{key}='{value}'" for key, value in attributes_to_include.items())

					# Build the line
					if node.is_new:
						highlight_indicator = f'*[{node.highlight_index}]*'
					else:
						highlight_indicator = f'[{node.highlight_index}]'

					line = f'{depth_str}{highlight_indicator}<{node.tag_name}'

					if attributes_html_str:
						line += f' {attributes_html_str}'

					if text:
						# Add space before >text only if there were NO attributes added before
						if not attributes_html_str:
							line += ' '
						line += f'>{text}'
					# Add space before /> only if neither attributes NOR text were added
					elif not attributes_html_str:
						line += ' '

					line += ' />'  # 1 token
					formatted_text.append(line)

				# Process children regardless
				for child in node.children:
					process_node(child, next_depth)

			elif isinstance(node, DOMTextNode):
				# Add text only if it doesn't have a highlighted parent
				if (
					not node.has_parent_with_highlight_index()
					and node.parent
					and node.parent.is_visible
					and node.parent.is_top_element
				):  # and node.is_parent_top_element()
					formatted_text.append(f'{depth_str}{node.text}')

		process_node(self, 0)
		return '\n'.join(formatted_text)

	# test-pilot
	def clickable_elements_to_string_with_tag(self, include_attributes: list[str] | None = None) -> str:
		"""Convert the processed DOM content to HTML."""
		formatted_text = []

		def process_node(node: DOMBaseNode, depth: int) -> None:
			next_depth = int(depth)
			depth_str = depth * '\t'

			if isinstance(node, DOMElementNode):
				# Add element with highlight_index
				if node.highlight_index is not None:
					next_depth += 1

					raw_text = node.get_all_text_till_next_clickable_element()
					text = re.sub(r'\s+', ' ', raw_text).strip()
					max_length = 30
					if len(text) > max_length:
						start_part = text[:15]
						end_part = text[-10:]
						text = f"{start_part}…{end_part}"
					attributes_html_str = ''
					if include_attributes:
						attributes_to_include = {
							key: str(value) for key, value in node.attributes.items() if key in include_attributes
						}

						# Easy LLM optimizations
						# if tag == role attribute, don't include it
						if node.tag_name == attributes_to_include.get('role'):
							del attributes_to_include['role']

						# if aria-label == text of the node, don't include it
						if (
							attributes_to_include.get('aria-label')
							and attributes_to_include.get('aria-label', '').strip() == text.strip()
						):
							del attributes_to_include['aria-label']

						# if placeholder == text of the node, don't include it
						if (
							attributes_to_include.get('placeholder')
							and attributes_to_include.get('placeholder', '').strip() == text.strip()
						):
							del attributes_to_include['placeholder']

						if attributes_to_include:
							# Format as key1='value1' key2='value2'
							attributes_html_str = ' '.join(f"{key}='{value}'" for key, value in attributes_to_include.items())

					# Build the line
					if node.is_new:
						highlight_indicator = f'*[{node.highlight_index}]*'
					else:
						highlight_indicator = f'[{node.highlight_index}]'

					line = f'{depth_str}{highlight_indicator}<{node.tag_name}'

					if attributes_html_str:
						line += f' {attributes_html_str}'

					if text:
						# Add space before >text only if there were NO attributes added before
						if not attributes_html_str:
							line += ' '
						line += f'>{text}'
					# Add space before /> only if neither attributes NOR text were added
					elif not attributes_html_str:
						line += ' '

					line += ' />'  # 1 token
					formatted_text.append(line)

				# Process children regardless
				for child in node.children:
					process_node(child, next_depth)

			elif isinstance(node, DOMTextNode):
				if (
					not node.has_parent_with_highlight_index()
					and node.parent
					and node.parent.is_visible
					and node.parent.is_top_element
				):
					parent_tag = node.parent.tag_name.lower()
					text = node.text
					if parent_tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
						formatted_text.append(f'{depth_str}<{parent_tag}>{text}</{parent_tag}>')
					else:
						formatted_text.append(f'{depth_str}{text}')

		process_node(self, 0)
		return '\n'.join(formatted_text)


	def get_file_upload_element(self, check_siblings: bool = True) -> Optional['DOMElementNode']:
		# Check if current element is a file input
		if self.tag_name == 'input' and self.attributes.get('type') == 'file':
			return self

		# Check children
		for child in self.children:
			if isinstance(child, DOMElementNode):
				result = child.get_file_upload_element(check_siblings=False)
				if result:
					return result

		# Check siblings only for the initial call
		if check_siblings and self.parent:
			for sibling in self.parent.children:
				if sibling is not self and isinstance(sibling, DOMElementNode):
					result = sibling.get_file_upload_element(check_siblings=False)
					if result:
						return result

		return None


SelectorMap = dict[int, DOMElementNode]


@dataclass
class DOMState:
	element_tree: DOMElementNode
	selector_map: SelectorMap

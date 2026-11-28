"""真实 AstrBot 栈测试辅助：标准库本地 HTTP 服务 + 真实核心组件装配。

不模拟任何 AstrBot 接口：ProviderManager、DecisionService、KnowledgeBase、
Context、AstrMessageEvent 全部使用真实类，LLM/Embedding/Decision/Rerank 请求由
标准库 http.server 提供的本地服务应答。
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
import tempfile
import threading
import unittest
import uuid
from asyncio import Queue
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_PARENT = _PROJECT_ROOT.parent
for _path in (_PROJECT_PARENT, _PROJECT_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

plugin_main = importlib.import_module("astrbot_plugin_simple_long_memory.main")
memory_manager_module = importlib.import_module(
    "astrbot_plugin_simple_long_memory.memory_manager"
)
MemoryPlugin = plugin_main.MemoryPlugin
MemoryManager = memory_manager_module.MemoryManager

EMBED_DIM = 8


def _embedding_payload(count=1, vectors=None):
    if vectors is None:
        vectors = [[0.1 * (i + 1)] * EMBED_DIM for i in range(count)]
    return {
        "object": "list",
        "model": "test-embed",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": vector,
            }
            for i, vector in enumerate(vectors)
        ],
    }


class _Handler(BaseHTTPRequestHandler):
    """按路径分发预置响应；未预置路径返回 500。"""

    def log_message(self, format, *args):  # noqa: A002
        pass

    def _reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        server = self.server
        try:
            request = json.loads(raw or b"{}")
        except Exception:
            request = {}
        path = self.path
        if path.endswith("/chat/completions"):
            content = server.next_chat_content(request)
            if content is None:
                self._reply(500, {"error": "no scripted chat response"})
                return
            if isinstance(content, dict) and "__http_error__" in content:
                self._reply(content["__http_error__"], content.get("payload") or {})
                return
            message = (
                {"role": "assistant", **content}
                if isinstance(content, dict)
                else {"role": "assistant", "content": content}
            )
            self._reply(
                200,
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": request.get("model", "test-model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )
        elif path.endswith("/embeddings"):
            status, payload = server.next_embedding_response(request)
            self._reply(status, payload)
        elif path.endswith("/decisions"):
            answers = server.next_decision_answers(request)
            if answers is None:
                self._reply(500, {"error": "no scripted decision response"})
                return
            self._reply(200, {"answers": answers})
        elif path.endswith("/rerank"):
            status, payload = server.next_rerank_response(request)
            self._reply(status, payload)
        else:
            self._reply(404, {"error": f"unexpected path {path}"})


class LocalAstrBotServer(ThreadingHTTPServer):
    """本地 HTTP 服务：脚本化 chat / decision / embedding / rerank 响应队列。"""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self._chat_contents: list[Any] = []
        self._default_chat_content = " {} "
        self._decision_factory = None
        self._embedding_errors: list[tuple[int, str]] = []
        self._embedding_vectors: dict[str, list[float]] | None = None
        self._rerank_errors: list[tuple[int, str]] = []
        self._rerank_handler = None
        self.request_log: list[dict] = []

    # ---------- chat ----------
    def next_chat_content(self, request=None):
        request = request or {}
        self.request_log.append(
            {
                "kind": "chat",
                "model": request.get("model"),
                "messages": request.get("messages"),
            }
        )
        if self._chat_contents:
            return self._chat_contents.pop(0)
        return self._default_chat_content

    def queue_chat(self, *contents):
        self._chat_contents.extend(contents)

    def queue_chat_message(self, *messages):
        """入队完整 assistant message（如带 tool_calls 的脚本化工具调用）。"""
        self._chat_contents.extend(messages)

    def queue_chat_error(self, status=500, payload=None):
        """入队一次 chat HTTP 错误响应。"""
        self._chat_contents.append(
            {
                "__http_error__": status,
                "payload": payload or {"error": {"message": "scripted chat error"}},
            }
        )

    def set_default_chat(self, content):
        self._default_chat_content = content

    # ---------- decisions ----------
    def next_decision_answers(self, request):
        self.request_log.append(
            {"kind": "decisions", "questions": request.get("questions")}
        )
        if self._decision_factory is None:
            return None
        return self._decision_factory(request)

    def set_decision_factory(self, factory):
        """factory(request) -> [{"name": "qN", "type": "predicate", "probability": f}]"""
        self._decision_factory = factory

    # ---------- embeddings ----------
    def queue_embedding_error(self, status=500, message="scripted embedding error"):
        """入队一次 HTTP 错误响应，仅影响下一次 embeddings 请求。"""
        self._embedding_errors.append((status, message))

    def set_embedding_vectors(self, mapping: dict[str, list[float]] | None):
        """按文本（精确或子串）指定向量；未命中的文本使用默认向量。"""
        self._embedding_vectors = dict(mapping) if mapping else None

    def next_embedding_response(self, request):
        inputs = request.get("input")
        self.request_log.append({"kind": "embeddings", "input": inputs})
        if self._embedding_errors:
            status, message = self._embedding_errors.pop(0)
            return status, {"error": {"message": message}}
        texts = inputs if isinstance(inputs, list) else [inputs]
        vectors = [
            self._embedding_vector(text, index) for index, text in enumerate(texts)
        ]
        return 200, _embedding_payload(len(vectors), vectors)

    def _embedding_vector(self, text, index):
        key = str(text)
        if self._embedding_vectors:
            if key in self._embedding_vectors:
                return list(self._embedding_vectors[key])
            for candidate, vector in self._embedding_vectors.items():
                if candidate and candidate in key:
                    return list(vector)
        return [0.1 * (index + 1)] * EMBED_DIM

    # ---------- rerank ----------
    def queue_rerank_error(self, status=500, message="scripted rerank error"):
        """入队一次 HTTP 错误响应，仅影响下一次 rerank 请求。"""
        self._rerank_errors.append((status, message))

    def set_rerank_handler(self, fn):
        """fn(request) -> [{"index": i, "relevance_score": score}, ...]"""
        self._rerank_handler = fn

    def next_rerank_results(self, request):
        documents = request.get("documents") or []
        if self._rerank_handler is not None:
            return self._rerank_handler(request)
        return [
            {"index": i, "relevance_score": round(1.0 / (i + 1), 4)}
            for i in range(len(documents))
        ]

    def next_rerank_response(self, request):
        self.request_log.append(
            {
                "kind": "rerank",
                "query": request.get("query"),
                "documents": request.get("documents"),
            }
        )
        if self._rerank_errors:
            status, message = self._rerank_errors.pop(0)
            return status, {"error": {"message": message}}
        return 200, {"results": self.next_rerank_results(request)}

    @property
    def base_url(self):
        return f"http://{self.server_address[0]}:{self.server_address[1]}/v1"


def decisions_by_request_choices(script):
    """跨请求按问题顺序返回 choice 应答；script 为 [(choice, probability), ...]。

    probabilities 覆盖该问题全部选项且和为 1，满足 Decision 契约校验。
    """
    cursor = {"next": 0}

    def factory(request):
        questions = request.get("questions") or []
        answers = []
        for index, question in enumerate(questions):
            position = cursor["next"]
            cursor["next"] += 1
            choice, probability = (
                script[position] if position < len(script) else script[-1]
            )
            options = [
                str(entry.get("value")) for entry in question.get("choices") or []
            ]
            if choice not in options:
                choice = options[0]
            remainder = max(0.0, 1.0 - probability) / max(1, len(options) - 1)
            answers.append(
                {
                    "name": question.get("name", f"q{index}"),
                    "type": "choice",
                    "choice": str(choice),
                    "confidence": min(1.0, probability),
                    "probabilities": [
                        {
                            "value": option,
                            "probability": probability
                            if option == choice
                            else remainder,
                        }
                        for option in options
                    ],
                }
            )
        return answers

    return factory


def decisions_by_request_probability(probabilities):
    """按请求问题顺序返回 predicate 概率；probabilities 长度需与问题数一致。"""

    def factory(request):
        questions = request.get("questions") or []
        answers = []
        for index, question in enumerate(questions):
            probability = (
                probabilities[index]
                if index < len(probabilities)
                else probabilities[-1]
            )
            answers.append(
                {
                    "name": question.get("name", f"q{index}"),
                    "type": "predicate",
                    "probability": probability,
                }
            )
        return answers

    return factory


def start_server():
    server = LocalAstrBotServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


CHAT_PROVIDER_ID = "test-chat"
"""本地脚本化 chat provider 的 ID，供装配方与判定端绑定保持一致。"""

EMBEDDING_PROVIDER_ID = "test-embedding"
"""本地脚本化 embedding provider 的 ID。"""

DECISION_PROVIDER_ID = "test-decision"
"""本地脚本化 decision provider 的 ID。"""

RERANK_PROVIDER_ID = "test-rerank"
"""本地脚本化 rerank provider 的 ID。"""

GATE_JUDGE_TIMEOUT_SECONDS = 20
"""闸门 Chat 判定端调用超时（秒）。"""


def provider_configs(base_url):
    return {
        "chat": {
            "id": CHAT_PROVIDER_ID,
            "type": "openai_chat_completion",
            "provider_type": "chat_completion",
            "enable": True,
            # openai_chat_completion 的 get_keys 要求 key 为列表；decision 要求字符串
            "key": ["test-key"],
            "api_base": base_url,
            "model": "test-model",
            "timeout": 20,
        },
        "decision": {
            "id": DECISION_PROVIDER_ID,
            "type": "openai_decision",
            "provider_type": "decision",
            "enable": True,
            "key": "test-key",
            "api_base": base_url,
            "model": "test-decision",
        },
        "embedding": {
            "id": EMBEDDING_PROVIDER_ID,
            "type": "openai_embedding",
            "provider_type": "embedding",
            "enable": True,
            "embedding_api_key": "test-key",
            "embedding_api_base": base_url,
            "embedding_model": "test-embed",
            "embedding_dimensions": str(EMBED_DIM),
        },
        "rerank": {
            "id": RERANK_PROVIDER_ID,
            "type": "vllm_rerank",
            "provider_type": "rerank",
            "enable": True,
            "rerank_api_key": "test-key",
            "rerank_api_base": base_url,
            "rerank_api_suffix": "/rerank",
            "rerank_model": "test-rerank",
            "timeout": 20,
        },
    }


def _disable_client_retries(provider_manager, *provider_ids):
    """关闭 SDK 与 AstrBot 传输层重试，使脚本化单次错误立即抛错且不退避等待。"""
    from astrbot.core.provider.sources import request_retry

    request_retry.REQUEST_RETRY_ATTEMPTS = 1
    for provider_id in provider_ids:
        provider = provider_manager.inst_map.get(provider_id)
        client = getattr(provider, "client", None)
        if client is not None:
            client.max_retries = 0


async def build_real_context(
    server, *, with_decision=True, with_rerank=False, kb_name=None
):
    """装配真实 Context：真实 ProviderManager/KB/ConversationManager 等。"""
    from astrbot.core import astrbot_config, db_helper, sp
    from astrbot.core.astrbot_config_mgr import AstrBotConfigManager
    from astrbot.core.conversation_mgr import ConversationManager
    from astrbot.core.cron.manager import CronJobManager
    from astrbot.core.knowledge_base.kb_mgr import KnowledgeBaseManager
    from astrbot.core.persona_mgr import PersonaManager
    from astrbot.core.platform.manager import PlatformManager
    from astrbot.core.provider.manager import ProviderManager
    from astrbot.core.star.context import Context
    from astrbot.core.umop_config_router import UmopConfigRouter

    await db_helper.initialize()
    acm = AstrBotConfigManager(
        default_config=astrbot_config,
        ucr=UmopConfigRouter(sp=sp),
        sp=sp,
    )
    persona_manager = PersonaManager(db_helper, acm)
    provider_manager = ProviderManager(acm, db_helper, persona_manager)
    configs = provider_configs(server.base_url)
    await provider_manager.load_provider(configs["chat"])
    await provider_manager.load_provider(configs["embedding"])
    _disable_client_retries(provider_manager, CHAT_PROVIDER_ID, EMBEDDING_PROVIDER_ID)
    if with_decision:
        await provider_manager.load_provider(configs["decision"])
    if with_rerank:
        await provider_manager.load_provider(configs["rerank"])

    kb_manager = KnowledgeBaseManager(provider_manager)
    await kb_manager.initialize()
    if kb_name:
        await kb_manager.create_kb(
            kb_name,
            embedding_provider_id=EMBEDDING_PROVIDER_ID,
            rerank_provider_id=RERANK_PROVIDER_ID if with_rerank else None,
        )

    platform_manager = PlatformManager(astrbot_config, Queue())
    conversation_manager = ConversationManager(db_helper, acm)
    from astrbot.core.platform_message_history_mgr import PlatformMessageHistoryManager

    message_history_manager = PlatformMessageHistoryManager(db_helper)
    cron_manager = CronJobManager(db_helper)
    context = Context(
        Queue(),
        astrbot_config,
        db_helper,
        provider_manager,
        platform_manager,
        conversation_manager,
        message_history_manager,
        persona_manager,
        acm,
        kb_manager,
        cron_manager,
    )
    return context


def real_message_event(
    *,
    umo="webchat:FriendMessage:tester",
    sender_id="tester",
    sender_name="测试用户",
    message_str="今天聊点什么",
    support_ui_interaction=False,
    admin=False,
    self_id="bot",
    group_id="",
):
    """构造真实 AstrMessageEvent（与 AgentWakeupEvent 同一构造面）。"""
    from astrbot.core.platform.astr_message_event import AstrMessageEvent
    from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
    from astrbot.core.platform.message_type import MessageType
    from astrbot.core.platform.platform_metadata import PlatformMetadata

    platform_id, kind, session_id = umo.split(":", 2)
    is_group = kind in ("GroupMessage", "group") or bool(group_id)
    message = AstrBotMessage()
    message.type = MessageType.GROUP_MESSAGE if is_group else MessageType.FRIEND_MESSAGE
    message.session_id = session_id
    message.sender = MessageMember(user_id=sender_id, nickname=sender_name)
    if is_group:
        message.group_id = group_id or session_id
    message.self_id = self_id
    platform_meta = PlatformMetadata(
        name=platform_id,
        description="test platform",
        id=platform_id,
        support_ui_interaction=support_ui_interaction,
    )
    event = AstrMessageEvent(
        message_str=message_str,
        message_obj=message,
        platform_meta=platform_meta,
        session_id=session_id,
    )
    if admin:
        event.role = "admin"
    return event


def make_memory_meta(
    uri: str,
    *,
    scope: str = "personal",
    owner: str = "qq_u1",
    umo: str = "qq:private:u1",
    domain: str = "facts",
    created_at: str = "2026-11-01T00:00:00+00:00",
    updated_at: str = "",
    recall: int = 0,
    created_by: str = "maintenance_curator",
    deprecated: bool = False,
    **extra,
) -> dict:
    """构造记忆记录元数据，字段与运行期写入保持一致。"""
    meta = {
        "uri": uri,
        "is_memory_record": True,
        "deprecated": deprecated,
        "memory_scope": scope,
        "owner_user_id": owner,
        "umo": umo,
        "domain": domain,
        "created_at": created_at,
        "updated_at": updated_at,
        "recall_count": recall,
        "created_by": created_by,
    }
    meta.update(extra)
    return meta


async def collect_command_text(generator) -> list[str]:
    """收集异步生成器产出的文本：事件结果取纯文本，其余转字符串。"""
    texts: list[str] = []
    async for item in generator:
        if hasattr(item, "get_plain_text"):
            texts.append(item.get_plain_text())
        else:
            texts.append(str(item))
    return texts


class RealStackTestCase(unittest.IsolatedAsyncioTestCase):
    """真实 Context + 真实 KB + 真实 MemoryManager 的公共装配基类。"""

    async def start_stack(
        self,
        *,
        with_decision: bool = True,
        with_rerank: bool = False,
        kb_name: str | None = None,
        config: dict | None = None,
        memories=(),
        with_plugin: bool = False,
    ):
        """装配真实栈并把 KB、MemoryManager、可选插件绑定到测试实例。"""
        self.server = start_server()
        self.kb_name = kb_name or f"stack-{uuid.uuid4().hex[:10]}"
        self.context = await build_real_context(
            self.server,
            with_decision=with_decision,
            with_rerank=with_rerank,
            kb_name=self.kb_name,
        )
        self.config = {"kb_name": self.kb_name, **(config or {})}
        self.mgr = MemoryManager(kb_mgr=self.context.kb_manager, config=self.config)
        self.mgr.initialize()
        await self.mgr.connect_kb()
        kb = await self.context.kb_manager.get_kb_by_name(self.kb_name)
        for doc_id, content, meta in memories:
            await kb.vec_db.insert(content, meta, doc_id)
        self.plugin = None
        if with_plugin:
            self.plugin = MemoryPlugin(self.context, dict(self.config))
            self.plugin.memory_mgr = self.mgr
        return self

    async def active_docs(self, limit: int = 100) -> list[dict]:
        docs = await self.mgr.vec_db.document_storage.get_documents(
            metadata_filters={"is_memory_record": True, "deprecated": False},
            limit=limit,
            offset=0,
        )
        return docs

    async def doc_meta(self, uri: str) -> dict:
        docs = await self.mgr.vec_db.document_storage.get_documents(
            metadata_filters={"uri": uri, "is_memory_record": True}, limit=1
        )
        raw = docs[0].get("metadata", {})
        if isinstance(raw, str):
            raw = json.loads(raw)
        return raw

    def make_llm(self, *, max_calls: int = 50, cache_enabled: bool = False):
        """构造走本地脚本化 chat provider 的后台整理 LLM。"""
        from maintenance.llm import MaintenanceLLM

        cache_dir = tempfile.mkdtemp(prefix="real-stack-llm-cache-")
        self.addCleanup(shutil.rmtree, cache_dir, True)
        return MaintenanceLLM(
            context=self.context,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
            default_model_id=CHAT_PROVIDER_ID,
            max_calls_per_cycle=max_calls,
        )

    def tearDown(self):
        server = getattr(self, "server", None)
        if server is not None:
            server.shutdown()
            server.server_close()

"""真实 AstrBot 栈测试辅助：标准库本地 HTTP 服务 + 真实核心组件装配。

不模拟任何 AstrBot 接口：ProviderManager、DecisionService、KnowledgeBase、
Context、AstrMessageEvent 全部使用真实类，LLM/Embedding/Decision 请求由
标准库 http.server 提供的本地服务应答。
"""

from __future__ import annotations

import json
import threading
from asyncio import Queue
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EMBED_DIM = 8


def _embedding_payload(count=1):
    return {
        "object": "list",
        "model": "test-embed",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": [0.1 * (i + 1)] * EMBED_DIM,
            }
            for i in range(count)
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
            content = server.next_chat_content()
            if content is None:
                self._reply(500, {"error": "no scripted chat response"})
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
            inputs = request.get("input")
            count = len(inputs) if isinstance(inputs, list) else 1
            self._reply(200, _embedding_payload(count))
        elif path.endswith("/decisions"):
            answers = server.next_decision_answers(request)
            if answers is None:
                self._reply(500, {"error": "no scripted decision response"})
                return
            self._reply(200, {"answers": answers})
        else:
            self._reply(404, {"error": f"unexpected path {path}"})


class LocalAstrBotServer(ThreadingHTTPServer):
    """本地 HTTP 服务：脚本化 chat / decision 响应队列。"""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self._chat_contents: list[str] = []
        self._default_chat_content = " {} "
        self._decision_factory = None
        self.request_log: list[dict] = []

    def next_chat_content(self):
        self.request_log.append({"kind": "chat"})
        if self._chat_contents:
            return self._chat_contents.pop(0)
        return self._default_chat_content

    def queue_chat(self, *contents):
        self._chat_contents.extend(contents)

    def queue_chat_message(self, *messages):
        """入队完整 assistant message（如带 tool_calls 的脚本化工具调用）。"""
        self._chat_contents.extend(messages)

    def set_default_chat(self, content):
        self._default_chat_content = content

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
            "id": "test-decision",
            "type": "openai_decision",
            "provider_type": "decision",
            "enable": True,
            "key": "test-key",
            "api_base": base_url,
            "model": "test-decision",
        },
        "embedding": {
            "id": "test-embedding",
            "type": "openai_embedding",
            "provider_type": "embedding",
            "enable": True,
            "embedding_api_key": "test-key",
            "embedding_api_base": base_url,
            "embedding_model": "test-embed",
            "embedding_dimensions": str(EMBED_DIM),
        },
    }


async def build_real_context(server, *, with_decision=True, kb_name=None):
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
    if with_decision:
        await provider_manager.load_provider(configs["decision"])

    kb_manager = KnowledgeBaseManager(provider_manager)
    await kb_manager.initialize()
    if kb_name:
        await kb_manager.create_kb(kb_name, embedding_provider_id="test-embedding")

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
):
    """构造真实 AstrMessageEvent（与 AgentWakeupEvent 同一构造面）。"""
    from astrbot.core.platform.astr_message_event import AstrMessageEvent
    from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
    from astrbot.core.platform.message_type import MessageType
    from astrbot.core.platform.platform_metadata import PlatformMetadata

    platform_id, _kind, session_id = umo.split(":", 2)
    message = AstrBotMessage()
    message.type = MessageType.FRIEND_MESSAGE
    message.session_id = session_id
    message.sender = MessageMember(user_id=sender_id, nickname=sender_name)
    message.self_id = "bot"
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
    return event

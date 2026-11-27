// 记忆待审卡片：只通过 window.AstrBotPluginView 官方桥接与宿主通信
(function () {
  const bridge = window.AstrBotPluginView;
  const itemsEl = document.querySelector("#items");
  const noticeEl = document.querySelector("#notice");
  const errorEl = document.querySelector("#error");
  const emptyEl = document.querySelector("#empty");
  let busy = false;

  function t(key, fallback) {
    return bridge.t(`views.memory_review.${key}`, fallback);
  }

  function renderItems(items) {
    itemsEl.replaceChildren();
    emptyEl.hidden = items.length > 0;
    emptyEl.textContent = t("empty", "无待审争议项");
    for (const item of items) {
      itemsEl.appendChild(renderItem(item));
    }
  }

  function renderItem(item) {
    const card = document.createElement("div");
    card.className = "item";

    const head = document.createElement("div");
    head.className = "item-head";
    const id = document.createElement("span");
    id.className = "item-id";
    id.textContent = `#${item.id}`;
    const opType = document.createElement("span");
    opType.className = "op-type";
    opType.textContent = String(item.op_type || "");
    head.append(id, opType);

    const reason = document.createElement("p");
    reason.className = "reason";
    reason.textContent = String(item.reason || "");
    const summary = document.createElement("p");
    summary.className = "summary";
    summary.textContent = String(item.summary || "");
    const createdAt = document.createElement("p");
    createdAt.className = "created-at";
    createdAt.textContent = String(item.created_at || "");

    const actions = document.createElement("div");
    actions.className = "actions";
    const approveButton = document.createElement("button");
    approveButton.type = "button";
    approveButton.className = "approve";
    approveButton.textContent = t("button_approve", "通过");
    approveButton.addEventListener("click", () => submit("approve", item.id));
    const rejectButton = document.createElement("button");
    rejectButton.type = "button";
    rejectButton.textContent = t("button_reject", "废案");
    rejectButton.addEventListener("click", () => submit("reject", item.id));
    actions.append(approveButton, rejectButton);

    card.append(head, reason, summary, createdAt, actions);
    return card;
  }

  function setButtonsDisabled(disabled) {
    for (const button of document.querySelectorAll("button")) {
      button.disabled = disabled;
    }
  }

  function render(data) {
    if (!data || typeof data !== "object") return;
    renderItems(Array.isArray(data.items) ? data.items : []);
    noticeEl.textContent = String(data.notice || "");
    errorEl.textContent = String(data.error || "");
  }

  async function submit(actionId, id) {
    if (busy) return;
    busy = true;
    noticeEl.textContent = "";
    errorEl.textContent = "";
    setButtonsDisabled(true);
    try {
      await bridge.view.action(actionId, { id });
    } catch (failure) {
      errorEl.textContent =
        failure.message || t("action_failed", "操作失败，请重试");
    } finally {
      busy = false;
      setButtonsDisabled(false);
    }
  }

  async function start() {
    await bridge.ready();
    bridge.view.subscribe(({ data }) => render(data));
    bridge.onContext(({ locale }) => {
      document.documentElement.lang = locale;
    });
  }
  start().catch((failure) => {
    const fallback = bridge
      ? t("action_failed", "操作失败，请重试")
      : "操作失败，请重试";
    errorEl.textContent = (failure && failure.message) || fallback;
  });
})();

(function (win, doc) {
  if (win.RagWidget) return;

  const WIDGETS = {};
  const EVENTS = {};

  function originOf(u) {
    try { return new URL(u).origin; } catch { return '*'; }
  }

  function createWrapper(cfg) {
    const id = cfg.id || 'rag-widget-' + Math.random().toString(36).slice(2, 9);
    if (doc.getElementById(id + '-wrapper')) return id;

    const wrapper = doc.createElement('div');
    wrapper.id = id + '-wrapper';
    wrapper.style.position = 'fixed';
    wrapper.style.bottom = cfg.bottom || '20px';
    wrapper.style.right = cfg.right || '20px';
    wrapper.style.zIndex = String(cfg.zIndex || 9999);

    // Shadow DOM isolation
    const shadowRoot = wrapper.attachShadow({ mode: 'open' });
    const style = doc.createElement('style');
    style.textContent = `
      .rag-widget-container {
        all: initial;
        width: auto;
        height: auto;
        background: transparent !important;
      }
      iframe {
        width: ${cfg.width || '340px'};
        height: ${cfg.height || '480px'};
        border: none;
        border-radius: ${cfg.borderRadius || '12px'};
        box-shadow: ${cfg.boxShadow || '0 2px 12px rgba(0,0,0,0.15)'};
        background: transparent !important;
      }
    `;
    const container = doc.createElement('div');
    container.className = 'rag-widget-container';

    const iframe = doc.createElement('iframe');
    iframe.id = id;
    iframe.setAttribute('title', cfg.ariaTitle || 'RAG Support Chat');
    iframe.allow = cfg.allow || 'clipboard-read; clipboard-write; microphone; camera;';
    iframe.src = (cfg.widgetUrl || '') +
      (cfg.widgetUrl && cfg.widgetUrl.indexOf('?') === -1 ? '?' : '&') +
      new URLSearchParams({ widgetId: id, hostOrigin: location.origin }).toString();

    container.appendChild(iframe);
    shadowRoot.appendChild(style);
    shadowRoot.appendChild(container);
    doc.body.appendChild(wrapper);

    WIDGETS[id] = {
      id,
      wrapperId: wrapper.id,
      iframeId: iframe.id,
      widgetUrl: cfg.widgetUrl,
      origin: originOf(cfg.widgetUrl)
    };

    return id;
  }

  function postToIframe(widgetId, msg) {
    const w = WIDGETS[widgetId];
    if (!w) return false;
    const iframe = doc.getElementById(w.iframeId);
    if (!iframe || !iframe.contentWindow) return false;
    iframe.contentWindow.postMessage(Object.assign({ __RAG_WIDGET__: true }, msg), w.origin || '*');
    return true;
  }

  function handleHostMessage(e) {
    if (!e.data) return;
    if (e.data.__RAG_WIDGET_EVENT__ === true) {
      const { widgetId, name, payload } = e.data;
      const listeners = EVENTS[widgetId] && EVENTS[widgetId][name];
      if (listeners) listeners.forEach(cb => { try { cb(payload); } catch (er) { console.error(er); } });
    }
  }
  win.addEventListener('message', handleHostMessage, false);

  win.RagWidget = {
    init(cfg = {}) {
      if (!cfg.widgetUrl) { console.error('RagWidget.init(): widgetUrl is required'); return null; }
      const id = createWrapper(cfg);

      // handle widget ready handshake
      const readyHandler = function (e) {
        if (!e.data || e.data.__RAG_WIDGET_READY__ !== true) return;
        if (e.origin !== originOf(cfg.widgetUrl)) return;
        if (e.data.widgetId !== id) return;

        const token = (typeof cfg.getToken === 'function') ? cfg.getToken() : (cfg.token || null);
        postToIframe(id, { type: 'init', payload: { apiBase: cfg.apiBase || '', token, config: cfg.config || {}, allowedHostOrigin: location.origin } });
        win.removeEventListener('message', readyHandler);
      };
      win.addEventListener('message', readyHandler);

      return {
        id,
        open() {
          postToIframe(id, { type: 'open' });
        },
        close() {
          postToIframe(id, { type: 'close' });
        },
        sendMessage(payload) {
          postToIframe(id, { type: 'message', payload });
        },
        on(eventName, cb) {
          EVENTS[id] = EVENTS[id] || {};
          EVENTS[id][eventName] = EVENTS[id][eventName] || [];
          EVENTS[id][eventName].push(cb);
        },
        off(eventName, cb) {
          if (!EVENTS[id] || !EVENTS[id][eventName]) return;
          EVENTS[id][eventName] = EVENTS[id][eventName].filter(f => f !== cb);
        },
        destroy() {
          const wrapper = doc.getElementById(id + '-wrapper');
          if (wrapper) wrapper.remove();
          delete WIDGETS[id];
          delete EVENTS[id];
        }
      };
    }
  };
})(window, document);

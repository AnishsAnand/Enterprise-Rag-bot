(function (win, doc) {
  if (win.RagWidget) return;

  const WIDGETS = {};
  const EVENTS = {};

  function originOf(u) {
    try { return new URL(u).origin; } catch { return '*'; }
  }

  function cssVal(v) {
    return (typeof v === 'number') ? (v + 'px') : (v || '');
  }

  function createWrapper(cfg) {
    const id = cfg.id || 'rag-widget-' + Math.random().toString(36).slice(2, 9);
    if (doc.getElementById(id + '-wrapper')) return id;

    // host wrapper (OUTSIDE iframe) - this will carry the visual border-radius & drop-shadow
    const wrapper = doc.createElement('div');
    wrapper.id = id + '-wrapper';
    wrapper.className = 'rag-widget-host';
    wrapper.style.position = 'fixed';
    wrapper.style.bottom = cssVal(cfg.bottom || '20px');
    wrapper.style.right = cssVal(cfg.right || '20px');
    wrapper.style.zIndex = String(cfg.zIndex || 9999);
    wrapper.style.pointerEvents = 'auto';
    wrapper.style.display = 'block';
    wrapper.style.margin = '0';
    wrapper.style.padding = '0';
    wrapper.style.overflow = 'visible';
    // visual on wrapper:
    const borderRadius = cssVal(cfg.borderRadius || '12px');
    const boxShadow = cfg.boxShadow || '0 10px 35px rgba(0,0,0,0.25)';
    wrapper.style.borderRadius = borderRadius;
    wrapper.style.boxShadow = boxShadow;
    // Keep wrapper background transparent so shadow composites over host page
    wrapper.style.background = 'transparent';
    wrapper.style.border = 'none';
    wrapper.style.transform = 'translateZ(0)';

    // Shadow DOM for isolation of the embed internals
    const shadowRoot = wrapper.attachShadow({ mode: 'open' });

    // Insert scoped stylesheet into shadow root
    const style = doc.createElement('style');
    const width = cssVal(cfg.width || '340px');
    const height = cssVal(cfg.height || '480px');

    style.textContent = `
      :host { all: initial; display:block; }
      .widget-container { width: ${width}; height: ${height}; display:block; background: transparent; border-radius: ${borderRadius}; overflow: visible; }
      .frame-wrap { width: ${width}; height: ${height}; display:block; background: transparent; border-radius: ${borderRadius}; overflow: visible; }
      /* IFRAME: must be transparent and have no visual shadow - wrapper has it */
      iframe {
        width: 100%;
        height: 100%;
        border: none;
        border-radius: ${borderRadius};
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        display:block;
        overflow:hidden;
        -webkit-transform: translateZ(0);
        transform: translateZ(0);
        backface-visibility: hidden;
        -webkit-backface-visibility: hidden;
      }
      iframe::-webkit-scrollbar { display: none !important; }
    `;

    const container = doc.createElement('div');
    container.className = 'widget-container';
    container.style.background = 'transparent';
    container.style.borderRadius = borderRadius;
    container.style.overflow = 'visible';

    const frameWrap = doc.createElement('div');
    frameWrap.className = 'frame-wrap';
    frameWrap.style.background = 'transparent';
    frameWrap.style.borderRadius = borderRadius;
    frameWrap.style.overflow = 'visible';

    const iframe = doc.createElement('iframe');
    iframe.id = id;
    iframe.setAttribute('title', cfg.ariaTitle || 'RAG Support Chat');
    iframe.allow = cfg.allow || 'clipboard-read; clipboard-write; microphone; camera;';
    // intentionally transparent iframe
    iframe.setAttribute('allowTransparency', 'true');
    iframe.setAttribute('frameborder', '0');
    iframe.setAttribute('scrolling', 'no');
    iframe.loading = cfg.loading || 'auto';
    iframe.style.background = 'transparent';
    iframe.style.backgroundColor = 'transparent';
    iframe.style.borderRadius = borderRadius;
    iframe.style.overflow = 'hidden';

    // construct iframe src
    const widgetUrl = (cfg.widgetUrl || '').trim();
    if (!widgetUrl) {
      console.error('RagWidget: widgetUrl is required');
      return id;
    }
    const sep = widgetUrl.indexOf('?') === -1 ? '?' : '&';
    iframe.src = widgetUrl + sep + new URLSearchParams({ widgetId: id, hostOrigin: location.origin }).toString();

    // Append into shadow DOM
    frameWrap.appendChild(iframe);
    container.appendChild(frameWrap);
    shadowRoot.appendChild(style);
    shadowRoot.appendChild(container);

    // Append wrapper to page (wrapper lives in host, carries shadow)
    doc.body.appendChild(wrapper);

    // Save metadata
    WIDGETS[id] = {
      id,
      wrapperId: wrapper.id,
      iframeId: iframe.id,
      widgetUrl: widgetUrl,
      origin: originOf(widgetUrl),
      wrapperEl: wrapper,
      iframeEl: iframe
    };

    // When iframe loads: attempt to ensure the widget interior panel is solid white (only if same-origin)
    iframe.addEventListener('load', function () {
      try {
        const docInside = iframe.contentDocument || iframe.contentWindow.document;
        if (!docInside) return;
        // Ensure the root inside iframe remains transparent but the visible panel (.widget-content/.widget-panel) is white.
        const s = docInside.createElement('style');
        s.setAttribute('data-rag-inject', 'true');
        s.textContent = `
          html, body { margin:0; padding:0; background: transparent !important; background-color: transparent !important; }
          /* The visible panel inside the widget must be solid */
          .widget-content, .widget-panel, .widget-root, .widget-body {
            background: #ffffff !important;
            background-color: #ffffff !important;
            border-radius: ${borderRadius} !important;
            box-shadow: none !important; /* avoid duplicate shadow inside iframe */
            isolation: isolate !important;
          }
        `;
        if (docInside.head) docInside.head.insertBefore(s, docInside.head.firstChild);
      } catch (e) {
        // cross-origin – cannot inject. That's OK: host wrapper provides the visual shadow; iframe remains transparent.
      }
    }, false);

    return id;
  }

  function postToIframe(widgetId, msg) {
    const info = WIDGETS[widgetId];
    if (!info) return false;
    const iframe = document.getElementById(info.iframeId);
    if (!iframe || !iframe.contentWindow) return false;
    try {
      iframe.contentWindow.postMessage(Object.assign({ __RAG_WIDGET__: true }, msg), info.origin || '*');
      return true;
    } catch (e) {
      try { iframe.contentWindow.postMessage(Object.assign({ __RAG_WIDGET__: true }, msg), '*'); return true; }
      catch (err) { console.error('postToIframe failed', err); return false; }
    }
  }

  function handleHostMessage(e) {
    if (!e || !e.data) return;
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

      const readyHandler = function (e) {
        if (!e || !e.data || e.data.__RAG_WIDGET_READY__ !== true) return;
        // allow either wildcard or exact origin (best-effort)
        const expectedOrigin = originOf(cfg.widgetUrl);
        if (expectedOrigin !== '*' && e.origin !== expectedOrigin) return;
        if (e.data.widgetId !== id) return;

        const token = (typeof cfg.getToken === 'function') ? cfg.getToken() : (cfg.token || null);
        postToIframe(id, { type: 'init', payload: { apiBase: cfg.apiBase || '', token, config: cfg.config || {}, allowedHostOrigin: location.origin } });
        win.removeEventListener('message', readyHandler);
      };
      win.addEventListener('message', readyHandler);

      return {
        id,
        open() { postToIframe(id, { type: 'open' }); },
        close() { postToIframe(id, { type: 'close' }); },
        sendMessage(payload) { postToIframe(id, { type: 'message', payload }); },
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

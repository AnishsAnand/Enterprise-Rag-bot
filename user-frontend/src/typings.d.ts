// src/typings.d.ts
declare interface RagInitPayload {
  apiBase?: string;
  token?: string | null;
  config?: any;
  allowedHostOrigin?: string;
}

declare interface Window {
  RagWidgetBridge?: {
    notifyHost: (name: string, payload?: any) => void;
    getInit?: () => RagInitPayload | undefined;
  };
  __RAG_INIT__?: RagInitPayload;
}

// widget-bridge.service.ts
import { Injectable, NgZone, OnDestroy } from '@angular/core';
import { Observable, Subject } from 'rxjs';

interface RagInitPayload {
  apiBase?: string;
  token?: string | null;
  config?: any;
  allowedHostOrigin?: string;
}

@Injectable({ providedIn: 'root' })
export class WidgetBridgeService implements OnDestroy {
  private initSubject = new Subject<RagInitPayload>();
  private openSubject = new Subject<void>();
  private closeSubject = new Subject<void>();
  private messageSubject = new Subject<any>();

  init$ = this.initSubject.asObservable();
  open$ = this.openSubject.asObservable();
  close$ = this.closeSubject.asObservable();
  message$ = this.messageSubject.asObservable();

  // in-memory token (do not store in localStorage)
  private _token: string | null = null;
  get token() { return this._token; }
  set token(t: string | null) { this._token = t; }

  constructor(private zone: NgZone) {
    // wire DOM events -> RxJS subjects inside Angular zone
    window.addEventListener('rag:init', (ev: any) => {
      this.zone.run(() => {
        const payload = ev.detail as RagInitPayload;
        this._token = payload?.token ?? null;
        this.initSubject.next(payload);
      });
    });

    window.addEventListener('rag:open', () => {
      this.zone.run(() => this.openSubject.next());
    });

    window.addEventListener('rag:close', () => {
      this.zone.run(() => this.closeSubject.next());
    });

    window.addEventListener('rag:message', (ev: any) => {
      this.zone.run(() => this.messageSubject.next(ev.detail));
    });
  }

  // send event back to host page
  notifyHost(name: string, payload: any) {
    try {
      // use RagWidgetBridge.notifyHost if available (host-bridge.js exposes it)
      const bridge: any = (window as any).RagWidgetBridge;
      if (bridge && typeof bridge.notifyHost === 'function') {
        bridge.notifyHost(name, payload);
        return;
      }
      // fallback
      const msg = { __RAG_WIDGET_EVENT__: true, widgetId: new URL(location.href).searchParams.get('widgetId') || 'unknown', name, payload };
      window.parent.postMessage(msg, '*');
    } catch (e) {
      console.warn('notifyHost error', e);
    }
  }

  // helper to clear token (logout)
  clearToken() { this._token = null; }

  ngOnDestroy() {
    this.initSubject.complete();
    this.openSubject.complete();
    this.closeSubject.complete();
    this.messageSubject.complete();
  }
}

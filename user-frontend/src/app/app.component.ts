import { AfterViewInit, Component, OnDestroy, ViewChild } from '@angular/core';
import { Subscription } from 'rxjs';
import { WidgetBridgeService } from './services/widget-bridge.service';
import { ChatComponent } from './components/chat/chat.component';

@Component({
  selector: 'app-root',
  // Keep the chat component as the only element. It contains the widget UI (FAB, header, body).
  template: `<app-chat #chat></app-chat>`
})
export class AppComponent implements AfterViewInit, OnDestroy {
  @ViewChild('chat') chat!: ChatComponent;

  private subs: Subscription[] = [];

  constructor(private widgetBridge: WidgetBridgeService) {}

  ngAfterViewInit(): void {
    // When the host posts init payload, set API base/token into your chat service via the bridge (WidgetBridgeService already stores token).
    this.subs.push(this.widgetBridge.init$.subscribe(init => {
      // init: { apiBase, token, config, allowedHostOrigin }
      // If you need to pass apiBase/token to ChatService, do so here.
      // Example (uncomment when you have chatService.setBase/setToken methods)
      // this.chatService.setBase(init.apiBase);
      // this.chatService.setToken(init.token);

      // Optionally auto open when configured
      if (init?.config?.autoOpen) {
        this.openWidget();
      }
    }));

    // Host asked to open the widget
    this.subs.push(this.widgetBridge.open$.subscribe(() => this.openWidget()));

    // Host asked to close the widget
    this.subs.push(this.widgetBridge.close$.subscribe(() => this.closeWidget()));

    // Host sent prefill / query messages
    this.subs.push(this.widgetBridge.message$.subscribe(msg => {
      if (!msg) return;
      if (msg.type === 'prefill' && typeof msg.query === 'string') {
        // set the chat input only
        this.safePatchQuery(msg.query);
      } else if (msg.type === 'query' && typeof msg.query === 'string') {
        // set input and immediately send
        this.safePatchQuery(msg.query);
        // small timeout to allow form binding to update
        setTimeout(() => {
          if (this.chat) this.chat.send();
        }, 50);
      } else {
        // any other types you can handle here
        console.debug('Widget message', msg);
      }
    }));
  }

  ngOnDestroy(): void {
    this.subs.forEach(s => s.unsubscribe());
  }

  // ----- helpers that control the ChatComponent safely -----
  private openWidget(): void {
    if (!this.chat) return;
    // ensure the chat is expanded (ChatComponent has isExpanded and toggleWidget)
    if (!this.chat.isExpanded) {
      // prefer changing state directly (safer than toggling blindly)
      this.chat.isExpanded = true;
    }
  }

  private closeWidget(): void {
    if (!this.chat) return;
    if (this.chat.isExpanded) {
      this.chat.isExpanded = false;
    }
  }

  private safePatchQuery(q: string): void {
    if (!this.chat || !this.chat.chatForm) return;
    // patch only the query form control; keep depth intact
    try {
      this.chat.chatForm.patchValue({ query: q });
    } catch (e) {
      console.warn('safePatchQuery error', e);
    }
  }
}

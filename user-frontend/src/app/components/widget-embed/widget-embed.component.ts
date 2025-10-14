import { Component, OnInit, OnDestroy, Renderer2, ElementRef, Inject, Injectable, } from '@angular/core';
import { DOCUMENT } from '@angular/common';
import { OverlayContainer } from '@angular/cdk/overlay';
import { Platform } from '@angular/cdk/platform';

@Injectable()
export class WidgetOverlayContainer extends OverlayContainer {
  constructor(@Inject(DOCUMENT) _doc: any, protected override _platform: Platform) {
    super(_doc, _platform);
  }

  override _createContainer(): void {
    super._createContainer();
    const container = this._containerElement;
    if (container) {
      container.style.background = 'transparent';
      container.style.backgroundColor = 'transparent';
      container.style.boxShadow = 'none';
      container.style.border = 'none';
    }
  }
}

@Component({
  selector: 'app-widget-embed',
  templateUrl: './widget-embed.component.html',
  styleUrls: ['./widget-embed.component.scss'],
  providers: [
    {
      provide: OverlayContainer,
      useClass: WidgetOverlayContainer
    }
  ]
})
export class WidgetEmbedComponent implements OnInit, OnDestroy {
  widgetId = '';
  hostOrigin = '*';
  boundHandler!: (e: MessageEvent) => void;

  constructor(
    private renderer: Renderer2,
    private elRef: ElementRef,
    @Inject(DOCUMENT) private document: Document
  ) {}

  ngOnInit(): void {
    // Transparent backgrounds
    this.renderer.setStyle(this.document.documentElement, 'background', 'transparent');
    this.renderer.setStyle(this.document.documentElement, 'background-color', 'transparent');
    this.renderer.setStyle(this.document.body, 'background', 'transparent');
    this.renderer.setStyle(this.document.body, 'background-color', 'transparent');
    this.renderer.setStyle(this.elRef.nativeElement, 'background', 'transparent');
    this.renderer.setStyle(this.elRef.nativeElement, 'background-color', 'transparent');

    const params = new URLSearchParams(location.search);
    this.widgetId = params.get('widgetId') || '';
    this.hostOrigin = params.get('hostOrigin') || '*';

    const ack = () => {
      window.parent.postMessage({ __RAG_WIDGET_READY__: true, widgetId: this.widgetId }, this.hostOrigin || '*');
    };
    ack();
    setTimeout(ack, 50);

    this.boundHandler = (e: MessageEvent) => {
      const originOk = !this.hostOrigin || this.hostOrigin === '*' || e.origin === this.hostOrigin;
      if (!originOk) return;
      const data = e.data || {};
      if (!data.__RAG_WIDGET__) return;

      const t = data.type;
      const p = data.payload;

      switch (t) {
        case 'init':
          if (p?.theme === 'dark') {
            this.renderer.addClass(this.document.body, 'dark-theme');
          } else {
            this.renderer.removeClass(this.document.body, 'dark-theme');
          }
          break;
      }
    };

    window.addEventListener('message', this.boundHandler, false);
  }

  ngOnDestroy(): void {
    if (this.boundHandler) window.removeEventListener('message', this.boundHandler);
  }
}

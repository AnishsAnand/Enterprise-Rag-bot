// src/app/components/chat/chat.component.ts
import {
  Component,
  OnInit,
  OnDestroy,
  AfterViewChecked,
  ViewChild,
  ElementRef
} from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { finalize } from 'rxjs';
import { ChatService, ChatResponse, UserQueryRequest, SummaryItem } from '../../services/chat.service';

interface Point {
  index?: number;
  text: string;
  image?: string | null;
  caption?: string | null;
}

interface Message {
  from: 'user' | 'bot' | 'system';
  text: string;
  ts?: string;
  images: string[];
  steps?: Point[];
  summary?: SummaryItem[];
  summaryTitle?: string;
}

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css']
})
export class ChatComponent implements OnInit, OnDestroy, AfterViewChecked {
  @ViewChild('chatMessages') chatMessages!: ElementRef<HTMLElement>;

  // Create control without template-disabled attribute to avoid Angular forms warning
  chatForm: FormGroup;
  messages: Message[] = [];
  loading = false;
  errorMsg = '';

  isExpanded = false;
  private widgetId = '';
  private allowedHostOrigin: string | null = null;
  private shouldScroll = true;

  private onInitHandler = (ev: any) => this.handleHostInit(ev);
  private onOpenHandler = () => (this.isExpanded = true);
  private onCloseHandler = () => (this.isExpanded = false);
  private onMessageHandler = (ev: any) => this.handleHostMessage(ev);

  constructor(private fb: FormBuilder, private chatSvc: ChatService) {
    // avoid using `disabled` attribute in template; set disabled in control creation if needed
    this.chatForm = this.fb.group({
      query: [{ value: '', disabled: false }, [Validators.required, Validators.minLength(1), Validators.maxLength(500)]]
    });
  }

  ngOnInit(): void {
    this.messages.push({
      from: 'system',
      text: 'Chat',
      images: [],
      summary: []
    });

    try {
      const params = new URLSearchParams(location.search);
      this.widgetId = params.get('widgetId') || '';
      const hostOrigin = params.get('hostOrigin');
      if (hostOrigin) this.allowedHostOrigin = hostOrigin;
    } catch {}

    window.addEventListener('rag:init', this.onInitHandler as EventListener);
    window.addEventListener('rag:open', this.onOpenHandler as EventListener);
    window.addEventListener('rag:close', this.onCloseHandler as EventListener);
    window.addEventListener('rag:message', this.onMessageHandler as EventListener);
  }

  ngAfterViewChecked(): void {
    if (this.shouldScroll) {
      this.scrollToBottom();
    }
  }

  ngOnDestroy(): void {
    window.removeEventListener('rag:init', this.onInitHandler as EventListener);
    window.removeEventListener('rag:open', this.onOpenHandler as EventListener);
    window.removeEventListener('rag:close', this.onCloseHandler as EventListener);
    window.removeEventListener('rag:message', this.onMessageHandler as EventListener);
  }

  private handleHostInit(ev: any) {
    const payload = ev.detail || {};
    if (payload?.allowedHostOrigin) {
      this.allowedHostOrigin = payload.allowedHostOrigin;
    }
  }

  private handleHostMessage(ev: any) {
    const p = ev.detail;
    if (p?.text) {
      this.messages.push({
        from: 'bot',
        text: p.text,
        ts: new Date().toISOString(),
        images: []
      });
      this.shouldScroll = true;
    }
  }

  toggleWidget(): void {
    this.isExpanded = !this.isExpanded;
    if (this.isExpanded) {
      this.shouldScroll = true;
    }
  }

  private normalizeImageToUrl(img: any): string | null {
    if (!img) return null;
    if (typeof img === 'string') return img.trim() || null;
    if (typeof img === 'object') {
      return (img.url || img.src || img.image || null);
    }
    return null;
  }

  formatTimestamp(ts: string | undefined): string {
    if (!ts) return '';
    const date = new Date(ts);
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const day = date.getDate().toString().padStart(2, '0');
    const year = date.getFullYear();
    const hours = date.getHours().toString().padStart(2, '0');
    const mins = date.getMinutes().toString().padStart(2, '0');
    return `${month}/${day}/${year}, ${hours}:${mins}`;
  }

  send(): void {
    if (this.chatForm.invalid) return;
    const q = this.chatForm.value.query.trim();
    if (!q) return;

    const userMsg: Message = {
      from: 'user',
      text: q,
      ts: new Date().toISOString(),
      images: [],
      summary: []
    };
    this.messages.push(userMsg);
    this.shouldScroll = true;
    this.loading = true;
    this.errorMsg = '';

    const req: UserQueryRequest = {
      query: q,
      max_results: 20,
      include_images: true
    };

    this.chatSvc.query(req)
      .pipe(finalize(() => {
        // always run this when stream completes (success/error)
        this.loading = false;
        this.shouldScroll = true;
      }))
      .subscribe({
        next: (res: ChatResponse) => {
          const answerText = (res.answer || '').trim() || 'I found some information for you.';
          const hasSteps = Array.isArray(res.steps) && res.steps.length > 0;

          // Process all images from response
          const allImages = (res.images ?? [])
            .map(i => this.normalizeImageToUrl(i))
            .filter((u): u is string => !!u);

          if (hasSteps) {
            // DISTRIBUTE IMAGES TO STEPS
            const stepsWithImages: Point[] = (res.steps ?? []).map((s: any, idx: number) => {
              // Check if step already has an image
              let stepImage = this.normalizeImageToUrl(s.image);

              // If step doesn't have image, assign from allImages array
              if (!stepImage && allImages.length > idx) {
                stepImage = allImages[idx];
              }

              return {
                index: s.index ?? idx + 1,
                text: s.text ?? '',
                image: stepImage,
                caption: s.caption ?? null
              };
            });

            // Show answer text as a separate bubble first
            this.messages.push({
              from: 'bot',
              text: answerText,
              ts: res.timestamp || new Date().toISOString(),
              images: [],
              summary: []
            });

            // Then show steps with distributed images
            this.messages.push({
              from: 'bot',
              text: res.stepsTitle || 'Step-by-step instructions',
              ts: new Date().toISOString(),
              images: [],
              steps: stepsWithImages,
              summary: []
            });
          } else {
            // No steps - show answer with all images
            this.messages.push({
              from: 'bot',
              text: answerText,
              ts: res.timestamp || new Date().toISOString(),
              images: allImages,
              summary: []
            });
          }

          // Handle summary
          if (res.summary) {
            let summaryItems: SummaryItem[] = [];
            if (typeof res.summary === 'string') {
              summaryItems = [{ content: res.summary }];
            } else if (Array.isArray(res.summary)) {
              summaryItems = res.summary;
            } else {
              summaryItems = [res.summary as SummaryItem];
            }

            this.messages.push({
              from: 'bot',
              text: res.summaryTitle || 'Quick Summary',
              ts: new Date().toISOString(),
              images: [],
              summary: summaryItems,
              summaryTitle: res.summaryTitle || 'Quick Summary'
            });
          }
        },
        error: (err: any) => {
          console.error('Chat query error', err);
          const message = err?.message || '⚠️ I could not answer your question at this time.';
          this.messages.push({
            from: 'bot',
            text: message,
            ts: new Date().toISOString(),
            images: [],
            summary: []
          });
        }
      });

    this.chatForm.patchValue({ query: '' });
  }

  private scrollToBottom(): void {
    try {
      const el = this.chatMessages?.nativeElement;
      if (el) {
        requestAnimationFrame(() => {
          try {
            el.scrollTop = el.scrollHeight;
          } catch {}
        });
        this.shouldScroll = false;
      }
    } catch {}
  }

  onImageError(event: Event): void {
    const img = event.target as HTMLImageElement;
    if (img) img.style.display = 'none';
  }

  openImage(url: string | undefined): void {
    if (url) window.open(url, '_blank');
  }

  trackByIndex(i: number): number {
    return i;
  }
}

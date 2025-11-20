import {
  Component,
  OnInit,
  OnDestroy,
  AfterViewChecked,
  ViewChild,
  ElementRef
} from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Subject } from 'rxjs';
import { takeUntil, finalize } from 'rxjs/operators';
import { ChatService, ChatResponse, UserQueryRequest, SummaryItem, SessionParameter } from '../../services/chat.service';

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
  isInteractive?: boolean;
  options?: string[];
  parameter?: SessionParameter;
  sessionId?: string;
  status?: string;
  executionTime?: number;
  details?: any;
}

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css']
})
export class ChatComponent implements OnInit, OnDestroy, AfterViewChecked {
  @ViewChild('chatMessages') chatMessages!: ElementRef<HTMLDivElement>;

  chatForm: FormGroup;
  messages: Message[] = [];
  loading = false;
  errorMsg = '';
  isExpanded = false;

  // Action agent session tracking
  currentSessionId: string | null = null;
  isInActionSession = false;
  currentActionStatus: string | null = null;

  private widgetId = '';
  private allowedHostOrigin: string | null = null;
  private shouldScroll = true;
  private destroy$ = new Subject<void>();

  private onInitHandler = (ev: any) => this.handleHostInit(ev);
  private onOpenHandler = () => (this.isExpanded = true);
  private onCloseHandler = () => (this.isExpanded = false);
  private onMessageHandler = (ev: any) => this.handleHostMessage(ev);

  constructor(private fb: FormBuilder, private chatSvc: ChatService) {
    this.chatForm = this.fb.group({
      query: [
        { value: '', disabled: false },
        [Validators.required, Validators.minLength(1), Validators.maxLength(500)]
      ]
    });
  }

  ngOnInit(): void {
    // Initial system message
    this.messages.push({
      from: 'system',
      text: '🤖 Unified RAG + Action Agent Ready\n\nCapabilities:\n• Answer questions with step-by-step guidance\n• Manage Kubernetes clusters\n• Execute automated tasks\n\nTry: "List all clusters" or "How do I...?"',
      images: [],
      summary: []
    });

    try {
      const params = new URLSearchParams(location.search);
      this.widgetId = params.get('widgetId') || '';
      const hostOrigin = params.get('hostOrigin');
      if (hostOrigin) this.allowedHostOrigin = hostOrigin;
    } catch (err) {
      console.error('Failed to parse URL parameters', err);
    }

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

    this.destroy$.next();
    this.destroy$.complete();
  }

  private handleHostInit(ev: any) {
    const payload = ev?.detail || {};
    if (payload?.allowedHostOrigin) {
      this.allowedHostOrigin = payload.allowedHostOrigin;
    }
  }

  private handleHostMessage(ev: any) {
    const p = ev?.detail || {};
    if (p?.text) {
      this.messages.push({
        from: 'bot',
        text: String(p.text),
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
    if (img == null) return null;
    if (typeof img === 'string') {
      const trimmed = img.trim();
      return trimmed || null;
    }
    if (typeof img === 'object') {
      const url = img.url || img.src || img.image || null;
      return url ? String(url).trim() : null;
    }
    return null;
  }

  formatTimestamp(ts: string | undefined): string {
    if (!ts) return '';
    try {
      const date = new Date(ts);
      const month = (date.getMonth() + 1).toString().padStart(2, '0');
      const day = date.getDate().toString().padStart(2, '0');
      const year = date.getFullYear();
      const hours = date.getHours().toString().padStart(2, '0');
      const mins = date.getMinutes().toString().padStart(2, '0');
      return `${month}/${day}/${year}, ${hours}:${mins}`;
    } catch {
      return '';
    }
  }

  /**
   * Handle query submission
   */
  send(): void {
    if (this.chatForm.invalid) return;

    const q = String(this.chatForm.value.query || '').trim();
    if (!q) return;

    // Add user message
    const userMsg: Message = {
      from: 'user',
      text: q,
      ts: new Date().toISOString(),
      images: []
    };
    this.messages.push(userMsg);
    this.shouldScroll = true;
    this.loading = true;
    this.errorMsg = '';

    // Build request
    const req: UserQueryRequest = {
      query: q,
      max_results: 20,
      include_images: true,
      session_id: this.currentSessionId || undefined
    };

    console.log('🚀 Sending query:', req);

    this.chatSvc.query(req)
      .pipe(
        takeUntil(this.destroy$),
        finalize(() => {
          this.loading = false;
          this.shouldScroll = true;
        })
      )
      .subscribe({
        next: (res: ChatResponse) => this.handleChatResponse(res),
        error: (err: any) => this.handleChatError(err)
      });

    this.chatForm.patchValue({ query: '' });
  }

  /**
   * Handle response from backend
   */
  private handleChatResponse(res: ChatResponse): void {
    console.log('📨 Processing response:', res);

    // Handle action agent interactions
    if (res.status === 'parameter_collection' || res.status === 'awaiting_confirmation') {
      this.isInActionSession = true;
      this.currentSessionId = res.session_id || null;
      this.currentActionStatus = res.status;

      const botMsg: Message = {
        from: 'bot',
        text: res.message || 'Please provide the required information',
        ts: new Date().toISOString(),
        images: [],
        isInteractive: true,
        options: res.options || [],
        parameter: res.parameter,
        sessionId: res.session_id,
        status: res.status
      };

      this.messages.push(botMsg);
      this.shouldScroll = true;
      return;
    }

    // Handle completed actions
    if (res.status === 'completed') {
      this.isInActionSession = false;
      const resultMsg: Message = {
        from: 'bot',
        text: res.message || 'Task completed successfully!',
        ts: new Date().toISOString(),
        images: [],
        status: 'completed',
        details: res.details,
        executionTime: res.execution_time
      };
      this.messages.push(resultMsg);
      this.shouldScroll = true;
      return;
    }

    // Handle failed actions
    if (res.status === 'failed' || res.status === 'cancelled') {
      this.isInActionSession = false;
      const resultMsg: Message = {
        from: 'bot',
        text: res.message || `Action ${res.status}`,
        ts: new Date().toISOString(),
        images: [],
        status: res.status
      };
      this.messages.push(resultMsg);
      this.shouldScroll = true;
      return;
    }

    // Handle normal chat/RAG responses
    const answerText = (res.answer || '').trim() || 'I found some information for you.';
    const hasSteps = Array.isArray(res.steps) && res.steps.length > 0;

    const allImages = (res.images ?? [])
      .map(i => this.normalizeImageToUrl(i))
      .filter((u): u is string => !!u);

    // Add main answer message
    this.messages.push({
      from: 'bot',
      text: answerText,
      ts: res.timestamp || new Date().toISOString(),
      images: hasSteps ? [] : allImages,
      summary: []
    });

    // Add steps if available
    if (hasSteps) {
      const stepsWithImages: Point[] = (res.steps ?? []).map((s: any, idx: number) => {
        let stepImage = this.normalizeImageToUrl(s?.image);
        if (!stepImage && allImages.length > idx) {
          stepImage = allImages[idx];
        }
        return {
          index: s?.index ?? idx + 1,
          text: s?.text ?? '',
          image: stepImage,
          caption: s?.caption ?? null
        };
      });

      this.messages.push({
        from: 'bot',
        text: res.stepsTitle || 'Step-by-step instructions',
        ts: new Date().toISOString(),
        images: [],
        steps: stepsWithImages,
        summary: []
      });
    }

    // Add summary if available
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

    this.shouldScroll = true;
  }

  /**
   * Handle error response
   */
  private handleChatError(err: any): void {
    console.error('❌ Chat error:', err);
    
    const message = err?.message || 'I could not process your request at this time.';
    
    this.messages.push({
      from: 'bot',
      text: `⚠️ ${message}`,
      ts: new Date().toISOString(),
      images: []
    });
    
    this.shouldScroll = true;
  }

  /**
   * Handle option selection (for action agent)
   */
  selectOption(option: string): void {
    if (!this.isInActionSession || !this.currentSessionId) {
      console.warn('❌ Not in active session');
      return;
    }

    console.log('📤 Selecting option:', option, 'SessionID:', this.currentSessionId);

    // Add user's choice as message
    this.messages.push({
      from: 'user',
      text: option,
      ts: new Date().toISOString(),
      images: []
    });

    this.shouldScroll = true;
    this.loading = true;
    this.errorMsg = '';

    // Continue conversation with selected option
    this.chatSvc.continueConversation(this.currentSessionId, option)
      .pipe(
        takeUntil(this.destroy$),
        finalize(() => {
          this.loading = false;
          this.shouldScroll = true;
        })
      )
      .subscribe({
        next: (res: ChatResponse) => this.handleContinuationResponse(res),
        error: (err: any) => this.handleChatError(err)
      });
  }

  /**
   * Handle continuation response from action agent
   */
  private handleContinuationResponse(res: ChatResponse): void {
    console.log('📨 Continuation response:', res);

    // Re-use the same response handler
    this.handleChatResponse(res);

    // If action is complete, clear session tracking
    if (res.status === 'completed' || res.status === 'failed' || res.status === 'cancelled') {
      this.isInActionSession = false;
      this.currentSessionId = null;
      this.currentActionStatus = null;
    }
  }

  /**
   * Scroll chat to bottom
   */
  private scrollToBottom(): void {
    try {
      const el = this.chatMessages?.nativeElement;
      if (el) {
        requestAnimationFrame(() => {
          try {
            el.scrollTop = el.scrollHeight;
          } catch (err) {
            console.error('Scroll error:', err);
          }
        });
        this.shouldScroll = false;
      }
    } catch (err) {
      console.error('scrollToBottom error:', err);
    }
  }

  /**
   * Handle image errors
   */
  onImageError(event: Event): void {
    const img = event.target as HTMLImageElement;
    if (img) img.style.display = 'none';
  }

  /**
   * Open image in new tab
   */
  openImage(url: string | undefined): void {
    if (url) window.open(url, '_blank');
  }

  /**
   * Track by index for ngFor
   */
  trackByIndex(index: number, _item: any): number {
    return index;
  }

  /**
   * Get CSS class for message
   */
  getMessageClass(message: Message): string {
    let cls = `message message-${message.from}`;
    if (message.isInteractive) cls += ' message-interactive';
    if (message.steps?.length) cls += ' message-with-steps';
    if (message.summary?.length) cls += ' message-with-summary';
    return cls;
  }

  /**
   * Check if message has images
   */
  hasImages(msg: Message): boolean {
    return Array.isArray(msg.images) && msg.images.length > 0;
  }

  /**
   * Check if message has steps
   */
  hasSteps(msg: Message): boolean {
    return Array.isArray(msg.steps) && msg.steps.length > 0;
  }

  /**
   * Check if message has summary
   */
  hasSummary(msg: Message): boolean {
    return Array.isArray(msg.summary) && msg.summary.length > 0;
  }
}
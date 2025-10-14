import { 
  Component,
  OnInit,
  ViewChild,
  ElementRef,
  AfterViewChecked,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatCardModule } from '@angular/material/card';
import { Router } from '@angular/router';

interface StepItem {
  text: string;
  type?: string;
}

interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
  steps?: StepItem[];
  summary?: string;
}

@Component({
  selector: 'app-user-chat',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatIconModule,
    MatButtonModule,
    MatInputModule,
    MatFormFieldModule,
    MatProgressSpinnerModule,
    MatCardModule,
  ],
  templateUrl: './user-chat.component.html',
  styleUrls: ['./user-chat.component.css'],
})
export class UserChatComponent implements OnInit, AfterViewChecked {
  @ViewChild('chatMessages') chatMessages!: ElementRef;

  isExpanded = true;
  currentMessage = '';
  isTyping = false;
  chatHistory: ChatMessage[] = [];

  private apiUrl = 'http://localhost:8001/api/chat';

  constructor(private http: HttpClient, private router: Router) {}

  ngOnInit() {
    this.initializeChat();
  }

  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  toggleWidget() {
    this.isExpanded = !this.isExpanded;
    if (this.isExpanded) {
      setTimeout(() => this.scrollToBottom(), 150);
    }
  }

  logout() {
    localStorage.removeItem('token');
    window.location.href = '/login';
  }

  initializeChat() {
    this.chatHistory = [
      {
        id: 'welcome-bot',
        type: 'bot',
        content: 'Hello! I\'m here to help you find information. Ask me anything!',
        timestamp: new Date(),
      },
    ];
  }

  async sendMessage() {
    const trimmed = this.currentMessage.trim();
    if (!trimmed) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: trimmed,
      timestamp: new Date(),
    };
    this.chatHistory.push(userMessage);
    this.currentMessage = '';
    this.isTyping = true;

    try {
      const response: any = await this.http
        .post<any>(`${this.apiUrl}/query`, {
          query: trimmed,
          max_results: 50,
          search_depth: 'balanced',
        })
        .toPromise();

      const botMessage: ChatMessage = {
        id: Date.now().toString() + '_bot',
        type: 'bot',
        content: response?.answer || 'No response received.',
        timestamp: new Date(),
        steps: response?.steps || [],
        summary: response?.summary || '',
      };
      this.chatHistory.push(botMessage);
    } catch (error) {
      this.chatHistory.push({
        id: Date.now().toString() + '_error',
        type: 'bot',
        content: 'I\'m having trouble connecting right now. Please try again in a moment.',
        timestamp: new Date(),
      });
    } finally {
      this.isTyping = false;
    }
  }

  scrollToBottom() {
    try {
      const el = this.chatMessages?.nativeElement;
      if (el) el.scrollTop = el.scrollHeight;
    } catch {}
  }

  get isLoginPage(): boolean {
    return this.router.url.includes('/login');
  }
}

import { 
  Component,
  OnInit,
  ViewChild,
  ElementRef,
  HostListener,
  AfterViewChecked,
  Input,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTabsModule } from '@angular/material/tabs';
import { MatCardModule } from '@angular/material/card';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatChipsModule } from '@angular/material/chips';
import { jwtDecode } from 'jwt-decode';
import { Router } from '@angular/router';

interface ImageData {
  url: string;
  alt?: string;
  type?: string;
  caption?: string;
  class?: string;
}

interface StepItem {
  text: string;
  image?: ImageData;
}

interface SourceItem {
  title?: string;
  url: string;
  relevance_score: number;
  images?: ImageData[];
}

interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
  images?: ImageData[];
  steps?: StepItem[];
  sources?: SourceItem[];
  summary?: string;
  expanded_context?: string;
}

interface KnowledgeStats {
  document_count: number;
  status: string;
  last_updated: string;
}

// API Configuration interfaces
interface APIKeyConfig {
  openrouter_api_key: string;
  voyage_api_key: string;
  ollama_base_url: string;
}

interface ServiceHealth {
  available: boolean;
  status: string;
  models?: string[];
}

interface APIKeyStatus {
  keys: APIKeyConfig;
  service_health: {
    services: {
      openrouter: ServiceHealth;
      voyage: ServiceHealth;
      ollama: ServiceHealth;
    };
    overall_status: string;
  };
  config_file_exists: boolean;
}

@Component({
  selector: 'app-rag-widget',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatIconModule,
    MatButtonModule,
    MatInputModule,
    MatFormFieldModule,
    MatProgressSpinnerModule,
    MatTabsModule,
    MatCardModule,
    MatTooltipModule,
    MatExpansionModule,
    MatChipsModule,
  ],
  templateUrl: './rag-widget.component.html',
  styleUrls: ['./rag-widget.component.css'],
})
export class RagWidgetComponent implements OnInit, AfterViewChecked {
  @ViewChild('chatMessages') chatMessages!: ElementRef;
  @Input() forceUserRole: 'admin' | 'user' | null = null;

  isExpanded = true;
  currentMessage = '';
  scrapeUrl = '';
  bulkScrapeUrl = '';

  isTyping = false;
  isScraping = false;
  isBulkScraping = false;

  chatHistory: ChatMessage[] = [];
  scrapeStatus: {
    title?: string;
    message?: string;
    details?: string;
    summary?: string;
  } | null = null;

  knowledgeStats: KnowledgeStats = {
    document_count: 0,
    status: 'unknown',
    last_updated: '',
  };
  
  uploading = false;
  progress = 0;
  uploadStatus = '';
  uploadError = '';
  uploadedFileName = '';
  uploadSummary = '';
  isAdmin = true;

  // API Configuration properties
  apiKeyStatus: APIKeyStatus = {
  keys: {
    openrouter_api_key: '',
    voyage_api_key: '',
    ollama_base_url: 'http://localhost:11434'
  },
  service_health: {
    services: {
      openrouter: { available: false, status: 'unknown', models: [] },
      voyage: { available: false, status: 'unknown', models: [] },
      ollama: { available: false, status: 'unknown', models: [] }
    },
    overall_status: 'unknown'
  },
  config_file_exists: false
};

  apiKeyLoading = false;
  apiKeyError = '';
  editingService: string | null = null;
  
  // Form inputs for API keys
  openrouterKey = '';
  voyageKey = '';
  ollamaUrl = 'http://localhost:11434';

  private apiUrl = 'http://localhost:8000/api';

  constructor(private http: HttpClient, private router: Router) {}

  async ngOnInit() {
    this.isExpanded = true;
    this.setRoleFromToken();
    this.initializeChat();
    this.refreshStats();
    
    // Initialize API configuration for admins
    if (this.isAdmin) {
      await this.loadAPIKeyStatus();
    }
  }

  setRoleFromToken() {
    if (this.forceUserRole) {
      this.isAdmin = this.forceUserRole === 'admin';
      return;
    }

    const token = localStorage.getItem('token');
    if (!token) {
      this.isAdmin = false;
      return;
    }

    try {
      const decoded: any = jwtDecode(token);
      this.isAdmin = decoded?.role === 'admin';
    } catch (err) {
      console.error('Invalid token:', err);
      this.isAdmin = false;
    }
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
    const welcomeMessage = this.isAdmin ? 'Welcome, Admin!' : 'Welcome';
    this.chatHistory = [
      {
        id: 'welcome-bot',
        type: 'bot',
        content: welcomeMessage,
        timestamp: new Date(),
      },
    ];
  }

  // API Configuration Methods
  async loadAPIKeyStatus() {
    this.apiKeyLoading = true;
    this.apiKeyError = '';
    
    try {
      const response = await this.http
        .get<APIKeyStatus>(`${this.apiUrl}/rag-widget/widget/api-keys`)
        .toPromise();
      
      this.apiKeyStatus = response!;
      
      // Pre-fill form fields with current values (masked)
      this.openrouterKey = this.apiKeyStatus.keys.openrouter_api_key || '';
      this.voyageKey = this.apiKeyStatus.keys.voyage_api_key || '';
      this.ollamaUrl = this.apiKeyStatus.keys.ollama_base_url || 'http://localhost:11434';
      
    } catch (error: any) {
      this.apiKeyError = 'Failed to load API configuration: ' + (error?.error?.detail || error.message);
    } finally {
      this.apiKeyLoading = false;
    }
  }

  async updateAPIKey(service: string) {
    this.apiKeyLoading = true;
    this.apiKeyError = '';
    
    try {
      const payload: any = { service };
      
      if (service === 'openrouter') {
        payload.api_key = this.openrouterKey || null;
      } else if (service === 'voyage') {
        payload.api_key = this.voyageKey || null;
      } else if (service === 'ollama') {
        payload.base_url = this.ollamaUrl || null;
      }
      
      const response = await this.http
        .post<any>(`${this.apiUrl}/rag-widget/widget/api-keys`, payload)
        .toPromise();
      
      // Show success message
      this.chatHistory.push({
        id: Date.now().toString(),
        type: 'bot',
        content: `✅ ${response?.message || 'API key updated successfully'}`,
        timestamp: new Date(),
      });
      
      // Refresh status
      await this.loadAPIKeyStatus();
      this.editingService = null;
      
    } catch (error: any) {
      this.apiKeyError = 'Failed to update API key: ' + (error?.error?.detail || error.message);
    } finally {
      this.apiKeyLoading = false;
    }
  }

  async testAPIKeys() {
    this.apiKeyLoading = true;
    this.apiKeyError = '';
    
    try {
      const response = await this.http
        .post<any>(`${this.apiUrl}/rag-widget/widget/api-keys/test`, {})
        .toPromise();
      
      this.apiKeyStatus!.service_health = response;
      
      // Show test results in chat
      const healthyServices = Object.values(response.services)
        .filter((s: any) => s.status === 'healthy' || s.status === 'assumed_healthy').length;
      const totalServices = Object.keys(response.services).length;
      
      this.chatHistory.push({
        id: Date.now().toString(),
        type: 'bot',
        content: `🔧 API Test Results: ${healthyServices}/${totalServices} services healthy. Overall status: ${response.overall_status}`,
        timestamp: new Date(),
      });
      
    } catch (error: any) {
      this.apiKeyError = 'Failed to test API keys: ' + (error?.error?.detail || error.message);
    } finally {
      this.apiKeyLoading = false;
    }
  }

  async clearAllAPIKeys() {
    if (!confirm('Are you sure you want to clear all API keys? This will disable all AI services.')) {
      return;
    }
    
    this.apiKeyLoading = true;
    this.apiKeyError = '';
    
    try {
      await this.http
        .delete(`${this.apiUrl}/rag-widget/widget/api-keys`)
        .toPromise();
      
      // Clear form fields
      this.openrouterKey = '';
      this.voyageKey = '';
      this.ollamaUrl = 'http://localhost:11434';
      
      this.chatHistory.push({
        id: Date.now().toString(),
        type: 'bot',
        content: '🗑️ All API keys have been cleared.',
        timestamp: new Date(),
      });
      
      await this.loadAPIKeyStatus();
      
    } catch (error: any) {
      this.apiKeyError = 'Failed to clear API keys: ' + (error?.error?.detail || error.message);
    } finally {
      this.apiKeyLoading = false;
    }
  }

  startEditing(service: string) {
    this.editingService = service;
    this.apiKeyError = '';
  }

  cancelEditing() {
    this.editingService = null;
    this.apiKeyError = '';
    
    // Reset form fields to current values
    if (this.apiKeyStatus) {
      this.openrouterKey = this.apiKeyStatus.keys.openrouter_api_key || '';
      this.voyageKey = this.apiKeyStatus.keys.voyage_api_key || '';
      this.ollamaUrl = this.apiKeyStatus.keys.ollama_base_url || 'http://localhost:11434';
    }
  }

  getServiceStatusIcon(service: ServiceHealth): string {
    switch (service.status) {
      case 'healthy':
      case 'assumed_healthy':
        return 'check_circle';
      case 'unhealthy':
      case 'error':
        return 'error';
      case 'no_models':
        return 'warning';
      default:
        return 'help';
    }
  }

  getServiceStatusColor(service: ServiceHealth): string {
    switch (service.status) {
      case 'healthy':
      case 'assumed_healthy':
        return 'green';
      case 'unhealthy':
      case 'error':
        return 'red';
      case 'no_models':
        return 'orange';
      default:
        return 'gray';
    }
  }

  isServiceConfigured(serviceName: string): boolean {
    if (!this.apiKeyStatus) return false;
    
    const key = this.apiKeyStatus.keys[serviceName as keyof APIKeyConfig];
    return Boolean(key && key.length > 0);
  }

  // Existing methods (chat, scraping, file upload, etc.)
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
        .post<any>(`${this.apiUrl}/rag-widget/widget/query`, {
          query: trimmed,
          max_results: 8,
          include_sources: true,
        })
        .toPromise();

      const botMessage: ChatMessage = {
        id: Date.now().toString() + '_bot',
        type: 'bot',
        content: response?.answer || 'No response received.',
        timestamp: new Date(),
        images: response?.images || [],
        steps: response?.steps || [],
        sources: response?.sources || [],
        summary: response?.summary || '',
        expanded_context: response?.expanded_context || '',
      };
      this.chatHistory.push(botMessage);
    } catch (error) {
      this.chatHistory.push({
        id: Date.now().toString() + '_error',
        type: 'bot',
        content:
          'Error while processing your question. Check if backend is running or try again later.',
        timestamp: new Date(),
      });
    } finally {
      this.isTyping = false;
    }
  }

  async scrapeSingleUrl() {
    if (!this.scrapeUrl) return;
    this.isScraping = true;
    this.scrapeStatus = null;

    try {
      const response = await this.http
        .post<any>(`${this.apiUrl}/rag-widget/widget/scrape`, {
          url: this.scrapeUrl,
          store_in_knowledge: true,
        })
        .toPromise();

      this.scrapeStatus = {
        title: 'Scraping Successful!',
        message: `Scraped "${response?.title}" and added to knowledge base.`,
        details: `Content length: ${response?.content_length}, Images: ${response?.images_count || 0}, Method: ${response?.method_used}`,
        summary: response?.summary || '',
      };
      this.scrapeUrl = '';
      this.refreshStats();
    } catch (error: any) {
      this.scrapeStatus = {
        title: 'Scraping Failed',
        message: error?.error?.detail || 'Unable to scrape URL. Please check and retry.',
      };
    } finally {
      this.isScraping = false;
    }
  }

  async bulkScrape() {
    if (!this.bulkScrapeUrl) return;
    this.isBulkScraping = true;
    this.scrapeStatus = null;

    try {
      const response = await this.http
        .post<any>(`${this.apiUrl}/rag-widget/widget/bulk-scrape`, {
          base_url: this.bulkScrapeUrl,
          max_depth: 2,
          max_urls: 50,
          auto_store: true,
        })
        .toPromise();

      this.scrapeStatus = {
        title: 'Bulk Scraping Started!',
        message: `Discovered ${response?.discovered_urls_count} URLs.`,
        details: `Estimated time: ${response?.estimated_time}`,
      };
      this.bulkScrapeUrl = '';

      setTimeout(() => this.refreshStats(), 10000);
    } catch (error: any) {
      this.scrapeStatus = {
        title: 'Bulk Scraping Failed',
        message: error?.error?.detail || 'Failed to bulk scrape. Check base URL.',
      };
    } finally {
      this.isBulkScraping = false;
    }
  }

  async refreshStats() {
    try {
      const stats = await this.http
        .get<KnowledgeStats>(`${this.apiUrl}/rag-widget/widget/knowledge-stats`)
        .toPromise();
      this.knowledgeStats = stats!;
    } catch (error) {
      console.error('Failed to load knowledge stats.');
    }
  }

  async clearKnowledge() {
    if (!confirm('Are you sure you want to clear the knowledge base?')) return;

    try {
      await this.http
        .delete(`${this.apiUrl}/rag-widget/widget/clear-knowledge`)
        .toPromise();

      this.knowledgeStats = {
        document_count: 0,
        status: 'cleared',
        last_updated: '',
      };

      this.chatHistory.push({
        id: Date.now().toString(),
        type: 'bot',
        content: 'Knowledge base cleared. You can now scrape fresh content.',
        timestamp: new Date(),
      });
    } catch (error) {
      alert('Failed to clear knowledge base.');
    }
  }

  scrollToBottom() {
    try {
      const el = this.chatMessages?.nativeElement;
      if (el) el.scrollTop = el.scrollHeight;
    } catch {}
  }

  @HostListener('document:keydown.escape', ['$event'])
  onEscapeKey() {
    if (this.isExpanded) {
      this.toggleWidget();
    }
  }

  handleFileInput(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.uploadFile(input.files[0]);
    }
  }

  handleDrop(event: DragEvent): void {
    event.preventDefault();
    if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
      this.uploadFile(event.dataTransfer.files[0]);
    }
  }

  allowDrop(event: DragEvent): void {
    event.preventDefault();
  }

  get isLoginPage(): boolean {
    return this.router.url.includes('/login');
  }

  uploadFile(file: File): void {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('store_in_knowledge', 'true');

    this.uploading = true;
    this.progress = 0;
    this.uploadStatus = '';
    this.uploadError = '';
    this.uploadedFileName = '';
    this.uploadSummary = '';

    this.http
      .post<any>(`${this.apiUrl}/rag-widget/widget/upload-file`, formData, {
        reportProgress: true,
        observe: 'events',
      })
      .subscribe({
        next: (event: any) => {
          if (event.type === 1 && event.total) {
            this.progress = Math.round((100 * event.loaded) / event.total);
          } else if (event.type === 4) {
            this.uploadStatus = `Uploaded: ${event.body?.filename} (${event.body?.format})`;
            this.uploadSummary = event.body?.summary || '';
            this.refreshStats();
            this.uploading = false;
          }
        },
        error: (err) => {
          this.uploadError = 'Upload failed: ' + (err?.error?.detail || err.message);
          this.uploading = false;
        },
      });
  }

  getImageCount(source: any): number {
    return source?.images?.length || 0;
  }

  hasImages(source: any): boolean {
    return this.getImageCount(source) > 0;
  }

  openImageInNewTab(imageUrl: string): void {
    if (imageUrl) window.open(imageUrl, '_blank');
  }

  onImageError(event: any): void {
    event.target.style.display = 'none';
  }

  getImageTypeIcon(imageType: string): string {
    switch (imageType) {
      case 'diagram':
        return 'analytics';
      case 'logo/icon':
        return 'account_circle';
      case 'banner':
        return 'photo_library';
      default:
        return 'image';
    }
  }

  hasAnySourceWithImages(sources?: any[]): boolean {
    return sources?.some((s) => this.hasImages(s)) || false;
  }

  getSourcesWithImages(sources?: any[]): any[] {
    return sources?.filter((s) => this.hasImages(s)) || [];
  }

  getSourcesWithoutImages(sources?: any[]): any[] {
    return sources?.filter((s) => !this.hasImages(s)) || [];
  }

  formatBotMessage(text: string): string {
    if (text.match(/\d+\./)) {
      const steps = text.split(/\d+\.\s+/).filter(Boolean);
      return `<ol>${steps.map((s) => `<li>${s.trim()}</li>`).join('')}</ol>`;
    }
    return `<p>${text}</p>`;
  }
  
}

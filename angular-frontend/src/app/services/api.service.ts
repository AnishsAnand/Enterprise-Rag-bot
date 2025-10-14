import { Injectable } from '@angular/core'; 
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import type { Observable } from 'rxjs';

export interface ScrapeRequest {
  url: string;
  extract_text: boolean;
  extract_links: boolean;
  extract_images: boolean;
  extract_tables: boolean;
  output_format: string;
  wait_for_element?: string;
  scroll_page: boolean;
}

export interface BulkScrapeRequest {
  base_url: string;
  max_depth: number;
  max_urls: number;
  output_format: string;
  store_in_rag: boolean;
  scrape_params: {
    extract_text: boolean;
    extract_links: boolean;
    extract_images: boolean;
    extract_tables: boolean;
  };
}

export interface QueryRequest {
  query: string;
  max_results: number;
  include_context?: boolean;
  include_sources?: boolean;
}

export interface RagStats {
  document_count: number;
  status: string;
  last_updated: string;
}

@Injectable({
  providedIn: 'root',
})
export class ApiService {
  private baseUrl = 'http://localhost:8000/api';

  constructor(private http: HttpClient) {}

  private getAuthHeaders(): HttpHeaders {
    const token = localStorage.getItem('token');
    return new HttpHeaders({
      Authorization: token ? `Bearer ${token}` : '',
    });
  }

  // 🔐 Auth
  login(username: string, password: string): Observable<any> {
    const body = new HttpParams()
      .set('username', username)
      .set('password', password);

    return this.http.post(`${this.baseUrl}/auth/login`, body.toString(), {
      headers: new HttpHeaders({
        'Content-Type': 'application/x-www-form-urlencoded',
      }),
    });
  }

  register(username: string, password: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/auth/register`, { username, password });
  }

  getProfile(): Observable<any> {
    return this.http.get(`${this.baseUrl}/auth/me`, {
      headers: this.getAuthHeaders(),
    });
  }

  // 💬 RAG Bot (Widget-based)
  sendRagMessage(request: QueryRequest): Observable<any> {
    return this.http.post(`${this.baseUrl}/rag-widget/widget/query`, request, {
      headers: this.getAuthHeaders(),
    });
  }

  uploadFileToRag(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('store_in_knowledge', 'true');

    return this.http.post(`${this.baseUrl}/rag-widget/widget/upload-file`, formData, {
      headers: this.getAuthHeaders(),
    });
  }

  getRagKnowledgeStats(): Observable<any> {
    return this.http.get(`${this.baseUrl}/rag-widget/widget/knowledge-stats`, {
      headers: this.getAuthHeaders(),
    });
  }

  clearRagKnowledge(): Observable<any> {
    return this.http.delete(`${this.baseUrl}/rag-widget/widget/clear-knowledge`, {
      headers: this.getAuthHeaders(),
    });
  }

  // 🔍 Scraper
  scrapeSingleUrl(url: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/rag-widget/widget/scrape`, {
      url,
      store_in_knowledge: true,
    }, {
      headers: this.getAuthHeaders(),
    });
  }

  bulkScrape(base_url: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/rag-widget/widget/bulk-scrape`, {
      base_url,
      max_depth: 2,
      max_urls: 50,
      auto_store: true,
    }, {
      headers: this.getAuthHeaders(),
    });
  }

  // ✅ ADD THIS: To support admin dashboard
  getRagStats(): Observable<RagStats> {
    return this.http.get<RagStats>(`${this.baseUrl}/rag/stats`, {
      headers: this.getAuthHeaders(),
    });
  }

  // 🛠️ Admin-only APIs
  getSystemStats(): Observable<any> {
    return this.http.get(`${this.baseUrl}/admin/stats`, {
      headers: this.getAuthHeaders(),
    });
  }

  getSystemHealth(): Observable<any> {
    return this.http.get(`${this.baseUrl}/admin/health`, {
      headers: this.getAuthHeaders(),
    });
  }

  // 📩 Support
  submitSupportTicket(ticket: any): Observable<any> {
    return this.http.post(`${this.baseUrl}/support/tickets`, ticket, {
      headers: this.getAuthHeaders(),
    });
  }

  getSupportArticles(): Observable<any> {
    return this.http.get(`${this.baseUrl}/support/articles`, {
      headers: this.getAuthHeaders(),
    });
  }

  sendChatMessage(message: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/support/chat`, { message }, {
      headers: this.getAuthHeaders(),
    });
  }
}

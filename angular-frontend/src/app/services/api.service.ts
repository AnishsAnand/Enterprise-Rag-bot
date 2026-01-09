// api.service.ts - COMPLETE FIXED VERSION
import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, retry, timeout } from 'rxjs/operators';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private readonly API_URL = environment.apiUrl;
  private readonly REQUEST_TIMEOUT = 30000; // 30 seconds

  constructor(private http: HttpClient) {}

  // ==============================================
  // SCRAPING ENDPOINTS
  // ==============================================

  getScrapingStatus(): Observable<any> {
    return this.http.get(`${this.API_URL}/api/scraper/status`).pipe(
      timeout(this.REQUEST_TIMEOUT),
      retry(1),
      catchError(this.handleError)
    );
  }

  scrapeUrl(url: string, storeInKnowledge: boolean = true): Observable<any> {
    return this.http.post(`${this.API_URL}/api/scraper/scrape`, {
      url,
      store_in_knowledge: storeInKnowledge
    }).pipe(
      timeout(this.REQUEST_TIMEOUT),
      catchError(this.handleError)
    );
  }

  bulkScrape(baseUrl: string, maxDepth: number = 2, maxUrls: number = 50): Observable<any> {
    return this.http.post(`${this.API_URL}/api/scraper/bulk-scrape`, {
      base_url: baseUrl,
      max_depth: maxDepth,
      max_urls: maxUrls,
      auto_store: true
    }).pipe(
      timeout(60000), // Longer timeout for bulk operations
      catchError(this.handleError)
    );
  }

  // ==============================================
  // RAG ENDPOINTS
  // ==============================================

  getRagStats(): Observable<any> {
    return this.http.get(`${this.API_URL}/api/rag/stats`).pipe(
      timeout(this.REQUEST_TIMEOUT),
      retry(1),
      catchError(this.handleError)
    );
  }

  queryRag(query: string, maxResults: number = 8): Observable<any> {
    return this.http.post(`${this.API_URL}/api/rag/query`, {
      query,
      max_results: maxResults,
      include_sources: true
    }).pipe(
      timeout(this.REQUEST_TIMEOUT),
      catchError(this.handleError)
    );
  }

  clearKnowledge(): Observable<any> {
    return this.http.delete(`${this.API_URL}/api/rag/clear-knowledge`).pipe(
      timeout(this.REQUEST_TIMEOUT),
      catchError(this.handleError)
    );
  }

  // ==============================================
  // SYSTEM HEALTH ENDPOINTS
  // ==============================================

  getSystemHealth(): Observable<any> {
    return this.http.get(`${this.API_URL}/health/readiness`).pipe(
      timeout(10000),
      retry(1),
      catchError(this.handleError)
    );
  }

  getSystemStats(): Observable<any> {
    return this.http.get(`${this.API_URL}/api/info`).pipe(
      timeout(this.REQUEST_TIMEOUT),
      retry(1),
      catchError(this.handleError)
    );
  }

  checkLiveness(): Observable<any> {
    return this.http.get(`${this.API_URL}/health/liveness`).pipe(
      timeout(5000),
      retry(1),
      catchError(this.handleError)
    );
  }

  // ==============================================
  // RAG WIDGET ENDPOINTS
  // ==============================================

  getKnowledgeStats(): Observable<any> {
    return this.http.get(`${this.API_URL}/api/rag-widget/widget/knowledge-stats`).pipe(
      timeout(this.REQUEST_TIMEOUT),
      retry(1),
      catchError(this.handleError)
    );
  }

  uploadFile(file: File, storeInKnowledge: boolean = true): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('store_in_knowledge', storeInKnowledge.toString());

    return this.http.post(`${this.API_URL}/api/rag-widget/widget/upload-file`, formData, {
      reportProgress: true,
      observe: 'events'
    }).pipe(
      timeout(60000), // Longer timeout for file uploads
      catchError(this.handleError)
    );
  }

  // ==============================================
  // API KEY MANAGEMENT (ADMIN)
  // ==============================================

  getAPIKeyStatus(): Observable<any> {
    return this.http.get(`${this.API_URL}/api/rag-widget/widget/api-keys`).pipe(
      timeout(this.REQUEST_TIMEOUT),
      catchError(this.handleError)
    );
  }

  updateAPIKey(service: string, apiKey?: string, baseUrl?: string): Observable<any> {
    const payload: any = { service };
    
    if (apiKey !== undefined) {
      payload.api_key = apiKey;
    }
    
    if (baseUrl !== undefined) {
      payload.base_url = baseUrl;
    }

    return this.http.post(`${this.API_URL}/api/rag-widget/widget/api-keys`, payload).pipe(
      timeout(this.REQUEST_TIMEOUT),
      catchError(this.handleError)
    );
  }

  testAPIKeys(): Observable<any> {
    return this.http.post(`${this.API_URL}/api/rag-widget/widget/api-keys/test`, {}).pipe(
      timeout(this.REQUEST_TIMEOUT),
      catchError(this.handleError)
    );
  }

  clearAllAPIKeys(): Observable<any> {
    return this.http.delete(`${this.API_URL}/api/rag-widget/widget/api-keys`).pipe(
      timeout(this.REQUEST_TIMEOUT),
      catchError(this.handleError)
    );
  }

  // ==============================================
  // ERROR HANDLING
  // ==============================================

  private handleError(error: HttpErrorResponse): Observable<never> {
    let errorMessage = 'An error occurred';

    if (error.error instanceof ErrorEvent) {
      // Client-side or network error
      errorMessage = `Client Error: ${error.error.message}`;
      console.error('Client-side error:', error.error);
    } else {
      // Backend returned an unsuccessful response code
      if (error.status === 0) {
        errorMessage = 'Cannot connect to server. Please check if the backend is running.';
        console.error('❌ Backend connection failed. Is the server running on ' + environment.apiUrl + '?');
      } else if (error.status === 401) {
        errorMessage = 'Unauthorized. Please login again.';
      } else if (error.status === 403) {
        errorMessage = 'Access forbidden. You do not have permission.';
      } else if (error.status === 404) {
        errorMessage = 'Resource not found.';
      } else if (error.status === 500) {
        errorMessage = 'Server error. Please try again later.';
      } else if (error.error?.detail) {
        errorMessage = error.error.detail;
      } else if (error.message) {
        errorMessage = error.message;
      }

      console.error(
        `❌ Backend returned code ${error.status}, ` +
        `body was: ${JSON.stringify(error.error)}`
      );
    }

    return throwError(() => ({
      status: error.status,
      message: errorMessage,
      error: error.error
    }));
  }

  // ==============================================
  // UTILITY METHODS
  // ==============================================

  /**
   * Check if backend is reachable
   */
  async checkBackendConnection(): Promise<boolean> {
    try {
      await this.http.get(`${this.API_URL}/health/liveness`, {
        headers: new HttpHeaders().set('skip-auth', 'true')
      }).toPromise();
      console.log('✅ Backend is reachable');
      return true;
    } catch (error) {
      console.error('❌ Backend is not reachable:', error);
      return false;
    }
  }

  /**
   * Get API URL for external use
   */
  getApiUrl(): string {
    return this.API_URL;
  }
}
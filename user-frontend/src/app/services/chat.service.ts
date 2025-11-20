import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import { Observable, map, catchError, throwError } from 'rxjs';
import { environment } from '../../environments/environments';

// ✅ Shared interfaces
export interface UserQueryRequest {
  query: string;
  max_results?: number;
  include_images?: boolean;
  session_id?: string;
}

export interface ContinueConversationRequest {
  session_id: string;
  user_input: string;
}

export interface SummaryItem {
  title?: string;
  content: string;
  bullets?: string[];
}

export interface SessionParameter {
  name: string;
  description: string;
  type: string;
  choices?: string[];
  default?: any;
}

export interface ActionResponse {
  status: string;
  message: string;
  intent_type?: string;
  parameter?: SessionParameter;
  options?: string[];
  session_id?: string;
  details?: any;
  execution_time?: number;
  collected_so_far?: { [key: string]: any };
  pending?: string[];
}

export interface ChatResponse extends ActionResponse {
  query?: string;
  answer?: string;
  steps?: Array<{ index?: number; text: string; image?: string; caption?: string }>;
  summary?: string | SummaryItem | SummaryItem[];
  confidence?: number;
  timestamp?: string;
  stepsTitle?: string;
  summaryTitle?: string;
  images?: any[];
  stepImages?: any[];
  routing_info?: any;
}

/**
 * ChatService — production-grade client for backend /api/chat/query
 */
@Injectable({
  providedIn: 'root',
})
export class ChatService {
  // ✅ Use environment configuration for API URL
  private baseUrl: string;

  constructor(private http: HttpClient) {
    // Remove trailing slashes and ensure proper format
    this.baseUrl = (environment.backendApi || 'http://localhost:8001')
      .replace(/\/+$/, '')
      .trim();
    
    console.log('✅ ChatService initialized with baseUrl:', this.baseUrl);
  }

  /**
   * Sends a chat query request to the backend.
   * Handles both new queries and continuing sessions.
   */
  query(request: UserQueryRequest): Observable<ChatResponse> {
    const payload = {
      query: request.query.trim(),
      max_results: request.max_results ?? 20,
      include_images: request.include_images ?? true,
      session_id: request.session_id || undefined
    };

    const httpOptions = {
      headers: new HttpHeaders({
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }),
    };

    const url = `${this.baseUrl}/api/chat/query`;
    console.log('📤 Sending query to:', url, 'Payload:', payload);

    return this.http.post<ChatResponse>(url, payload, httpOptions).pipe(
      map((res) => {
        console.log('✅ Received response:', res);
        return this.normalizeResponse(res, request);
      }),
      catchError((err: HttpErrorResponse) => {
        console.error('❌ Query error:', err);
        return this.handleError(err);
      })
    );
  }

  /**
   * Continue an ongoing conversation with the action agent
   */
  continueConversation(sessionId: string, userInput: string): Observable<ChatResponse> {
    const payload: ContinueConversationRequest = {
      session_id: sessionId,
      user_input: userInput.trim()
    };

    const httpOptions = {
      headers: new HttpHeaders({
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }),
    };

    const url = `${this.baseUrl}/api/chat/continue`;
    console.log('📤 Continuing conversation at:', url, 'Payload:', payload);

    return this.http.post<ChatResponse>(url, payload, httpOptions).pipe(
      map((res) => {
        console.log('✅ Received continuation response:', res);
        return this.normalizeResponse(res, { query: userInput });
      }),
      catchError((err: HttpErrorResponse) => {
        console.error('❌ Continuation error:', err);
        return this.handleError(err);
      })
    );
  }

  /**
   * Get list of available cluster endpoints
   */
  getClusterEndpoints(): Observable<any> {
    const url = `${this.baseUrl}/api/clusters/endpoints`;
    console.log('📤 Getting cluster endpoints from:', url);

    return this.http.get(url).pipe(
      catchError((err: HttpErrorResponse) => {
        console.error('❌ Cluster endpoints error:', err);
        return this.handleError(err);
      })
    );
  }

  /**
   * Get active sessions
   */
  getActiveSessions(): Observable<any> {
    const url = `${this.baseUrl}/api/sessions`;
    console.log('📤 Getting active sessions from:', url);

    return this.http.get(url).pipe(
      catchError((err: HttpErrorResponse) => {
        console.error('❌ Sessions error:', err);
        return this.handleError(err);
      })
    );
  }

  /**
   * Clear a session
   */
  clearSession(sessionId: string): Observable<any> {
    const url = `${this.baseUrl}/api/sessions/${sessionId}`;
    console.log('📤 Clearing session at:', url);

    return this.http.delete(url).pipe(
      catchError((err: HttpErrorResponse) => {
        console.error('❌ Clear session error:', err);
        return this.handleError(err);
      })
    );
  }

  /**
   * Check service health
   */
  getServiceHealth(): Observable<any> {
    const url = `${this.baseUrl}/health`;
    console.log('📤 Checking health at:', url);

    return this.http.get(url).pipe(
      catchError((err: HttpErrorResponse) => {
        console.error('❌ Health check error:', err);
        return this.handleError(err);
      })
    );
  }

  /**
   * Normalize backend response for UI safety
   */
  private normalizeResponse(res: ChatResponse | null, request: UserQueryRequest): ChatResponse {
    if (!res) throw new Error('Empty response received from server.');

    const makeAbsolute = (url: string): string => {
      if (!url) return url;
      if (typeof url !== 'string') return url;
      
      try {
        new URL(url);
        return url;
      } catch {
        // Relative URL - make it absolute
        if (url.startsWith('/')) {
          return `${this.baseUrl}${url}`;
        }
        return `${this.baseUrl}/${url}`;
      }
    };

    // Normalize images array
    if (Array.isArray(res.images)) {
      res.images = res.images
        .map((img: any) => {
          if (typeof img === 'string') return makeAbsolute(img);
          if (img?.url) return { ...img, url: makeAbsolute(img.url) };
          return img;
        })
        .filter(img => img); // Remove nulls
    }

    // Normalize step images
    if (Array.isArray(res.stepImages)) {
      res.stepImages = res.stepImages
        .map((img: any) =>
          typeof img === 'string' ? makeAbsolute(img) : img
        )
        .filter(img => img);
    }

    // Normalize images within steps
    if (Array.isArray(res.steps)) {
      res.steps.forEach((step) => {
        if (step.image) step.image = makeAbsolute(step.image);
      });
    }

    // Ensure required fields
    res.answer = res.answer?.trim() || '';
    res.query = res.query || request.query || '';
    res.timestamp = res.timestamp || new Date().toISOString();
    res.status = res.status || 'unknown';

    return res;
  }

  /**
   * Centralized error handler with detailed diagnostics
   */
  private handleError(err: HttpErrorResponse) {
    let msg = 'An unexpected error occurred.';
    let details = '';

    if (err.error?.detail) {
      msg = err.error.detail;
    } else if (err.error?.message) {
      msg = err.error.message;
    } else if (err.status === 404) {
      msg = 'Chat endpoint not found.';
      details = `Backend URL: ${this.baseUrl}/api/chat/query`;
    } else if (err.status === 422) {
      msg = 'Invalid query format.';
      details = JSON.stringify(err.error);
    } else if (err.status === 500) {
      msg = 'Server error. Check backend logs.';
      details = err.error?.details || err.message;
    } else if (err.status === 0) {
      msg = `Cannot connect to backend server at ${this.baseUrl}`;
      details = 'Check if backend is running and CORS is configured.';
    } else {
      msg = err.message || msg;
    }

    const fullMessage = details ? `${msg}\n\n${details}` : msg;
    console.error('❌ ChatService error:', fullMessage, err);
    
    return throwError(() => new Error(fullMessage));
  }
}
import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import { Observable, map, catchError, throwError } from 'rxjs';
import { environment } from '../../environments/environments';

// ✅ Define and export shared interfaces

export interface UserQueryRequest {
  query: string;
  max_results?: number;
  include_images?: boolean;
}

export interface SummaryItem {
  title?: string;
  content: string;
  bullets?: string[];
}

export interface ChatResponse {
  query: string;
  answer: string;
  steps?: Array<{ index?: number; text: string; image?: string }>;
  summary?: string | SummaryItem | SummaryItem[];
  confidence?: number;
  timestamp?: string;
  stepsTitle?: string;
  summaryTitle?: string;
  images?: any[];
  stepImages?: any[];
}

/**
 * ChatService — production-grade client for backend /api/chat/query
 */
@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private baseUrl = environment.backendApi.replace(/\/$/, '');

  constructor(private http: HttpClient) {}

  /**
   * Sends a chat query request to the backend.
   */
  query(request: UserQueryRequest): Observable<ChatResponse> {
    const payload = {
      query: request.query,
      max_results: request.max_results ?? 10,
      include_images: request.include_images ?? true,
    };

    const httpOptions = {
      headers: new HttpHeaders({
        'Content-Type': 'application/json',
      }),
    };

    const url = `${this.baseUrl}/api/chat/query`;

    return this.http.post<ChatResponse>(url, payload, httpOptions).pipe(
      map((res) => this.normalizeResponse(res, request)),
      catchError((err: HttpErrorResponse) => this.handleError(err))
    );
  }

  /**
   * Normalize backend response for UI safety
   */
  private normalizeResponse(
    res: ChatResponse | null,
    request: UserQueryRequest
  ): ChatResponse {
    if (!res) throw new Error('Empty response received from server.');

    const makeAbsolute = (url: string): string => {
      if (!url) return url;
      try {
        new URL(url); // absolute
        return url;
      } catch {
        return `${this.baseUrl}${url.startsWith('/') ? '' : '/'}${url}`;
      }
    };

    if (Array.isArray(res.images)) {
      res.images = res.images.map((img: any) => {
        if (typeof img === 'string') return makeAbsolute(img);
        if (img && typeof img === 'object' && img.url) {
          return { ...img, url: makeAbsolute(img.url) };
        }
        return img;
      });
    }

    if (Array.isArray(res.stepImages)) {
      res.stepImages = res.stepImages.map((img: any) =>
        typeof img === 'string' ? makeAbsolute(img) : img
      );
    }

    if (Array.isArray(res.steps)) {
      res.steps.forEach((step) => {
        if (step.image) step.image = makeAbsolute(step.image);
      });
    }

    res.answer = res.answer?.trim() || 'No answer available.';
    res.query = res.query || request.query;
    res.timestamp = res.timestamp || new Date().toISOString();

    return res;
  }

  /**
   * Centralized error handler
   */
  private handleError(err: HttpErrorResponse) {
    let msg = 'An unexpected error occurred.';
    if (err.error?.detail) msg = err.error.detail;
    else if (err.status === 422) msg = 'Invalid query format.';
    else if (err.status === 0) msg = 'Cannot connect to backend server.';
    else msg = err.message || msg;

    console.error('ChatService error:', msg, err);
    return throwError(() => new Error(msg));
  }
}

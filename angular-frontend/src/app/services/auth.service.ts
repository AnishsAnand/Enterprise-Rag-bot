// auth.service.ts - COMPLETE FIXED VERSION
import { Injectable } from "@angular/core";
import { BehaviorSubject, Observable, throwError } from "rxjs";
import { catchError, tap, map } from "rxjs/operators";
import { Router } from "@angular/router";
import { HttpClient, HttpParams, HttpErrorResponse } from "@angular/common/http";
import { environment } from "../../environments/environment";

export interface User {
  id: number;
  username: string;
  email?: string;
  role: "user" | "admin";
  is_verified?: boolean;
  created_at?: string;
  last_login?: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

@Injectable({ providedIn: "root" })
export class AuthService {
  /** Use environment variable for API URL */
  private readonly API_URL = `${environment.apiUrl}/api/auth`;

  private readonly tokenKey = "token";
  private currentUserSubject = new BehaviorSubject<User | null>(null);
  public currentUser$ = this.currentUserSubject.asObservable();

  constructor(
    private http: HttpClient,
    private router: Router
  ) {
    this.restoreSession();
  }

  /** ===================== SESSION ===================== */

  private restoreSession(): void {
    const token = localStorage.getItem(this.tokenKey);
    const userRaw = localStorage.getItem("user");

    if (token && userRaw) {
      try {
        const user = JSON.parse(userRaw);
        this.currentUserSubject.next(user);
        console.log('✅ Session restored:', user.username);
      } catch (error) {
        console.error('❌ Failed to restore session:', error);
        this.logout();
      }
    }
  }

  /** ===================== AUTH ===================== */

  /**
   * ✅ FastAPI OAuth2PasswordRequestForm compatible login
   */
  login(username: string, password: string): Observable<AuthResponse> {
    console.log('🔐 Attempting login:', username);
    
    const body = new HttpParams()
      .set("username", username)
      .set("password", password);

    return this.http.post<AuthResponse>(
      `${this.API_URL}/login`,
      body.toString(),
      {
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
      }
    ).pipe(
      tap(res => {
        console.log('✅ Login successful:', res.user?.username);
        
        // Store token and user data
        localStorage.setItem(this.tokenKey, res.access_token);
        
        if (res.user) {
          localStorage.setItem("user", JSON.stringify(res.user));
          this.currentUserSubject.next(res.user);
        }
      }),
      catchError((error: HttpErrorResponse) => {
        console.error('❌ Login failed:', error);
        
        // Enhanced error handling
        let errorMessage = 'Login failed';
        
        if (error.status === 0) {
          errorMessage = 'Cannot connect to server. Please check if backend is running.';
        } else if (error.status === 401) {
          errorMessage = 'Invalid username or password';
        } else if (error.status === 403) {
          errorMessage = 'Account disabled';
        } else if (error.error?.detail) {
          errorMessage = error.error.detail;
        }
        
        return throwError(() => ({
          ...error,
          message: errorMessage
        }));
      })
    );
  }

  register(data: {
    username: string;
    email: string;
    password: string;
  }): Observable<AuthResponse> {
    console.log('📝 Attempting registration:', data.username);
    
    return this.http.post<AuthResponse>(`${this.API_URL}/register`, data).pipe(
      tap(res => {
        console.log('✅ Registration successful:', res.user?.username);
        
        // Store token and user data
        localStorage.setItem(this.tokenKey, res.access_token);
        
        if (res.user) {
          localStorage.setItem("user", JSON.stringify(res.user));
          this.currentUserSubject.next(res.user);
        }
      }),
      catchError((error: HttpErrorResponse) => {
        console.error('❌ Registration failed:', error);
        
        let errorMessage = 'Registration failed';
        
        if (error.status === 0) {
          errorMessage = 'Cannot connect to server';
        } else if (error.status === 400) {
          errorMessage = error.error?.detail || 'User already exists';
        } else if (error.error?.detail) {
          errorMessage = error.error.detail;
        }
        
        return throwError(() => ({
          ...error,
          message: errorMessage
        }));
      })
    );
  }

  logout(): void {
    console.log('🚪 Logging out user');
    localStorage.removeItem(this.tokenKey);
    localStorage.removeItem("user");
    this.currentUserSubject.next(null);
    this.router.navigateByUrl("/login");
  }

  /** ===================== USER ===================== */

  getCurrentUser(): Observable<User> {
    return this.http.get<User>(`${this.API_URL}/me`).pipe(
      tap(user => {
        // Update local user data
        localStorage.setItem("user", JSON.stringify(user));
        this.currentUserSubject.next(user);
      }),
      catchError((error: HttpErrorResponse) => {
        console.error('❌ Failed to get current user:', error);
        
        // If unauthorized, logout
        if (error.status === 401) {
          this.logout();
        }
        
        return throwError(() => error);
      })
    );
  }

  getCurrentUserValue(): User | null {
    return this.currentUserSubject.value;
  }

  isAuthenticated(): boolean {
    return !!this.currentUserSubject.value && !!this.getToken();
  }

  isAdmin(): boolean {
    return this.currentUserSubject.value?.role === "admin";
  }

  getToken(): string | null {
    return localStorage.getItem(this.tokenKey);
  }

  /** Handle social authentication callback */
  handleSocialCallback(token: string, refreshToken?: string): void {
    localStorage.setItem(this.tokenKey, token);
    
    if (refreshToken) {
      localStorage.setItem("refresh_token", refreshToken);
    }
    
    // Fetch user details
    this.getCurrentUser().subscribe({
      next: (user) => {
        console.log('✅ Social auth successful:', user.username);
      },
      error: (error) => {
        console.error('❌ Failed to fetch user after social auth:', error);
        this.logout();
      }
    });
  }

  /** Refresh token (if implementing refresh tokens) */
  refreshToken(): Observable<AuthResponse> {
    const refreshToken = localStorage.getItem("refresh_token");
    
    if (!refreshToken) {
      return throwError(() => new Error('No refresh token available'));
    }
    
    return this.http.post<AuthResponse>(
      `${this.API_URL}/refresh`,
      { refresh_token: refreshToken }
    ).pipe(
      tap(res => {
        localStorage.setItem(this.tokenKey, res.access_token);
        
        if (res.user) {
          localStorage.setItem("user", JSON.stringify(res.user));
          this.currentUserSubject.next(res.user);
        }
      }),
      catchError((error) => {
        console.error('❌ Token refresh failed:', error);
        this.logout();
        return throwError(() => error);
      })
    );
  }
}